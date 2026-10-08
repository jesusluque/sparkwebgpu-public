// The plugin host of the WebGPU splat renderer (plan-athenea.md §3): a
// registry of SplatPlugin manifests that decides, per device and per asset,
// which plugins run, in what order, and with which compiled kernels.
//
//   const host = new PluginHost({ capabilities: splats.capabilities });
//   host.register(tintPlugin()).register(fadePlugin());
//   host.attach(splats);              // the renderer now asks the host
//   host.disable("example.tint", mesh); // per-asset state
//   await host.ready();               // optional: kernel variants loaded
//
// Resolution: a plugin is active for an asset when the device's tier and
// features allow it, the asset carries the sections and attributes it
// requires, and it is not switched off (per asset, else globally).
//
// Kernels: plugin Slang is compiled offline into presets (slang/plugins/
// presets.json, tools/slang-build). generate is dispatched per mesh, so each
// mesh gets the variant of its own active colour plugins; the draw and the
// tile rasterizer run once a frame, with the variant of the blend plugins
// active on any mesh. A list with no compiled variant falls back to the
// longest compiled one inside it (the rest are reported "no-variant"); a
// variant still loading falls back to the default kernels for that frame.
// No active plugin: the default kernels, exactly as without a host.

import type { KernelModule } from "../KernelModule";
import type { WgpuSplatMesh, WgpuSplatRenderer } from "../WgpuSplatRenderer";
import type { GpuCapabilities } from "../capabilities";
import type { WgpuDyno, WgpuDynoFrame } from "../dyno/DynoKernels";
import type { FxChain, FxStep } from "../fx/FxChain";
import attribDrawModule from "../generated/draw/splat_attrib_draw";
import drawModule from "../generated/draw/splat_draw";
import generateModule from "../generated/kernels/generate";
import {
  type KernelVariant,
  blendVariants,
  colourVariants,
  pluginSlang,
} from "../generated/presets";
import tileModule from "../generated/tiles/tile_raster";
import { createUniform } from "../gpuBuffers";
import { UniformWriter } from "../uniforms";
import { hasFeature, tierOf } from "./tiers";
import type {
  ControlSpec,
  InactiveReason,
  PluginAsset,
  PluginFrame,
  PluginResolution,
  PluginUniformContext,
  SplatPlugin,
  Tier,
} from "./types";

/** The kernels a frame's draw uses, with the blend plugins' resources. */
export interface PixelKernels {
  /** Blend plugin ids joined by "+". */
  key: string;
  draw: KernelModule;
  attribDraw: KernelModule;
  tiles: KernelModule;
  /** Uniform blocks as GPU buffers (render pipelines), by block name. */
  buffers: Record<string, GPUBuffer>;
  /** The same blocks' bytes (compute dispatches: the tile rasterizer). */
  uniforms: Record<string, ArrayBuffer>;
}

/** A mesh's generate variant, with its colour plugins' resources. */
export interface GenerateKernel {
  key: string;
  module: KernelModule;
  buffers: Record<string, GPUBuffer>;
  uniforms: Record<string, ArrayBuffer>;
}

export interface PluginHostOptions {
  capabilities?: GpuCapabilities | null;
  /** Overrides the tier measured from capabilities. */
  tier?: Tier;
  /** Kernel variant tables (default: generated/presets.ts), for tests. */
  variants?: {
    colour: Record<string, KernelVariant>;
    blend: Record<string, KernelVariant>;
  };
}

export const DEFAULT_KERNELS = {
  "kernels/generate": generateModule,
  "draw/splat_draw": drawModule,
  "draw/splat_attrib_draw": attribDrawModule,
  "tiles/tile_raster": tileModule,
} as const;

interface Entry {
  plugin: SplatPlugin;
  index: number;
}

interface ManagedDyno {
  dyno: WgpuDyno;
  ids: string;
}

/** Plugins in stage order: `after`/`before` edges, then `order`, then registration. */
export function orderPlugins<T extends SplatPlugin>(
  plugins: readonly T[],
): T[] {
  const ids = new Map(plugins.map((p, i) => [p.id, i]));
  const deps = plugins.map(() => new Set<number>());
  plugins.forEach((p, i) => {
    for (const a of p.after ?? []) {
      const j = ids.get(a);
      if (j !== undefined && j !== i) deps[i].add(j);
    }
    for (const b of p.before ?? []) {
      const j = ids.get(b);
      if (j !== undefined && j !== i) deps[j].add(i);
    }
  });
  const rank = (i: number) => [plugins[i].order ?? 0, i] as const;
  const less = (a: number, b: number) => {
    const [oa, ia] = rank(a);
    const [ob, ib] = rank(b);
    return oa !== ob ? oa < ob : ia < ib;
  };
  const done = new Set<number>();
  const out: T[] = [];
  while (out.length < plugins.length) {
    let next = -1;
    for (let i = 0; i < plugins.length; i++) {
      if (done.has(i) || [...deps[i]].some((d) => !done.has(d))) continue;
      if (next < 0 || less(i, next)) next = i;
    }
    if (next < 0) {
      const left = plugins.filter((_, i) => !done.has(i)).map((p) => p.id);
      throw new Error(`plugins: ordering cycle among ${left.join(", ")}`);
    }
    done.add(next);
    out.push(plugins[next]);
  }
  return out;
}

/** Whether `sub` is `list` with some entries left out, in order. */
function isSubsequence(sub: readonly string[], list: readonly string[]) {
  let k = 0;
  for (const x of list) if (k < sub.length && sub[k] === x) k++;
  return k === sub.length;
}

/**
 * The compiled variant for `ids` (in stage order): the exact one, else the
 * longest whose plugins are a subsequence of them. "" when none fits.
 */
export function pickVariant(
  table: Record<string, KernelVariant>,
  ids: readonly string[],
): string {
  const exact = ids.join("+");
  if (ids.length === 0 || table[exact]) return ids.length ? exact : "";
  let best = "";
  let bestLength = 0;
  for (const [key, v] of Object.entries(table)) {
    if (v.plugins.length > bestLength && isSubsequence(v.plugins, ids)) {
      best = key;
      bestLength = v.plugins.length;
    }
  }
  return best;
}

export class PluginHost {
  readonly capabilities: GpuCapabilities | null;
  readonly tier: Tier;
  renderer: WgpuSplatRenderer | null = null;
  /** Bumped by every change that can alter the image (registry, state, a loaded variant). */
  version = 0;

  private entries = new Map<string, Entry>();
  private registered = 0;
  private sorted: SplatPlugin[] = [];
  private global = new Map<string, boolean>();
  private perAsset = new WeakMap<object, Map<string, boolean>>();
  private assets = new WeakMap<object, PluginAsset>();
  private variants: NonNullable<PluginHostOptions["variants"]>;
  private loaded = new Map<string, Record<string, KernelModule>>();
  private loading = new Map<string, Promise<void>>();
  private warned = new Set<string>();
  private seenVersion = -1;
  private lastFrame: PluginFrame | null = null;
  private dynos = new WeakMap<WgpuSplatMesh, ManagedDyno>();
  private pixelBuffers = new Map<string, GPUBuffer>();
  private fxSteps = new Map<string, FxStep[]>();

  constructor(options: PluginHostOptions = {}) {
    this.capabilities = options.capabilities ?? null;
    this.tier = options.tier ?? tierOf(this.capabilities);
    this.variants = options.variants ?? {
      colour: colourVariants,
      blend: blendVariants,
    };
  }

  // Registry

  /** Adds a plugin; throws on a repeated id, mismatched Slang or an ordering cycle. */
  register(plugin: SplatPlugin): this {
    if (this.entries.has(plugin.id)) {
      throw new Error(`PluginHost: '${plugin.id}' is already registered`);
    }
    this.checkSlang(plugin);
    const entries = new Map(this.entries);
    entries.set(plugin.id, { plugin, index: this.registered });
    // Throws before anything changes if the order has a cycle.
    this.sorted = orderPlugins(
      [...entries.values()]
        .sort((a, b) => a.index - b.index)
        .map((e) => e.plugin),
    );
    this.entries = entries;
    this.registered += 1;
    this.changed();
    if (this.renderer) plugin.attach?.(this.renderer);
    return this;
  }

  unregister(id: string): boolean {
    const entry = this.entries.get(id);
    if (!entry) return false;
    if (this.renderer) entry.plugin.detach?.(this.renderer);
    this.entries.delete(id);
    this.sorted = this.sorted.filter((p) => p.id !== id);
    this.fxSteps.delete(id);
    this.changed();
    return true;
  }

  get(id: string): SplatPlugin | undefined {
    return this.entries.get(id)?.plugin;
  }

  /** Registered plugins in stage order. */
  get plugins(): readonly SplatPlugin[] {
    return this.sorted;
  }

  /** Whether an enabled plugin makes linear light past 1 (SplatPlugin.hdr). */
  get hdr(): boolean {
    return this.sorted.some(
      (p) =>
        p.hdr &&
        !this.deviceReason(p) &&
        (this.global.get(p.id) ?? p.enabled ?? true),
    );
  }

  // A plugin naming Slang must match what slang-build compiled for its id.
  private checkSlang(plugin: SplatPlugin) {
    const s = plugin.slang;
    if (!s) return;
    const compiled = pluginSlang[plugin.id];
    if (!compiled) {
      throw new Error(
        `PluginHost: '${plugin.id}' has Slang but no preset compiles it (slang/plugins/presets.json)`,
      );
    }
    if (
      compiled.module !== s.module ||
      compiled.colour !== (s.colour ?? null) ||
      compiled.blend !== (s.blend ?? null)
    ) {
      throw new Error(
        `PluginHost: '${plugin.id}' Slang differs from presets.json (${JSON.stringify(compiled)}); run npm run build:slang`,
      );
    }
  }

  // Assets and active state

  /** What `asset` carries, for requirement checks (else read from a mesh's source). */
  setAsset(asset: object, info: PluginAsset) {
    this.assets.set(asset, info);
    this.changed();
  }

  /** The asset's sections and attributes. */
  assetInfo(asset: object): PluginAsset {
    const info = this.assets.get(asset);
    if (info) return info;
    const pool = (asset as Partial<WgpuSplatMesh>).source?.attribs;
    return { attributes: pool?.columns.map((c) => c.spec.name) ?? [] };
  }

  /**
   * Switches a plugin on or off for `asset` (a drawn WgpuSplatMesh, or the
   * object it draws: a SplatMesh), or for all assets without their own
   * setting.
   */
  setActive(id: string, active: boolean, asset?: object) {
    let map = this.global;
    if (asset) {
      map = this.perAsset.get(asset) ?? new Map();
      this.perAsset.set(asset, map);
    }
    if (map.get(id) === active) return;
    map.set(id, active);
    this.changed();
  }

  enable(id: string, asset?: object) {
    this.setActive(id, true, asset);
  }

  disable(id: string, asset?: object) {
    this.setActive(id, false, asset);
  }

  /** Drops a setting, so `asset` (or every asset) follows the default again. */
  clearActive(id: string, asset?: object) {
    const map = asset ? this.perAsset.get(asset) : this.global;
    if (map?.delete(id)) this.changed();
  }

  /** Why the device rules `plugin` out, or null. */
  private deviceReason(
    plugin: SplatPlugin,
  ): { reason: InactiveReason; detail?: string } | null {
    const minTier = plugin.minTier ?? 1;
    if (this.tier < minTier) {
      return {
        reason: "tier",
        detail: `needs T${minTier}, have T${this.tier}`,
      };
    }
    const missing = (plugin.requires?.features ?? []).filter(
      (f) => !hasFeature(this.capabilities, f),
    );
    if (missing.length) return { reason: "features", detail: missing.join() };
    return null;
  }

  /** Why `plugin` is not active for `asset` (or globally), or null when it is. */
  private reason(
    plugin: SplatPlugin,
    asset?: object,
  ): { reason: InactiveReason; detail?: string } | null {
    const device = this.deviceReason(plugin);
    if (device) return device;
    // A setting for the drawn mesh, or for its object (the SplatMesh a page
    // holds: SparkWebGPU may make a new WgpuSplatMesh for the same one).
    const object = asset && (asset as Partial<WgpuSplatMesh>).object;
    const on =
      (asset && this.perAsset.get(asset)?.get(plugin.id)) ??
      (object && this.perAsset.get(object)?.get(plugin.id)) ??
      this.global.get(plugin.id) ??
      plugin.enabled ??
      true;
    if (!on) return { reason: "disabled" };
    if (asset) {
      const info = this.assetInfo(asset);
      const lack = (need?: readonly string[], have?: readonly string[]) =>
        (need ?? []).filter((n) => !(have ?? []).includes(n));
      const attributes = lack(plugin.requires?.attributes, info.attributes);
      if (attributes.length) {
        return { reason: "attributes", detail: attributes.join() };
      }
      const sections = lack(plugin.requires?.sections, info.sections);
      if (sections.length)
        return { reason: "sections", detail: sections.join() };
    }
    return null;
  }

  /**
   * The attributes the registered plugins this device runs (and not
   * switched off for every asset) require or read: what a pager that loads
   * streams on demand loads (WgpuSplatPager, attribPaging.ts). Asset
   * requirements are not checked: an asset carries an attribute only once
   * it is loaded.
   */
  attributeDemand(): string[] {
    const names = new Set<string>();
    for (const plugin of this.sorted) {
      if (this.deviceReason(plugin)) continue;
      const on = this.global.get(plugin.id) ?? plugin.enabled ?? true;
      if (!on) continue;
      for (const n of plugin.requires?.attributes ?? []) names.add(n);
      for (const n of plugin.requires?.reads ?? []) names.add(n);
    }
    return [...names];
  }

  isActive(id: string, asset?: object): boolean {
    const plugin = this.get(id);
    return Boolean(plugin && !this.reason(plugin, asset));
  }

  /** Active plugins for `asset` in stage order, and why the rest are not. */
  resolve(asset?: object): PluginResolution {
    const active: SplatPlugin[] = [];
    const inactive: PluginResolution["inactive"] = [];
    for (const plugin of this.sorted) {
      const r = this.reason(plugin, asset);
      if (r) inactive.push({ id: plugin.id, ...r });
      else active.push(plugin);
    }
    const wanted = active.filter((p) => p.slang?.colour).map((p) => p.id);
    const colour = pickVariant(this.variants.colour, wanted);
    for (const id of wanted) {
      if (!colour.split("+").includes(id)) {
        inactive.push({ id, reason: "no-variant", detail: "colour" });
        this.warnOnce(`colour:${wanted.join("+")}`);
      }
    }
    return {
      tier: this.tier,
      active: active.filter(
        (p) => !p.slang?.colour || colour.split("+").includes(p.id),
      ),
      inactive,
      colour,
    };
  }

  /** The blend variant for the assets drawn together (active on any of them). */
  blendKey(assets: readonly object[]): string {
    const wanted = this.sorted
      .filter(
        (p) =>
          p.slang?.blend &&
          (p.blendActive?.() ?? true) &&
          (assets.length === 0
            ? !this.reason(p)
            : assets.some((a) => !this.reason(p, a))),
      )
      .map((p) => p.id);
    const key = pickVariant(this.variants.blend, wanted);
    if (key !== wanted.join("+")) this.warnOnce(`blend:${wanted.join("+")}`);
    return key;
  }

  private warnOnce(what: string) {
    if (this.warned.has(what)) return;
    this.warned.add(what);
    console.warn(
      `PluginHost: no compiled variant for ${what}; add a preset to slang/plugins/presets.json`,
    );
  }

  private changed() {
    this.version += 1;
  }

  // Kernel variants

  /** The variant's modules, loading it if needed (null until loaded). */
  private variant(
    kind: "colour" | "blend",
    key: string,
  ): Record<string, KernelModule> | null {
    const id = `${kind}:${key}`;
    const modules = this.loaded.get(id);
    if (modules) return modules;
    this.load(kind, key);
    return null;
  }

  private load(kind: "colour" | "blend", key: string): Promise<void> {
    const id = `${kind}:${key}`;
    if (this.loaded.has(id)) return Promise.resolve();
    let p = this.loading.get(id);
    if (!p) {
      const table =
        kind === "colour" ? this.variants.colour : this.variants.blend;
      const v = table[key];
      if (!v)
        return Promise.reject(
          new Error(`PluginHost: no ${kind} variant '${key}'`),
        );
      p = v.load().then((modules) => {
        this.loaded.set(id, modules);
        this.loading.delete(id);
        this.changed();
      });
      this.loading.set(id, p);
    }
    return p;
  }

  /**
   * Loads the variants the attached renderer's meshes (or `assets`) need
   * now, so no frame falls back to the default kernels while they load.
   */
  async ready(assets?: readonly object[]): Promise<void> {
    const list = assets ?? this.renderer?.meshes ?? [];
    const jobs: Promise<void>[] = [];
    for (const a of list.length ? list : [undefined]) {
      const key = this.resolve(a).colour;
      if (key) jobs.push(this.load("colour", key));
    }
    const blend = this.blendKey(list);
    if (blend) jobs.push(this.load("blend", blend));
    await Promise.all(jobs);
  }

  // Renderer integration

  /** Makes `renderer` consult this host (renderer.plugins) and attaches the plugins. */
  attach(renderer: WgpuSplatRenderer): this {
    if (this.renderer === renderer) return this;
    this.detach();
    this.renderer = renderer;
    renderer.plugins = this;
    for (const p of this.sorted) p.attach?.(renderer);
    this.changed();
    return this;
  }

  detach() {
    const r = this.renderer;
    if (!r) return;
    for (const p of this.sorted) p.detach?.(r);
    for (const mesh of r.meshes) {
      const managed = this.dynos.get(mesh);
      if (managed && mesh.dyno === managed.dyno) mesh.dyno = undefined;
      this.dynos.delete(mesh);
    }
    if (r.plugins === this) r.plugins = null;
    this.renderer = null;
    for (const b of this.pixelBuffers.values()) b.destroy();
    this.pixelBuffers.clear();
  }

  /**
   * Once a frame, before generate: plugins' onFrame, their dyno graphs onto
   * the meshes. True when the splats must be regenerated.
   */
  frame(frame: PluginFrame): boolean {
    this.lastFrame = frame;
    let dirty = false;
    for (const p of this.sorted) {
      if (this.deviceReason(p)) continue;
      if (p.onFrame?.(frame) === "dirty") dirty = true;
    }
    this.syncDynos(frame.renderer.meshes);
    if (this.version !== this.seenVersion) {
      this.seenVersion = this.version;
      dirty = true;
    }
    return dirty;
  }

  /** The active plugins' pre-passes, in stage order (after frame()). */
  encodePasses(encoder: GPUCommandEncoder) {
    const frame = this.lastFrame;
    if (!frame) return;
    for (const p of this.sorted) {
      if (!p.passes?.length || this.reason(p)) continue;
      for (const pass of p.passes) pass.encode(encoder, frame);
    }
  }

  // Each mesh's dyno: the graphs of its active dyno plugins, chained in stage
  // order. Meshes with a dyno of their own (renderer.add(..., dyno)) keep it.
  private syncDynos(meshes: readonly WgpuSplatMesh[]) {
    for (const mesh of meshes) {
      const managed = this.dynos.get(mesh);
      if (mesh.dyno && mesh.dyno !== managed?.dyno) continue;
      const plugins = this.sorted.filter(
        (p) => p.dyno && !this.reason(p, mesh),
      );
      if (plugins.length === 0) {
        if (managed) {
          mesh.dyno = undefined;
          this.dynos.delete(mesh);
        }
        continue;
      }
      const dynos = plugins.map((p) => p.dyno as WgpuDyno);
      const ids = plugins.map((p) => p.id).join("+");
      const dyno: WgpuDyno = managed?.dyno ?? {};
      dyno.generator = dynos.find((d) => d.generator)?.generator;
      dyno.objectModifiers = dynos.flatMap((d) => d.objectModifiers ?? []);
      dyno.worldModifiers = dynos.flatMap((d) => d.worldModifiers ?? []);
      dyno.covObjectModifiers = dynos.flatMap(
        (d) => d.covObjectModifiers ?? [],
      );
      dyno.covWorldModifiers = dynos.flatMap((d) => d.covWorldModifiers ?? []);
      dyno.worldSpace = dynos.some((d) => d.worldSpace);
      dyno.update = (f: WgpuDynoFrame) => {
        for (const d of dynos) d.update?.(f);
      };
      if (!managed || managed.ids !== ids) this.dynos.set(mesh, { dyno, ids });
      mesh.dyno = dyno;
    }
  }

  /** `mesh`'s generate variant and its colour plugins' resources; null: the default kernel. */
  generateKernel(mesh: WgpuSplatMesh): GenerateKernel | null {
    const { colour, active } = this.resolve(mesh);
    if (!colour) return null;
    const modules = this.variant("colour", colour);
    if (!modules) return null;
    const module = modules["kernels/generate"];
    const ctx: PluginUniformContext = {
      frame: this.lastFrame,
      mesh,
      asset: mesh,
    };
    const uniforms: Record<string, ArrayBuffer> = {};
    const buffers: Record<string, GPUBuffer> = {};
    for (const p of active) {
      if (!p.slang?.colour) continue;
      for (const u of p.uniforms ?? []) {
        if (u.stage !== "splat") continue;
        const w = UniformWriter.for(module, u.block);
        u.write(w, ctx);
        uniforms[u.block] = w.data;
      }
      Object.assign(buffers, p.buffers?.("splat", ctx));
    }
    return { key: colour, module, buffers, uniforms };
  }

  /** The frame's draw and tile kernels with the blend plugins' resources; null: the defaults. */
  pixelKernels(device: GPUDevice): PixelKernels | null {
    const meshes = this.renderer?.meshes ?? [];
    const key = this.blendKey(meshes);
    if (!key) return null;
    const modules = this.variant("blend", key);
    if (!modules) return null;
    const draw = modules["draw/splat_draw"];
    const ctx: PluginUniformContext = { frame: this.lastFrame };
    const buffers: Record<string, GPUBuffer> = {};
    const uniforms: Record<string, ArrayBuffer> = {};
    for (const id of key.split("+")) {
      const p = this.get(id) as SplatPlugin;
      for (const u of p.uniforms ?? []) {
        if (u.stage !== "pixel") continue;
        const w = UniformWriter.for(draw, u.block);
        u.write(w, ctx);
        uniforms[u.block] = w.data;
        let buffer = this.pixelBuffers.get(u.block);
        if (!buffer || buffer.size !== w.data.byteLength) {
          buffer?.destroy();
          buffer = createUniform(device, w.data.byteLength, u.block);
          this.pixelBuffers.set(u.block, buffer);
        }
        device.queue.writeBuffer(buffer, 0, w.data);
        buffers[u.block] = buffer;
      }
      Object.assign(buffers, p.buffers?.("pixel", ctx));
    }
    return {
      key,
      draw,
      attribDraw: modules["draw/splat_attrib_draw"],
      tiles: modules["tiles/tile_raster"],
      buffers,
      uniforms,
    };
  }

  // Post effects and UI

  /**
   * Puts the active plugins' effects at the end of `chain`, in stage order,
   * after the chain's own steps (which stay as they are).
   */
  applyFx(chain: FxChain) {
    const ours = new Set([...this.fxSteps.values()].flat());
    chain.steps = chain.steps.filter((s) => !ours.has(s));
    for (const p of this.sorted) {
      if (!p.fx?.length || this.reason(p)) continue;
      let steps = this.fxSteps.get(p.id);
      if (!steps) {
        steps = p.fx.map((fx, k) => ({
          effect: fx.effect,
          // The plugin's own object, so it can change a parameter live.
          params: fx.params ?? {},
          name: fx.name,
          instance: `${p.id}.${fx.name ?? k}`,
        }));
        this.fxSteps.set(p.id, steps);
      }
      chain.steps.push(...steps);
    }
  }

  /** The active plugins' controls, tagged with their plugin. */
  controls(): (ControlSpec & { plugin: string })[] {
    return this.sorted
      .filter((p) => p.ui?.length && !this.reason(p))
      .flatMap((p) => (p.ui ?? []).map((c) => ({ ...c, plugin: p.id })));
  }
}

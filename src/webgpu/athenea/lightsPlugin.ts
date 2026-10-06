// athenea's light groups on the PluginHost (plan-athenea.md phase 6,
// relighting stage 2; proposals 062, 066, 067, 069, 076): a car's lamps
// switched, dimmed and sequenced over its relit cloud without touching any
// baked data -- only the weights w_k change.
//
//   const lights = atheneaLightsPlugin({ sidecar: await (await fetch(usda)).text() });
//   lights.setLights(car, await decodeAthl(athlBytes), { cloudHash });
//   // or paged: each page's layers fetched by range as the cloud's page arrives
//   lights.setLights(car, await openAthl(urlRange(url)), { pageChunks: () => athlPagerChunks(pager, paged) });
//   host.register(relight).register(lights).register(atheneaOutputPlugin());
//   lights.setLightState("noche_ciudad");
//   lights.playSequence("bienvenida");
//
// Two halves, as 066 ships them beside the cloud:
//   - the .lights.usda (lightSidecar.ts): groups, states, rules, sequences,
//     lamp curves; a LightRig evaluates w_k(t) once a frame on the CPU, on
//     the frame's clock;
//   - the .athl (athl.ts): per group, its emitter polygons, lens profile and
//     sparse layers, one storage buffer per asset.
// Per splat, in generate after the relight (or raster) colour
// (slang/athenea_adapter/lights.slang):
//   colour = base + sum_k w_k (direct_k profile_k + layer_k)
// A frame whose weights did not change is not regenerated (onFrame).
//
// The .athl's groups are matched to the sidecar's by name. Its layers are
// per unit of the radiance they were baked for (AthlGroup.radiance), so
// w_k = the sidecar's radiance x level x colour / nitsPerUnit / that.
// A sidecar baked for another cloud (066 §5.3) is refused: setLights with
// the cloud's hash (athcCloudHash of its first 4096 bytes) checks both.

import * as THREE from "three";
import type { WgpuSplatMesh } from "../WgpuSplatRenderer";
import { ATTRIB_NONE, type AttribPool } from "../attributes/schema";
import { atheneaAdapterLights as C } from "../generated/constants";
import { upload } from "../gpuBuffers";
import type { ControlSpec, PluginFrame, SplatPlugin } from "../plugins/types";
import type { UniformWriter } from "../uniforms";
import { ATHL_NONE, type AthlData, AthlPager, AthlStore } from "./athl";
import {
  LightRig,
  type LightSidecar,
  type LightWeights,
  parseLightSidecar,
} from "./lightSidecar";
import { ATHENEA_RASTER_ID } from "./rasterPlugin";
import { ATHENEA_RELIGHT_ID } from "./relightPlugin";

export const ATHENEA_LIGHTS_ID = "athenea.lights";

type Rgb = [number, number, number];

export interface AtheneaLightTerms {
  /** The emitter polygons' direct light (LTC, 060) through the lens (076). */
  direct?: boolean;
  /** The baked light the group sends on after bouncing (062's layer). */
  indirect?: boolean;
  /** The emitter's own glow. */
  emission?: boolean;
  /** The view-dependent reflected field. */
  field?: boolean;
}

export interface AtheneaLightsOptions {
  /** The .lights.usda, as text or read. */
  sidecar?: LightSidecar | string;
  terms?: AtheneaLightTerms;
  /** A dimmer over every group. */
  master?: number;
}

export interface AtheneaLightsAssetOptions {
  /**
   * The cloud's hash (athcCloudHash of its first 4096 bytes): checked
   * against the .athl's and the sidecar's; a mismatch refuses the lights.
   */
  cloudHash?: string;
  /**
   * For a paged cloud: each pool page's .athl chunk (ATHL_NONE when the
   * page holds none of this cloud), e.g. athlPagerChunks(pager, paged).
   */
  pageChunks?: () => ArrayLike<number>;
}

export interface LightGroupState {
  name: string;
  /** The curve's output (0 off, 1 on; a blinker's square wave). */
  level: number;
  /** Where it is going, after the state and the rules. */
  target: number;
  /** w_k, linear RGB in scene units. */
  weight: Rgb;
  /** Has baked data for some asset (its .athl). */
  baked: boolean;
}

export interface AtheneaLightsPlugin extends SplatPlugin {
  readonly rig: LightRig | null;
  readonly sidecar: LightSidecar | null;
  /** The frame clock the commands are stamped with (seconds). */
  readonly time: number;
  setSidecar(sidecar: LightSidecar | string): void;
  /**
   * `asset`'s .athl (a SplatMesh or WgpuSplatMesh); null removes it. An
   * AthlPager (openAthl) fetches the layers of the chunks the cloud holds:
   * its pages' (`pageChunks`) or, for a whole cloud, all of them.
   */
  setLights(
    asset: object,
    athl: AthlData | AthlStore | AthlPager | null,
    options?: AtheneaLightsAssetOptions,
  ): void;
  storeOf(asset: object): AthlStore | null;
  /** Whether `asset`'s stored colours are linear light (else sRGB). */
  setStoredLinear(asset: object, linear: boolean): void;
  /** 066's named states: "aparcado", "diurno", "noche_ciudad", "frenando"... */
  setLightState(name: string | null): void;
  /** A group's target by hand (null: back to the state's). */
  setGroupLevel(name: string, level: number | null): void;
  playSequence(name: string): void;
  stopSequence(): void;
  setTerms(terms: AtheneaLightTerms): void;
  readonly terms: Required<AtheneaLightTerms>;
  master: number;
  /** Every group now, for a UI. */
  readonly lightGroups: LightGroupState[];
  /** Evaluates w_k at `time` (the frame does this itself). */
  evaluate(time: number): LightWeights | null;
  readonly stats: { evaluations: number; changes: number };
}

interface AssetLights {
  store: AthlStore;
  pager: AthlPager | null;
  options: AtheneaLightsAssetOptions;
  gpu: { buffer: GPUBuffer; version: number } | null;
  refused: string | null;
}

interface PoolOnGpu {
  version: number;
  buffer: GPUBuffer;
  normalId: number;
  owned: boolean;
}

function keysOf(mesh: WgpuSplatMesh | undefined): object[] {
  if (!mesh) return [];
  return mesh.object ? [mesh, mesh.object] : [mesh];
}

function pagedOf(mesh: WgpuSplatMesh): {
  athcStoredLinear?: boolean;
} | null {
  return (
    (mesh.object as { paged?: { athcStoredLinear?: boolean } })?.paged ?? null
  );
}

/**
 * A paged cloud's pool pages to .athl chunks: the pager's page -> (splats,
 * chunk) table, for the pages of `splats`.
 */
export function athlPagerChunks(
  pager: {
    pageToSplatsChunk: ({ splats: unknown; chunk: number } | undefined)[];
  },
  splats: unknown,
): Uint32Array {
  const out = new Uint32Array(C.kAtheneaLightsMaxPages).fill(ATHL_NONE);
  pager.pageToSplatsChunk.forEach((e, page) => {
    if (e && e.splats === splats && page < out.length) out[page] = e.chunk;
  });
  return out;
}

export function atheneaLightsPlugin(
  initial: AtheneaLightsOptions = {},
): AtheneaLightsPlugin {
  const assets = new WeakMap<object, AssetLights>();
  const paging = new Set<AssetLights>();
  const linear = new WeakMap<object, boolean>();
  const pools = new WeakMap<AttribPool, PoolOnGpu>();
  const warned = new Set<string>();
  const stats = { evaluations: 0, changes: 0 };
  const terms: Required<AtheneaLightTerms> = {
    direct: true,
    indirect: true,
    emission: true,
    field: true,
    ...initial.terms,
  };
  let sidecar: LightSidecar | null = null;
  let rig: LightRig | null = null;
  let device: GPUDevice | null = null;
  let empty: GPUBuffer | null = null;
  let now = 0;
  let weights: LightWeights | null = null;
  let previous = "";
  let dirty = true;
  let master = initial.master ?? 1;
  const baked = new Set<string>();

  const warnOnce = (key: string, message: string) => {
    if (warned.has(key)) return;
    warned.add(key);
    console.warn(`atheneaLights: ${message}`);
  };

  const lookup = <T>(map: WeakMap<object, T>, mesh?: WgpuSplatMesh) => {
    for (const k of keysOf(mesh)) {
      const v = map.get(k);
      if (v !== undefined) return v;
    }
    return undefined;
  };

  const check = (a: AssetLights) => {
    const athlHash = a.store.header.cloudHash;
    const want = a.options.cloudHash;
    const zero = "0000000000000000";
    a.refused = null;
    if (want && athlHash && athlHash !== zero && want !== athlHash) {
      a.refused = `the .athl was baked for cloud ${athlHash}, not ${want}`;
    } else if (
      sidecar?.cloudHash &&
      athlHash &&
      athlHash !== zero &&
      sidecar.cloudHash !== athlHash
    ) {
      a.refused = `the .lights.usda is for cloud ${sidecar.cloudHash}, the .athl for ${athlHash}`;
    } else if (want && sidecar?.cloudHash && sidecar.cloudHash !== want) {
      a.refused = `the .lights.usda is for cloud ${sidecar.cloudHash}, not ${want}`;
    }
    if (a.refused) console.warn(`atheneaLights: refused: ${a.refused}`);
    if (sidecar) {
      const names = new Set(sidecar.groups.map((g) => g.name));
      for (const g of a.store.groups) {
        if (!names.has(g.name)) {
          warnOnce(
            `athl:${g.name}`,
            `.athl group '${g.name}' is not in the sidecar (kept off)`,
          );
        }
      }
    }
  };

  const evaluate = (time: number): LightWeights | null => {
    if (!rig) return null;
    rig.master = master;
    weights = rig.evaluate(time, weights ?? undefined);
    stats.evaluations += 1;
    const key = weights.weights.join(",");
    if (key !== previous) {
      previous = key;
      stats.changes += 1;
      dirty = true;
    }
    return weights;
  };

  /** w for the .athl's group (by name), per unit of its baked radiance. */
  const weightOf = (name: string, bakedRadiance: number): Rgb => {
    if (!rig || !weights) return [0, 0, 0];
    const k = rig.groupNames.indexOf(name);
    if (k < 0) return [0, 0, 0];
    const s = 1 / (bakedRadiance > 0 ? bakedRadiance : 1);
    const w = weights.weights;
    return [w[4 * k] * s, w[4 * k + 1] * s, w[4 * k + 2] * s];
  };

  const poolOf = (mesh?: WgpuSplatMesh): PoolOnGpu | null => {
    const pool = mesh?.source.attribs;
    if (!pool || !device) return null;
    if (pool.gpuBuffer) {
      const k = pool.id("normalOct");
      return {
        version: pool.version,
        buffer: pool.gpuBuffer,
        normalId: k < 0 ? ATTRIB_NONE : k,
        owned: false,
      };
    }
    let g = pools.get(pool);
    if (!g || g.version !== pool.version) {
      g?.buffer.destroy();
      const { layout, words } = pool.pack((s) => s.name === "normalOct");
      const k = layout.specs.findIndex((s) => s.name === "normalOct");
      g = {
        version: pool.version,
        buffer: upload(device, words, "athenea lights attribs"),
        normalId: k < 0 ? ATTRIB_NONE : k,
        owned: true,
      };
      pools.set(pool, g);
    }
    return g;
  };

  const lightsOf = (mesh?: WgpuSplatMesh): AssetLights | null => {
    const a = lookup(assets, mesh);
    if (!a || !device) return null;
    if (!a.gpu || a.gpu.version !== a.store.version) {
      a.gpu?.buffer.destroy();
      a.gpu = {
        buffer: upload(device, a.store.pack(), "athenea lights athl"),
        version: a.store.version,
      };
    }
    return a;
  };

  /** How the mesh's indices map to the .athl's; null when they do not. */
  const layoutOf = (
    mesh: WgpuSplatMesh,
    a: AssetLights,
  ): { layout: number; merged: number; splatBase: number } | null => {
    const h = a.store.header;
    if (pagedOf(mesh)) {
      if (!a.options.pageChunks) {
        warnOnce("paged", "a paged cloud needs setLights(..., { pageChunks })");
        return null;
      }
      return { layout: C.kLightsPaged, merged: 0, splatBase: 0 };
    }
    const n = mesh.source.count;
    if (n === h.merged + h.splatCount) {
      return {
        layout: C.kLightsWhole,
        merged: h.merged,
        splatBase: h.splatBase,
      };
    }
    if (n === h.elementCount) {
      return { layout: C.kLightsWhole, merged: 0, splatBase: 0 };
    }
    warnOnce(
      `count:${n}`,
      `a mesh of ${n} splats does not match the .athl (${h.merged} merged + ${h.splatCount} splats)`,
    );
    return null;
  };

  const writeArrays = (
    w: UniformWriter,
    groups: Rgb[],
    pages: ArrayLike<number> | null,
  ) => {
    const view = new DataView(w.data);
    const field = (name: string) => {
      const f = w.block.fields?.find((x) => x.name === name);
      if (!f) throw new Error(`atheneaLights: no ${name} field`);
      return f.offset;
    };
    const wAt = field("weights");
    for (let k = 0; k < C.kAtheneaLightsMaxGroups; k++) {
      const c = groups[k] ?? [0, 0, 0];
      for (let i = 0; i < 4; i++) {
        view.setFloat32(wAt + 16 * k + 4 * i, i < 3 ? c[i] : 0, true);
      }
    }
    const pAt = field("pageChunk");
    for (let p = 0; p < C.kAtheneaLightsMaxPages; p++) {
      view.setUint32(
        pAt + 4 * p,
        pages ? (pages[p] ?? ATHL_NONE) : ATHL_NONE,
        true,
      );
    }
  };

  const groupStates = (): LightGroupState[] => {
    if (!rig) return [];
    const w = weights;
    return rig.groupNames.map((name, k) => ({
      name,
      level: w ? w.weights[4 * k + 3] : 0,
      target: w ? w.targets[k] : 0,
      weight: w
        ? [w.weights[4 * k], w.weights[4 * k + 1], w.weights[4 * k + 2]]
        : [0, 0, 0],
      baked: baked.has(name),
    }));
  };

  const termBits = () =>
    (terms.direct ? C.kLightsDirect : 0) |
    (terms.indirect ? C.kLightsIndirect : 0) |
    (terms.emission ? C.kLightsEmission : 0) |
    (terms.field ? C.kLightsField : 0);

  const ui: ControlSpec[] = [];
  const rebuildUi = () => {
    ui.length = 0;
    if (!rig) return;
    const r = rig;
    ui.push({
      id: "state",
      label: "Light state",
      type: "select",
      options: r.stateNames,
      get: () => r.currentState ?? "",
      set: (v) => plugin.setLightState(String(v)),
    });
    if (r.sequenceNames.length) {
      ui.push({
        id: "sequence",
        label: "Sequence",
        type: "select",
        options: ["", ...r.sequenceNames],
        get: () => r.playing ?? "",
        set: (v) =>
          v ? plugin.playSequence(String(v)) : plugin.stopSequence(),
      });
    }
    for (const name of r.groupNames) {
      ui.push({
        id: `group:${name}`,
        label: name,
        type: "number",
        min: 0,
        max: 4,
        step: 0.05,
        get: () => groupStates().find((g) => g.name === name)?.target ?? 0,
        set: (v) => plugin.setGroupLevel(name, Number(v)),
      });
    }
    ui.push({
      id: "master",
      label: "Lights dimmer",
      type: "number",
      min: 0,
      max: 4,
      step: 0.05,
      get: () => master,
      set: (v) => {
        plugin.master = Number(v);
      },
    });
    for (const t of ["direct", "indirect", "emission", "field"] as const) {
      ui.push({
        id: `term:${t}`,
        label: `${t} term`,
        type: "boolean",
        get: () => terms[t],
        set: (v) => plugin.setTerms({ [t]: Boolean(v) }),
      });
    }
    ui.push({
      id: "reset",
      label: "Clear manual levels",
      type: "action",
      set: () => {
        r.clearLevels(now);
        dirty = true;
      },
    });
  };

  const plugin: AtheneaLightsPlugin = {
    id: ATHENEA_LIGHTS_ID,
    minTier: 1,
    after: [ATHENEA_RELIGHT_ID, ATHENEA_RASTER_ID],
    requires: { reads: ["normalOct"] },
    slang: {
      module: "athenea_adapter.lights",
      colour: "AtheneaLightsColour",
    },
    stats,
    ui,
    get rig() {
      return rig;
    },
    get sidecar() {
      return sidecar;
    },
    get time() {
      return now;
    },
    get terms() {
      return terms;
    },
    get master() {
      return master;
    },
    set master(v: number) {
      master = v;
      evaluate(now);
    },
    get lightGroups() {
      return groupStates();
    },
    uniforms: [
      {
        block: "atheneaLights",
        stage: "splat",
        write(w, ctx) {
          const mesh = ctx.mesh;
          const a = mesh ? lightsOf(mesh) : null;
          const where = a && mesh && !a.refused ? layoutOf(mesh, a) : null;
          if (!a || !mesh || !where || !rig) {
            w.setAll({ groupCount: 0 });
            writeArrays(w, [], null);
            return;
          }
          const pool = poolOf(mesh);
          w.setAll({
            groupCount: a.store.groups.length,
            terms: termBits(),
            normalId: pool?.normalId ?? ATTRIB_NONE,
            storedLinear:
              lookup(linear, mesh) ?? pagedOf(mesh)?.athcStoredLinear ?? false,
            layout: where.layout,
            merged: where.merged,
            splatBase: where.splatBase,
          });
          writeArrays(
            w,
            a.store.groups.map((g) => weightOf(g.name, g.radiance)),
            where.layout === C.kLightsPaged
              ? (a.options.pageChunks?.() ?? null)
              : null,
          );
        },
      },
    ],
    buffers(stage, ctx): Record<string, GPUBuffer> {
      if (stage !== "splat" || !device) return {};
      empty ??= upload(
        device,
        new Uint32Array([0, 0, 4, 0]),
        "athenea lights empty",
      );
      const a = lightsOf(ctx.mesh);
      const pool = poolOf(ctx.mesh);
      return {
        atheneaLightData: a?.gpu?.buffer ?? empty,
        atheneaLightsPool: pool?.buffer ?? empty,
      };
    },
    attach(renderer) {
      device = renderer.device;
      dirty = true;
    },
    detach() {
      empty?.destroy();
      empty = null;
      device = null;
    },
    onFrame(frame: PluginFrame) {
      now = frame.time;
      for (const a of paging) {
        const chunks = a.options.pageChunks?.();
        const all = Array.from({ length: a.store.chunkCount }, (_, k) => k);
        (a.pager as AthlPager).update(
          chunks
            ? [...new Set(Array.from(chunks))].filter((c) => c !== ATHL_NONE)
            : all,
        );
      }
      evaluate(now);
      const was = dirty;
      dirty = false;
      return was ? "dirty" : "clean";
    },
    evaluate,
    setSidecar(s) {
      sidecar = typeof s === "string" ? parseLightSidecar(s) : s;
      for (const m of sidecar.warnings) console.warn(`atheneaLights: ${m}`);
      rig = new LightRig(sidecar, now);
      weights = null;
      previous = "";
      rebuildUi();
      evaluate(now);
      dirty = true;
    },
    setLights(asset, athl, options = {}) {
      const old = assets.get(asset);
      if (old) {
        old.gpu?.buffer.destroy();
        paging.delete(old);
        if (old.pager) old.pager.onChange = null;
      }
      if (!athl) {
        assets.delete(asset);
        dirty = true;
        return;
      }
      const pager = athl instanceof AthlPager ? athl : null;
      const store = pager
        ? pager.store
        : athl instanceof AthlStore
          ? athl
          : new AthlStore(athl as AthlData);
      const a: AssetLights = {
        store,
        pager,
        options,
        gpu: null,
        refused: null,
      };
      check(a);
      for (const g of store.groups) baked.add(g.name);
      if (pager) {
        pager.onChange = () => {
          dirty = true;
        };
        paging.add(a);
      }
      assets.set(asset, a);
      dirty = true;
    },
    storeOf(asset) {
      return assets.get(asset)?.store ?? null;
    },
    setStoredLinear(asset, value) {
      linear.set(asset, value);
      dirty = true;
    },
    setLightState(name) {
      if (!rig) throw new Error("atheneaLights: no sidecar");
      rig.setState(name, now);
      evaluate(now);
    },
    setGroupLevel(name, level) {
      if (!rig) throw new Error("atheneaLights: no sidecar");
      rig.setLevel(name, level, now);
      evaluate(now);
    },
    playSequence(name) {
      if (!rig) throw new Error("atheneaLights: no sidecar");
      rig.play(name, now);
      evaluate(now);
    },
    stopSequence() {
      rig?.stop(now);
      evaluate(now);
    },
    setTerms(t) {
      Object.assign(terms, t);
      dirty = true;
    },
  };
  if (initial.sidecar) plugin.setSidecar(initial.sidecar);
  return plugin;
}

/** A colour for a group's swatch in a UI (its weight, normalised, sRGB). */
export function lightGroupSwatch(g: LightGroupState): string {
  const m = Math.max(...g.weight, 1e-6);
  const c = new THREE.Color(g.weight[0] / m, g.weight[1] / m, g.weight[2] / m);
  return `#${c.getHexString()}`;
}

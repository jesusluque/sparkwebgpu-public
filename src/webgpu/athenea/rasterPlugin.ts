// athenea's base raster on the PluginHost (plan-athenea.md, phase 3): the
// splats' colours made as athenea's splat_project makes them, in linear
// light (slang/athenea_adapter/raster.slang, an ISplatColour), blended in
// linear light into a float target, and shown through athenea's display
// transform (AtheneaDisplay, an FxChain effect: atheneaOutputPlugin).
//
//   const host = new PluginHost({ capabilities: spark.capabilities });
//   const raster = atheneaRasterPlugin();
//   host.register(raster).register(atheneaOutputPlugin({ view: "agx" }));
//   host.attach(splats);
//   raster.setOverrides(mesh, [{ groups: [0, 40], tint: [1, 0.2, 0.2] }]);
//
// Per asset (the SplatMesh, or the WgpuSplatMesh):
//   - the colours' storage: sRGB-encoded (every capture, and a linear .athc
//     the loader encoded for Spark's blend) or linear light (a .athc flagged
//     linear and loaded with PagedSplats({ athcKeepLinear: true })), read
//     from the mesh where it can be and settable with setStoredLinear;
//   - emission (the .athc stream) added to the colour, times `emission`;
//   - overrides by finest-group range (the athcGroup attribute: a .athc
//     keeps no Cryptomatte id, so a pick returns athenea's finest LoD group
//     and any coarser group is a range of them): a tint or a replacement
//     colour, and an opacity scale (0 hides the range).
//
// The draw blends linear light only into a linear target (a HalfFloat
// RenderTarget, as SparkRenderer renders into for three's output pass); on
// an 8-bit sRGB canvas it blends the encoded colours as Spark does.

import type { WgpuSplatMesh } from "../WgpuSplatRenderer";
import { ATTRIB_NONE, type AttribPool } from "../attributes/schema";
import { atheneaAdapterRaster } from "../generated/constants";
import { upload } from "../gpuBuffers";
import type { EffectDesc, SplatPlugin } from "../plugins/types";
import type { UniformWriter } from "../uniforms";
import {
  ATHENEA_DISPLAYS,
  ATHENEA_VIEWS,
  AtheneaDisplay,
  type AtheneaDisplayEncoding,
  type AtheneaView,
} from "./AtheneaDisplay";

export const ATHENEA_RASTER_ID = "athenea.raster";
export const ATHENEA_OUTPUT_ID = "athenea.output";
/** The .athc attribute of finest-group ranges (spark-lib athc GROUP_ATTRIBUTE). */
export const ATHC_GROUP_ATTRIBUTE = "athcGroup";
export const MAX_ATHENEA_OVERRIDES = atheneaAdapterRaster.kAtheneaOverrideRows;

type Rgb = [number, number, number];

/** An opinion the frame holds over a range of finest groups. */
export interface AtheneaGroupOverride {
  /** Finest groups lo .. hi - 1 (a pick's group g is [g, g + 1]). */
  groups: [number, number];
  /** Multiplies the linear colour, or is it (`replace`). Default white. */
  tint?: Rgb;
  replace?: boolean;
  /** Opacity scale; 0 hides the range. Default 1. */
  opacity?: number;
}

export interface AtheneaRasterOptions {
  /** Emission added to the colour, times this (default 1; 0: none). */
  emission?: number;
}

export interface AtheneaRasterPlugin extends SplatPlugin {
  emission: number;
  /** The overrides of `asset` (a SplatMesh or WgpuSplatMesh), first match wins. */
  setOverrides(asset: object, overrides: readonly AtheneaGroupOverride[]): void;
  overridesOf(asset: object): readonly AtheneaGroupOverride[];
  /** Whether `asset`'s stored colours are linear light (else sRGB-encoded). */
  setStoredLinear(asset: object, linear: boolean): void;
  storedLinearOf(mesh: WgpuSplatMesh): boolean;
  setEmission(scale: number): void;
}

interface PoolOnGpu {
  version: number;
  buffer: GPUBuffer;
  emissionId: number;
  groupId: number;
}

/** What a mesh is known by: the WgpuSplatMesh, or its object (the SplatMesh). */
function keysOf(mesh: WgpuSplatMesh | undefined): object[] {
  if (!mesh) return [];
  return mesh.object ? [mesh, mesh.object] : [mesh];
}

/** A paged .athc that kept its linear colours (PagedSplats.athcStoredLinear). */
function pagedStoredLinear(mesh: WgpuSplatMesh): boolean | undefined {
  const paged = (mesh.object as { paged?: { athcStoredLinear?: boolean } })
    ?.paged;
  return paged?.athcStoredLinear;
}

/**
 * The athenea raster as an ISplatColour plugin of generate (variant
 * "athenea-raster" in slang/plugins/presets.json).
 */
export function atheneaRasterPlugin(
  options: AtheneaRasterOptions = {},
): AtheneaRasterPlugin {
  const overrides = new WeakMap<object, readonly AtheneaGroupOverride[]>();
  const linear = new WeakMap<object, boolean>();
  const pools = new WeakMap<AttribPool, PoolOnGpu>();
  let device: GPUDevice | null = null;
  let empty: GPUBuffer | null = null;
  let dirty = false;

  const lookup = <T>(map: WeakMap<object, T>, mesh?: WgpuSplatMesh) => {
    for (const k of keysOf(mesh)) {
      const v = map.get(k);
      if (v !== undefined) return v;
    }
    return undefined;
  };

  // The pool generate reads: a paged one as it is on the GPU, else the two
  // columns the raster reads, uploaded again when the pool changes.
  const poolOf = (mesh?: WgpuSplatMesh): PoolOnGpu | null => {
    const pool = mesh?.source.attribs;
    if (!pool || !device) return null;
    if (pool.gpuBuffer) {
      const id = (name: string) => {
        const k = pool.id(name);
        return k < 0 ? ATTRIB_NONE : k;
      };
      return {
        version: pool.version,
        buffer: pool.gpuBuffer,
        emissionId: id("emission"),
        groupId: id(ATHC_GROUP_ATTRIBUTE),
      };
    }
    let g = pools.get(pool);
    if (!g || g.version !== pool.version) {
      g?.buffer.destroy();
      const wanted = ["emission", ATHC_GROUP_ATTRIBUTE];
      const { layout, words } = pool.pack((s) => wanted.includes(s.name));
      const id = (name: string) => {
        const k = layout.specs.findIndex((s) => s.name === name);
        return k < 0 ? ATTRIB_NONE : k;
      };
      g = {
        version: pool.version,
        buffer: upload(device, words, "athenea raster attribs"),
        emissionId: id("emission"),
        groupId: id(ATHC_GROUP_ATTRIBUTE),
      };
      pools.set(pool, g);
    }
    return g;
  };

  const writeOverrides = (
    w: UniformWriter,
    list: readonly AtheneaGroupOverride[],
  ) => {
    const field = w.block.fields?.find((f) => f.name === "overrides");
    if (!field) throw new Error("atheneaRaster: no overrides field");
    const view = new DataView(w.data);
    list.slice(0, MAX_ATHENEA_OVERRIDES).forEach((o, k) => {
      const at = field.offset + k * 32;
      view.setUint32(at, o.groups[0] >>> 0, true);
      view.setUint32(at + 4, o.groups[1] >>> 0, true);
      view.setFloat32(at + 8, o.replace ? 1 : 0, true);
      view.setFloat32(at + 12, o.opacity ?? 1, true);
      const tint = o.tint ?? [1, 1, 1];
      tint.forEach((c, i) => view.setFloat32(at + 16 + 4 * i, c, true));
    });
    return Math.min(list.length, MAX_ATHENEA_OVERRIDES);
  };

  const plugin: AtheneaRasterPlugin = {
    id: ATHENEA_RASTER_ID,
    // Read where a cloud has them: a pager loads them for it.
    requires: { reads: ["emission", ATHC_GROUP_ATTRIBUTE] },
    slang: {
      module: "athenea_adapter.raster",
      colour: "AtheneaRasterColour",
    },
    emission: options.emission ?? 1,
    uniforms: [
      {
        block: "atheneaRaster",
        stage: "splat",
        write(w, ctx) {
          const mesh = ctx.mesh;
          const pool = poolOf(mesh);
          w.setAll({
            storedLinear: mesh ? plugin.storedLinearOf(mesh) : false,
            emissionId: pool?.emissionId ?? ATTRIB_NONE,
            groupId: pool?.groupId ?? ATTRIB_NONE,
            emissionScale: plugin.emission,
          });
          const list = lookup(overrides, mesh) ?? [];
          // Without group ids no row can match: leave the table empty.
          const rows =
            pool && pool.groupId !== ATTRIB_NONE ? writeOverrides(w, list) : 0;
          w.set("overrideCount", rows);
        },
      },
    ],
    buffers(stage, ctx): Record<string, GPUBuffer> {
      if (stage !== "splat") return {};
      const pool = poolOf(ctx.mesh);
      if (pool) return { atheneaPool: pool.buffer };
      // An empty pool (core/attrib.slang): no attributes, no splats.
      empty ??= upload(
        device as GPUDevice,
        new Uint32Array([0, 0, 4, 0]),
        "athenea raster empty attribs",
      );
      return { atheneaPool: empty };
    },
    ui: [
      {
        id: "emission",
        label: "Emission",
        type: "number",
        min: 0,
        max: 4,
        step: 0.05,
        get: () => plugin.emission,
        set: (v) => plugin.setEmission(Number(v)),
      },
    ],
    attach(renderer) {
      device = renderer.device;
    },
    detach() {
      empty?.destroy();
      empty = null;
      device = null;
    },
    onFrame() {
      const was = dirty;
      dirty = false;
      return was ? "dirty" : "clean";
    },
    setOverrides(asset, list) {
      if (list.length > MAX_ATHENEA_OVERRIDES) {
        console.warn(
          `atheneaRaster: ${list.length} overrides, the first ${MAX_ATHENEA_OVERRIDES} are kept`,
        );
      }
      overrides.set(asset, [...list]);
      dirty = true;
    },
    overridesOf(asset) {
      return overrides.get(asset) ?? [];
    },
    setStoredLinear(asset, value) {
      linear.set(asset, value);
      dirty = true;
    },
    storedLinearOf(mesh) {
      return lookup(linear, mesh) ?? pagedStoredLinear(mesh) ?? false;
    },
    setEmission(scale) {
      plugin.emission = scale;
      dirty = true;
    },
  };
  return plugin;
}

export interface AtheneaOutputOptions {
  view?: AtheneaView;
  display?: AtheneaDisplayEncoding;
  /** Stops. */
  exposure?: number;
  /** Linear light under what the picture leaves uncovered. */
  background?: Rgb;
  /** ACES 2.0's peak, nits (default 100). */
  peakLuminance?: number;
}

export interface AtheneaOutputPlugin extends SplatPlugin {
  /** The FxChain step's parameters (live: edit them and redraw). */
  readonly params: Record<string, number | number[]>;
  set(options: AtheneaOutputOptions): void;
}

/**
 * athenea's display transform as the last post effect (PluginHost.applyFx):
 * exposure in stops, then Standard, AgX or ACES 2.0, onto a display.
 */
export function atheneaOutputPlugin(
  options: AtheneaOutputOptions = {},
): AtheneaOutputPlugin {
  const params: Record<string, number | number[]> = {};
  const fx: EffectDesc = {
    effect: new AtheneaDisplay(),
    params,
    name: "display",
  };
  const plugin: AtheneaOutputPlugin = {
    id: ATHENEA_OUTPUT_ID,
    // After every other effect: the display transform ends the chain.
    order: 1000,
    fx: [fx],
    params,
    set(o) {
      if (o.view) params.view = ATHENEA_VIEWS.indexOf(o.view);
      if (o.display) params.display = ATHENEA_DISPLAYS.indexOf(o.display);
      if (o.exposure !== undefined) params.exposure = o.exposure;
      if (o.background) params.background = [...o.background];
      if (o.peakLuminance !== undefined) params.peakLuminance = o.peakLuminance;
    },
    ui: [
      {
        id: "exposure",
        label: "Exposure (stops)",
        type: "number",
        min: -8,
        max: 8,
        step: 0.1,
        get: () => params.exposure ?? 0,
        set: (v) => {
          params.exposure = Number(v);
        },
      },
      {
        id: "view",
        label: "View",
        type: "select",
        options: ATHENEA_VIEWS,
        get: () => ATHENEA_VIEWS[(params.view as number) ?? 1],
        set: (v) => {
          params.view = Math.max(0, ATHENEA_VIEWS.indexOf(v as AtheneaView));
        },
      },
      {
        id: "display",
        label: "Display",
        type: "select",
        options: ATHENEA_DISPLAYS,
        get: () => ATHENEA_DISPLAYS[(params.display as number) ?? 0],
        set: (v) => {
          params.display = Math.max(
            0,
            ATHENEA_DISPLAYS.indexOf(v as AtheneaDisplayEncoding),
          );
        },
      },
    ],
  };
  plugin.set(options);
  return plugin;
}

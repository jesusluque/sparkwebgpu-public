// The plugin manifest (plan-athenea.md §3): what a feature of the splat
// renderer declares so a PluginHost can decide where it runs. Plain data and
// callbacks; nothing here touches the GPU. Every field but `id` is optional,
// so a plugin can be as small as a dyno modifier or a post effect.

import type * as THREE from "three";
import type { WgpuSplatMesh, WgpuSplatRenderer } from "../WgpuSplatRenderer";
import type { AttributeSpec } from "../attributes/schema";
import type { WgpuDyno } from "../dyno/DynoKernels";
import type { Effect } from "../fx/Effect";
import type { ParamValues } from "../fx/types";
import type { UniformWriter } from "../uniforms";

/**
 * Device tiers, as athenea's (plan-athenea.md §3), measured by tierOf():
 * 0 no usable WebGPU (Spark's WebGL), 1 default limits, 2 raised limits,
 * 3 the largest.
 */
export type Tier = 0 | 1 | 2 | 3;

export interface PluginRequirements {
  /** Sections the asset must carry (.athc v3 sections, from phase 2 on). */
  sections?: readonly string[];
  /** Per-Gaussian attributes the asset must carry, by name. */
  attributes?: readonly string[];
  /**
   * Attributes it reads where the asset has them, without needing them:
   * they do not rule the plugin out, but a pager that loads streams on
   * demand (WgpuSplatPager, .athc v3 sections) loads them for it.
   */
  reads?: readonly string[];
  /**
   * Device features: WebGPU feature names ("shader-f16") or GpuCapabilities
   * flags ("subgroups", "timestampQuery", "gpuSort"...).
   */
  features?: readonly string[];
}

/** What an asset carries, for requirement checks. */
export interface PluginAsset {
  attributes?: readonly string[];
  sections?: readonly string[];
}

/** How a plugin's data is read from a sectioned file (phase 2). */
export interface SectionDecoder {
  /** Section name in the file. */
  readonly section: string;
  /** The section's bytes for `count` splats, as attribute columns by name. */
  decode(
    data: ArrayBuffer,
    info: { count: number },
  ): Record<string, ArrayBufferView>;
}

/**
 * The plugin's Slang, as listed in slang/plugins/presets.json: its module
 * and the extension structs it adds. The kernels are compiled offline into
 * presets; the host checks that these names match what was compiled.
 */
export interface PluginSlang {
  module: string;
  /** An ISplatColour struct, run in generate per splat. */
  colour?: string;
  /** An IBlendTerm struct, run in the draw and the tile rasterizer per pixel. */
  blend?: string;
}

export interface PluginFrame {
  renderer: WgpuSplatRenderer;
  camera: THREE.Camera;
  /** Seconds. */
  time: number;
  deltaTime: number;
}

export interface PluginUniformContext {
  frame: PluginFrame | null;
  /** The mesh being generated ("splat" blocks); undefined for "pixel" blocks. */
  mesh?: WgpuSplatMesh;
  /** The asset key the mesh resolves to (PluginHost.assetKey). */
  asset?: object;
}

/**
 * A uniform block the plugin's Slang module declares (a global
 * ConstantBuffer), filled by name through its reflection.
 */
export interface UniformBlockSpec {
  /** The block's global name in Slang, e.g. "tintParams". */
  block: string;
  /**
   * "splat": read by the colour term in generate, written per mesh.
   * "pixel": read by the blend term in the draw and tiles, once a frame.
   */
  stage: "splat" | "pixel";
  write(w: UniformWriter, ctx: PluginUniformContext): void;
}

/** GPU work before generate each frame (prefilter an environment, a shadow map). */
export interface PrePass {
  name: string;
  encode(encoder: GPUCommandEncoder, frame: PluginFrame): void;
}

/** A post effect (aofx) the plugin adds to the host's FxChain. */
export interface EffectDesc {
  effect: Effect;
  params?: ParamValues;
  name?: string;
}

/** A control the plugin publishes for a UI (light states, exposure...). */
export interface ControlSpec {
  id: string;
  label?: string;
  type: "number" | "boolean" | "select" | "color" | "action";
  min?: number;
  max?: number;
  step?: number;
  options?: readonly string[];
  get?(): unknown;
  set?(value: unknown): void;
}

export interface SplatPlugin {
  /** Unique, dotted by convention ("athenea.tx", "spark.attributes"). */
  readonly id: string;
  /** Lowest device tier the plugin runs on (default 1). */
  minTier?: Tier;
  requires?: PluginRequirements;
  /** Active unless switched off (default true). */
  enabled?: boolean;
  /** Stage ordering: runs after / before these plugin ids, then by order. */
  after?: readonly string[];
  before?: readonly string[];
  order?: number;

  /** What it registers in the AttribPool (format and lodMerge). */
  attributes?: readonly AttributeSpec[];
  sections?: readonly SectionDecoder[];
  slang?: PluginSlang;
  uniforms?: readonly UniformBlockSpec[];
  /**
   * Storage buffers its Slang module declares, by name, for the stage
   * ("splat": generate per mesh, "pixel": draw and tiles).
   */
  buffers?(
    stage: "splat" | "pixel",
    ctx: PluginUniformContext,
  ): Record<string, GPUBuffer>;
  passes?: readonly PrePass[];
  fx?: readonly EffectDesc[];
  ui?: readonly ControlSpec[];
  /**
   * Dyno graphs run in generate for each asset it is active on; the host
   * chains those of several plugins in stage order.
   */
  dyno?: WgpuDyno;

  /** Once a frame while registered; "dirty" regenerates the splats. */
  onFrame?(frame: PluginFrame): "dirty" | "clean" | undefined;
  /** When the host attaches to a renderer (or the plugin registers on an attached host). */
  attach?(renderer: WgpuSplatRenderer): void;
  detach?(renderer: WgpuSplatRenderer): void;
}

/** Why a plugin is not active. */
export type InactiveReason =
  | "unregistered"
  | "tier"
  | "features"
  | "attributes"
  | "sections"
  | "disabled"
  | "no-variant";

export interface PluginResolution {
  tier: Tier;
  /** Active plugins, in stage order. */
  active: SplatPlugin[];
  inactive: { id: string; reason: InactiveReason; detail?: string }[];
  /** generate variant key (colour plugin ids joined by "+"; "" default). */
  colour: string;
}

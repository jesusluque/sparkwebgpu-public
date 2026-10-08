// Splat rendering on three.js's WebGPURenderer with the Slang kernels:
// generate (compute) -> sort -> draw (own render pass after three's).
//
// Usage:
//   const renderer = new WebGPURenderer({ antialias: false });
//   await renderer.init();
//   const splats = new WgpuSplatRenderer(renderer);
//   splats.add(GpuSplatSource.fromExt(device, a, b, count), mesh);
//   renderer.setAnimationLoop(() => {
//     renderer.setRenderTarget(target);     // RenderTarget with a DepthTexture
//     renderer.render(scene, camera);
//     splats.render(camera, target);
//     renderer.setRenderTarget(null);
//     output.render(renderer);              // QuadMesh showing target.texture
//   });
//
// The draw pass loads the texture three just rendered into and blends
// premultiplied splats over it. Into a RenderTarget with a depthTexture the
// splats are occluded by the scene (three's canvas depth buffer is internal,
// so on the canvas they draw over it). Sorting is on the CPU for now, from a
// metric readback one frame behind, as SparkRenderer does with its WASM sort.

import * as THREE from "three";
import {
  AutoRasterizer,
  type AutoRasterizerState,
  DrawTimer,
  type RasterPath,
} from "./AutoRasterizer";
import { DepthResolve } from "./DepthResolve";
import { GpuProfiler } from "./GpuProfiler";
import { GpuSorter, type SortDepthRange } from "./GpuSorter";
import { GpuSplatSource } from "./GpuSplatSource";
import type { KernelModule } from "./KernelModule";
import { KernelRegistry } from "./KernelRegistry";
import {
  SRGB_LAYER_BLEND,
  SRGB_LAYER_FORMAT,
  SrgbComposite,
} from "./SrgbComposite";
import { TileRasterizer } from "./TileRasterizer";
import { type GpuCapabilities, capabilitiesOf } from "./capabilities";
import { sortBackToFront } from "./cpuSort";
import {
  type DynoDispatch,
  DynoKernels,
  type WgpuDyno,
} from "./dyno/DynoKernels";
import {
  drawSplatDraw,
  drawSplatShape,
  kernelsGenerate,
} from "./generated/constants";
import drawModule from "./generated/draw/splat_draw";
import generateModule from "./generated/kernels/generate";
import {
  createReadback,
  createStorage,
  createUniform,
  upload,
} from "./gpuBuffers";
import type { PixelKernels, PluginHost } from "./plugins/PluginHost";
import {
  type ReflectedRenderPipeline,
  createBindGroups,
  createReflectedRenderPipeline,
  depthTestAttachment,
} from "./renderPipeline";
import {
  type ThreeWebGPURenderer,
  canvasContext,
  gpuDevice,
  gpuTexture,
  webgpuBackend,
} from "./threeInternals";
import type { WebGPURendererLike } from "./threeRenderer";
import { UniformWriter } from "./uniforms";

// The projector's record of a sorted slot (splat_draw.slang).
const PROJECTED_BYTES = 16 * drawSplatDraw.PROJECTED_WORDS;

const {
  GEN_SRC_EXT,
  GEN_OUT_EXT,
  GEN_USE_LOD,
  GEN_LOD_OPACITY,
  GEN_SORT_RADIAL,
  GEN_DYNO_SOURCE,
  GEN_OUT_COV,
  GEN_CULL,
  GEN_COV_TRANSFORM,
  GEN_LOD_FADE,
  GEN_ADDITIVE,
} = kernelsGenerate;
const {
  DRAW_EXT,
  DRAW_COV,
  DRAW_2DGS,
  DRAW_LOD_INFLATE,
  DRAW_ORTHOGRAPHIC,
  DRAW_ENCODE_LINEAR,
  DRAW_PREMULTIPLIED,
  DRAW_DISK_CLIP,
  DRAW_ADDITIVE_ONLY,
  DRAW_NO_ADDITIVE,
} = drawSplatShape;

/** A portal disk in view space that clips the splats (WgpuSplatRenderer.diskClip). */
export interface SplatDiskClip {
  center: THREE.Vector3;
  normal: THREE.Vector3;
  /**
   * > 0: draw only the splats behind the disk, seen through it; < 0: drop
   * the splats behind it where it is seen; 0: no clip.
   */
  radius: number;
  /** Clip seen from either side (else only facing against the normal). */
  twoSided: boolean;
}

// The ext accumulator's RGBA (packSplatExt: rg, b and alpha as halves) into
// an RgbaArray texture, at splatTexCoord(index).
const BAKE_RGBA_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> splats: array<vec4u>;
@group(0) @binding(1) var rgba: texture_storage_2d_array<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> count: vec4u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= count.x) { return; }
  let a = splats[2u * i];
  let b = splats[2u * i + 1u];
  let value = vec4f(unpack2x16float(b.x), unpack2x16float(b.y).x, unpack2x16float(a.w).x);
  textureStore(rgba, vec2u(i & 2047u, (i >> 11u) & 2047u), i >> 22u, clamp(value, vec4f(0.0), vec4f(1.0)));
}
`;

export interface WgpuSplatMesh {
  source: GpuSplatSource;
  object: THREE.Object3D;
  recolor: THREE.Vector4;
  /** Source indices to draw, from LOD traversal; all splats when null. */
  lodIndices: Uint32Array | null;
  lodBuffer: GPUBuffer | null;
  /**
   * The GPU copy of lodIndices carries a LoD transition's fade in each
   * index's top byte (LodFade; generate's GEN_LOD_FADE).
   */
  lodFaded?: boolean;
  /** Dyno generator and modifiers run in the generate kernel. */
  dyno?: WgpuDyno;
  /**
   * Additive splats (light added, nothing covered; generate's
   * GEN_ADDITIVE). Default: the object's `additive` (SplatMesh.additive).
   * Ext accumulator and Gaussian splats only; ignored otherwise.
   */
  additive?: boolean;
}

export interface WgpuSplatRendererOptions {
  /** Test against the render target's depthTexture, if it has one (default true). */
  depthTest?: boolean;
  sortRadial?: boolean;
  /**
   * "gpu" (default): radix sort on the GPU in the same frame, drawn with an
   * indirect count. "cpu": metric readback and a JS radix sort, a frame behind.
   */
  sort?: "gpu" | "cpu";
  /**
   * Key bits the GPU sort orders by: 32 (exact), 24 or 16 (4 passes; merges
   * splats within 2^-7 relative distance). Default 32.
   */
  sortBits?: 16 | 24 | 32;
  /**
   * GPU sort with the cull on and a perspective camera: key the metrics by
   * their place between the camera's planes on 24 bits (6 radix passes, not
   * 8), instead of their float bits (sortBits), while that step is at most
   * 2.5e-4 of a unit (a range under ~4.2k units: with the default radial
   * metric, a far plane of ~2k at a 60 degree field). The step is uniform:
   * finer than 24 float bits beyond a few units, and the order differs
   * from the 32-bit one only between splats closer than that.
   * Default true.
   */
  depthKeys?: boolean;
  /**
   * Regenerate and re-sort every frame even when nothing moved. Off by
   * default: like SparkRenderer, an unchanged frame redraws the last order.
   * Turn on (or call markDirty()) for splats animated on the GPU.
   */
  alwaysGenerate?: boolean;
  /**
   * GPU sort: while only the camera and mesh transforms move, re-sort at
   * most every this many ms and draw the last order in between, as
   * SparkRenderer draws its last worker sort (same option name). A frame
   * where nothing moved sorts a stale order (see sortStale). Default 0:
   * every regenerated frame is sorted.
   */
  minSortIntervalMs?: number;
  maxStdDev?: number;
  minPixelRadius?: number;
  maxPixelRadius?: number;
  minAlpha?: number;
  blurAmount?: number;
  preBlurAmount?: number;
  falloff?: number;
  clipXY?: number;
  focalAdjustment?: number;
  /** Depth of field: distance to the focal plane (0: off). */
  focalDistance?: number;
  /** Depth of field: full aperture angle in radians (0: off). */
  apertureAngle?: number;
  /** Trade LOD opacity above 1 for size (SparkRenderer.lodInflate). */
  lodInflate?: boolean;
  /**
   * Time the generate, sort and draw passes with timestamp queries into
   * stats.gpuMs (needs the timestamp-query feature; ignored without it).
   * Splits the sort into a pass per stage, so it costs a little.
   */
  profile?: boolean;
  /**
   * Project each sorted splat once, in a compute pass after the sort
   * (draw/splat_draw.slang projectSplats), and have the quads' vertices read
   * that record, instead of projecting it in each of its four vertices.
   * Same image. Applies to the GPU sort's quad draw (not 2DGS, attribute
   * draw stages or the tile rasterizer) while the records (64 B a splat)
   * fit one storage binding. Default false: on an Apple GPU (Chrome) the
   * draw took as long either way (the quads' raster, not their vertex
   * shading, is what costs; docs/docs/webgpu-performance-audit.md), and the
   * records take 64 B a splat.
   */
  projectOnce?: boolean;
  /**
   * Accumulator format between generate and draw. "ext" (32 B a splat):
   * float centers and colours. "packed" (16 B, as SparkRenderer's default):
   * half-float centers relative to the camera, 8-bit colour clamped to the
   * packed range, so precision drops far from the camera. "auto" (default):
   * ext while it fits one storage binding (capabilities.maxSplats, 4.19M
   * splats with default limits), packed above that (maxSplatsPacked). The
   * draw costs the same with either on Apple GPUs.
   */
  accumulator?: "ext" | "packed" | "auto";
  /**
   * Drop splats in generate that the draw would skip (center outside the
   * clipXY frustum, alpha under minAlpha), so the sort and the draw handle
   * only the rest. Same image; default true. GPU sort only.
   */
  cull?: boolean;
  /**
   * Accumulate covariance splats (SparkRenderer.covSplats): mesh transforms
   * may scale non-uniformly or shear, and meshes may use CovSplat modifiers
   * (WgpuDyno.covObjectModifiers, linear-blend SplatSkinning). Default false.
   */
  covSplats?: boolean;
  /**
   * Surfels (2DGS): how a splat with exactly one scale 0, as decoded, is
   * drawn. Detected per splat, so clouds may mix discs and 3D Gaussians.
   * "auto" (default): the exact ray-splat intersection with Huang et al.'s
   * screen filter (sigma sqrt(2)/2 px, as athenea's raster), on every path
   * (quads, projectOnce, tiles). "ewa": as a projected 3D Gaussian of rank 2
   * with the anti-alias blur (the 3D path; edge-on discs keep the blur).
   */
  surfels?: "auto" | "ewa";
  /**
   * SparkRenderer.enable2DGS: forces `surfels: "auto"`. Default false.
   */
  enable2DGS?: boolean;
  /**
   * render(): blend the splats in sRGB space into linear targets too, as
   * WebGL Spark does on the canvas (see SrgbComposite), rather than in the
   * target's linear space. Default false.
   */
  srgbBlend?: boolean;
  /**
   * Keep the splats' light in float end to end: colours above 1 kept per
   * splat (the ext accumulator, half-float colour, even when `accumulator`
   * says "packed", while it fits a binding), and no 8-bit sRGB layer
   * (`srgbBlend` is ignored): splats blend in linear light into the float
   * target. Implied while a plugin with `hdr` is registered and enabled
   * (the athenea raster, relight, lights and output plugins). Default false.
   */
  hdr?: boolean;
  /**
   * How render() draws the sorted splats. "hardware" (default): instanced
   * quads blended back to front. "tiles" (experimental): a compute tile
   * rasterizer (TileRasterizer) blending each 16 x 16 tile front to back,
   * stopping where transmittance drops under 1/255, then composited. Needs
   * the GPU sort; 2DGS, draw stages and renderInPass stay on hardware.
   * "auto": whichever of the two the GPU draws faster, timed with timestamp
   * queries and re-probed now and then (AutoRasterizer); hardware without
   * the timestamp-query feature. stats.rasterizer says which drew.
   */
  rasterizer?: "hardware" | "tiles" | "auto";
}

/** A mesh's output range in the accumulator this frame. */
export interface SplatMeshRange {
  mesh: WgpuSplatMesh;
  base: number;
  count: number;
}

export interface SplatDrawContext {
  colorTarget: GPUColorTargetState;
  depthStencil?: GPUDepthStencilState;
  width: number;
  height: number;
  /** The frame's blend-plugin kernels and resources (PluginHost), if any. */
  plugin?: PixelKernels | null;
}

/** A render pass someone else opened (three's), for renderInPass. */
export interface SplatPassTarget {
  format: GPUTextureFormat;
  depthFormat: GPUTextureFormat | null;
  sampleCount: number;
  width: number;
  height: number;
  /** The pass's color is linear (encode splat colors to linear). */
  linear: boolean;
  /** Formats of the pass's other color attachments (MRT), left unwritten. */
  extraFormats?: GPUTextureFormat[];
  /** For reversed-depth buffers. @default "less-equal" */
  depthCompare?: GPUCompareFunction;
  /**
   * The pass draws into an SrgbComposite layer: alpha accumulates the
   * transmittance. Colours are written as sRGB (linear is ignored).
   */
  layer?: boolean;
}

/** A draw pipeline replacing the default one, with its extra resources. */
export interface SplatDrawVariant {
  pipeline: ReflectedRenderPipeline;
  /** Bindings besides ordering, splats and params. */
  buffers: Record<string, GPUBuffer>;
  /** Color attachments after the main one (null for unused targets). */
  attachments: (GPURenderPassColorAttachment | null)[];
}

/**
 * Optional work around the core passes, e.g. per-Gaussian attributes
 * (src/webgpu/attributes): extra compute after generate, a draw variant.
 */
export interface SplatRendererStage {
  generate?(
    encoder: GPUCommandEncoder,
    ranges: readonly SplatMeshRange[],
  ): void;
  draw?(context: SplatDrawContext): SplatDrawVariant | null;
}

export class WgpuSplatRenderer {
  readonly device: GPUDevice;
  readonly registry: KernelRegistry;
  readonly meshes: WgpuSplatMesh[] = [];
  /** Counters for debugging and benchmarks. */
  readonly stats = {
    frames: 0,
    draws: 0,
    sorts: 0,
    drawn: 0,
    sortMs: 0,
    /** Frames that ran generate and the GPU sort (the rest redrew). */
    generated: 0,
    /** JS time of the last render() call. */
    cpuMs: 0,
    /** Smoothed GPU ms per pass with options.profile (GpuProfiler labels). */
    gpuMs: {} as Record<string, number>,
    /** The path the last draw took. */
    rasterizer: "hardware" as RasterPath,
    /** rasterizer "auto": its choice and timings. */
    auto: null as AutoRasterizerState | null,
    /** The last draw drew additive splats as a layer of their own. */
    additiveLayer: false,
    /** The last frame ran the projector (projectOnce). */
    projected: false,
  };
  options: Required<WgpuSplatRendererOptions>;
  /** What the device allows; consulted for the sort path and sizes. */
  readonly capabilities: GpuCapabilities;
  readonly stages: SplatRendererStage[] = [];
  /**
   * The portal clip of SparkPortals' DISK_PORTAL_FRAGMENT_SHADER, for the
   * next draws (default draw and attribute variants).
   */
  diskClip: SplatDiskClip | null = null;
  /** Plugins (src/webgpu/plugins), set by PluginHost.attach. */
  plugins: PluginHost | null = null;
  /**
   * Called when a render defers its sort (minSortIntervalMs), so on-demand
   * apps render once more after motion stops and get the exact order.
   */
  onSortDeferred: (() => void) | null = null;

  private capacity = 0;
  /**
   * The size in pixels of the last target drawn (null before the first
   * draw): what a pre-pass that projects per splat (the athenea relight's
   * per-pixel slope) takes the frame to be.
   */
  lastDrawSize: { width: number; height: number } | null = null;
  private accumBytes = 0;
  private accumulator: GPUBuffer | null = null;
  private metric: GPUBuffer | null = null;
  private ordering: GPUBuffer | null = null;
  private sorter: GpuSorter;
  // What the last generate saw; an identical frame skips generate and sort.
  private lastSignature: number[] = [];
  // The signature without camera and mesh transforms, and whether only
  // those moved in the last change (minSortIntervalMs).
  private lastStructure: number[] = [];
  private posesOnly = false;
  private lastSortTime = Number.NEGATIVE_INFINITY;
  private stale = false;
  private dirty = true;
  private drawUniform: GPUBuffer;
  private additiveUniform?: GPUBuffer;
  private additiveLayer?: GPUTexture;
  private additiveAdd = new Map<string, GPURenderPipeline>();
  /** Single-sample copies of multisampled targets' depth (render()). */
  private depthResolve?: DepthResolve;
  /** The projector's records (projectOnce), PROJECTED_BYTES a slot. */
  private projected: GPUBuffer | null = null;
  private pipelines = new Map<string, ReflectedRenderPipeline>();
  private emptyBuffer: GPUBuffer;
  private dynoKernels: DynoKernels;
  private profiler: GpuProfiler | null = null;
  // The accumulator as the last generate wrote it: packed centers are
  // relative to `origin` (the camera then).
  private written: {
    packed: boolean;
    origin: THREE.Vector3;
    /** The range the generated metrics lie in (depthKeys), or null. */
    depth: SortDepthRange | null;
  } = { packed: false, origin: new THREE.Vector3(), depth: null };
  private lastTime = performance.now() / 1000;
  private dynoDirty = false;
  private bakePipeline?: GPUComputePipeline;
  private srgb?: SrgbComposite;
  private tiles?: TileRasterizer;
  // This frame's blend-plugin kernels (null: the default ones).
  private pixel: PixelKernels | null = null;
  // rasterizer "auto": the policy, its draw timer (without options.profile),
  // and the first tile encode a probe needs the pair count of.
  private auto?: AutoRasterizer;
  private autoTimer: DrawTimer | null = null;
  private warmFrom = 0;

  // Sort state: a readback in flight, and the order last uploaded.
  private sortPending = false;
  private mappingVersion = 0;
  private drawCount = 0;
  private drawVersion = -1;

  constructor(
    readonly renderer: WebGPURendererLike,
    options: WgpuSplatRendererOptions = {},
  ) {
    if (!webgpuBackend(renderer)) {
      throw new Error(
        "WgpuSplatRenderer needs WebGPURenderer on its WebGPU backend (await renderer.init())",
      );
    }
    this.device = gpuDevice(renderer);
    this.registry = new KernelRegistry(this.device);
    this.sorter = new GpuSorter(this.registry);
    this.options = {
      depthTest: true,
      sortRadial: true,
      sort: "gpu",
      sortBits: 32,
      depthKeys: true,
      alwaysGenerate: false,
      minSortIntervalMs: 0,
      maxStdDev: Math.sqrt(8),
      minPixelRadius: 0,
      maxPixelRadius: 512,
      minAlpha: 0.5 / 255,
      blurAmount: 0.3,
      preBlurAmount: 0,
      falloff: 1,
      clipXY: 1.4,
      focalAdjustment: 1,
      focalDistance: 0,
      apertureAngle: 0,
      lodInflate: false,
      profile: false,
      projectOnce: false,
      accumulator: "auto",
      cull: true,
      covSplats: false,
      surfels: "auto",
      enable2DGS: false,
      srgbBlend: false,
      hdr: false,
      rasterizer: "hardware",
      ...options,
    };
    this.capabilities = capabilitiesOf(this.device);
    if (this.options.sort === "gpu" && !this.capabilities.gpuSort) {
      console.warn(
        `WgpuSplatRenderer: GPU sort unavailable (${this.capabilities.gpuSortReason}); sorting on the CPU`,
      );
      this.options.sort = "cpu";
    }
    this.drawUniform = createUniform(
      this.device,
      UniformWriter.for(drawModule).data.byteLength,
      "splat draw params",
    );
    this.emptyBuffer = createStorage(this.device, 16, "empty");
    this.dynoKernels = new DynoKernels(
      this.registry,
      generateModule,
      "generate",
    );
  }

  add(
    source: GpuSplatSource,
    object: THREE.Object3D = new THREE.Object3D(),
    dyno?: WgpuDyno,
  ): WgpuSplatMesh {
    const mesh: WgpuSplatMesh = {
      source,
      object,
      recolor: new THREE.Vector4(1, 1, 1, 1),
      lodIndices: null,
      lodBuffer: null,
      dyno,
    };
    this.meshes.push(mesh);
    this.mappingVersion += 1;
    return mesh;
  }

  /**
   * Whether any mesh's dyno output may differ from the last call (new graph,
   * uniform values or textures), after running the dyno updaters: frames
   * that skip unchanged generates must still generate when this is true.
   */
  dynoChanged(): boolean {
    let changed = false;
    for (const mesh of [...this.meshes]) {
      if (DynoKernels.active(mesh.dyno)) {
        try {
          changed = this.dynoKernels.changed(mesh, mesh.dyno) || changed;
        } catch (error) {
          // A graph that can't compile (GLSL-only code...) drops its mesh,
          // as SparkRenderer drops a generator whose update throws, rather
          // than failing every frame.
          console.error("WgpuSplatRenderer: dyno failed, mesh removed", error);
          this.remove(mesh);
          changed = true;
        }
      }
    }
    return changed;
  }

  /** `numSplats` splats made by a dyno generator (SplatGenerator-style). */
  addGenerator(
    numSplats: number,
    dyno: WgpuDyno,
    object: THREE.Object3D = new THREE.Object3D(),
  ): WgpuSplatMesh {
    const source = new GpuSplatSource("packed", numSplats, this.emptyBuffer);
    return this.add(source, object, dyno);
  }

  remove(mesh: WgpuSplatMesh) {
    const i = this.meshes.indexOf(mesh);
    if (i >= 0) {
      this.meshes.splice(i, 1);
      mesh.lodBuffer?.destroy();
      this.mappingVersion += 1;
    }
  }

  /** Restricts a mesh to the given source indices (a LOD traversal result). */
  /**
   * The source indices `mesh` draws (null: all). `faded`, the same indices
   * with a fade in their top byte (LodFade.encode), is what the GPU reads
   * instead while a LoD transition fades; `indices` stay the plain ones.
   */
  setLodIndices(
    mesh: WgpuSplatMesh,
    indices: Uint32Array | null,
    faded?: Uint32Array,
  ) {
    mesh.lodIndices = indices;
    const gpu = indices && faded?.length === indices.length ? faded : indices;
    mesh.lodFaded = gpu !== indices;
    if (gpu && mesh.lodBuffer && mesh.lodBuffer.size >= gpu.byteLength) {
      // LOD updates arrive often; reuse the buffer while it is large enough.
      this.device.queue.writeBuffer(
        mesh.lodBuffer,
        0,
        gpu.buffer,
        gpu.byteOffset,
        gpu.byteLength,
      );
    } else {
      mesh.lodBuffer?.destroy();
      mesh.lodBuffer = gpu ? upload(this.device, gpu, "lod indices") : null;
    }
    this.mappingVersion += 1;
  }

  private meshCount(m: WgpuSplatMesh) {
    return m.lodIndices ? m.lodIndices.length : m.source.count;
  }

  /**
   * Whether the splats' light stays in float end to end (options.hdr, or a
   * plugin that makes linear light past 1).
   */
  get hdr(): boolean {
    return Boolean(this.options.hdr || this.plugins?.hdr);
  }

  private warnedHdrPacked = false;

  // Whether generate writes the packed accumulator for `total` splats. Its
  // colour is 8 bits clamped to [0, 1] (sRGB-encoded), so HDR keeps the ext
  // one (half-float colour) whenever it fits.
  private packedFor(total: number) {
    const a = this.options.accumulator;
    const over = total > this.capabilities.maxSplats;
    if (this.hdr) {
      if (over && !this.warnedHdrPacked) {
        this.warnedHdrPacked = true;
        console.warn(
          `WgpuSplatRenderer: ${total} splats is over the ext accumulator's ${this.capabilities.maxSplats} (maxStorageBufferBindingSize; see splatRequiredLimits): HDR colours are clamped to 1 in the packed one`,
        );
      }
      return over;
    }
    return a === "packed" || (a === "auto" && over);
  }

  private ensureCapacity(total: number) {
    const bytes = this.packedFor(total) ? 16 : 32;
    if (total <= this.capacity && this.capacity * bytes <= this.accumBytes) {
      return;
    }
    const { maxSplats, maxSplatsPacked } = this.capabilities;
    const max = bytes === 16 ? maxSplatsPacked : maxSplats;
    if (total > max) {
      throw new Error(
        `WgpuSplatRenderer: ${total} splats is over this device's ${max} (maxStorageBufferBindingSize; see splatRequiredLimits${bytes === 32 ? ', or accumulator "auto"' : ""})`,
      );
    }
    this.capacity = Math.min(
      Math.max(total, this.capacity, Math.ceil(this.capacity * 1.5)),
      max,
    );
    this.accumulator?.destroy();
    this.metric?.destroy();
    this.ordering?.destroy();
    this.accumBytes = this.capacity * bytes;
    this.accumulator = createStorage(
      this.device,
      this.accumBytes,
      "accumulator",
    );
    this.metric = createStorage(this.device, this.capacity * 4, "sort metric");
    this.ordering = createStorage(this.device, this.capacity * 4, "ordering");
    this.drawCount = 0;
    this.drawVersion = -1;
    this.dirty = true;
  }

  /** Forces the next render to regenerate and re-sort. */
  markDirty() {
    this.dirty = true;
  }

  /**
   * Whether the last render drew an order sorted for an earlier pose
   * (minSortIntervalMs): render again once nothing moves to sort it.
   */
  get sortStale(): boolean {
    return this.stale;
  }

  // True when the camera, a mesh transform or colour, the mesh set or the
  // sort settings changed since the last generate.
  private changedSince(camera: THREE.Camera, total: number): boolean {
    const structure: number[] = [
      total,
      this.mappingVersion,
      this.options.sortBits,
      this.options.depthKeys ? 1 : 0,
      this.options.sortRadial ? 1 : 0,
      this.packedFor(total) ? 1 : 0,
      this.options.cull ? 1 : 0,
      this.options.clipXY,
      this.options.minAlpha,
      this.options.covSplats ? 1 : 0,
      // A stage added later (attributes) has gathered nothing yet.
      this.stages.length,
    ];
    const sig: number[] = [
      ...camera.matrixWorld.elements,
      ...camera.projectionMatrix.elements,
    ];
    for (const m of this.meshes) {
      m.object.updateMatrixWorld();
      sig.push(...m.object.matrixWorld.elements);
      structure.push(...m.recolor.toArray(), m.source.version);
    }
    const equal = (a: number[], b: number[]) =>
      a.length === b.length && a.every((v, i) => v === b[i]);
    const sameStructure =
      !this.dirty &&
      !this.dynoDirty &&
      !this.options.alwaysGenerate &&
      equal(structure, this.lastStructure);
    const same = sameStructure && equal(sig, this.lastSignature);
    this.posesOnly = sameStructure;
    this.lastStructure = structure;
    this.lastSignature = sig;
    this.dirty = false;
    return !same;
  }

  // The GPU-sort frame before the draw: generate on any change, and sort
  // unless only poses moved within minSortIntervalMs of the last sort. An
  // unchanged frame sorts a stale order: its metric is still current.
  private generateAndSort(
    encoder: GPUCommandEncoder,
    camera: THREE.Camera,
    total: number,
    cameraPos: THREE.Vector3,
    cameraDir: THREE.Vector3,
  ) {
    const changed = this.changedSince(camera, total);
    if (changed) {
      this.generateAll(encoder, camera, cameraPos, cameraDir);
      this.stats.generated += 1;
    }
    const now = performance.now();
    const deferred =
      changed &&
      this.posesOnly &&
      now - this.lastSortTime < this.options.minSortIntervalMs;
    if ((changed || this.stale) && !deferred) {
      this.encodeSort(encoder, total);
      this.lastSortTime = now;
    }
    this.stale = deferred;
    if (deferred) this.onSortDeferred?.();
  }

  /**
   * Generates, sorts and draws all meshes over what three rendered this frame:
   * into `target` when given (a RenderTarget three rendered into first; its
   * depthTexture, if any, occludes the splats), else onto the canvas.
   */
  render(camera: THREE.Camera, target?: THREE.RenderTarget) {
    const t0 = performance.now();
    this.renderFrame(camera, target);
    this.stats.cpuMs = performance.now() - t0;
  }

  // The profiler while options.profile is on, created on first use.
  private activeProfiler(): GpuProfiler | null {
    if (!this.options.profile) return null;
    if (!this.profiler) {
      this.profiler = GpuProfiler.create(this.device);
      if (!this.profiler) {
        console.warn("WgpuSplatRenderer: no timestamp-query; not profiling");
        this.options.profile = false;
        return null;
      }
      this.stats.gpuMs = this.profiler.ms;
    }
    return this.profiler;
  }

  private timestampWrites(label: string) {
    return this.options.profile
      ? this.profiler?.timestampWrites(label)
      : undefined;
  }

  /**
   * Timestamp writes for a plugin's own pass, timed into stats.gpuMs under
   * `label` with options.profile; undefined otherwise.
   */
  passTimestampWrites(
    label: string,
  ):
    | (GPUComputePassTimestampWrites & GPURenderPassTimestampWrites)
    | undefined {
    return this.timestampWrites(label);
  }

  // The GPU sort of this frame's metric, a pass per stage when profiling.
  private encodeSort(encoder: GPUCommandEncoder, total: number) {
    const metric = this.metric as GPUBuffer;
    const bits = this.options.sortBits;
    const depth = this.written.depth;
    if (this.options.profile && this.profiler) {
      this.sorter.encodeProfiled(
        encoder,
        this.profiler,
        metric,
        total,
        bits,
        depth,
      );
      return;
    }
    const pass = encoder.beginComputePass({ label: "sort" });
    this.sorter.encode(pass, metric, total, bits, depth);
    pass.end();
  }

  private submit(encoder: GPUCommandEncoder) {
    const after =
      this.profiler && this.options.profile
        ? this.profiler.resolve(encoder)
        : null;
    const afterAuto = this.autoTimer?.resolve(encoder);
    this.registry.submit(encoder.finish());
    after?.();
    afterAuto?.();
  }

  // The draw path of a frame that may take either; "warm": a tile probe's
  // frame drawn on hardware while the tile stages run untimed, until the
  // pair buffers are sized for this view (an undersized frame drops pairs).
  private rasterPath(): RasterPath | "warm" {
    if (this.options.rasterizer !== "auto") {
      return this.options.rasterizer === "tiles" ? "tiles" : "hardware";
    }
    if (!this.auto) {
      this.auto = new AutoRasterizer();
      const auto = this.auto;
      if (this.options.profile && this.activeProfiler()) {
        // The profiler times every pass (a tile frame has tiles.* stages),
        // which may overlap: the frame's span, generate and sort included.
        (this.profiler as GpuProfiler).onFrame = (frame) => {
          if (frame.draw === undefined) return;
          const tiles = frame["tiles.blend"] !== undefined;
          auto.measured(tiles ? "tiles" : "hardware", frame.span);
        };
      } else {
        // Readbacks for the frames an unthrottled loop has in flight.
        const profiler = GpuProfiler.create(this.device, 0, 32);
        this.autoTimer =
          profiler &&
          new DrawTimer(profiler, (path, ms) => auto.measured(path, ms));
      }
    }
    if (!this.autoTimer && !this.profiler?.onFrame) return "hardware";
    const auto = this.auto;
    const probes = auto.probes;
    const path = auto.next();
    this.stats.auto = auto.state();
    if (path === "tiles" && auto.path === "hardware") {
      this.tiles ??= new TileRasterizer(this.registry);
      if (auto.probes !== probes) this.warmFrom = this.tiles.encodes;
      if (this.tiles.readFrom < this.warmFrom) {
        // Waits for the readback, on hardware, after one warm-up encode.
        auto.extendProbe();
        return this.tiles.encodes === this.warmFrom ? "warm" : "hardware";
      }
    }
    return path;
  }

  private renderFrame(camera: THREE.Camera, target?: THREE.RenderTarget) {
    this.activeProfiler();
    const total = this.meshes.reduce((n, m) => n + this.meshCount(m), 0);
    this.stats.frames += 1;
    if (total === 0) return;
    this.ensureCapacity(total);
    const { cameraPos, cameraDir } = this.updateDynos(camera);

    const encoder = this.device.createCommandEncoder({ label: "splats" });
    this.plugins?.encodePasses(encoder);
    if (this.options.sort === "gpu") {
      this.generateAndSort(encoder, camera, total, cameraPos, cameraDir);
      this.draw(encoder, camera, target, total);
      this.submit(encoder);
      return;
    }
    this.generateAll(encoder, camera, cameraPos, cameraDir);
    this.draw(encoder, camera, target);
    const version = this.mappingVersion;
    const readback = this.sortPending ? null : this.copyMetric(encoder, total);
    this.submit(encoder);
    if (readback) this.sortFrom(readback, total, version);
  }

  /**
   * Like render, but records the draw into `pass`, a render pass the caller
   * (three's WebGPURenderer) opened and submits later: generate and sort are
   * submitted now, so they run first. Draw stages (attribute variants) are
   * not applied here, as they need attachments of their own.
   */
  renderInPass(
    camera: THREE.Camera,
    pass: GPURenderPassEncoder,
    target: SplatPassTarget,
  ) {
    // Times generate and sort; the draw is inside three's pass.
    this.activeProfiler();
    const total = this.meshes.reduce((n, m) => n + this.meshCount(m), 0);
    this.stats.frames += 1;
    if (total === 0) return;
    this.ensureCapacity(total);
    const { cameraPos, cameraDir } = this.updateDynos(camera);
    const encoder = this.device.createCommandEncoder({ label: "splats" });
    this.plugins?.encodePasses(encoder);
    const gpu = this.options.sort === "gpu";
    let readback: GPUBuffer | null = null;
    let projected: GPUBuffer | null = null;
    if (gpu) {
      this.generateAndSort(encoder, camera, total, cameraPos, cameraDir);
      if (this.projects(total)) {
        // The draw params now, so the projector (submitted with the sort,
        // before the caller's pass) projects for this target.
        const drawParams = this.writeDrawParams(
          camera,
          target.width,
          target.height,
          target.linear && !target.layer,
        );
        projected = this.encodeProject(encoder, total, drawParams);
      }
    } else {
      this.generateAll(encoder, camera, cameraPos, cameraDir);
      readback = this.sortPending ? null : this.copyMetric(encoder, total);
    }
    this.stats.projected = projected !== null;
    // Additive splats as a layer of their own (drawAdditiveLayer), drawn
    // now without the scene's depth (three's pass has not run yet), added
    // in three's pass before the other splats.
    const additiveLayer =
      gpu &&
      !target.layer &&
      !this.written.packed &&
      this.meshes.some((m) => this.isAdditive(m));
    this.stats.additiveLayer = additiveLayer;
    const pixelKernels = this.plugins?.pixelKernels(this.device) ?? null;
    if (additiveLayer) {
      this.drawAdditiveLayer(
        encoder,
        target,
        undefined,
        null,
        camera,
        target.linear && !target.layer,
        projected ? "splatVertexProjected" : "splatVertex",
        projected,
        pixelKernels,
      );
    }
    const version = this.mappingVersion;
    this.submit(encoder);
    if (readback) this.sortFrom(readback, total, version);
    if (
      !gpu &&
      (this.drawVersion !== this.mappingVersion || this.drawCount === 0)
    ) {
      return;
    }

    this.writeDrawParams(
      camera,
      target.width,
      target.height,
      target.linear && !target.layer,
      additiveLayer ? DRAW_NO_ADDITIVE : 0,
    );
    if (additiveLayer) {
      this.addAdditiveLayer(
        pass,
        target.format,
        target.extraFormats ?? [],
        target.depthFormat ?? null,
        target.sampleCount,
      );
    }
    const depthFormat = this.options.depthTest ? target.depthFormat : null;
    const pixel = this.plugins?.pixelKernels(this.device) ?? null;
    const module = pixel?.draw ?? drawModule;
    const vertex = projected ? "splatVertexProjected" : "splatVertex";
    const key = [
      module.name,
      vertex,
      target.format,
      target.depthFormat,
      depthFormat ? "test" : "",
      target.sampleCount,
      target.depthCompare,
      target.layer ? "layer" : "",
      ...(target.extraFormats ?? []),
    ].join("/");
    let rp = this.pipelines.get(key);
    if (!rp) {
      const { colorTarget } = this.pipelineStates(target.format, null);
      if (target.layer) colorTarget.blend = SRGB_LAYER_BLEND;
      rp = createReflectedRenderPipeline(this.device, module, {
        vertex,
        fragment: "splatFragment",
        targets: [
          colorTarget,
          ...(target.extraFormats ?? []).map((format) => ({
            format,
            writeMask: 0,
          })),
        ],
        // The pass's depth attachment must match even when not testing.
        depthStencil: target.depthFormat
          ? {
              format: target.depthFormat,
              depthWriteEnabled: false,
              depthCompare: depthFormat
                ? (target.depthCompare ?? "less-equal")
                : "always",
            }
          : undefined,
        multisample: { count: target.sampleCount },
      });
      this.pipelines.set(key, rp);
    }
    const groups = createBindGroups(this.device, rp, {
      ordering: gpu ? this.sorter.ordering : (this.ordering as GPUBuffer),
      ...(projected ? { projected } : {}),
      splats: this.accumulator as GPUBuffer,
      params: this.drawUniform,
      ...pixel?.buffers,
    });
    pass.setPipeline(rp.pipeline);
    groups.forEach((g, i) => pass.setBindGroup(i, g));
    if (gpu) {
      pass.drawIndirect(this.sorter.drawArgs, 0);
    } else {
      pass.draw(4, this.drawCount);
    }
    this.stats.draws += 1;
    this.stats.drawn = gpu ? total : this.drawCount;
  }

  // Runs the dyno updaters for this frame; the camera's world position and
  // direction for generate.
  private updateDynos(camera: THREE.Camera) {
    camera.updateMatrixWorld();
    const cameraPos = new THREE.Vector3().setFromMatrixPosition(
      camera.matrixWorld,
    );
    const cameraDir = new THREE.Vector3(0, 0, -1).transformDirection(
      camera.matrixWorld,
    );
    const time = performance.now() / 1000;
    const deltaTime = time - this.lastTime;
    this.lastTime = time;
    if (this.plugins?.frame({ renderer: this, camera, time, deltaTime })) {
      this.dirty = true;
    }
    for (const mesh of this.meshes) {
      mesh.dyno?.update?.({
        camera,
        object: mesh.object,
        time,
        deltaTime,
        lod: mesh.lodIndices != null,
      });
    }
    // Generate-skipping frames must regenerate when this is set (dyno
    // uniforms animated by time, edited SDFs...).
    this.dynoDirty = this.dynoChanged();
    return { cameraPos, cameraDir };
  }

  // The GPU textures to draw into: three's for a RenderTarget, or the canvas.
  private resolveTarget(target?: THREE.RenderTarget): {
    color: GPUTexture;
    depth: GPUTexture | null;
    linear: boolean;
  } {
    const { renderer } = this;
    if (!target) {
      return {
        color: canvasContext(renderer).getCurrentTexture(),
        depth: null,
        linear: false,
      };
    }
    const color = gpuTexture(renderer, target.texture);
    if (!color) {
      throw new Error(
        "WgpuSplatRenderer: render the scene into the target first",
      );
    }
    // A multisampled target (samples > 1): three resolves its colour into
    // target.texture's single-sample texture at the end of its pass, which
    // is what `color` is; its depth stays multisampled, resolved in draw().
    const depth = target.depthTexture
      ? (gpuTexture(renderer, target.depthTexture) ?? null)
      : null;
    return {
      color,
      depth,
      // An -srgb format encodes on store (three picks one for 8-bit sRGB
      // targets): the shader writes linear values to it too.
      linear:
        target.texture.colorSpace !== THREE.SRGBColorSpace ||
        color.format.endsWith("-srgb"),
    };
  }

  // generate, then the stages that follow it (attribute gathering...), on
  // every frame that regenerates.
  private generateAll(
    encoder: GPUCommandEncoder,
    camera: THREE.Camera,
    cameraPos: THREE.Vector3,
    cameraDir: THREE.Vector3,
  ) {
    this.generate(encoder, camera, cameraPos, cameraDir);
    if (this.stages.length) {
      let base = 0;
      const ranges = this.meshes.map((mesh) => {
        const count = this.meshCount(mesh);
        base += count;
        return { mesh, base: base - count, count };
      });
      for (const s of this.stages) s.generate?.(encoder, ranges);
    }
  }

  private generate(
    encoder: GPUCommandEncoder,
    camera: THREE.Camera,
    cameraPos: THREE.Vector3,
    cameraDir: THREE.Vector3,
  ) {
    // The CPU sort draws a later frame in this metric's order: splats culled
    // for this camera would be missing when they come into view.
    const cull =
      this.options.cull && this.options.sort === "gpu"
        ? this.cullParams(camera)
        : undefined;
    this.written.depth = cull ? this.depthRange(camera) : null;
    const packed = this.packedFor(
      this.meshes.reduce((n, m) => n + this.meshCount(m), 0),
    );
    this.written.packed = packed;
    const origin = this.written.origin;
    if (packed) origin.copy(cameraPos);
    else origin.set(0, 0, 0);
    const pass = encoder.beginComputePass({
      label: "generate",
      timestampWrites: this.timestampWrites("generate"),
    });
    let base = 0;
    for (const mesh of this.meshes) {
      const count = this.meshCount(mesh);
      this.encodeGenerate(pass, mesh, {
        base,
        count,
        lod: mesh.lodIndices != null,
        cameraPos,
        cameraDir,
        outSplats: this.accumulator as GPUBuffer,
        sortMetric: this.metric as GPUBuffer,
        packed,
        origin,
        cull,
      });
      base += count;
    }
    pass.end();
  }

  // One mesh's generate dispatch: `count` splats into outSplats from `base`.
  private encodeGenerate(
    pass: GPUComputePassEncoder,
    mesh: WgpuSplatMesh,
    out: {
      base: number;
      count: number;
      /** Through the mesh's LOD indices (else source indices 0..count). */
      lod: boolean;
      cameraPos: THREE.Vector3;
      cameraDir: THREE.Vector3;
      outSplats: GPUBuffer;
      sortMetric: GPUBuffer;
      /** Plain Gsplats even when the accumulator holds covariance splats. */
      noCov?: boolean;
      /** The 16-byte accumulator, centres relative to `origin` (else ext). */
      packed?: boolean;
      origin?: THREE.Vector3;
      /** Frustum culling params (cullParams); no culling when absent. */
      cull?: ReturnType<WgpuSplatRenderer["cullParams"]>;
    },
  ) {
    const { base, count, cameraPos, cameraDir } = out;
    const plugin = this.plugins?.generateKernel(mesh) ?? null;
    const module = plugin?.module ?? generateModule;
    const kernel = this.registry.get(module, "generate");
    const position = new THREE.Vector3();
    const rotation = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    const inverse = new THREE.Matrix4();
    const basis = new THREE.Matrix4();
    const { source, object } = mesh;
    object.updateMatrixWorld();
    object.matrixWorld.decompose(position, rotation, scale);
    let dyno: DynoDispatch | null = null;
    if (DynoKernels.active(mesh.dyno)) {
      dyno = this.dynoKernels.prepare(mesh, mesh.dyno, module);
      if (mesh.dyno.worldSpace) {
        position.set(0, 0, 0);
        rotation.identity();
        scale.set(1, 1, 1);
      }
    }
    const viewObject = cameraPos
      .clone()
      .applyMatrix4(inverse.copy(object.matrixWorld).invert());
    let flags = (out.packed ? 0 : GEN_OUT_EXT) | (out.cull ? GEN_CULL : 0);
    if (source.format === "ext") flags |= GEN_SRC_EXT;
    if (out.lod) flags |= GEN_USE_LOD;
    if (out.lod && mesh.lodFaded) flags |= GEN_LOD_FADE;
    if (this.isAdditive(mesh) && !out.packed) flags |= GEN_ADDITIVE;
    if (source.lodOpacity) flags |= GEN_LOD_OPACITY;
    if (this.options.sortRadial) flags |= GEN_SORT_RADIAL;
    if (dyno && mesh.dyno?.generator) flags |= GEN_DYNO_SOURCE;
    if (this.options.covSplats && !out.noCov) {
      flags |= GEN_OUT_COV;
      // Gsplat world modifiers need the similarity transform before them.
      if (!mesh.dyno?.worldModifiers?.length) flags |= GEN_COV_TRANSFORM;
    }
    if (dyno && mesh.dyno?.worldSpace) basis.identity();
    else basis.copy(object.matrixWorld);
    const b = basis.elements;
    const origin = out.origin ?? new THREE.Vector3();
    const params = UniformWriter.for(generateModule).setAll({
      numSplats: count,
      outBase: base,
      flags,
      numSh: source.numSh,
      srcCount: source.count,
      shStride: source.shStride,
      rotate: [rotation.x, rotation.y, rotation.z, rotation.w],
      translateScale: [
        position.x,
        position.y,
        position.z,
        (scale.x + scale.y + scale.z) / 3,
      ],
      recolor: mesh.recolor.toArray(),
      encoding: source.encoding,
      shMax: source.shMax,
      viewObject: [viewObject.x, viewObject.y, viewObject.z, 0],
      viewCenter: [cameraPos.x, cameraPos.y, cameraPos.z, 0],
      viewDir: [cameraDir.x, cameraDir.y, cameraDir.z, 0],
      outOrigin: [origin.x, origin.y, origin.z, 0],
      covBasis0: [b[0], b[1], b[2], 0],
      covBasis1: [b[4], b[5], b[6], 0],
      covBasis2: [b[8], b[9], b[10], 0],
      ...out.cull,
    });
    (dyno?.kernel ?? kernel).dispatch(pass, {
      bindings: dyno?.bindings,
      grid: [count],
      buffers: {
        src: source.src,
        sh: source.sh ?? this.emptyBuffer,
        lodIndices: mesh.lodBuffer ?? this.emptyBuffer,
        outSplats: out.outSplats,
        sortMetric: out.sortMetric,
        ...plugin?.buffers,
      },
      uniforms: plugin
        ? { ...plugin.uniforms, params: params.data }
        : params.data,
    });
  }

  /**
   * `mesh`'s splats as generated now (its modifiers and recolor, SH for
   * `camera`), their RGBA by source index into `texture`: an rgba8unorm
   * 2d-array with STORAGE_BINDING, splatTexCoord layout (RgbaArray's). For
   * SparkRenderer.getRgba, which bakes painted colours into SplatMesh.splatRgba.
   */
  bakeRgba(mesh: WgpuSplatMesh, camera: THREE.Camera, texture: GPUTexture) {
    const count = mesh.source.count;
    if (count === 0) return;
    camera.updateMatrixWorld();
    const cameraPos = new THREE.Vector3().setFromMatrixPosition(
      camera.matrixWorld,
    );
    const cameraDir = new THREE.Vector3(0, 0, -1).transformDirection(
      camera.matrixWorld,
    );
    const { device } = this;
    const outSplats = createStorage(device, count * 32, "baked splats");
    const sortMetric = createStorage(device, count * 4, "baked metric");
    const encoder = device.createCommandEncoder({ label: "bake rgba" });
    const pass = encoder.beginComputePass({ label: "bake rgba" });
    this.encodeGenerate(pass, mesh, {
      base: 0,
      count,
      lod: false,
      cameraPos,
      cameraDir,
      outSplats,
      sortMetric,
      noCov: true,
    });
    this.bakePipeline ??= device.createComputePipeline({
      label: "bake rgba",
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: BAKE_RGBA_WGSL }),
        entryPoint: "main",
      },
    });
    const countBuffer = createUniform(device, 16, "bake count");
    device.queue.writeBuffer(countBuffer, 0, new Uint32Array([count, 0, 0, 0]));
    pass.setPipeline(this.bakePipeline);
    pass.setBindGroup(
      0,
      device.createBindGroup({
        layout: this.bakePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: outSplats } },
          {
            binding: 1,
            resource: texture.createView({ dimension: "2d-array" }),
          },
          { binding: 2, resource: { buffer: countBuffer } },
        ],
      }),
    );
    pass.dispatchWorkgroups(Math.ceil(count / 256));
    pass.end();
    this.registry.submit(encoder.finish());
    outSplats.destroy();
    sortMetric.destroy();
    countBuffer.destroy();
  }

  // The range of the metrics generate's cull lets through for `camera`
  // (depthKeys): the view depth between the planes, + 100 along the view
  // axis; radially, up to the far plane's corners past clipXY.
  private depthRange(camera: THREE.Camera): SortDepthRange | null {
    const cam = camera as THREE.PerspectiveCamera;
    if (!this.options.depthKeys || !cam.isPerspectiveCamera) return null;
    const { near, far } = cam;
    const p = cam.projectionMatrix.elements;
    if (!(near > 0 && far > near && p[0] && p[5])) return null;
    let range: SortDepthRange;
    if (this.options.sortRadial) {
      const x = (this.options.clipXY + Math.abs(p[8])) / Math.abs(p[0]);
      const y = (this.options.clipXY + Math.abs(p[9])) / Math.abs(p[5]);
      range = { lo: near, hi: far * Math.sqrt(1 + x * x + y * y) };
    } else {
      range = { lo: near + 100, hi: far + 100 };
    }
    return (range.hi - range.lo) / 2 ** 24 <= 2.5e-4 ? range : null;
  }

  // generate's copy of the draw's early outs (GEN_CULL), for this camera.
  private cullParams(camera: THREE.Camera) {
    const view = camera.matrixWorldInverse;
    const vp = new THREE.Matrix4().multiplyMatrices(
      camera.projectionMatrix,
      view,
    ).elements;
    const v = view.elements;
    return {
      cullProj0: vp.slice(0, 4),
      cullProj1: vp.slice(4, 8),
      cullProj2: vp.slice(8, 12),
      cullProj3: vp.slice(12, 16),
      cullViewZ: [v[2], v[6], v[10], v[14]],
      cullParams: [this.options.clipXY, this.options.minAlpha, 0, 0],
    };
  }

  private copyMetric(encoder: GPUCommandEncoder, total: number): GPUBuffer {
    const staging = createReadback(
      this.device,
      total * 4,
      "sort metric readback",
    );
    encoder.copyBufferToBuffer(
      this.metric as GPUBuffer,
      0,
      staging,
      0,
      total * 4,
    );
    this.sortPending = true;
    return staging;
  }

  // Back to front: largest metric first; inactive splats (+inf) left out.
  private async sortFrom(staging: GPUBuffer, total: number, version: number) {
    try {
      await staging.mapAsync(GPUMapMode.READ);
      const metric = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      if (version !== this.mappingVersion || !this.ordering) return;
      const t0 = performance.now();
      const order = sortBackToFront(metric, total);
      this.stats.sortMs = performance.now() - t0;
      this.stats.sorts += 1;
      this.device.queue.writeBuffer(
        this.ordering,
        0,
        order.buffer,
        0,
        order.byteLength,
      );
      this.drawCount = order.length;
      this.drawVersion = version;
    } finally {
      staging.destroy();
      this.sortPending = false;
    }
  }

  // Whether this frame's quads read the projector's records (projectOnce):
  // GPU sort, while the records fit one binding.
  private projects(total: number): boolean {
    return (
      this.options.projectOnce &&
      this.options.sort === "gpu" &&
      total * PROJECTED_BYTES <= this.device.limits.maxStorageBufferBindingSize
    );
  }

  // The projector over the sorted slots, after the sort and with this
  // frame's draw params (already written), into `encoder`.
  private encodeProject(
    encoder: GPUCommandEncoder,
    total: number,
    drawParams: ArrayBuffer,
  ): GPUBuffer {
    const bytes = Math.max(total, 1) * PROJECTED_BYTES;
    if (!this.projected || this.projected.size < bytes) {
      this.projected?.destroy();
      this.projected = createStorage(
        this.device,
        Math.min(
          Math.ceil(bytes * (this.projected ? 1.5 : 1)),
          this.device.limits.maxStorageBufferBindingSize,
        ),
        "projected splats",
      );
    }
    const pass = encoder.beginComputePass({
      label: "project",
      timestampWrites: this.timestampWrites("project"),
    });
    this.registry.get(drawModule, "projectSplats").dispatchIndirect(
      pass,
      {
        buffers: {
          ordering: this.sorter.ordering,
          sortCount: this.sorter.drawArgs,
          projectedOut: this.projected,
          splats: this.accumulator as GPUBuffer,
        },
        uniforms: drawParams,
      },
      this.sorter.dispatchArgs,
    );
    pass.end();
    return this.projected;
  }

  private pipelineStates(
    format: GPUTextureFormat,
    depthFormat: GPUTextureFormat | null,
  ): { colorTarget: GPUColorTargetState; depthStencil?: GPUDepthStencilState } {
    const blend: GPUBlendState = {
      color: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
      alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
    };
    return {
      colorTarget: { format, blend },
      depthStencil: depthFormat
        ? {
            format: depthFormat,
            depthWriteEnabled: false,
            depthCompare: "less-equal",
          }
        : undefined,
    };
  }

  private pipeline(
    format: GPUTextureFormat,
    depthFormat: GPUTextureFormat | null,
    layer = false,
    module: KernelModule = drawModule,
    vertex = "splatVertex",
  ) {
    const key = `${format}/${depthFormat}${layer ? "/layer" : ""}/${module.name}/${vertex}`;
    let p = this.pipelines.get(key);
    if (!p) {
      const { colorTarget, depthStencil } = this.pipelineStates(
        format,
        depthFormat,
      );
      if (layer) colorTarget.blend = SRGB_LAYER_BLEND;
      p = createReflectedRenderPipeline(this.device, module, {
        vertex,
        fragment: "splatFragment",
        targets: [colorTarget],
        depthStencil,
      });
      this.pipelines.set(key, p);
    }
    return p;
  }

  private draw(
    encoder: GPUCommandEncoder,
    camera: THREE.Camera,
    renderTarget?: THREE.RenderTarget,
    gpuSorted?: number,
  ) {
    const gpu = gpuSorted !== undefined;
    if (
      !gpu &&
      (this.drawVersion !== this.mappingVersion || this.drawCount === 0)
    )
      return;
    const { color: target, depth, linear } = this.resolveTarget(renderTarget);
    const size = { x: target.width, y: target.height };
    let depthTexture = this.options.depthTest ? depth : null;
    if (depthTexture && depthTexture.sampleCount > 1) {
      // Splats draw single-sampled over the resolved colour, tested against
      // a single-sample copy of the multisampled depth.
      this.depthResolve ??= new DepthResolve(this.device);
      depthTexture = this.depthResolve.resolve(encoder, depthTexture);
    }
    const depthView = depthTexture?.createView();
    const depthFormat = depthTexture?.format ?? null;

    const pixel = this.plugins?.pixelKernels(this.device) ?? null;
    this.pixel = pixel;
    let variant: SplatDrawVariant | null = null;
    for (const s of this.stages) {
      variant ??=
        s.draw?.({
          ...this.pipelineStates(target.format, depthFormat),
          width: size.x,
          height: size.y,
          plugin: pixel,
        }) ?? null;
    }
    // Blended in sRGB in a layer of their own, composited after.
    const layer = linear && this.options.srgbBlend && !variant && !this.hdr;
    const drawParams = this.writeDrawParams(
      camera,
      size.x,
      size.y,
      linear && !layer,
    );
    this.tiles?.poll();
    // Additive splats are drawn as a layer of their own (drawAdditiveLayer):
    // the quad draw only. A draw variant (attribute targets) draws the rest;
    // the layer has the colour alone.
    const additiveLayer =
      gpu &&
      !layer &&
      !this.written.packed &&
      this.meshes.some((m) => this.isAdditive(m));
    const either = gpu && !variant && !additiveLayer;
    const path = either ? this.rasterPath() : "hardware";
    this.stats.additiveLayer = additiveLayer;
    // Times the draw for rasterizer "auto" (options.profile times it anyway).
    let drawEnd: GPURenderPassTimestampWrites | undefined;
    if (either && this.autoTimer && !this.options.profile) {
      if (path === "warm") this.autoTimer.skip();
      else
        drawEnd = this.autoTimer.begin(encoder, path, path !== this.auto?.path);
    }
    if (path === "tiles") {
      this.drawTiles(encoder, target, depthTexture, layer, drawParams, drawEnd);
      this.stats.draws += 1;
      this.stats.drawn = gpuSorted as number;
      this.stats.rasterizer = "tiles";
      this.stats.projected = false;
      return;
    }
    if (path === "warm") {
      // Untimed, so options.profile's frame reads as a hardware one.
      this.encodeTiles(encoder, target, depthTexture, layer, drawParams, false);
    }
    this.stats.rasterizer = "hardware";
    const projected =
      gpu && !variant && this.projects(gpuSorted as number)
        ? this.encodeProject(encoder, gpuSorted as number, drawParams)
        : null;
    const vertex = projected ? "splatVertexProjected" : "splatVertex";
    this.stats.projected = projected !== null;
    if (additiveLayer) {
      this.drawAdditiveLayer(
        encoder,
        target,
        depthView,
        depthFormat,
        camera,
        linear,
        vertex,
        projected,
        pixel,
      );
      // The layer's light added to the frame (its own coverage dropped).
      const composite = encoder.beginRenderPass({
        label: "additive splats add",
        colorAttachments: [
          { view: target.createView(), loadOp: "load", storeOp: "store" },
        ],
      });
      this.addAdditiveLayer(composite, target.format, [], null, 1);
      composite.end();
      this.writeDrawParams(camera, size.x, size.y, linear, DRAW_NO_ADDITIVE);
    }
    const timestampWrites = this.timestampWrites("draw") ?? drawEnd;
    const rp =
      variant?.pipeline ??
      (layer
        ? this.pipeline(
            SRGB_LAYER_FORMAT,
            depthFormat,
            true,
            pixel?.draw,
            vertex,
          )
        : this.pipeline(
            target.format,
            depthFormat,
            false,
            pixel?.draw,
            vertex,
          ));
    const groups = createBindGroups(this.device, rp, {
      ordering: gpu ? this.sorter.ordering : (this.ordering as GPUBuffer),
      ...(projected ? { projected } : {}),
      splats: this.accumulator as GPUBuffer,
      params: this.drawUniform,
      ...pixel?.buffers,
      ...variant?.buffers,
    });
    const depthAttachment: GPURenderPassDepthStencilAttachment | undefined =
      depthView ? depthTestAttachment(depthView, depthFormat) : undefined;
    const pass = layer
      ? this.srgbComposite.beginLayer(
          encoder,
          target,
          depthAttachment,
          timestampWrites,
        )
      : encoder.beginRenderPass({
          label: "splats",
          timestampWrites,
          colorAttachments: [
            { view: target.createView(), loadOp: "load", storeOp: "store" },
            ...(variant?.attachments ?? []),
          ],
          depthStencilAttachment: depthAttachment,
        });
    pass.setPipeline(rp.pipeline);
    groups.forEach((g, i) => pass.setBindGroup(i, g));
    if (gpu) {
      pass.drawIndirect(this.sorter.drawArgs, 0);
    } else {
      pass.draw(4, this.drawCount);
    }
    this.stats.draws += 1;
    this.stats.drawn = gpu ? (gpuSorted as number) : this.drawCount;
    pass.end();
    if (layer) {
      this.srgbComposite.composite(encoder, target, {
        view: target.createView(),
      });
    }
  }

  // The tile rasterizer's draw: its compute stages, then its image blended
  // over the target as the quads would be.
  // `end`: timestamp writes for the composite pass.
  private drawTiles(
    encoder: GPUCommandEncoder,
    target: GPUTexture,
    depth: GPUTexture | null,
    layer: boolean,
    drawParams: ArrayBuffer,
    end?: GPURenderPassTimestampWrites,
  ) {
    const format = this.encodeTiles(encoder, target, depth, layer, drawParams);
    const timestampWrites = this.timestampWrites("draw") ?? end;
    const tiles = this.tiles as TileRasterizer;
    const pass = layer
      ? this.srgbComposite.beginLayer(
          encoder,
          target,
          undefined,
          timestampWrites,
        )
      : encoder.beginRenderPass({
          label: "splats",
          timestampWrites,
          colorAttachments: [
            { view: target.createView(), loadOp: "load", storeOp: "store" },
          ],
        });
    const blend = layer
      ? SRGB_LAYER_BLEND
      : (this.pipelineStates(format, null).colorTarget.blend as GPUBlendState);
    tiles.composite(pass, format, blend);
    pass.end();
    if (layer) {
      this.srgbComposite.composite(encoder, target, {
        view: target.createView(),
      });
    }
  }

  // The tile stages, leaving their image for composite; the target format.
  // `profile`: time them with options.profile.
  private encodeTiles(
    encoder: GPUCommandEncoder,
    target: GPUTexture,
    depth: GPUTexture | null,
    layer: boolean,
    drawParams: ArrayBuffer,
    profile = true,
  ) {
    this.tiles ??= new TileRasterizer(this.registry);
    const format = layer ? SRGB_LAYER_FORMAT : target.format;
    this.tiles.encode(encoder, {
      ordering: this.sorter.ordering,
      sortCount: this.sorter.drawArgs,
      slots: this.meshes.reduce((n, m) => n + this.meshCount(m), 0),
      splats: this.accumulator as GPUBuffer,
      drawParams,
      width: target.width,
      height: target.height,
      depth,
      clamp: format.includes("unorm"),
      profiler: this.options.profile && profile ? this.profiler : null,
      module: this.pixel?.tiles,
      pluginUniforms: this.pixel?.uniforms,
    });
    return format;
  }

  /** Composites splat layers over linear targets (srgbBlend). */
  get srgbComposite(): SrgbComposite {
    this.srgb ??= new SrgbComposite(this.device);
    return this.srgb;
  }

  // The draw uniforms for this camera and target size.
  /** Whether `mesh`'s splats are additive (WgpuSplatMesh.additive). */
  private isAdditive(mesh: WgpuSplatMesh): boolean {
    return Boolean(
      mesh.additive ?? (mesh.object as { additive?: boolean }).additive,
    );
  }

  /**
   * The additive splats' layer (athenea's light catcher): drawn alone into
   * a cleared float target, blended among themselves as any splats (each
   * covers the additive splats behind it), and that layer added to the
   * frame before the other splats are drawn over it. A light catcher lies
   * on the floor under everything else, so the rest covering the sum is
   * athenea's blend (splat_blend.slang's lampTransmittance).
   */
  private drawAdditiveLayer(
    encoder: GPUCommandEncoder,
    target: { width: number; height: number },
    depthView: GPUTextureView | undefined,
    depthFormat: GPUTextureFormat | null,
    camera: THREE.Camera,
    linear: boolean,
    vertex: string,
    projected: GPUBuffer | null,
    pixel: { draw?: KernelModule; buffers?: Record<string, GPUBuffer> } | null,
  ) {
    const size = [target.width, target.height];
    const format: GPUTextureFormat = "rgba16float";
    if (
      !this.additiveLayer ||
      this.additiveLayer.width !== size[0] ||
      this.additiveLayer.height !== size[1]
    ) {
      this.additiveLayer?.destroy();
      this.additiveLayer = this.device.createTexture({
        label: "additive splats layer",
        size,
        format,
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
    }
    this.additiveUniform ??= createUniform(
      this.device,
      UniformWriter.for(drawModule).data.byteLength,
      "splat draw params (additive layer)",
    );
    this.writeDrawParams(
      camera,
      size[0],
      size[1],
      linear,
      DRAW_ADDITIVE_ONLY,
      this.additiveUniform,
    );
    const rp = this.pipeline(format, depthFormat, false, pixel?.draw, vertex);
    const groups = createBindGroups(this.device, rp, {
      ordering: this.sorter.ordering,
      ...(projected ? { projected } : {}),
      splats: this.accumulator as GPUBuffer,
      params: this.additiveUniform,
      ...pixel?.buffers,
    });
    const pass = encoder.beginRenderPass({
      label: "additive splats",
      colorAttachments: [
        {
          view: this.additiveLayer.createView(),
          loadOp: "clear",
          clearValue: [0, 0, 0, 0],
          storeOp: "store",
        },
      ],
      depthStencilAttachment: depthView
        ? depthTestAttachment(depthView, depthFormat)
        : undefined,
    });
    pass.setPipeline(rp.pipeline);
    groups.forEach((g, i) => pass.setBindGroup(i, g));
    pass.drawIndirect(this.sorter.drawArgs, 0);
    pass.end();
  }

  /**
   * Adds the additive layer's light (drawAdditiveLayer's texture) in `pass`:
   * a full-screen draw blending one, one into the colour, its alpha and
   * the other attachments untouched (renderInPass draws it in three's
   * pass, before the splats).
   */
  private addAdditiveLayer(
    pass: GPURenderPassEncoder,
    format: GPUTextureFormat,
    extraFormats: GPUTextureFormat[],
    depthFormat: GPUTextureFormat | null,
    samples: number,
  ) {
    if (!this.additiveLayer) return;
    const key = [format, ...extraFormats, depthFormat, samples].join("/");
    let add = this.additiveAdd.get(key);
    if (!add) {
      const module = this.device.createShaderModule({
        label: "additive layer add",
        code: /* wgsl */ `
@group(0) @binding(0) var layer: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  return vec4f(textureLoad(layer, vec2u(p.xy), 0).rgb, 0.0);
}`,
      });
      add = this.device.createRenderPipeline({
        label: "additive layer add",
        layout: "auto",
        vertex: { module, entryPoint: "vs" },
        fragment: {
          module,
          entryPoint: "fs",
          targets: [
            {
              format,
              blend: {
                color: { srcFactor: "one", dstFactor: "one" },
                alpha: { srcFactor: "zero", dstFactor: "one" },
              },
            },
            ...extraFormats.map((f) => ({ format: f, writeMask: 0 })),
          ],
        },
        depthStencil: depthFormat
          ? {
              format: depthFormat,
              depthWriteEnabled: false,
              depthCompare: "always",
            }
          : undefined,
        multisample: { count: samples },
        primitive: { topology: "triangle-list" },
      });
      this.additiveAdd.set(key, add);
    }
    pass.setPipeline(add);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: add.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: this.additiveLayer.createView() }],
      }),
    );
    pass.draw(3);
  }

  private writeDrawParams(
    camera: THREE.Camera,
    width: number,
    height: number,
    linear: boolean,
    extraFlags = 0,
    buffer: GPUBuffer = this.drawUniform,
  ) {
    this.lastDrawSize = { width, height };
    const view = camera.matrixWorldInverse;
    const viewQuat = new THREE.Quaternion().setFromRotationMatrix(view);
    // World = stored + origin, so the origin moves into the translation.
    const viewPos = new THREE.Vector3()
      .copy(this.written.origin)
      .applyMatrix4(view);
    const p = camera.projectionMatrix.elements;
    const basis = new THREE.Matrix3().setFromMatrix4(view).elements;
    const o = this.options;
    const disk = this.diskClip?.radius ? this.diskClip : null;
    const params = UniformWriter.for(drawModule).setAll({
      proj0: p.slice(0, 4),
      proj1: p.slice(4, 8),
      proj2: p.slice(8, 12),
      proj3: p.slice(12, 16),
      renderToViewQuat: [viewQuat.x, viewQuat.y, viewQuat.z, viewQuat.w],
      renderToViewPos: [viewPos.x, viewPos.y, viewPos.z, 0],
      renderToViewBasis0: basis.slice(0, 3),
      renderToViewBasis1: basis.slice(3, 6),
      renderToViewBasis2: basis.slice(6, 9),
      renderWidth: width,
      renderHeight: height,
      maxStdDev: o.maxStdDev,
      minPixelRadius: o.minPixelRadius,
      maxPixelRadius: o.maxPixelRadius,
      minAlpha: o.minAlpha,
      blurAmount: o.blurAmount,
      preBlurAmount: o.preBlurAmount,
      focalDistance: o.focalDistance,
      apertureAngle: o.apertureAngle,
      clipXY: o.clipXY,
      focalAdjustment: o.focalAdjustment,
      falloff: o.falloff,
      diskCenter: disk ? [...disk.center.toArray(), disk.radius] : [0, 0, 0, 0],
      diskNormal: disk
        ? [...disk.normal.toArray(), disk.twoSided ? 1 : 0]
        : [0, 0, 0, 0],
      flags:
        (this.written.packed ? 0 : DRAW_EXT) |
        DRAW_PREMULTIPLIED |
        (o.covSplats ? DRAW_COV : 0) |
        (o.surfels !== "ewa" || o.enable2DGS ? DRAW_2DGS : 0) |
        (o.lodInflate ? DRAW_LOD_INFLATE : 0) |
        (linear ? DRAW_ENCODE_LINEAR : 0) |
        (disk ? DRAW_DISK_CLIP : 0) |
        ((camera as THREE.OrthographicCamera).isOrthographicCamera
          ? DRAW_ORTHOGRAPHIC
          : 0) |
        extraFlags,
    });
    this.device.queue.writeBuffer(buffer, 0, params.data);
    return params.data;
  }

  dispose() {
    for (const m of this.meshes) m.lodBuffer?.destroy();
    this.accumulator?.destroy();
    this.metric?.destroy();
    this.ordering?.destroy();
    this.projected?.destroy();
    this.drawUniform.destroy();
    this.additiveUniform?.destroy();
    this.additiveLayer?.destroy();
    this.emptyBuffer.destroy();
    this.sorter.destroy();
    this.srgb?.dispose();
    this.tiles?.destroy();
    this.depthResolve?.dispose();
    this.profiler?.destroy();
    this.autoTimer?.destroy();
    this.registry.destroy();
  }
}

export { GpuSplatSource } from "./GpuSplatSource";
export type { WgpuDyno, WgpuDynoFrame } from "./dyno/DynoKernels";
export { sortBackToFront } from "./cpuSort";

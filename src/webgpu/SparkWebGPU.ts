// SparkRenderer on three's WebGPURenderer: the scene's SplatMeshes and
// SplatGenerators drawn by WgpuSplatRenderer inside three's own render pass.
//
// SparkRenderer stays a transparent THREE.Mesh in the scene, but with an
// empty instanced geometry three never draws. three calls its onBeforeRender
// while recording its render pass, between the opaque objects and the
// transparent ones sorted after it; there we submit generate and sort (they
// run before three's pass, submitted when the render finishes) and record the
// splat draw into three's open pass. The splats then test against the depth
// three is using, on the canvas, in render targets and in post-processing
// passes alike, and the user's renderer.render(scene, camera) is unchanged.
//
// Each frame the scene is walked as SparkRenderer.prepareGenerate does:
// visible SplatMeshes become WgpuSplatRenderer meshes (their PackedSplats or
// ExtSplats uploaded once, shared by meshes drawing the same splats), LoD and
// paged ones go through WgpuLod, their modifiers, edits and skinning through
// splatMeshDyno, and other SplatGenerators through splatGeneratorDyno.

import * as THREE from "three";
import { ExtSplats } from "../ExtSplats";
import { PackedSplats } from "../PackedSplats";
import { PagedSplats } from "../PagedSplats";
import type { RgbaArray } from "../RgbaArray";
import type { SparkRenderer } from "../SparkRenderer";
import { type SplatEdit, isSplatEdit } from "../SplatEdit";
import { SplatGenerator } from "../SplatGenerator";
import { EmptySplatSource, SplatMesh, type SplatSource } from "../SplatMesh";
import { DepthResolve } from "./DepthResolve";
import { GpuSplatSource } from "./GpuSplatSource";
import { type CompositeToneMapping, SRGB_LAYER_FORMAT } from "./SrgbComposite";
import { WgpuLod, type WgpuLodMesh } from "./WgpuLod";
import {
  type SplatDiskClip,
  type SplatPassTarget,
  type WgpuSplatMesh,
  WgpuSplatRenderer,
} from "./WgpuSplatRenderer";
import type { WgpuDyno } from "./dyno/DynoKernels";
import {
  splatGeneratorDyno,
  splatMeshDyno,
  splatSourceMeshDyno,
} from "./dyno/adapters";
import { depthTestAttachment } from "./renderPipeline";
import {
  type OpenPass,
  type ThreeWebGPURenderer,
  canvasContext,
  openPass,
  webgpuBackend,
} from "./threeInternals";
import type { WebGPURendererLike } from "./threeRenderer";
import { INVERTIBLE_TONE_MAPPINGS } from "./toneMapping";

export type { WebGPURendererLike };

/** Whether `renderer` is three's WebGPURenderer (either backend). */
export function isWebGPURenderer(
  renderer: unknown,
): renderer is WebGPURendererLike {
  return !!(renderer as WebGPURendererLike | undefined)?.isWebGPURenderer;
}

type BaseSplats = PackedSplats | ExtSplats;

// A SplatMesh or SplatGenerator as drawn this frame.
interface Entry {
  /** What the entry was built from; rebuilt when it changes. */
  key: unknown[];
  dyno: WgpuDyno;
  mesh?: WgpuSplatMesh;
  lodMesh?: WgpuLodMesh;
  /** Uploaded splats shared with other meshes drawing the same ones. */
  shared?: SharedSource;
  /** Drawn from a dyno generator of this many splats (a custom SplatSource). */
  generated?: number;
  /** A WgpuLod.add in flight. */
  pending?: boolean;
  detached?: boolean;
}

interface SharedSource {
  splats: BaseSplats;
  source: GpuSplatSource;
  users: number;
}

const warned = new Set<string>();
function warnOnce(message: string) {
  if (warned.has(message)) return;
  warned.add(message);
  console.warn(`SparkRenderer (WebGPU): ${message}`);
}

export class SparkWebGPU {
  splats?: WgpuSplatRenderer;
  lod?: WgpuLod;
  private entries = new Map<SplatGenerator, Entry>();
  private shared = new Map<BaseSplats, SharedSource>();
  private initWatched = new WeakSet<SplatMesh>();
  private globalEdits: SplatEdit[] = [];
  private lastFrame = -1;
  private lastTime = performance.now() / 1000;
  private lastCamera: THREE.Camera | null = null;
  private failed = false;
  private depthResolve?: DepthResolve;
  /** A visible SplatMesh has covSplats (see applyOptions). */
  private covMeshes = false;
  /** WgpuSplatRenderer's accumulator option before accumExtSplats. */
  private baseAccumulator: WgpuSplatRenderer["options"]["accumulator"] = "auto";
  private pmrem?: {
    Class: unknown;
    generator: {
      fromCubemap(t: THREE.Texture): THREE.RenderTarget;
      dispose(): void;
    };
  };

  constructor(
    readonly spark: SparkRenderer,
    readonly renderer: WebGPURendererLike,
  ) {}

  /**
   * LoD is traversed for this camera when set (else each frame's first
   * render camera): SparkPortals drives it from the main view.
   */
  lodCamera: THREE.Camera | null = null;

  /**
   * SparkRenderer.onBeforeRender on WebGPU: syncs the scene and draws.
   * `host` is the SparkRenderer in the scene, when this one draws in its
   * place (SparkRenderer.render / sparkOverride): its portal disk uniforms
   * apply, as its fragment shader would on WebGL.
   */
  onBeforeRender(
    scene: THREE.Scene,
    camera: THREE.Camera,
    host: SparkRenderer = this.spark,
  ) {
    if (this.failed) return;
    const { renderer } = this;
    if (!webgpuBackend(renderer)) {
      this.failed = true;
      console.error(
        "SparkRenderer: WebGPURenderer fell back to WebGL; splats need its WebGPU backend",
      );
      return;
    }
    if (renderer.xr?.isPresenting) {
      warnOnce("XR is not supported");
      return;
    }
    const open = openPass(renderer);
    if (!open) {
      warnOnce("no open render pass to draw into (render bundles?)");
      return;
    }
    const rc = open.context;

    const splats = this.ensureRenderer();
    // An ArrayCamera (with sub-cameras) draws each into its viewport, as
    // three does; its first one is the frame's camera for LoD and getRgba.
    const views = (camera as THREE.ArrayCamera).isArrayCamera
      ? (camera as THREE.ArrayCamera).cameras
      : [];
    const mainCamera: THREE.Camera = views[0] ?? camera;
    this.lastCamera = mainCamera;
    const width = rc.viewport ? rc.viewportValue.z : rc.width;
    const height = rc.viewport ? rc.viewportValue.w : rc.height;

    const frame = renderer.info.frame;
    if (frame !== this.lastFrame) {
      // Once per frame, as SparkRenderer updates on a new frame.
      this.lastFrame = frame;
      this.sync(scene, camera);
      this.applyOptions();
      if (this.lod?.active && this.spark.enableDriveLod) {
        const size = views.length
          ? this.viewportOf(views[0], width, height)
          : { z: width, w: height };
        this.lod.update(this.lodCamera ?? mainCamera, {
          x: size.z,
          y: size.w,
        });
      }
    }
    if (views.length) {
      splats.diskClip = diskClip(host);
      this.spark.dirty = false;
      this.drawViewsAfterRender(scene, views, open);
      return;
    }

    const format = open.colorFormat;
    // An -srgb format (three's for 8-bit sRGB targets) encodes on store, so
    // the shader writes linear values to it, as three's materials do.
    const srgbFormat = format.endsWith("-srgb");
    // HDR (SparkRendererOptions.hdr, or a plugin making linear light past
    // 1): the light stays in float, blended linearly into a float target.
    const hdr = splats.hdr && !this.spark.rawColor;
    const floatFormat = format.includes("float");
    const target: SplatPassTarget = {
      format,
      depthFormat: open.depthFormat,
      sampleCount: open.sampleCount,
      width,
      height,
      linear: this.spark.rawColor
        ? srgbFormat
        : srgbFormat ||
          (hdr && floatFormat) ||
          open.colorSpace !== THREE.SRGBColorSpace,
      extraFormats: open.extraFormats,
      depthCompare: (camera as { reversedDepth?: boolean }).reversedDepth
        ? "greater-equal"
        : "less-equal",
    };
    splats.diskClip = diskClip(host);
    this.spark.dirty = false;
    // On the canvas three renders into a linear half-float target, then
    // converts it in an output pass.
    const canvas =
      open.isFrameBufferTarget &&
      !renderer.getOutputRenderTarget() &&
      (scene as THREE.Scene).isScene;
    // three tone-maps the canvas in its output pass; WebGL Spark doesn't
    // tone map splats. The composite inverts three's operators (see
    // SrgbComposite); for others (a custom node) splats go on the canvas
    // after the output pass.
    // With HDR into a float target the splats are light like three's own
    // objects: three's output pass tone-maps them with the rest.
    const floatHdr = hdr && floatFormat;
    const toneMapping =
      canvas && !floatHdr
        ? (renderer.toneMapping ?? THREE.NoToneMapping)
        : THREE.NoToneMapping;
    const invertible =
      INVERTIBLE_TONE_MAPPINGS.includes(toneMapping) &&
      renderer.outputColorSpace === THREE.SRGBColorSpace &&
      !this.spark.rawColor;
    // With MSAA too: drawOnCanvas tests against the resolved depth. Raw
    // colours too, which three's output pass would encode, unlike WebGL's
    // canvas.
    if (
      (toneMapping !== THREE.NoToneMapping && !invertible) ||
      (canvas && this.spark.rawColor)
    ) {
      this.drawAfterOutput(scene, camera, open);
      return;
    }
    // The 8-bit sRGB layer (WebGL's canvas, for parity) clamps the
    // background and the splats' light to 1 and quantises both: never with
    // HDR into a float target.
    if (
      rc.textures &&
      target.linear &&
      !this.spark.rawColor &&
      !floatHdr &&
      (canvas || this.spark.srgbBlend)
    ) {
      this.drawSrgbBlended(camera, open, target, {
        toneMapping: invertible ? toneMapping : THREE.NoToneMapping,
        exposure: renderer.toneMappingExposure ?? 1,
      });
      return;
    }
    splats.renderInPass(camera, open.pass, target);
    // three skips setting a pipeline or bind group it believes is still set.
    open.resetState();
  }

  // Splats blend in sRGB space, as WebGL Spark blends them on the canvas and
  // as they were trained, but three's target is linear. So three's pass is
  // ended where the splats come in the transparent order, the splats are
  // composited over its colour in sRGB (SrgbComposite), and the pass
  // resumes, as three's own copyFramebufferToTexture does: transparent
  // objects in front of the splats still draw over them.
  private drawSrgbBlended(
    camera: THREE.Camera,
    open: OpenPass,
    target: SplatPassTarget,
    toneMapping: CompositeToneMapping,
  ) {
    const splats = this.splats as WgpuSplatRenderer;
    const { descriptor, encoder, context: rc } = open;
    const attachment = [
      ...descriptor.colorAttachments,
    ][0] as GPURenderPassColorAttachment;
    const color = open.colorTexture();
    open.end();

    const depth = descriptor.depthStencilAttachment;
    const composite = splats.srgbComposite;
    const layer = composite.beginLayer(
      encoder,
      color,
      depth ? depthTestAttachment(depth.view, target.depthFormat) : undefined,
      undefined,
      toneMapping,
    );
    const viewport = rc.viewport ? rc.viewportValue : null;
    if (viewport) {
      layer.setViewport(viewport.x, viewport.y, viewport.z, viewport.w, 0, 1);
    }
    splats.renderInPass(camera, layer, {
      ...target,
      format: SRGB_LAYER_FORMAT,
      extraFormats: undefined,
      linear: false,
      layer: true,
    });
    layer.end();
    composite.composite(
      encoder,
      color,
      { view: attachment.view, resolveTarget: attachment.resolveTarget },
      viewport,
    );
    open.resume();
  }

  // With tone mapping the composite can't invert (a custom node, a linear
  // output colour space, rawColor), splats (which WebGL Spark doesn't tone
  // map) go on the canvas after three's output pass, blended in sRGB,
  // tested against the scene's depth, which three keeps. Transparent
  // objects in front of the splats are then drawn under them.
  private drawAfterOutput(
    scene: THREE.Scene,
    camera: THREE.Camera,
    open: OpenPass,
  ) {
    const rc = open.context;
    const depth = open.depthTexture();
    const viewport = rc.viewport ? rc.viewportValue.clone() : null;
    const size = { x: rc.width, y: rc.height };
    const previous = scene.onAfterRender;
    // Restored first (as an own property when it was the prototype's).
    scene.onAfterRender = (...args) => {
      scene.onAfterRender = previous;
      previous.apply(scene, args);
      this.drawOnCanvas(camera, depth, viewport, size);
    };
  }

  private drawOnCanvas(
    camera: THREE.Camera,
    sceneDepth: GPUTexture | null,
    viewport: THREE.Vector4 | null,
    size: { x: number; y: number },
  ) {
    const { renderer } = this;
    this.drawAfter(
      camera,
      canvasContext(renderer).getCurrentTexture(),
      sceneDepth,
      viewport,
      size,
      !this.spark.rawColor &&
        renderer.outputColorSpace !== THREE.SRGBColorSpace,
    );
  }

  // The viewport three gives an ArrayCamera's sub-camera (its viewport in
  // CSS pixels times the pixel ratio), else the whole target.
  private viewportOf(
    camera: THREE.Camera,
    width: number,
    height: number,
  ): THREE.Vector4 {
    const vp = (camera as { viewport?: THREE.Vector4 }).viewport;
    if (!vp) return new THREE.Vector4(0, 0, width, height);
    const ratio = this.renderer.getPixelRatio?.() ?? 1;
    return new THREE.Vector4(
      Math.floor(vp.x * ratio),
      Math.floor(vp.y * ratio),
      Math.floor(vp.z * ratio),
      Math.floor(vp.w * ratio),
    );
  }

  // An ArrayCamera: every sub-camera needs its own generate and sort, which
  // are submitted ahead of three's pass, so they can't share it. The views
  // are drawn once three has submitted its frame (scene.onAfterRender), one
  // after another, each over three's colour in its viewport and tested
  // against three's depth (resolved when multisampled): on the canvas after
  // the output pass, as drawAfterOutput, or into the render target.
  // Transparent objects in front of the splats are then drawn under them.
  private drawViewsAfterRender(
    scene: THREE.Scene,
    views: THREE.Camera[],
    open: OpenPass,
  ) {
    const { renderer, spark } = this;
    const rc = open.context;
    const canvas =
      open.isFrameBufferTarget &&
      !renderer.getOutputRenderTarget() &&
      (scene as THREE.Scene).isScene;
    if (spark.srgbBlend && !canvas) {
      warnOnce("srgbBlend is ignored with an ArrayCamera");
    }
    if (splatsSortCpu(this.splats)) {
      warnOnce(
        'an ArrayCamera with sort "cpu" draws its views in one sort order',
      );
    }
    const depth = open.depthTexture();
    // three's frame buffer target is converted onto the canvas by its output
    // pass; other targets keep what three drew in their own texture.
    const color = canvas ? null : open.resolvedColorTexture();
    const format = open.colorFormat;
    const linear = canvas
      ? !spark.rawColor && renderer.outputColorSpace !== THREE.SRGBColorSpace
      : spark.rawColor
        ? format.endsWith("-srgb")
        : format.endsWith("-srgb") ||
          (this.splats as WgpuSplatRenderer).hdr ||
          open.colorSpace !== THREE.SRGBColorSpace;
    const viewports = views.map((v) => this.viewportOf(v, rc.width, rc.height));
    const previous = scene.onAfterRender;
    scene.onAfterRender = (...args) => {
      scene.onAfterRender = previous;
      previous.apply(scene, args);
      const target = color ?? canvasContext(renderer).getCurrentTexture();
      const splats = this.splats as WgpuSplatRenderer;
      // Each view sorts for itself, whatever minSortIntervalMs says.
      const interval = splats.options.minSortIntervalMs;
      splats.options.minSortIntervalMs = 0;
      let resolved = depth;
      if (depth && depth.sampleCount > 1) {
        const encoder = splats.device.createCommandEncoder({
          label: "depth resolve",
        });
        this.depthResolve ??= new DepthResolve(splats.device);
        resolved = this.depthResolve.resolve(encoder, depth);
        splats.device.queue.submit([encoder.finish()]);
      }
      try {
        views.forEach((view, i) => {
          // As three draws an object only for sub-cameras seeing its layers.
          if (spark.layers.test(view.layers)) {
            const vp = viewports[i];
            this.drawAfter(
              view,
              target,
              resolved,
              vp,
              { x: vp.z, y: vp.w },
              linear,
            );
          }
        });
      } finally {
        splats.options.minSortIntervalMs = interval;
      }
    };
  }

  // Draws the splats over `color` in a pass of their own, submitted now:
  // after three's frame.
  private drawAfter(
    camera: THREE.Camera,
    color: GPUTexture,
    sceneDepth: GPUTexture | null,
    viewport: THREE.Vector4 | null,
    size: { x: number; y: number },
    linear: boolean,
  ) {
    const splats = this.splats as WgpuSplatRenderer;
    let depth = sceneDepth;
    const encoder = splats.device.createCommandEncoder({
      label: "splats after three",
    });
    if (depth && depth.sampleCount > 1) {
      this.depthResolve ??= new DepthResolve(splats.device);
      depth = this.depthResolve.resolve(encoder, depth);
    }
    const depthFormat = depth?.format ?? null;
    const pass = encoder.beginRenderPass({
      label: "splats",
      colorAttachments: [
        { view: color.createView(), loadOp: "load", storeOp: "store" },
      ],
      depthStencilAttachment: depth
        ? depthTestAttachment(depth.createView(), depthFormat)
        : undefined,
    });
    if (viewport) {
      pass.setViewport(viewport.x, viewport.y, viewport.z, viewport.w, 0, 1);
    }
    splats.renderInPass(camera, pass, {
      format: color.format,
      depthFormat,
      sampleCount: 1,
      width: viewport ? viewport.z : size.x,
      height: viewport ? viewport.w : size.y,
      linear,
      depthCompare: (camera as { reversedDepth?: boolean }).reversedDepth
        ? "greater-equal"
        : "less-equal",
    });
    pass.end();
    splats.device.queue.submit([encoder.finish()]);
  }

  /** SparkRenderer.getRgba: bakes the mesh's generated RGBA on the GPU. */
  getRgba(generator: SplatGenerator, rgba: RgbaArray) {
    const entry = this.entries.get(generator);
    const mesh = entry?.mesh ?? entry?.lodMesh?.mesh;
    if (!mesh || !this.splats || !this.lastCamera) {
      throw new Error(
        "SparkRenderer.getRgba: render the generator with this SparkRenderer first",
      );
    }
    const texture = rgba.gpuTexture(this.splats.device, mesh.source.count);
    this.splats.bakeRgba(mesh, this.lastCamera, texture);
  }

  /**
   * SparkRenderer.renderEnvMap: prefilters a cube map with three/webgpu's
   * PMREMGenerator (three's WebGL one can't run on WebGPURenderer), as WebGL
   * does, into a texture of its own. That is THREE.PMREMGenerator when the
   * "three" Spark imports is three/webgpu, or the one passed. Otherwise
   * (Spark resolving "three" to three's WebGL build, as Vite does) the cube
   * texture itself is returned: three/webgpu's materials prefilter a cube
   * envMap themselves, but the next renderCubeMap or renderEnvMap redraws it.
   */
  prefilterEnvMap(cube: THREE.Texture, PMREMGenerator?: unknown) {
    type Generator = {
      fromCubemap(t: THREE.Texture): THREE.RenderTarget;
      dispose(): void;
    };
    const Class =
      PMREMGenerator ??
      ("WebGPURenderer" in THREE ? THREE.PMREMGenerator : undefined);
    if (!Class) {
      // Re-prefiltered where it is used.
      cube.needsPMREMUpdate = true;
      return cube;
    }
    if (this.pmrem?.Class !== Class) {
      this.pmrem?.generator.dispose();
      const generator = new (Class as new (renderer: unknown) => Generator)(
        this.renderer,
      );
      this.pmrem = { Class, generator };
    }
    return this.pmrem.generator.fromCubemap(cube).texture;
  }

  private ensureRenderer() {
    if (!this.splats) {
      const spark = this.spark;
      this.splats = new WgpuSplatRenderer(
        this.renderer as unknown as ConstructorParameters<
          typeof WgpuSplatRenderer
        >[0],
        { depthTest: spark.material.depthTest },
      );
      this.splats.onSortDeferred = () => spark.setDirty();
      this.lod = new WgpuLod(this.splats, {
        pagedExtSplats: spark.pagedExtSplats,
        maxPagedSplats: spark.maxPagedSplats,
        numLodFetchers: spark.numLodFetchers,
        pagedAttributes: spark.pagedAttributes,
        onDirty: () => spark.setDirty(),
      });
      this.baseAccumulator = this.splats.options.accumulator;
    }
    return this.splats;
  }

  // SparkRenderer's settings, which apps change at any time.
  private applyOptions() {
    const spark = this.spark;
    const o = (this.splats as WgpuSplatRenderer).options;
    // Covariance splats in the accumulator (generate writes them, the draw
    // reads the covariance directly): SparkRenderer.covSplats, or a visible
    // SplatMesh with covSplats, whose CovSplat pipeline needs them (WebGL
    // throws there instead).
    o.covSplats = spark.covSplats || this.covMeshes;
    o.enable2DGS = spark.enable2DGS;
    // accumExtSplats: always the ext accumulator (float centres, half-float
    // colour). Otherwise WgpuSplatRenderer's own choice ("auto": ext while it
    // fits a binding, packed above), not WebGL's packed default: the ext one
    // is more precise and draws as fast.
    o.accumulator = spark.accumExtSplats ? "ext" : this.baseAccumulator;
    o.maxStdDev = spark.maxStdDev;
    o.minPixelRadius = spark.minPixelRadius;
    o.maxPixelRadius = spark.maxPixelRadius;
    o.minAlpha = spark.minAlpha;
    o.blurAmount = spark.blurAmount;
    o.preBlurAmount = spark.preBlurAmount;
    o.falloff = spark.falloff;
    o.clipXY = spark.clipXY;
    o.focalAdjustment = spark.focalAdjustment;
    o.focalDistance = spark.focalDistance;
    o.apertureAngle = spark.apertureAngle;
    o.sortRadial = spark.sortRadial;
    o.minSortIntervalMs = spark.minSortIntervalMs;
    o.lodInflate = spark.lodInflate;
    o.depthTest = spark.material.depthTest;
    o.hdr = spark.hdr;
    const l = (this.lod as WgpuLod).options;
    l.lodSplatCount = spark.lodSplatCount;
    l.lodSplatScale = spark.lodSplatScale;
    l.lodRenderScale = spark.lodRenderScale;
    l.lodTraverseMode = spark.lodTraverseMode;
    l.behindFoveate = spark.behindFoveate;
    l.coneFov0 = spark.coneFov0;
    l.coneFov = spark.coneFov;
    l.coneFoveate = spark.coneFoveate;
    l.enableLodFetching = spark.enableLodFetching;
    l.lodCleanupTimeoutMs = spark.lodCleanupTimeoutMs;
    l.lodRaycast = spark.lodRaycast;
    l.lodRaycastIntervalMs = spark.lodRaycastIntervalMs;
  }

  // Matches the WgpuSplatRenderer meshes to the scene's visible generators.
  private sync(scene: THREE.Scene, camera: THREE.Camera) {
    const inLayers = (node: THREE.Object3D) =>
      !camera.layers || camera.layers.test(node.layers);
    const all = new Set<SplatGenerator>();
    scene.traverse((node) => {
      if (node instanceof SplatGenerator && inLayers(node)) all.add(node);
    });
    const visible: SplatGenerator[] = [];
    const globalEdits: SplatEdit[] = [];
    scene.traverseVisible((node) => {
      if (node instanceof SplatGenerator && inLayers(node)) visible.push(node);
      if (isSplatEdit(node)) {
        let ancestor = node.parent;
        while (ancestor && !(ancestor instanceof SplatMesh)) {
          ancestor = ancestor.parent;
        }
        if (!ancestor) globalEdits.push(node);
      }
    });
    this.globalEdits = globalEdits;
    const covMeshes = visible.some(
      (n) => n instanceof SplatMesh && n.covSplats,
    );
    if (covMeshes && !this.spark.covSplats) {
      warnOnce(
        "SplatMeshes with covSplats turn on covariance splats (SparkRenderer covSplats)",
      );
    }
    this.covMeshes = covMeshes;

    const time = performance.now() / 1000;
    const deltaTime = time - this.lastTime;
    this.lastTime = time;
    for (const node of all) {
      if (node instanceof SplatMesh) {
        // SplatMesh.update runs onFrame; frameUpdate is replaced here.
        node.onFrame?.({ mesh: node, time, deltaTime });
      }
    }

    const shown = new Set<SplatGenerator>();
    for (const node of visible) {
      if (this.updateEntry(node)) shown.add(node);
    }
    for (const [node, entry] of this.entries) {
      if (shown.has(node)) continue;
      // Hidden LoD meshes keep their WgpuLod mesh, to be shown again.
      this.detach(entry, all.has(node));
      if (!all.has(node)) this.entries.delete(node);
    }
  }

  // Brings a visible generator's entry up to date; whether it is drawn.
  private updateEntry(node: SplatGenerator): boolean {
    let entry = this.entries.get(node);
    let key: unknown[];
    let build: () => Entry | null;
    if (node instanceof SplatMesh) {
      if (!node.isInitialized) {
        this.watchInit(node);
        return false;
      }
      const { base, paged, custom } = meshSplats(node);
      if (custom) {
        // A custom SplatSource: its fetchSplat graph generates the splats.
        const count = custom.getNumSplats();
        if (count <= 0) return false;
        key = [custom, count];
        build = () => {
          const dyno = splatSourceMeshDyno(node, custom, {
            globalEdits: () => this.globalEdits,
          });
          const splats = this.splats as WgpuSplatRenderer;
          return {
            key,
            dyno,
            mesh: splats.addGenerator(count, dyno, node),
            generated: count,
          };
        };
      } else {
        const lodSplats = base?.lodSplats;
        const useLod = this.spark.enableLod && !!(paged || lodSplats);
        const splats = paged ?? base;
        if (!splats || (!useLod && (base as BaseSplats).numSplats === 0)) {
          return false;
        }
        key = [useLod, splats, lodSplats];
        build = () => {
          const dyno = splatMeshDyno(node, {
            globalEdits: () => this.globalEdits,
          });
          if (useLod) return this.addLod(node, { key: [], dyno });
          return this.addBase(node, base as BaseSplats, dyno);
        };
      }
    } else {
      if (!node.generator || node.numSplats <= 0) return false;
      key = [node.numSplats];
      build = () => {
        const dyno = splatGeneratorDyno(node);
        const splats = this.splats as WgpuSplatRenderer;
        return {
          key,
          dyno,
          mesh: splats.addGenerator(node.numSplats, dyno, node),
        };
      };
    }

    if (entry && !sameKey(entry.key, key)) {
      this.detach(entry);
      entry = undefined;
    }
    const fresh = !entry;
    if (!entry) {
      const built = build();
      if (!built) return false;
      built.key = key;
      entry = built;
      this.entries.set(node, entry);
    } else if (entry.detached) {
      this.reattach(node, entry);
    }

    if (node instanceof SplatMesh) {
      // SplatMesh.updateGenerator(): construct its dyno blocks again.
      if (node.generatorDirty) {
        node.generatorDirty = false;
        if (!fresh) this.renewMesh(node, entry);
      }
      const mesh = entry.mesh ?? entry.lodMesh?.mesh;
      mesh?.recolor.set(
        node.recolor.r,
        node.recolor.g,
        node.recolor.b,
        node.opacity,
      );
      if (entry.shared && mesh) {
        this.refreshShared(entry.shared);
        if (mesh.source !== entry.shared.source) {
          mesh.source = entry.shared.source;
          this.splats?.markDirty();
        }
      }
      const lodMesh = entry.lodMesh;
      if (lodMesh) {
        lodMesh.lodScale = node.lodScale;
        lodMesh.behindFoveate = node.behindFoveate;
        lodMesh.coneFov0 = node.coneFov0;
        lodMesh.coneFov = node.coneFov;
        lodMesh.coneFoveate = node.coneFoveate;
        this.lod?.setEnableLod(lodMesh, node.enableLod !== false);
      }
      this.updateRaycast(node, entry);
    }
    return true;
  }

  // What SplatMesh.update leaves for raycast(): for a LoD selection, the
  // raycast traversal's indices into lodSplats (paged: into the page pool,
  // which WgpuSplatPager mirrors on the CPU), else the splats drawn.
  private updateRaycast(node: SplatMesh, entry: Entry) {
    const lodMesh = entry.lodMesh;
    const drawn = lodMesh?.mesh.lodIndices;
    node.raycastIndices =
      lodMesh && drawn
        ? (lodMesh.raycastIndices ?? {
            numSplats: drawn.length,
            indices: drawn,
          })
        : undefined;
    const { base, paged, custom } = meshSplats(node);
    if (paged) return;
    node.context.enableLod.value = false;
    node.context.numSplats.value = entry.pending
      ? 0
      : (custom?.getNumSplats() ?? base?.numSplats ?? 0);
  }

  // A new WgpuSplatMesh for the same splats: DynoKernels compiles a mesh's
  // graphs once, keyed by the mesh.
  private renewMesh(node: SplatMesh, entry: Entry) {
    const splats = this.splats as WgpuSplatRenderer;
    const old = entry.mesh ?? entry.lodMesh?.mesh;
    if (!old) return;
    const mesh = splats.add(old.source, node, entry.dyno);
    if (old.lodIndices) splats.setLodIndices(mesh, old.lodIndices);
    splats.remove(old);
    if (entry.lodMesh) entry.lodMesh.mesh = mesh;
    else entry.mesh = mesh;
  }

  private addBase(node: SplatMesh, base: BaseSplats, dyno: WgpuDyno): Entry {
    const shared = this.acquire(base);
    const mesh = (this.splats as WgpuSplatRenderer).add(
      shared.source,
      node,
      dyno,
    );
    return { key: [], dyno, mesh, shared };
  }

  // Fills `entry` once WgpuLod has the mesh (after loading .rad metadata).
  private addLod(node: SplatMesh, entry: Entry): Entry {
    entry.pending = true;
    const lod = this.lod as WgpuLod;
    const { base, paged } = meshSplats(node);
    const splats = paged ?? base;
    lod
      .add(splats as PackedSplats, node, { lodScale: node.lodScale })
      .then((lodMesh) => {
        entry.pending = false;
        // Dropped, or rebuilt for new splats, while loading.
        if (this.entries.get(node) !== entry) {
          lod.remove(lodMesh);
          return;
        }
        lodMesh.mesh.dyno = entry.dyno;
        entry.lodMesh = lodMesh;
        if (entry.detached) lod.setVisible(lodMesh, false);
        this.spark.setDirty();
      })
      .catch((error) => {
        console.error("SparkRenderer (WebGPU): LoD mesh failed", error);
      });
    return entry;
  }

  private reattach(node: SplatGenerator, entry: Entry) {
    entry.detached = false;
    const splats = this.splats as WgpuSplatRenderer;
    if (entry.shared) {
      // Uploaded again if no other mesh kept the splats.
      entry.shared = this.acquire(entry.shared.splats);
      entry.mesh = splats.add(entry.shared.source, node, entry.dyno);
    } else if (entry.lodMesh) {
      // LoD: shown again, with its sources and tree.
      this.lod?.setVisible(entry.lodMesh, true);
    } else if (entry.generated !== undefined) {
      entry.mesh = splats.addGenerator(entry.generated, entry.dyno, node);
    } else if (node instanceof SplatMesh) {
      if (!entry.pending) this.addLod(node, entry);
    } else {
      entry.mesh = splats.addGenerator(node.numSplats, entry.dyno, node);
    }
  }

  // Stops drawing an entry. `keep`: a hidden LoD mesh stays in WgpuLod,
  // hidden, as SparkRenderer keeps a LoD mesh's tree and textures.
  private detach(entry: Entry, keep = false) {
    if (entry.lodMesh && !keep) {
      this.lod?.remove(entry.lodMesh);
      entry.lodMesh = undefined;
    }
    if (entry.detached) return;
    entry.detached = true;
    if (entry.mesh) this.splats?.remove(entry.mesh);
    entry.mesh = undefined;
    if (entry.lodMesh) this.lod?.setVisible(entry.lodMesh, false);
    if (entry.shared) this.release(entry.shared);
  }

  private acquire(splats: BaseSplats): SharedSource {
    let shared = this.shared.get(splats);
    if (!shared) {
      shared = { splats, source: this.upload(splats), users: 0 };
      this.shared.set(splats, shared);
    }
    shared.users += 1;
    return shared;
  }

  private release(shared: SharedSource) {
    shared.users -= 1;
    if (shared.users > 0) return;
    shared.source.destroy();
    this.shared.delete(shared.splats);
  }

  private upload(splats: BaseSplats): GpuSplatSource {
    const device = (this.splats as WgpuSplatRenderer).device;
    if (splats instanceof PackedSplats) {
      // The WebGL path consumes needsUpdate when it uploads its texture.
      splats.needsUpdate = false;
      return GpuSplatSource.fromPackedSplats(device, splats);
    }
    return GpuSplatSource.fromExtSplats(device, splats);
  }

  // Re-uploads splats edited since (PackedSplats.needsUpdate, a new count).
  private refreshShared(shared: SharedSource) {
    const { splats } = shared;
    const edited =
      (splats instanceof PackedSplats && splats.needsUpdate) ||
      splats.numSplats !== shared.source.count;
    if (!edited) return;
    shared.source.destroy();
    shared.source = this.upload(splats);
  }

  private watchInit(mesh: SplatMesh) {
    if (this.initWatched.has(mesh)) return;
    this.initWatched.add(mesh);
    const onInitialized = () => {
      mesh.removeEventListener("initialized", onInitialized);
      this.initWatched.delete(mesh);
      this.spark.setDirty();
    };
    mesh.addEventListener("initialized", onInitialized);
  }

  dispose() {
    for (const entry of this.entries.values()) this.detach(entry);
    this.entries.clear();
    this.lod?.dispose();
    this.splats?.dispose();
    this.depthResolve?.dispose();
    this.depthResolve = undefined;
    this.pmrem?.generator.dispose();
    this.pmrem = undefined;
    this.lod = undefined;
    this.splats = undefined;
  }
}

// The portal disk of SparkPortals' DISK_PORTAL_FRAGMENT_SHADER, from the
// extraUniforms it reads (diskCenter, diskNormal, diskRadius, diskTwoSided),
// which WebGPU's splat shader implements itself.
function diskClip(spark: SparkRenderer): SplatDiskClip | null {
  const u = spark.uniforms as unknown as Record<
    string,
    { value: unknown } | undefined
  >;
  const radius = u.diskRadius?.value;
  if (typeof radius !== "number" || radius === 0) return null;
  return {
    center: u.diskCenter?.value as THREE.Vector3,
    normal: u.diskNormal?.value as THREE.Vector3,
    radius,
    twoSided: !!u.diskTwoSided?.value,
  };
}

// What a SplatMesh draws, as SplatMesh.update picks it (its splats first):
// PackedSplats or ExtSplats, PagedSplats, or another SplatSource.
function meshSplats(node: SplatMesh): {
  base?: BaseSplats;
  paged?: PagedSplats;
  custom?: SplatSource;
} {
  const own = node.splats;
  if (own instanceof PackedSplats || own instanceof ExtSplats) {
    return { base: own };
  }
  if (own instanceof PagedSplats) return { paged: own };
  if (own && !(own instanceof EmptySplatSource)) return { custom: own };
  const base = node.packedSplats ?? node.extSplats;
  return node.paged ? { base, paged: node.paged } : { base };
}

function splatsSortCpu(splats?: WgpuSplatRenderer) {
  return splats?.options.sort === "cpu";
}

function sameKey(a: unknown[], b: unknown[]) {
  return a.length === b.length && a.every((x, i) => x === b[i]);
}

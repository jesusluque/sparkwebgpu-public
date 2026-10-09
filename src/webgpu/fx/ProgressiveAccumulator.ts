// Progressive accumulation while the view holds still (a path tracer's
// "progressive refinement", for a rasterizer's anti-aliasing).
//
// One sample a pixel cannot be both sharp and steady: a small anti-alias blur
// matches a supersampled render but makes sub-pixel HDR glints shimmer as
// the camera moves (research/simplify-measurements.md, BO and BS). This
// keeps the ordinary frame while anything moves and, once the camera and
// the scene have stayed the same for a frame, renders the frame again with
// the projection jittered by a sub-pixel Halton (2, 3) offset and a sharper
// splat blur, folding each render into a float running mean (`frames`
// samples, 16 by default) that is shown in its place. Any change starts
// over from the ordinary frame. The mean is linear light, taken before the
// FxChain (display transform, grade, bloom) runs on it.
//
//   const accum = new fx.ProgressiveAccumulator(chain, { frames: 16 });
//   renderer.setAnimationLoop(() => {
//     accum.begin({ camera, width, height, spark });   // may jitter camera
//     renderer.setRenderTarget(target);
//     renderer.render(scene, camera);
//     accum.end(gpuTexture(renderer, target.texture)); // + the chain
//     ...
//   });
//
// What counts as a change: the camera's pose and projection, the target's
// size, the splats' contentVersion (mesh set, poses, LoD and paging, plugins,
// dynos -- anything animated never holds still) and `version`, which the
// page bumps for whatever else is in view (a dome, a GUI), or reset().

import type * as THREE from "three";
import accumulateModule from "../generated/fx/accumulate";
import { type ThreeWebGPURenderer, gpuTexture } from "../threeInternals";
import { UniformWriter } from "../uniforms";
import { FxChain } from "./FxChain";
import { Gpu } from "./Gpu";
import { TextureBridge } from "./TextureBridge";
import { type FxBuffer, PIXEL_BYTES, rect } from "./types";

export interface AccumulateFrame {
  camera: THREE.Camera;
  /** The target's size in pixels. */
  width: number;
  height: number;
  /**
   * A SparkRenderer (or anything with blurAmount): its blur is `blur` while
   * sampling, and its splats' contentVersion counts as a change.
   */
  spark?: {
    blurAmount: number;
    webgpu?: { splats?: { contentVersion: number } | null } | null;
  } | null;
  /** Anything else in view; bump it on a change (or call reset()). */
  version?: number;
}

/**
 * What a frame is: "off" (frames 0), "plain" (something changed: the
 * ordinary frame), "sample" (jittered and folded into the mean) or "hold"
 * (the mean has all its samples; the ordinary frame is drawn, the mean shown).
 */
export type AccumulateMode = "off" | "plain" | "sample" | "hold";

export interface ProgressiveOptions {
  /** Samples in the mean (16); 0 turns accumulation off. */
  frames?: number;
  /** The splats' anti-alias blur in px^2 while sampling (0.1). */
  blur?: number;
}

/** The radical inverse of `i` in `base`, in [0, 1). */
export function halton(i: number, base: number): number {
  let f = 1;
  let r = 0;
  let n = i;
  while (n > 0) {
    f /= base;
    r += f * (n % base);
    n = Math.floor(n / base);
  }
  return r;
}

/** Sample k's offset in pixels, in [-0.5, 0.5)^2: Halton (2, 3) from index 1. */
export function jitterOffset(k: number): [number, number] {
  return [halton(k + 1, 2) - 0.5, halton(k + 1, 3) - 0.5];
}

export class ProgressiveAccumulator {
  readonly gpu: Gpu;
  readonly bridge: TextureBridge;
  /** Applied after the mean when given (constructed from a chain). */
  readonly chain: FxChain | null;
  frames: number;
  blur: number;
  /** Samples in the mean. */
  samples = 0;
  /** What the last begin() decided. */
  mode: AccumulateMode = "off";
  /** The last sample's offset in pixels (null outside "sample"). */
  jitter: [number, number] | null = null;

  private mean: FxBuffer | null = null;
  private lastSignature: number[] = [];
  private restart = true;
  private frame: AccumulateFrame | null = null;
  private saved: {
    projection: THREE.Matrix4;
    blur: number | null;
  } | null = null;
  private contentAtBegin = 0;

  constructor(
    target: GPUDevice | Gpu | FxChain,
    options: ProgressiveOptions = {},
  ) {
    if (target instanceof FxChain) {
      this.chain = target;
      this.gpu = target.gpu;
      this.bridge = target.bridge;
    } else {
      this.chain = null;
      this.gpu = target instanceof Gpu ? target : Gpu.forDevice(target);
      this.bridge = new TextureBridge(this.gpu);
    }
    this.frames = Math.max(0, Math.floor(options.frames ?? 16));
    this.blur = options.blur ?? 0.1;
  }

  /** The mean has all its samples. */
  get converged(): boolean {
    return this.frames > 0 && this.samples >= this.frames;
  }

  /** Starts over on the next frame (something changed out of sight of begin). */
  reset() {
    this.restart = true;
    this.samples = 0;
  }

  private content(f: AccumulateFrame): number {
    return f.spark?.webgpu?.splats?.contentVersion ?? 0;
  }

  /**
   * Before rendering: decides the frame and, for a sample, jitters the
   * camera's projection and sharpens the splats' blur until end().
   */
  begin(f: AccumulateFrame): AccumulateMode {
    this.frame = f;
    this.jitter = null;
    this.contentAtBegin = this.content(f);
    if (this.frames <= 0) {
      this.samples = 0;
      this.mode = "off";
      return this.mode;
    }
    const sig = [
      ...f.camera.matrixWorld.elements,
      ...f.camera.projectionMatrix.elements,
      f.width,
      f.height,
      f.version ?? 0,
      this.contentAtBegin,
      this.frames,
      this.blur,
    ];
    const still =
      !this.restart &&
      sig.length === this.lastSignature.length &&
      sig.every((v, i) => v === this.lastSignature[i]);
    this.lastSignature = sig;
    this.restart = false;
    if (!still) {
      this.samples = 0;
      this.mode = "plain";
      return this.mode;
    }
    if (this.samples >= this.frames) {
      this.mode = "hold";
      return this.mode;
    }
    this.mode = "sample";
    const [dx, dy] = jitterOffset(this.samples);
    this.jitter = [dx, dy];
    const camera = f.camera;
    const projection = camera.projectionMatrix.clone();
    this.saved = { projection, blur: f.spark ? f.spark.blurAmount : null };
    // Clip space moved by (dx, dy) pixels: rows 0 and 1 gain the w row.
    const e = camera.projectionMatrix.elements;
    const nx = (2 * dx) / f.width;
    const ny = (2 * dy) / f.height;
    for (let c = 0; c < 4; c++) {
      e[c * 4] += nx * e[c * 4 + 3];
      e[c * 4 + 1] += ny * e[c * 4 + 3];
    }
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    camera.userData.unjitteredProjectionMatrix = projection;
    if (f.spark) f.spark.blurAmount = this.blur;
    return this.mode;
  }

  /**
   * After rendering into `texture` (the frame's linear target): restores the
   * camera, folds a sample into the mean, and leaves the mean (or the frame
   * as drawn) in the texture, through the chain when there is one. Submits.
   */
  end(texture: GPUTexture) {
    const f = this.frame;
    this.frame = null;
    if (this.saved && f) {
      f.camera.projectionMatrix.copy(this.saved.projection);
      f.camera.projectionMatrixInverse.copy(this.saved.projection).invert();
      f.camera.userData.unjitteredProjectionMatrix = undefined;
      if (f.spark && this.saved.blur !== null)
        f.spark.blurAmount = this.saved.blur;
    }
    this.saved = null;
    let mode = this.mode;
    // The splats changed while this frame drew (a page streamed in, a
    // plugin animated): the frame stands as drawn, and the next one is new.
    if (f && (mode === "sample" || mode === "hold")) {
      if (this.content(f) !== this.contentAtBegin) {
        this.reset();
        mode = "plain";
      }
    }
    if (mode === "off" || mode === "plain") {
      if (this.chain) this.chain.apply(texture);
      return;
    }
    const mean = this.meanFor(texture.width, texture.height);
    if (mode === "sample") {
      const sample = this.bridge.read(texture);
      const u = UniformWriter.for(accumulateModule).setAll({
        width: mean.width,
        height: mean.height,
        srcStride: sample.stride,
        dstStride: mean.stride,
        weight: 1 / (this.samples + 1),
      });
      this.gpu.run(
        this.gpu.load(accumulateModule, "accumulateMain"),
        [mean.width, mean.height, 1],
        { src: sample, mean },
        u,
      );
      this.samples += 1;
    }
    if (this.chain) {
      this.chain.applyPicture(mean, texture);
    } else {
      this.bridge.write(mean, texture);
      this.gpu.flush();
    }
  }

  /**
   * Writes the mean as it stands (linear, no chain) into `texture` and
   * submits; false when it holds no sample yet. For readbacks.
   */
  writeMean(texture: GPUTexture): boolean {
    if (!this.mean || this.samples === 0) return false;
    this.bridge.write(this.mean, texture);
    this.gpu.flush();
    return true;
  }

  /** `end` on a three.js RenderTarget's colour texture (WebGPURenderer). */
  endRenderTarget(
    renderer: ThreeWebGPURenderer,
    target: { texture: THREE.Texture },
  ) {
    const texture = gpuTexture(renderer, target.texture);
    if (!texture) {
      throw new Error(
        "ProgressiveAccumulator: render into the target before end()",
      );
    }
    this.end(texture);
  }

  // The mean's picture, kept across frames; a new size starts over.
  private meanFor(width: number, height: number): FxBuffer {
    const m = this.mean;
    if (m && m.width === width && m.height === height) return m;
    if (m) this.gpu.drop(`accumulate.mean.${m.width}x${m.height}`);
    const buffer = this.gpu.keep(
      `accumulate.mean.${width}x${height}`,
      null,
      Math.max(1, width * height) * PIXEL_BYTES,
    ) as GPUBuffer;
    this.mean = {
      buffer,
      width,
      height,
      stride: width,
      rect: rect(0, 0, width, height),
    };
    this.samples = 0;
    return this.mean;
  }

  /** Frees the mean. */
  dispose() {
    if (this.mean)
      this.gpu.drop(`accumulate.mean.${this.mean.width}x${this.mean.height}`);
    this.mean = null;
  }
}

// An ordered list of aofx effects applied to a rendered frame.
//
//   const chain = new FxChain(splats.device);
//   chain.add(new Grade(), { gamma: [1.2, 1.2, 1.2, 1] });
//   chain.add(new Blur(), { size: [6, 6] });
//   renderer.setAnimationLoop(() => {
//     renderer.setRenderTarget(target);
//     renderer.render(scene, camera);
//     splats.render(camera, target);
//     chain.applyToRenderTarget(renderer, target);   // in place
//     renderer.setRenderTarget(null);
//     output.render(renderer);
//   });
//
// Each step's pass-through input is the step before it (the first sees the
// frame); its other clips can be wired to any FxNode, such as a generator
// for Merge's A. A step with no inputs replaces the picture.

import type * as THREE from "three";
import { type ThreeWebGPURenderer, gpuTexture } from "../threeInternals";
import type { Effect } from "./Effect";
import { type FxContext, FxGraph, type FxInput, type FxNode } from "./FxGraph";
import { Gpu } from "./Gpu";
import { type BridgeOptions, TextureBridge } from "./TextureBridge";
import {
  type FxBuffer,
  type ParamValues,
  defaultParams,
  passThroughClip,
} from "./types";

export interface FxStep extends FxNode {
  /** For a UI; the effect's label when left out. */
  name?: string;
}

interface RenderTargetLike {
  texture: THREE.Texture;
  samples?: number;
}

export class FxChain {
  readonly gpu: Gpu;
  readonly bridge: TextureBridge;
  readonly graph: FxGraph;
  steps: FxStep[] = [];
  /**
   * Scale and project size for every frame. Parameters in pixels are
   * canonical (say CSS pixels) and the frame renders at `scaleX` of them
   * (the device pixel ratio), as an aofx host renders a proxy.
   */
  context: FxContext = {};
  /** Values the last frame's effects attached, by id (ImageStatistics). */
  values = new Map<string, readonly number[]>();

  constructor(device: GPUDevice | Gpu) {
    this.gpu = device instanceof Gpu ? device : Gpu.forDevice(device);
    this.bridge = new TextureBridge(this.gpu);
    this.graph = new FxGraph(this.gpu);
  }

  /** Appends a step with the effect's defaults, overridden by `params`. */
  add(
    effect: Effect,
    params: ParamValues = {},
    options: Omit<FxStep, "effect" | "params"> = {},
  ): FxStep {
    const step: FxStep = {
      ...options,
      effect,
      params: { ...params },
      instance:
        options.instance ??
        `${effect.describe().label}${this.steps.length + 1}`,
    };
    this.steps.push(step);
    return step;
  }

  /** Every parameter at its default (normalised ones against `width` x `height`). */
  static defaults(effect: Effect, width = 0, height = 0) {
    return defaultParams(effect.describe(), width, height);
  }

  get active() {
    return this.steps.some((s) => s.enabled !== false);
  }

  /** The steps linked into one graph over `source`. */
  link(source: FxInput): FxInput {
    let prev = source;
    for (const step of this.steps) {
      const desc = step.effect.describe();
      const main = passThroughClip(desc);
      const node: FxNode = {
        ...step,
        inputs: main ? { ...step.inputs, [main.name]: prev } : step.inputs,
      };
      prev = node;
    }
    return prev;
  }

  /** Runs the chain over a picture; recorded into the Gpu's frame, not flushed. */
  render(
    picture: FxBuffer,
    ctx: FxContext = {},
    window = picture.rect,
  ): FxBuffer | null {
    const top = this.link({ picture });
    const out = this.graph.render(top, window, {
      projectWidth: picture.width,
      projectHeight: picture.height,
      ...this.context,
      ...ctx,
    });
    for (const [k, v] of out.values) this.values.set(k, v);
    for (const [k, p] of out.pending) {
      p.then((v) => this.values.set(k, v));
    }
    return out.picture;
  }

  /** Applies the chain to a texture in place and submits. */
  apply(texture: GPUTexture, opts: BridgeOptions = {}) {
    if (!this.active) return;
    const picture = this.bridge.read(texture, opts);
    const out = this.render(picture);
    // All identities: the texture already holds the answer.
    if (out !== picture) {
      this.bridge.write(out, texture, opts);
    }
    this.gpu.flush();
  }

  /**
   * Runs the chain over `picture` (rather than the texture's own pixels)
   * and writes the result into `texture`, then submits: a picture made
   * earlier in this Gpu's frame, such as ProgressiveAccumulator's mean.
   */
  applyPicture(
    picture: FxBuffer,
    texture: GPUTexture,
    opts: BridgeOptions = {},
  ) {
    const out = this.active ? this.render(picture) : picture;
    this.bridge.write(out, texture, opts);
    this.gpu.flush();
  }

  /** `apply` on a three.js RenderTarget's colour texture (WebGPURenderer). */
  applyToRenderTarget(renderer: ThreeWebGPURenderer, target: RenderTargetLike) {
    const texture = gpuTexture(renderer, target.texture);
    if (!texture) {
      throw new Error(
        "FxChain: render into the target before applying effects",
      );
    }
    this.apply(texture, {
      srgb: target.texture.colorSpace === "srgb" ? true : undefined,
    });
  }

  dispose() {
    this.gpu.destroy();
  }
}

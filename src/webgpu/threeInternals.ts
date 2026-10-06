// Every use Spark makes of three.js internals on WebGPU, in one place:
// WebGPURenderer's current render context and frame buffer target, and
// WebGPUBackend's per-resource data (GPU textures, the open render pass,
// its descriptor and encoder), pass state cache, pipeline cache and utils.
// None of it is API: it is three r180's and may change in any release. So
// it is checked here, against the revision it was written for and its
// shape, and a three upgrade fails in this module with an error naming
// what changed, not as a broken frame elsewhere. To support a new
// revision, check each access below against its sources and add it to
// THREE_REVISIONS.
//
// Public three API (render, setRenderTarget, initTexture,
// readRenderTargetPixelsAsync...) is used directly elsewhere.

import * as THREE from "three";

/** The three revisions whose internals this module matches. */
export const THREE_REVISIONS: readonly string[] = ["180"];

/** three's backend data for a texture (backend.get(texture)). */
export interface TextureData {
  texture?: GPUTexture;
  msaaTexture?: GPUTexture;
  format?: GPUTextureFormat;
}

/** The render context three is recording (renderer._currentRenderContext). */
export interface RenderContextLike {
  textures: THREE.Texture[] | null;
  scissor?: boolean;
  scissorValue?: THREE.Vector4;
  renderTarget?: THREE.RenderTarget | null;
  depthTexture: THREE.Texture | null;
  width: number;
  height: number;
  viewport: boolean;
  viewportValue: THREE.Vector4;
}

// backend.get(renderContext) while its pass is open.
interface RenderContextData {
  currentPass?: GPURenderPassEncoder | null;
  descriptor?: GPURenderPassDescriptor;
  encoder?: GPUCommandEncoder;
  currentSets?: unknown;
}

interface Backend {
  isWebGPUBackend?: boolean;
  device: GPUDevice;
  context: GPUCanvasContext;
  get(resource: object): object | undefined;
  generateMipmaps?(texture: THREE.Texture): void;
  updateViewport?(rc: RenderContextLike): void;
  utils?: {
    getCurrentColorFormat(rc: RenderContextLike): GPUTextureFormat;
    getCurrentDepthStencilFormat(
      rc: RenderContextLike,
    ): GPUTextureFormat | undefined;
    getSampleCountRenderContext(rc: RenderContextLike): number;
    getCurrentColorSpace(rc: RenderContextLike): string;
  };
  pipelineUtils?: { _activePipelines?: WeakMap<object, unknown> };
}

/** What Spark reads of a WebGPURenderer: `backend` and two internals. */
export interface ThreeWebGPURenderer {
  backend: Backend;
  _currentRenderContext?: RenderContextLike | null;
  _frameBufferTarget?: THREE.RenderTarget | null;
}

export class ThreeInternalsError extends Error {
  constructor(message: string) {
    super(
      `Spark (WebGPU): ${message}. Spark relies on three r${THREE_REVISIONS.join(", r")} internals (src/webgpu/threeInternals.ts); update that module for this three.`,
    );
    this.name = "ThreeInternalsError";
  }
}

/** Throws unless `revision` (default: the three Spark imports) is supported. */
export function checkThreeRevision(revision: string = THREE.REVISION) {
  if (!THREE_REVISIONS.includes(revision)) {
    throw new ThreeInternalsError(`three r${revision} is not supported`);
  }
}

const checked = new WeakSet<object>();
const checkedPass = new WeakSet<object>();

/**
 * The renderer's WebGPU backend, or null on its WebGL fallback. Checks the
 * three revision and the backend's shape once per renderer.
 */
export function webgpuBackend(renderer: ThreeWebGPURenderer): Backend | null {
  const backend = renderer.backend;
  if (!backend?.isWebGPUBackend) return null;
  if (!checked.has(renderer)) {
    checkThreeRevision();
    expectShape(backend, "renderer.backend", {
      get: "function",
      device: "object",
      context: "object",
    });
    checked.add(renderer);
  }
  return backend;
}

function requireBackend(renderer: ThreeWebGPURenderer): Backend {
  const backend = webgpuBackend(renderer);
  if (!backend) {
    throw new ThreeInternalsError(
      "WebGPURenderer is not on its WebGPU backend (await renderer.init())",
    );
  }
  return backend;
}

function expectShape(
  object: object,
  name: string,
  shape: Record<string, "function" | "object">,
) {
  for (const [key, type] of Object.entries(shape)) {
    const value = (object as Record<string, unknown>)[key];
    const actual: string = value === null ? "null" : typeof value;
    if (actual !== type) {
      throw new ThreeInternalsError(`${name}.${key} is ${actual}, not ${type}`);
    }
  }
}

/** renderer.backend.device */
export function gpuDevice(renderer: ThreeWebGPURenderer): GPUDevice {
  return requireBackend(renderer).device;
}

/** renderer.backend.context: the canvas's GPUCanvasContext. */
export function canvasContext(renderer: ThreeWebGPURenderer): GPUCanvasContext {
  return requireBackend(renderer).context;
}

/** three's backend data for a texture: its GPUTexture(s) and format. */
export function textureData(
  renderer: ThreeWebGPURenderer,
  texture: THREE.Texture,
): TextureData {
  return (requireBackend(renderer).get(texture) ?? {}) as TextureData;
}

/** The GPUTexture three made for `texture`, once it has been used. */
export function gpuTexture(
  renderer: ThreeWebGPURenderer,
  texture: THREE.Texture,
): GPUTexture | undefined {
  return textureData(renderer, texture).texture;
}

/** backend.generateMipmaps, when this backend has it. */
export function generateMipmaps(
  renderer: ThreeWebGPURenderer,
  texture: THREE.Texture,
) {
  requireBackend(renderer).generateMipmaps?.(texture);
}

/**
 * The render pass three has open while drawing (from an object's
 * onBeforeRender), with what Spark needs of it; null when none is open
 * (render bundles...).
 */
export function openPass(renderer: ThreeWebGPURenderer): OpenPass | null {
  const backend = requireBackend(renderer);
  if (!checkedPass.has(renderer)) {
    for (const key of ["_currentRenderContext", "_frameBufferTarget"]) {
      if (!(key in renderer)) {
        throw new ThreeInternalsError(`WebGPURenderer.${key} is missing`);
      }
    }
    expectShape(backend, "renderer.backend", {
      utils: "object",
      updateViewport: "function",
      pipelineUtils: "object",
    });
    expectShape(backend.utils as object, "renderer.backend.utils", {
      getCurrentColorFormat: "function",
      getCurrentDepthStencilFormat: "function",
      getSampleCountRenderContext: "function",
      getCurrentColorSpace: "function",
    });
    expectShape(
      backend.pipelineUtils as object,
      "renderer.backend.pipelineUtils",
      { _activePipelines: "object" },
    );
    checkedPass.add(renderer);
  }
  const rc = renderer._currentRenderContext;
  const data = rc
    ? (backend.get(rc) as RenderContextData | undefined)
    : undefined;
  if (!rc || !data?.currentPass) return null;
  return new OpenPass(renderer, backend, rc, data);
}

export class OpenPass {
  constructor(
    private readonly renderer: ThreeWebGPURenderer,
    private readonly backend: Backend,
    /** three's render context: target size, viewport, attachments. */
    readonly context: RenderContextLike,
    private readonly data: RenderContextData,
  ) {}

  get pass(): GPURenderPassEncoder {
    return this.data.currentPass as GPURenderPassEncoder;
  }

  get descriptor(): GPURenderPassDescriptor {
    return this.data.descriptor as GPURenderPassDescriptor;
  }

  get encoder(): GPUCommandEncoder {
    return this.data.encoder as GPUCommandEncoder;
  }

  private get utils() {
    return this.backend.utils as NonNullable<Backend["utils"]>;
  }

  get colorFormat(): GPUTextureFormat {
    return this.utils.getCurrentColorFormat(this.context);
  }

  /** The depth attachment's format, null without one. */
  get depthFormat(): GPUTextureFormat | null {
    return this.descriptor?.depthStencilAttachment
      ? (this.utils.getCurrentDepthStencilFormat(this.context) ?? null)
      : null;
  }

  get sampleCount(): number {
    return this.utils.getSampleCountRenderContext(this.context);
  }

  get colorSpace(): string {
    return this.utils.getCurrentColorSpace(this.context);
  }

  /** Formats of the colour attachments after the first (MRT). */
  get extraFormats(): GPUTextureFormat[] | undefined {
    return this.context.textures
      ?.slice(1)
      .map((t) => textureData(this.renderer, t).format as GPUTextureFormat);
  }

  /**
   * Whether this is three's own frame buffer target: the linear half-float
   * target three draws the canvas into before its output pass.
   */
  get isFrameBufferTarget(): boolean {
    const target = this.context.renderTarget;
    return !!target && target === this.renderer._frameBufferTarget;
  }

  /** The first colour attachment's texture (multisampled one if any). */
  colorTexture(): GPUTexture {
    const textures = this.context.textures as THREE.Texture[];
    const data = textureData(this.renderer, textures[0]);
    return (data.msaaTexture ?? data.texture) as GPUTexture;
  }

  /**
   * The first colour attachment's single-sample texture: with MSAA, the one
   * three resolves into at the end of its pass.
   */
  resolvedColorTexture(): GPUTexture {
    const textures = this.context.textures as THREE.Texture[];
    return textureData(this.renderer, textures[0]).texture as GPUTexture;
  }

  /** The depth texture three is drawing with, when it has one. */
  depthTexture(): GPUTexture | null {
    const depth = this.context.depthTexture;
    return depth ? (gpuTexture(this.renderer, depth) ?? null) : null;
  }

  /** Ends three's pass, to run passes of our own on its encoder. */
  end() {
    this.pass.end();
  }

  /**
   * Opens a new pass on three's encoder in place of the one ended,
   * loading what it drew, and gives it back to three with its viewport and
   * scissor.
   */
  resume(): GPURenderPassEncoder {
    const { descriptor, context: rc } = this;
    for (const a of descriptor.colorAttachments) {
      if (a) a.loadOp = "load";
    }
    const depth = descriptor.depthStencilAttachment;
    if (depth?.depthLoadOp) depth.depthLoadOp = "load";
    if (depth?.stencilLoadOp) depth.stencilLoadOp = "load";
    const pass = this.encoder.beginRenderPass(descriptor);
    this.data.currentPass = pass;
    this.resetState();
    if (rc.viewport) this.backend.updateViewport?.(rc);
    if (rc.scissor && rc.scissorValue) {
      const { x, y, z, w } = rc.scissorValue;
      pass.setScissorRect(x, y, z, w);
    }
    return pass;
  }

  /**
   * Forgets the pipeline and bindings three believes are set on the pass:
   * after drawing into it ourselves, three would otherwise skip setting
   * them again.
   */
  resetState() {
    this.backend.pipelineUtils?._activePipelines?.delete(this.pass);
    this.data.currentSets = {
      attributes: {},
      bindingGroups: [],
      pipeline: null,
      index: null,
    };
  }
}

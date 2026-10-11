// Textures in and out of the picture layout, with slang/fx/bridge.slang.
//
// copyTextureToBuffer, then `unpackTexels` into a float4 picture (flipped y
// up, decoded to linear); `packTexels`, then copyBufferToTexture. Copies
// rather than texture bindings, so the same kernels serve rgba16float,
// rgba8unorm(-srgb), bgra8unorm(-srgb) and rgba32float, and no storage
// texture format -- bgra8unorm-storage least of all -- is needed. The
// texture needs COPY_SRC to be read and COPY_DST to be written, which
// three's RenderTarget textures have.

import { fxBridge } from "../generated/constants";
import bridgeModule from "../generated/fx/bridge";
import { copyBytesPerRow } from "../gpuBuffers";
import { UniformWriter } from "../uniforms";
import type { Gpu } from "./Gpu";
import { type FxBuffer, type Rect, placed, rect } from "./types";

const FORMATS: Record<string, { code: number; bytes: number; srgb: boolean }> =
  {
    rgba16float: { code: fxBridge.kRgba16Float, bytes: 8, srgb: false },
    rgba8unorm: { code: fxBridge.kRgba8Unorm, bytes: 4, srgb: false },
    "rgba8unorm-srgb": { code: fxBridge.kRgba8Unorm, bytes: 4, srgb: true },
    bgra8unorm: { code: fxBridge.kBgra8Unorm, bytes: 4, srgb: false },
    "bgra8unorm-srgb": { code: fxBridge.kBgra8Unorm, bytes: 4, srgb: true },
    rgba32float: { code: fxBridge.kRgba32Float, bytes: 16, srgb: false },
  };

export const bridgeFormats = Object.keys(FORMATS) as GPUTextureFormat[];

export interface BridgeOptions {
  /** Texels are sRGB-encoded; -srgb formats always are. */
  srgb?: boolean;
  /** Texture row 0 is the top of the picture (WebGPU's convention); default true. */
  flipY?: boolean;
  /** Texels hold straight alpha. Pictures are always premultiplied. */
  unpremultiplied?: boolean;
}

function formatOf(texture: GPUTexture) {
  const f = FORMATS[texture.format];
  if (!f) {
    throw new Error(
      `TextureBridge: ${texture.format} is not supported (${bridgeFormats.join(", ")})`,
    );
  }
  if (texture.sampleCount > 1) {
    throw new Error("TextureBridge: resolve multisampled textures first");
  }
  return f;
}

export class TextureBridge {
  constructor(readonly gpu: Gpu) {}

  private params(
    texture: GPUTexture,
    opts: BridgeOptions,
    rowWords: number,
  ): UniformWriter {
    const f = formatOf(texture);
    return UniformWriter.for(bridgeModule).setAll({
      width: texture.width,
      height: texture.height,
      rowWords,
      format: f.code,
      srgb: f.srgb || !!opts.srgb,
      flipY: opts.flipY ?? true,
      unpremultiplied: !!opts.unpremultiplied,
    });
  }

  private layout(texture: GPUTexture) {
    const f = formatOf(texture);
    const bytesPerRow = copyBytesPerRow(texture.width * f.bytes);
    return { bytesPerRow, bytes: bytesPerRow * texture.height };
  }

  /** The texture as a picture at (0, 0, width, height); recorded, not submitted. */
  read(texture: GPUTexture, opts: BridgeOptions = {}): FxBuffer {
    if (!(texture.usage & GPUTextureUsage.COPY_SRC)) {
      throw new Error("TextureBridge.read: the texture lacks COPY_SRC usage");
    }
    const gpu = this.gpu;
    const { bytesPerRow, bytes } = this.layout(texture);
    const texels = gpu.lend(bytes);
    gpu
      .commandEncoder()
      .copyTextureToBuffer(
        { texture },
        { buffer: texels, bytesPerRow, rowsPerImage: texture.height },
        { width: texture.width, height: texture.height },
      );
    const picture = gpu.scratch(texture.width, texture.height);
    const u = this.params(texture, opts, bytesPerRow / 4).set(
      "stride",
      picture.stride,
    );
    gpu.run(
      gpu.load(bridgeModule, "unpackTexels"),
      [texture.width, texture.height, 1],
      { texels, picture },
      u,
    );
    // Unpacked: the texels' buffer serves the frame's write (same size).
    gpu.release(texels);
    return picture;
  }

  /**
   * Writes `picture` into the texture, which covers `frame` of the image
   * (by default (0, 0, width, height)). Outside the picture is transparent.
   */
  write(
    picture: FxBuffer | null,
    texture: GPUTexture,
    opts: BridgeOptions & { frame?: Rect } = {},
  ) {
    if (!(texture.usage & GPUTextureUsage.COPY_DST)) {
      throw new Error("TextureBridge.write: the texture lacks COPY_DST usage");
    }
    const gpu = this.gpu;
    const frame = opts.frame ?? rect(0, 0, texture.width, texture.height);
    // Nothing at all is a transparent frame, which a 1x1 empty picture is.
    const src = picture ?? placed(gpu.scratch(1, 1), rect(-2, -2, -1, -1));
    const { bytesPerRow, bytes } = this.layout(texture);
    const texels = gpu.lend(bytes);
    const u = this.params(texture, opts, bytesPerRow / 4).setAll({
      srcOffsetX: frame.x1 - src.rect.x1,
      srcOffsetY: frame.y1 - src.rect.y1,
      srcWidth: src.width,
      srcHeight: src.height,
      srcStride: src.stride,
    });
    gpu.run(
      gpu.load(bridgeModule, "packTexels"),
      [texture.width, texture.height, 1],
      { src, texelsOut: texels },
      u,
    );
    gpu
      .commandEncoder()
      .copyBufferToTexture(
        { buffer: texels, bytesPerRow, rowsPerImage: texture.height },
        { texture },
        { width: texture.width, height: texture.height },
      );
  }
}

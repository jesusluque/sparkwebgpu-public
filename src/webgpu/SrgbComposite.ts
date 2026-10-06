// Splats blended in sRGB space over a linear target, as WebGL Spark blends
// them on the canvas. Hardware blending works on the stored (linear) values,
// so the splats blend in a layer of their own: an 8-bit copy of the target
// as WebGL's canvas would hold it, q(srgb(dst)), with the transmittance T
// left by the splats in alpha (cleared to 1). Each blend then rounds as
// WebGL's does. A layer that starts empty and holds L and T apart instead
// is biased: T stalls at a few levels behind dense splats (T(1 - a) rounds
// back to T), and the background shows through them. A full-screen pass
// writes back
//   linear(layer + T * (srgb(dst) - q(srgb(dst))))
// over a copy of the target: the term in T restores what quantisation
// (and clamping, for HDR) took from the background where it still shows.
// srgb() and linear() are three's own transfer functions; pixels no splat
// covers are left as they were.
//
// With tone mapping, three tone-maps the target in its output pass, after
// everything is blended, where WebGL Spark blends untone-mapped splats over
// the tone-mapped picture. So the layer starts from the picture as it will
// show, srgb(T(dst)), and is written back as T's inverse
// (toneMapping.ts), which the output pass maps to the composited colour.
// Transparent objects drawn after the splats then still blend over them,
// in three's linear space. Colours T never produces (AgX and ACES desaturate
// bright colours, and AgX's white is 0.997) come out as the nearest it
// does.

import * as THREE from "three";
import { reuseTexture } from "./gpuBuffers";
import { FULLSCREEN_TRIANGLE_WGSL } from "./renderPipeline";
import { toneMappingWgsl } from "./toneMapping";

const COMPOSITE_WGSL = (
  multisampled: boolean,
  toneMapping: number,
) => /* wgsl */ `
@group(0) @binding(0) var dstTex: ${multisampled ? "texture_multisampled_2d" : "texture_2d"}<f32>;
@group(0) @binding(1) var layerTex: ${multisampled ? "texture_multisampled_2d" : "texture_2d"}<f32>;
// x: three's toneMappingExposure.
@group(0) @binding(2) var<uniform> params: vec4f;
${toneMappingWgsl(toneMapping)}
${FULLSCREEN_TRIANGLE_WGSL}
// three's sRGBTransferOETF / sRGBTransferEOTF.
fn toSrgb(c: vec3f) -> vec3f {
  return select(1.055 * pow(c, vec3f(0.41666)) - 0.055, c * 12.92, c <= vec3f(0.0031308));
}
fn toLinear(c: vec3f) -> vec3f {
  return select(pow(c * 0.9478672986 + 0.0521327014, vec3f(2.4)), c * 0.0773993808, c <= vec3f(0.04045));
}

// The background as shown, in sRGB: clamped as the display (and WebGL's
// canvas, before splats blend over it) clamps it, or an HDR sky shows
// through the splats' transmittance brighter than white.
fn background(dst: vec4f) -> vec3f {
  return toSrgb(saturate(toneMap(max(dst.rgb, vec3f(0.0)), params.x)));
}
// The layer's 8-bit unorm rounding of c.
fn quantize(c: vec3f) -> vec3f {
  return round(saturate(c) * 255.0) / 255.0;
}

@fragment
fn initMain(@builtin(position) p: vec4f${multisampled ? ", @builtin(sample_index) s: u32" : ""}) -> @location(0) vec4f {
  let dst = textureLoad(dstTex, vec2i(p.xy), ${multisampled ? "s" : "0"});
  return vec4f(background(dst), 1.0);
}

@fragment
fn fragmentMain(@builtin(position) p: vec4f${multisampled ? ", @builtin(sample_index) s: u32" : ""}) -> @location(0) vec4f {
  let xy = vec2i(p.xy);
  let dst = textureLoad(dstTex, xy, ${multisampled ? "s" : "0"});
  let layer = textureLoad(layerTex, xy, ${multisampled ? "s" : "0"});
  let t = layer.a;
  let bg = background(dst);
  let q = quantize(bg);
  if (t == 1.0 && all(layer.rgb == q)) {
    return dst;
  }
  let rgb = layer.rgb + t * (bg - q);
  return vec4f(inverseToneMap(toLinear(rgb), params.x), 1.0 - t + t * dst.a);
}
`;

/**
 * The splat layer's format: 8-bit, so that each blend rounds as on WebGL's
 * 8-bit canvas (dense scenes drift a few levels from half floats).
 */
export const SRGB_LAYER_FORMAT: GPUTextureFormat = "rgba8unorm";

/** Blend state for the layer: colour over, alpha the transmittance. */
export const SRGB_LAYER_BLEND: GPUBlendState = {
  color: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
  alpha: { srcFactor: "zero", dstFactor: "one-minus-src-alpha" },
};

interface Scratch {
  layer: GPUTexture;
  copy: GPUTexture;
}

/** three's tone mapping, as its output pass will apply it. */
export interface CompositeToneMapping {
  /** A THREE.*ToneMapping in INVERTIBLE_TONE_MAPPINGS. */
  toneMapping: number;
  exposure: number;
}

export class SrgbComposite {
  private pipelines = new Map<string, GPURenderPipeline>();
  private scratch: Scratch | null = null;
  private readonly params: GPUBuffer;
  // The layer's, from beginLayer.
  private toneMapping: CompositeToneMapping = {
    toneMapping: THREE.NoToneMapping,
    exposure: 1,
  };

  constructor(readonly device: GPUDevice) {
    this.params = device.createBuffer({
      label: "splat layer composite params",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  /**
   * A layer for `color` (the target's attachment texture), holding its
   * picture as shown under three's `toneMapping` when its output pass
   * tone-maps `color`. `composite` must follow before `color` changes.
   */
  beginLayer(
    encoder: GPUCommandEncoder,
    color: GPUTexture,
    depth: GPURenderPassDepthStencilAttachment | undefined,
    timestampWrites?: GPURenderPassTimestampWrites,
    toneMapping: CompositeToneMapping = {
      toneMapping: THREE.NoToneMapping,
      exposure: 1,
    },
  ): GPURenderPassEncoder {
    const { layer, copy } = this.ensureScratch(color);
    this.toneMapping = toneMapping;
    encoder.copyTextureToTexture({ texture: color }, { texture: copy }, [
      color.width,
      color.height,
    ]);
    this.device.queue.writeBuffer(
      this.params,
      0,
      new Float32Array([toneMapping.exposure, 0, 0, 0]),
    );
    const init = this.pipeline(
      "initMain",
      SRGB_LAYER_FORMAT,
      color.sampleCount,
      toneMapping.toneMapping,
    );
    const pass = encoder.beginRenderPass({
      label: "splat layer background",
      colorAttachments: [
        { view: layer.createView(), loadOp: "clear", storeOp: "store" },
      ],
    });
    pass.setPipeline(init);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: init.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: copy.createView() },
          { binding: 2, resource: { buffer: this.params } },
        ],
      }),
    );
    pass.draw(3);
    pass.end();
    return encoder.beginRenderPass({
      label: "splat layer",
      timestampWrites,
      colorAttachments: [
        { view: layer.createView(), loadOp: "load", storeOp: "store" },
      ],
      depthStencilAttachment: depth,
    });
  }

  /**
   * Composites the layer over `color`, through `view` (its attachment view,
   * which may resolve), within `viewport` when given.
   */
  composite(
    encoder: GPUCommandEncoder,
    color: GPUTexture,
    attachment: Pick<GPURenderPassColorAttachment, "view" | "resolveTarget">,
    viewport?: { x: number; y: number; z: number; w: number } | null,
  ) {
    const { layer, copy } = this.ensureScratch(color);
    const pipeline = this.pipeline(
      "fragmentMain",
      color.format,
      color.sampleCount,
      this.toneMapping.toneMapping,
    );
    const pass = encoder.beginRenderPass({
      label: "splat layer composite",
      colorAttachments: [{ ...attachment, loadOp: "load", storeOp: "store" }],
    });
    if (viewport) {
      pass.setViewport(viewport.x, viewport.y, viewport.z, viewport.w, 0, 1);
      // Clamped: the viewport may reach past the target, a scissor can't.
      const x0 = Math.max(0, Math.floor(viewport.x));
      const y0 = Math.max(0, Math.floor(viewport.y));
      const x1 = Math.min(color.width, Math.ceil(viewport.x + viewport.z));
      const y1 = Math.min(color.height, Math.ceil(viewport.y + viewport.w));
      pass.setScissorRect(x0, y0, Math.max(0, x1 - x0), Math.max(0, y1 - y0));
    }
    pass.setPipeline(pipeline);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: copy.createView() },
          { binding: 1, resource: layer.createView() },
          { binding: 2, resource: { buffer: this.params } },
        ],
      }),
    );
    pass.draw(3);
    pass.end();
  }

  private ensureScratch(color: GPUTexture): Scratch {
    const size = [color.width, color.height];
    const sampleCount = color.sampleCount;
    this.scratch = {
      layer: reuseTexture(this.device, this.scratch?.layer, {
        label: "splat layer",
        size,
        format: SRGB_LAYER_FORMAT,
        sampleCount,
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      }),
      copy: reuseTexture(this.device, this.scratch?.copy, {
        label: "splat layer background",
        size,
        format: color.format,
        sampleCount,
        // Multisampled textures must be attachments.
        usage:
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.TEXTURE_BINDING |
          (sampleCount > 1 ? GPUTextureUsage.RENDER_ATTACHMENT : 0),
      }),
    };
    return this.scratch;
  }

  private pipeline(
    entryPoint: "initMain" | "fragmentMain",
    format: GPUTextureFormat,
    sampleCount: number,
    toneMapping: number,
  ) {
    const key = `${entryPoint}/${format}/${sampleCount}/${toneMapping}`;
    let pipeline = this.pipelines.get(key);
    if (!pipeline) {
      const module = this.device.createShaderModule({
        code: COMPOSITE_WGSL(sampleCount > 1, toneMapping),
      });
      pipeline = this.device.createRenderPipeline({
        label: "splat layer composite",
        layout: "auto",
        vertex: { module, entryPoint: "fullscreenVertex" },
        fragment: { module, entryPoint, targets: [{ format }] },
        multisample: { count: sampleCount },
      });
      this.pipelines.set(key, pipeline);
    }
    return pipeline;
  }

  dispose() {
    this.scratch?.layer.destroy();
    this.scratch?.copy.destroy();
    this.scratch = null;
    this.params.destroy();
  }
}

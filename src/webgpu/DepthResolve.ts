// A single-sample copy of a multisampled depth texture (its sample 0).
//
// With antialias: true three renders the scene multisampled. Splats drawn on
// the single-sample canvas after three's output pass (SparkWebGPU) can't
// attach that depth, and WebGPU can't resolve depth textures, so a
// full-screen pass copies it into a depth32float texture they test against.

import { reuseTexture } from "./gpuBuffers";
import { FULLSCREEN_TRIANGLE_WGSL } from "./renderPipeline";

const SHADER = /* wgsl */ `
@group(0) @binding(0) var depthMs: texture_depth_multisampled_2d;
${FULLSCREEN_TRIANGLE_WGSL}
@fragment
fn fs(@builtin(position) pos: vec4f) -> @builtin(frag_depth) f32 {
  return textureLoad(depthMs, vec2i(pos.xy), 0);
}
`;

export const RESOLVED_DEPTH_FORMAT: GPUTextureFormat = "depth32float";

export class DepthResolve {
  private pipeline?: GPURenderPipeline;
  private texture?: GPUTexture;

  constructor(readonly device: GPUDevice) {}

  /** Records the copy of `source` into `encoder`; returns the copy. */
  resolve(encoder: GPUCommandEncoder, source: GPUTexture): GPUTexture {
    const { device } = this;
    if (!this.pipeline) {
      const module = device.createShaderModule({
        label: "depth resolve",
        code: SHADER,
      });
      this.pipeline = device.createRenderPipeline({
        label: "depth resolve",
        layout: "auto",
        vertex: { module, entryPoint: "fullscreenVertex" },
        fragment: { module, entryPoint: "fs", targets: [] },
        depthStencil: {
          format: RESOLVED_DEPTH_FORMAT,
          depthWriteEnabled: true,
          depthCompare: "always",
        },
      });
    }
    const texture = reuseTexture(device, this.texture, {
      label: "resolved depth",
      size: [source.width, source.height],
      format: RESOLVED_DEPTH_FORMAT,
      // Sampled too: the tile rasterizer reads the depth it tests against.
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.TEXTURE_BINDING,
    });
    this.texture = texture;
    const bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: source.createView({ aspect: "depth-only" }),
        },
      ],
    });
    const pass = encoder.beginRenderPass({
      label: "depth resolve",
      colorAttachments: [],
      depthStencilAttachment: {
        view: texture.createView(),
        depthLoadOp: "clear",
        depthClearValue: 0,
        depthStoreOp: "store",
      },
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    return texture;
  }

  dispose() {
    this.texture?.destroy();
    this.texture = undefined;
  }
}

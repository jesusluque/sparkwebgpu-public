// Render pipelines from a module with a vertex and a fragment entry: the
// bind-group layout is the union of the bindings either entry uses, visible
// to the stages that use them.

import type { BindingReflection, KernelModule } from "./KernelModule";
import { bindGroupLayouts, usedBindings } from "./KernelRegistry";

/**
 * A vertex entry `fullscreenVertex` for `draw(3)`: one triangle covering the
 * target, for full-screen passes.
 */
export const FULLSCREEN_TRIANGLE_WGSL = /* wgsl */ `
@vertex
fn fullscreenVertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}
`;

/**
 * A depth attachment to test against without writing it (the splat
 * pipelines don't write depth). Loaded and stored rather than
 * `depthReadOnly`: Safari 26 doesn't test against a read-only attachment,
 * and splats then cover three's opaque objects.
 */
export function depthTestAttachment(
  view: GPURenderPassDepthStencilAttachment["view"],
  format: GPUTextureFormat | null | undefined,
): GPURenderPassDepthStencilAttachment {
  return {
    view,
    depthLoadOp: "load",
    depthStoreOp: "store",
    ...(format?.includes("stencil")
      ? { stencilLoadOp: "load", stencilStoreOp: "store" }
      : {}),
  };
}

export interface RenderPipelineOptions {
  vertex: string;
  fragment: string;
  targets: (GPUColorTargetState | null)[];
  primitive?: GPUPrimitiveState;
  depthStencil?: GPUDepthStencilState;
  multisample?: GPUMultisampleState;
  label?: string;
}

export interface ReflectedRenderPipeline {
  pipeline: GPURenderPipeline;
  layouts: GPUBindGroupLayout[];
  bindings: BindingReflection[];
}

export function createReflectedRenderPipeline(
  device: GPUDevice,
  module: KernelModule,
  options: RenderPipelineOptions,
): ReflectedRenderPipeline {
  const entry = (name: string) => {
    const e = module.reflection.entries.find((x) => x.name === name);
    if (!e) throw new Error(`${module.name}: no entry '${name}'`);
    return e;
  };
  const vs = entry(options.vertex);
  const fs = entry(options.fragment);
  const visibility = new Map<string, number>();
  for (const b of usedBindings(module, vs)) {
    visibility.set(
      b.name,
      (visibility.get(b.name) ?? 0) | GPUShaderStage.VERTEX,
    );
  }
  for (const b of usedBindings(module, fs)) {
    visibility.set(
      b.name,
      (visibility.get(b.name) ?? 0) | GPUShaderStage.FRAGMENT,
    );
  }
  const bindings = module.reflection.bindings
    .filter((b) => visibility.has(b.name))
    .sort((a, b) => a.group - b.group || a.binding - b.binding);

  for (const b of bindings) {
    if (b.kind === "unsupported" || b.kind === "external") {
      throw new Error(`${module.name}: unsupported binding '${b.name}'`);
    }
  }
  const layouts = bindGroupLayouts(
    device,
    module.name,
    bindings,
    (b) => visibility.get(b.name) as number,
  );
  const shaderModule = device.createShaderModule({
    label: module.name,
    code: module.wgsl,
  });
  const pipeline = device.createRenderPipeline({
    label: options.label ?? module.name,
    layout: device.createPipelineLayout({ bindGroupLayouts: layouts }),
    vertex: { module: shaderModule, entryPoint: vs.name },
    fragment: {
      module: shaderModule,
      entryPoint: fs.name,
      targets: options.targets,
    },
    primitive: options.primitive ?? { topology: "triangle-strip" },
    depthStencil: options.depthStencil,
    multisample: options.multisample,
  });
  return { pipeline, layouts, bindings };
}

/** Bind groups for a reflected pipeline, buffers by binding name. */
export function createBindGroups(
  device: GPUDevice,
  rp: ReflectedRenderPipeline,
  buffers: Record<string, GPUBuffer | GPUBufferBinding>,
): GPUBindGroup[] {
  return rp.layouts.map((layout, group) =>
    device.createBindGroup({
      layout,
      entries: rp.bindings
        .filter((b) => b.group === group)
        .map((b) => {
          const r = buffers[b.name];
          if (!r) throw new Error(`missing buffer '${b.name}'`);
          return {
            binding: b.binding,
            resource:
              "size" in r && !("buffer" in r)
                ? { buffer: r as GPUBuffer }
                : (r as GPUBufferBinding),
          };
        }),
    }),
  );
}

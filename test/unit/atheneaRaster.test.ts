// The athenea raster and output plugins on a PluginHost, without a GPU: the
// compiled preset they name, the per-mesh uniform block the raster writes
// (ids, storage, overrides at their reflected offsets), and the output
// stage as the last, live step of an FxChain.

import { describe, expect, it } from "vitest";
import type { WgpuSplatMesh } from "../../src/webgpu/WgpuSplatRenderer";
import {
  MAX_ATHENEA_OVERRIDES,
  atheneaOutputPlugin,
  atheneaRasterPlugin,
} from "../../src/webgpu/athenea/rasterPlugin";
import { ATTRIB_NONE, AttribPool } from "../../src/webgpu/attributes/schema";
import type { FxChain } from "../../src/webgpu/fx/FxChain";
import { colourVariants } from "../../src/webgpu/generated/presets";
import { PluginHost, fxPlugin } from "../../src/webgpu/plugins";
import { UniformWriter } from "../../src/webgpu/uniforms";

function fakeMesh(pool: AttribPool | null): WgpuSplatMesh {
  return {
    source: { attribs: pool },
    object: {},
  } as unknown as WgpuSplatMesh;
}

describe("athenea raster plugin", () => {
  it("registers against its compiled preset and picks its variant", () => {
    const host = new PluginHost({ tier: 2 });
    host.register(atheneaRasterPlugin());
    expect(host.resolve(fakeMesh(null)).colour).toBe("athenea.raster");
  });

  it("writes ids, storage and overrides into its uniform block", async () => {
    const generate = (await colourVariants["athenea.raster"].load())[
      "kernels/generate"
    ];
    const raster = atheneaRasterPlugin({ emission: 0.5 });
    const block = raster.uniforms?.[0];
    expect(block?.block).toBe("atheneaRaster");
    // A paged pool is bound as it is: ids are its column order.
    const pool = new AttribPool(4);
    pool.setAttribute("normalOct", [0, 0, 0, 0], "u32");
    pool.setAttribute("athcGroup", [0, 1, 1, 2, 2, 3, 3, 4], "u32", 2);
    pool.setAttribute("emission", [0, 0, 0, 0], "u32");
    pool.gpuBuffer = {} as GPUBuffer;
    const mesh = fakeMesh(pool);
    raster.attach?.({ device: {} } as never);
    raster.setStoredLinear(mesh.object, true);
    raster.setOverrides(mesh, [
      { groups: [1, 3], tint: [0.5, 0.25, 1], replace: true, opacity: 0.5 },
    ]);
    const w = UniformWriter.for(generate, "atheneaRaster");
    block?.write(w, { frame: null, mesh, asset: mesh });
    const view = new DataView(w.data);
    const field = (name: string) =>
      w.block.fields?.find((f) => f.name === name)?.offset as number;
    expect(view.getUint32(field("storedLinear"), true)).toBe(1);
    expect(view.getUint32(field("emissionId"), true)).toBe(2);
    expect(view.getUint32(field("groupId"), true)).toBe(1);
    expect(view.getUint32(field("overrideCount"), true)).toBe(1);
    expect(view.getFloat32(field("emissionScale"), true)).toBe(0.5);
    const rows = field("overrides");
    expect(view.getUint32(rows, true)).toBe(1);
    expect(view.getUint32(rows + 4, true)).toBe(3);
    expect(view.getFloat32(rows + 8, true)).toBe(1);
    expect(view.getFloat32(rows + 12, true)).toBe(0.5);
    expect(view.getFloat32(rows + 20, true)).toBe(0.25);
    expect(raster.buffers?.("splat", { frame: null, mesh })).toEqual({
      atheneaPool: pool.gpuBuffer,
    });
  });

  it("leaves the override table empty for a cloud without group ids", async () => {
    const generate = (await colourVariants["athenea.raster"].load())[
      "kernels/generate"
    ];
    const raster = atheneaRasterPlugin();
    const pool = new AttribPool(1);
    pool.setAttribute("normalOct", [0], "u32");
    pool.gpuBuffer = {} as GPUBuffer;
    const mesh = fakeMesh(pool);
    raster.attach?.({ device: {} } as never);
    raster.setOverrides(mesh, [{ groups: [0, 1], opacity: 0 }]);
    const w = UniformWriter.for(generate, "atheneaRaster");
    raster.uniforms?.[0].write(w, { frame: null, mesh, asset: mesh });
    const view = new DataView(w.data);
    const at = (name: string) =>
      w.block.fields?.find((f) => f.name === name)?.offset as number;
    expect(view.getUint32(at("groupId"), true)).toBe(ATTRIB_NONE);
    expect(view.getUint32(at("emissionId"), true)).toBe(ATTRIB_NONE);
    expect(view.getUint32(at("overrideCount"), true)).toBe(0);
    expect(MAX_ATHENEA_OVERRIDES).toBe(16);
  });

  it("ends the FxChain with the display transform, its parameters live", () => {
    const host = new PluginHost({ tier: 2 });
    const output = atheneaOutputPlugin({ view: "aces2", exposure: 1.5 });
    host.register(output);
    host.register(fxPlugin("other", [{ effect: {} as never, name: "x" }]));
    const chain = { steps: [] } as unknown as FxChain;
    host.applyFx(chain);
    expect(chain.steps.map((s) => s.instance)).toEqual([
      "other.x",
      "athenea.output.display",
    ]);
    const step = chain.steps[1];
    expect(step.params).toMatchObject({ view: 2, exposure: 1.5 });
    output.set({ exposure: -2, display: "displayP3" });
    expect(step.params).toMatchObject({ view: 2, exposure: -2, display: 2 });
    const exposure = host.controls().find((c) => c.id === "exposure");
    exposure?.set?.(0.25);
    expect(step.params.exposure).toBe(0.25);
  });
});

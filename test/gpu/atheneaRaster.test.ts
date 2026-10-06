// The athenea raster plugin (atheneaRasterPlugin, slang/athenea_adapter/
// raster.slang) through WgpuSplatRenderer on Dawn, against athenea's own
// arithmetic: a pixel's colour as splat_blend composes it -- front to back,
// each splat's light `cloudLight(colour)` (common/color.slang) plus its
// emission, weighted by transmittance * alpha -- over the background, in
// linear light. Overrides (splatOverrideColour, by finest-group range) and
// a hidden range change exactly the splats they name.
//
// The scene is a stack of large splats facing the camera, so at the centre
// pixel each one's alpha is its opacity (the footprints of Spark and
// athenea differ at the edges, not at the centre): what is left to compare
// is the colour path and the blend, which is what this plugin is.

import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { encodeExtSplat } from "../../src/utils";
import {
  GpuSplatSource,
  WgpuSplatRenderer,
  type WgpuSplatRendererOptions,
} from "../../src/webgpu/WgpuSplatRenderer";
import {
  atheneaGroupOf,
  pickAtheneaGroup,
} from "../../src/webgpu/athenea/pick";
import {
  type AtheneaGroupOverride,
  atheneaRasterPlugin,
} from "../../src/webgpu/athenea/rasterPlugin";
import { SplatAttributes } from "../../src/webgpu/attributes/SplatAttributes";
import { AttribPool } from "../../src/webgpu/attributes/schema";
import { upload } from "../../src/webgpu/gpuBuffers";
import { PluginHost } from "../../src/webgpu/plugins";
import { device } from "./device";

const W = 64;
const H = 48;
const BG = [0.05, 0.1, 0.2];

// athenea common/color.slang.
const srgbToLinear = (c: number) => {
  const x = Math.max(c, 0);
  return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
};
const cloudLight = (c: number[], linear: boolean) =>
  c.map((x) => (linear ? Math.max(x, 0) : srgbToLinear(x)));

/** packing.slang packRgb9e5 for values that are exact at exponent 15. */
const rgb9e5 = (r: number, g: number, b: number) =>
  (Math.round(r * 512) |
    (Math.round(g * 512) << 9) |
    (Math.round(b * 512) << 18) |
    (15 << 27)) >>>
  0;

interface Layer {
  z: number;
  opacity: number;
  rgb: [number, number, number];
  emission?: [number, number, number];
}

const LAYERS: Layer[] = [
  { z: -2, opacity: 0.45, rgb: [0.9, 0.3, 0.1] },
  { z: -3, opacity: 0.5, rgb: [0.2, 0.8, 0.35], emission: [0.5, 0.25, 0] },
  { z: -4, opacity: 0.6, rgb: [0.1, 0.2, 0.95] },
  { z: -5, opacity: 0.8, rgb: [1.6, 1.2, 0.7] }, // past 1: light, not a code
];

/** splat_blend's walk at one pixel: nearest first, over the background. */
function athenea(
  layers: Layer[],
  linear: boolean,
  overrides: AtheneaGroupOverride[] = [],
  emission = 1,
) {
  let T = 1;
  const rgb = [0, 0, 0];
  layers.forEach((l, k) => {
    let light = cloudLight(l.rgb, linear);
    let alpha = l.opacity;
    const o = overrides.find((o) => k >= o.groups[0] && k + 1 <= o.groups[1]);
    if (o) {
      const tint = o.tint ?? [1, 1, 1];
      light = o.replace ? [...tint] : light.map((c, i) => c * tint[i]);
      alpha *= o.opacity ?? 1;
    }
    if (l.emission)
      light = light.map((c, i) => c + emission * (l.emission?.[i] ?? 0));
    const w = T * alpha;
    for (let c = 0; c < 3; c++) rgb[c] += w * light[c];
    T *= 1 - alpha;
  });
  return rgb.map((c, i) => c + T * BG[i]);
}

describe.skipIf(!device)("athenea raster plugin", () => {
  const d = device as GPUDevice;

  function stack(layers: Layer[]) {
    const n = layers.length;
    const a = new Uint32Array(n * 4);
    const b = new Uint32Array(n * 4);
    layers.forEach((l, i) =>
      encodeExtSplat(
        [a, b],
        i,
        0,
        0,
        l.z,
        3,
        3,
        0.01,
        0,
        0,
        0,
        1,
        l.opacity,
        ...l.rgb,
      ),
    );
    const source = GpuSplatSource.fromExt(d, a, b, n);
    // As a decoded .athc carries them: each splat its own finest group.
    source.setAttribute(
      "athcGroup",
      layers.flatMap((_, k) => [k, k + 1]),
      "u32",
      2,
    );
    source.setAttribute(
      "emission",
      layers.map((l) => (l.emission ? rgb9e5(...l.emission) : 0)),
      "u32",
      1,
    );
    return source;
  }

  const color = d.createTexture({
    size: [W, H],
    format: "rgba16float",
    usage:
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.TEXTURE_BINDING,
  });
  const target = {
    texture: { colorSpace: THREE.LinearSRGBColorSpace },
    samples: 0,
  } as unknown as THREE.RenderTarget;
  const fakeRenderer = {
    backend: {
      isWebGPUBackend: true,
      device: d,
      context: {} as GPUCanvasContext,
      get: (r: object) =>
        r === target.texture ? { texture: color } : undefined,
    },
  };

  const camera = new THREE.PerspectiveCamera(50, W / H, 0.05, 100);
  camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();

  function clear() {
    const enc = d.createCommandEncoder();
    enc
      .beginRenderPass({
        colorAttachments: [
          {
            view: color.createView(),
            loadOp: "clear",
            storeOp: "store",
            clearValue: [...BG, 1],
          },
        ],
      })
      .end();
    d.queue.submit([enc.finish()]);
  }

  async function centre(): Promise<number[]> {
    const bytesPerRow = 256 * Math.ceil((W * 8) / 256);
    const buf = d.createBuffer({
      size: bytesPerRow * H,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = d.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: color }, { buffer: buf, bytesPerRow }, [
      W,
      H,
    ]);
    d.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const halves = new Uint16Array(buf.getMappedRange());
    const at = ((H / 2) * bytesPerRow) / 2 + (W / 2) * 4;
    const out = [0, 1, 2].map((c) =>
      THREE.DataUtils.fromHalfFloat(halves[at + c]),
    );
    buf.unmap();
    buf.destroy();
    return out;
  }

  async function render(
    layers: Layer[],
    options: WgpuSplatRendererOptions,
    setup?: (
      host: PluginHost,
      mesh: ReturnType<WgpuSplatRenderer["add"]>,
    ) => void,
  ) {
    const splats = new WgpuSplatRenderer(fakeRenderer as never, {
      depthTest: false,
      alwaysGenerate: true,
      ...options,
    });
    const mesh = splats.add(stack(layers));
    if (setup) {
      const host = new PluginHost({ capabilities: splats.capabilities });
      host.attach(splats);
      setup(host, mesh);
      await host.ready();
    }
    for (let frame = 0; frame < 3; frame++) {
      clear();
      splats.render(camera, target);
      await d.queue.onSubmittedWorkDone();
      await new Promise((r) => setTimeout(r, 5));
    }
    const px = await centre();
    splats.plugins?.detach();
    splats.dispose();
    return px;
  }

  const worst = (a: number[], b: number[]) =>
    Math.max(...a.map((x, i) => Math.abs(x - b[i]) / Math.max(1, b[i])));

  for (const rasterizer of ["hardware", "tiles"] as const) {
    for (const linear of [false, true]) {
      it(`blends as athenea's splat_blend (${rasterizer}, ${linear ? "linear" : "sRGB"} cloud)`, async () => {
        const got = await render(LAYERS, { rasterizer }, (host, mesh) => {
          const raster = atheneaRasterPlugin();
          host.register(raster);
          raster.setStoredLinear(mesh, linear);
        });
        const want = athenea(LAYERS, linear);
        if (process.env.ATHENEA_LOG)
          process.stderr.write(
            `WORST ${rasterizer} ${linear} ${got} / ${want} ${worst(got, want)}\n`,
          );
        // Half floats, and the colour's trip through Spark's transfer.
        expect(worst(got, want)).toBeLessThan(2e-3);
      });
    }
  }

  it("differs from Spark's colour path where athenea's does", async () => {
    // Spark decodes with gamma 2.2 and adds no emission: the plugin is
    // what moves the pixel onto athenea's value.
    const spark = await render(LAYERS, {});
    const want = athenea(LAYERS, false);
    expect(worst(spark, want)).toBeGreaterThan(1e-2);
  });

  it("overrides by finest-group range: tint, replace, hide; emission scale", async () => {
    const overrides: AtheneaGroupOverride[] = [
      { groups: [0, 1], tint: [0.2, 1, 1] },
      { groups: [2, 3], opacity: 0 },
      { groups: [3, 4], tint: [0.3, 0.3, 0.9], replace: true },
      // Never reached: the first row that applies is taken.
      { groups: [0, 4], tint: [0, 0, 0] },
    ];
    const got = await render(LAYERS, {}, (host, mesh) => {
      const raster = atheneaRasterPlugin({ emission: 2 });
      host.register(raster);
      // Keyed on the mesh's object as a SplatMesh would be.
      raster.setOverrides(mesh.object, overrides);
    });
    const want = athenea(LAYERS, false, overrides, 2);
    expect(worst(got, want)).toBeLessThan(2e-3);
    const plain = athenea(LAYERS, false, [], 2);
    expect(worst(got, plain)).toBeGreaterThan(5e-2);
  });

  it("picks finest groups through the id target, hidden ranges picked through", async () => {
    const splats = new WgpuSplatRenderer(fakeRenderer as never, {
      depthTest: false,
      alwaysGenerate: true,
    });
    const mesh = splats.add(stack(LAYERS));
    const host = new PluginHost({ capabilities: splats.capabilities });
    const raster = atheneaRasterPlugin();
    host.register(raster).attach(splats);
    await host.ready();
    const attributes = new SplatAttributes(splats, { targets: { id: true } });
    const frame = async () => {
      for (let k = 0; k < 2; k++) {
        clear();
        splats.render(camera, target);
        await d.queue.onSubmittedWorkDone();
      }
    };
    await frame();
    // The nearest splat covers the centre past pickAlpha (0.45 > 0.3).
    let hit = await pickAtheneaGroup(attributes, W / 2, H / 2);
    expect(hit?.mesh).toBe(mesh);
    expect(hit?.groups).toEqual([0, 1]);
    // Hide it: the pick goes through to the next group.
    raster.setOverrides(mesh, [{ groups: hit?.groups ?? [0, 1], opacity: 0 }]);
    await frame();
    hit = await pickAtheneaGroup(attributes, W / 2, H / 2);
    expect(hit?.groups).toEqual([1, 2]);
    attributes.dispose();
    host.detach();
    splats.dispose();
  });

  it("reads a paged pool's group back from the GPU", async () => {
    const pool = new AttribPool(5);
    pool.setAttribute("normalOct", [1, 2, 3, 4, 5], "u32");
    pool.setAttribute(
      "athcGroup",
      [0, 9, 9, 10, 10, 11, 11, 12, 12, 13],
      "u32",
      2,
    );
    // As PagedAttribPool keeps it: packed on the GPU, columns empty.
    pool.gpuBuffer = upload(d, pool.pack().words, "paged");
    for (const c of pool.columns) c.words = new Uint32Array(0);
    const mesh = { source: { attribs: pool } } as never;
    expect(await atheneaGroupOf(d, mesh, 3)).toEqual([11, 12]);
    expect(await atheneaGroupOf(d, mesh, 0)).toEqual([0, 9]);
    pool.gpuBuffer.destroy();
  });
});

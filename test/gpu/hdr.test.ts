// HDR end to end (SparkRendererOptions.hdr, SplatPlugin.hdr) on Dawn, with
// tiny scenes: one large splat facing the camera, so at the centre pixel
// its alpha is its opacity and the pixel is alpha * light + (1 - alpha) *
// background, read back from a HalfFloat (rgba16float) target as floats.
//
//   - linear light past 1 (4.0) survives generate, the accumulator and the
//     draw into the target, for the athenea raster colour and for a relit
//     splat (athenea's relight pass), packed accumulator asked for or not;
//   - a background past 1 is not clipped under splats: no 8-bit sRGB layer
//     (srgbBlend, WebGL parity) in HDR, through WgpuSplatRenderer and
//     through SparkRenderer on three's half-float canvas frame.

import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { PackedSplats } from "../../src/PackedSplats";
import { SparkRenderer } from "../../src/SparkRenderer";
import { SplatMesh } from "../../src/SplatMesh";
import { encodeExtSplat } from "../../src/utils";
import {
  GpuSplatSource,
  WgpuSplatRenderer,
  type WgpuSplatRendererOptions,
} from "../../src/webgpu/WgpuSplatRenderer";
import { atheneaRasterPlugin } from "../../src/webgpu/athenea/rasterPlugin";
import { atheneaRelightPlugin } from "../../src/webgpu/athenea/relightPlugin";
import { createReadback, readAndDestroy } from "../../src/webgpu/gpuBuffers";
import { PluginHost } from "../../src/webgpu/plugins";
import { loadWebGPU } from "../../src/webgpuLoader";
import { device, wideDevice } from "./device";

await loadWebGPU();

const W = 32;
const H = 32;
const OPACITY = 0.9;

/** A target, a fake three renderer around it, and its centre pixel. */
function harness(d: GPUDevice) {
  const color = d.createTexture({
    size: [W, H],
    format: "rgba16float",
    usage:
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.COPY_DST |
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
  camera.position.set(0, 0, 3);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();

  function clear(bg: number[]) {
    const enc = d.createCommandEncoder();
    enc
      .beginRenderPass({
        colorAttachments: [
          {
            view: color.createView(),
            loadOp: "clear",
            storeOp: "store",
            clearValue: [bg[0], bg[1], bg[2], 1],
          },
        ],
      })
      .end();
    d.queue.submit([enc.finish()]);
  }

  return { color, target, fakeRenderer, camera, clear };
}

async function centre(d: GPUDevice, texture: GPUTexture): Promise<number[]> {
  const bytesPerRow = 256;
  const buf = d.createBuffer({
    size: bytesPerRow * H,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = d.createCommandEncoder();
  enc.copyTextureToBuffer({ texture }, { buffer: buf, bytesPerRow }, [W, H]);
  d.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const halves = new Uint16Array(buf.getMappedRange());
  const at = ((H / 2) * bytesPerRow) / 2 + (W / 2) * 4;
  const px = [0, 1, 2].map((c) =>
    THREE.DataUtils.fromHalfFloat(halves[at + c]),
  );
  buf.unmap();
  buf.destroy();
  return px;
}

/** One splat 6 wide facing +Z at the origin, its ext RGBA as given. */
function card(d: GPUDevice, rgb: number[], opacity = OPACITY) {
  const a = new Uint32Array(4);
  const b = new Uint32Array(4);
  encodeExtSplat(
    [a, b],
    0,
    0,
    0,
    0,
    3,
    3,
    0.01,
    0,
    0,
    0,
    1,
    opacity,
    rgb[0],
    rgb[1],
    rgb[2],
  );
  return GpuSplatSource.fromExt(d, a, b, 1);
}

const relErr = (got: number[], want: number[]) =>
  Math.max(...got.map((v, i) => Math.abs(v - want[i]) / Math.max(want[i], 1)));

describe.skipIf(!device)("HDR splats into a float target", () => {
  const d = device as GPUDevice;
  const { color, target, fakeRenderer, camera, clear } = harness(d);

  async function render(
    options: WgpuSplatRendererOptions,
    source: GpuSplatSource,
    bg: number[],
    plugins?: (
      host: PluginHost,
      mesh: ReturnType<WgpuSplatRenderer["add"]>,
    ) => void,
  ) {
    const splats = new WgpuSplatRenderer(fakeRenderer as never, {
      depthTest: false,
      alwaysGenerate: true,
      ...options,
    });
    const mesh = splats.add(source);
    if (plugins) {
      const host = new PluginHost({ capabilities: splats.capabilities });
      host.attach(splats);
      plugins(host, mesh);
      await host.ready();
    }
    for (let frame = 0; frame < 2; frame++) {
      clear(bg);
      splats.render(camera, target);
      await d.queue.onSubmittedWorkDone();
    }
    const px = await centre(d, color);
    splats.plugins?.detach();
    splats.dispose();
    return px;
  }

  const LIGHT = [4, 3, 2];
  const BLACK = [0, 0, 0];
  const want = (light: number[], bg: number[], alpha = OPACITY) =>
    light.map((c, i) => alpha * c + (1 - alpha) * bg[i]);

  for (const accumulator of ["ext", "packed"] as const) {
    it(`keeps the athenea raster's linear light 4.0 (${accumulator} asked for)`, async () => {
      const got = await render(
        { accumulator },
        card(d, LIGHT),
        BLACK,
        (host, mesh) => {
          const raster = atheneaRasterPlugin();
          host.register(raster);
          raster.setStoredLinear(mesh, true);
        },
      );
      // Half floats and the trip through Spark's transfer (gamma 2.2).
      expect(relErr(got, want(LIGHT, BLACK))).toBeLessThan(2e-3);
      expect(got[0]).toBeGreaterThan(3.5);
    });
  }

  it("clamps in the packed accumulator without HDR, keeps 4.0 with it", async () => {
    // Spark's own colour path (no plugin): the source holds gamma-encoded
    // colour, decoded by the draw into the linear target.
    const encoded = LIGHT.map((c) => c ** (1 / 2.2));
    const packed = await render(
      { accumulator: "packed" },
      card(d, encoded),
      BLACK,
    );
    // Clamped to 1 (alpha is 8 bits there too).
    expect(Math.max(...packed)).toBeLessThan(OPACITY + 0.01);
    const hdr = await render(
      { accumulator: "packed", hdr: true },
      card(d, encoded),
      BLACK,
    );
    expect(relErr(hdr, want(LIGHT, BLACK))).toBeLessThan(2e-3);
  });

  // A dome past 1 behind a half-transparent splat.
  const DOME = [8, 6, 300];
  const GREY = [0.25, 0.25, 0.25];

  it("blends over a background past 1 unclipped with HDR, even with srgbBlend", async () => {
    const encoded = GREY.map((c) => c ** (1 / 2.2));
    const got = await render(
      { srgbBlend: true, hdr: true },
      card(d, encoded, 0.5),
      DOME,
    );
    expect(relErr(got, want(GREY, DOME, 0.5))).toBeLessThan(2e-3);
    // Without HDR, srgbBlend (WebGL parity) shows the dome through the
    // splat clamped, as an 8-bit canvas would.
    const parity = await render(
      { srgbBlend: true },
      card(d, encoded, 0.5),
      DOME,
    );
    expect(Math.max(...parity)).toBeLessThan(1.01);
  });

  it("blends over a background past 1 unclipped while an hdr plugin is registered", async () => {
    const got = await render(
      { srgbBlend: true },
      card(d, GREY, 0.5),
      DOME,
      (host, mesh) => {
        const raster = atheneaRasterPlugin();
        host.register(raster);
        raster.setStoredLinear(mesh, true);
      },
    );
    expect(relErr(got, want(GREY, DOME, 0.5))).toBeLessThan(2e-3);
  });
});

// SparkRenderer on three's canvas frame: a half-float target three tone-maps
// and encodes in its output pass. Without HDR the splats go through the
// 8-bit sRGB layer over a clamped copy of it (WebGL parity); with it they
// blend straight into it.
describe.skipIf(!device)(
  "SparkRenderer hdr on the half-float canvas frame",
  () => {
    const d = device as GPUDevice;
    const frame = d.createTexture({
      size: [W, H],
      format: "rgba16float",
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.TEXTURE_BINDING,
    });
    const frameTexture = {};
    const frameBufferTarget = {};
    const rc = {
      textures: [frameTexture],
      depthTexture: null,
      renderTarget: frameBufferTarget,
      width: W,
      height: H,
      viewport: false,
      viewportValue: new THREE.Vector4(0, 0, W, H),
    };
    const rcData: Record<string, unknown> = {};
    const fakeRenderer = {
      isWebGPURenderer: true,
      info: { frame: 0 },
      outputColorSpace: THREE.SRGBColorSpace,
      toneMapping: THREE.NoToneMapping,
      getOutputRenderTarget: () => null,
      _currentRenderContext: rc,
      _frameBufferTarget: frameBufferTarget,
      backend: {
        isWebGPUBackend: true,
        device: d,
        context: {
          getCurrentTexture: () => frame,
          canvas: { width: W, height: H },
        },
        get: (resource: object) =>
          resource === rc
            ? rcData
            : resource === frameTexture
              ? { texture: frame }
              : {},
        utils: {
          getCurrentColorFormat: () => frame.format,
          getCurrentDepthStencilFormat: () => undefined,
          getSampleCountRenderContext: () => 1,
          getCurrentColorSpace: () => THREE.LinearSRGBColorSpace,
        },
        updateViewport: () => {},
        pipelineUtils: { _activePipelines: new WeakMap() },
      },
    };

    async function render(spark: SparkRenderer, scene: THREE.Scene) {
      const camera = new THREE.PerspectiveCamera(50, W / H, 0.05, 100);
      camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
      camera.position.set(0, 0, 3);
      camera.updateProjectionMatrix();
      camera.updateMatrixWorld();
      for (let k = 0; k < 2; k++) {
        fakeRenderer.info.frame += 1;
        const encoder = d.createCommandEncoder();
        const descriptor: GPURenderPassDescriptor = {
          colorAttachments: [
            {
              view: frame.createView(),
              loadOp: "clear",
              storeOp: "store",
              clearValue: [8, 6, 300, 1],
            },
          ],
        };
        rcData.descriptor = descriptor;
        rcData.encoder = encoder;
        rcData.currentPass = encoder.beginRenderPass(descriptor);
        spark.onBeforeRender(fakeRenderer as never, scene, camera);
        (rcData.currentPass as GPURenderPassEncoder).end();
        d.queue.submit([encoder.finish()]);
        await d.queue.onSubmittedWorkDone();
      }
      return centre(d, frame);
    }

    function grey() {
      const packed = new PackedSplats();
      packed.pushSplat(
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(3, 3, 0.01),
        new THREE.Quaternion(),
        0.5,
        new THREE.Color(0.5, 0.5, 0.5),
      );
      return new SplatMesh({ packedSplats: packed });
    }

    for (const hdr of [false, true]) {
      it(`${hdr ? "keeps" : "clamps (WebGL parity)"} the background past 1 under splats`, async () => {
        const spark = new SparkRenderer({
          renderer: fakeRenderer as never,
          hdr,
        });
        const scene = new THREE.Scene();
        scene.add(spark);
        scene.add(grey());
        try {
          const px = await render(spark, scene);
          // PackedSplats keeps 8 bits: 0.5 is 128/255, decoded with gamma 2.2
          // into linear light.
          const q = 128 / 255;
          const g = q ** 2.2;
          if (hdr) {
            const expected = [8, 6, 300].map((c) => q * g + (1 - q) * c);
            expect(relErr(px, expected)).toBeLessThan(3e-3);
          } else {
            expect(Math.max(...px)).toBeLessThan(1.01);
          }
        } finally {
          spark.dispose();
        }
      });
    }
  },
);

// Athenea's relight pass: a matte card lit by a strong sun past 4 in linear
// light, its relit colour (the pass's own output) in the target unclipped.
describe.skipIf(!wideDevice)("relit light past 1 into a float target", () => {
  const d = wideDevice as GPUDevice;
  const { color, target, fakeRenderer, camera, clear } = harness(d);
  const TRANSFER = 112;
  const CELLS = 8;

  it("draws a relit radiance of 4 or more as it is", async () => {
    const source = card(d, [0.8, 0.8, 0.8]);
    // pbr: matte (metallic 0, roughness 1).
    source.setAttribute("pbr", [255 << 8], "u32", 1);
    // A transfer whose direct half is the clamped cosine about +Z
    // (degree-1 band), nothing indirect, no field; every cell open.
    const transfer = new Array(TRANSFER).fill(0);
    transfer[0] = 0.28209479177387814;
    transfer[2] = (2 / 3) * 0.4886025119029199;
    source.setAttribute("transfer", transfer, "f16", TRANSFER);
    source.setAttribute(
      "shadowBits",
      new Array(CELLS).fill(0xffffffff),
      "u32",
      CELLS,
    );

    const splats = new WgpuSplatRenderer(fakeRenderer as never, {
      depthTest: false,
      alwaysGenerate: true,
      // Asked for, but HDR (the relight plugin) keeps the ext one.
      accumulator: "packed",
    });
    const mesh = splats.add(source);
    const host = new PluginHost({ capabilities: splats.capabilities, tier: 2 });
    const sky = {
      width: 16,
      height: 8,
      data: new Float32Array(16 * 8 * 4).fill(0.2),
    };
    const relight = atheneaRelightPlugin({
      hdri: sky,
      sun: { direction: [0, 0, 1], intensity: 60 },
    });
    relight.setStoredLinear(mesh, true);
    host.register(relight).attach(splats);
    await host.ready();
    expect(splats.hdr).toBe(true);

    const draw = async () => {
      for (let k = 0; k < 2; k++) {
        clear([0, 0, 0]);
        splats.render(camera, target);
        await d.queue.onSubmittedWorkDone();
      }
    };
    const relitColour = async () => {
      const relit = relight.buffers?.("splat", { frame: null, mesh } as never)
        .atheneaRelit as GPUBuffer;
      const staging = createReadback(d, relit.size, "relit");
      const enc = d.createCommandEncoder();
      enc.copyBufferToBuffer(relit, 0, staging, 0, relit.size);
      d.queue.submit([enc.finish()]);
      return [...new Float32Array(await readAndDestroy(staging)).slice(0, 3)];
    };
    await draw();
    let c = await relitColour();
    // Past 4 whatever the sun's exact reading: scale it there.
    if (Math.min(...c) < 4) {
      const scale = 5 / Math.max(Math.min(...c), 1e-3);
      relight.set({ sun: { direction: [0, 0, 1], intensity: 60 * scale } });
      await draw();
      c = await relitColour();
    }
    expect(Math.min(...c)).toBeGreaterThanOrEqual(4);
    const px = await centre(d, color);
    expect(
      relErr(
        px,
        c.map((v) => OPACITY * v),
      ),
    ).toBeLessThan(3e-3);
    expect(Math.min(...px)).toBeGreaterThan(3.5);
    host.detach();
    splats.dispose();
  }, 120_000);
});

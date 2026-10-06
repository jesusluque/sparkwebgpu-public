// The tile rasterizer (rasterizer: "tiles") against the quad draw on the
// same scene: a dense cloud of overlapping splats of mixed sizes, onto an
// 8-bit canvas and, with a depth texture occluding half of it, into a
// float target.

import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { encodeExtSplat } from "../../src/utils";
import {
  GpuSplatSource,
  WgpuSplatRenderer,
  type WgpuSplatRendererOptions,
} from "../../src/webgpu/WgpuSplatRenderer";
import { device } from "./device";

const W = 160;
const H = 96;

describe.skipIf(!device)("TileRasterizer", () => {
  const d = device as GPUDevice;

  function cloud(count: number) {
    const a = new Uint32Array(count * 4);
    const b = new Uint32Array(count * 4);
    let s = 7;
    const rnd = () => {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      return s / 2 ** 32;
    };
    for (let i = 0; i < count; i++) {
      const q = new THREE.Quaternion(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5, 0.5)
        .normalize()
        .toArray() as number[];
      const big = rnd() < 0.05;
      const sc = big ? 0.15 + 0.2 * rnd() : 0.01 + 0.05 * rnd();
      encodeExtSplat(
        [a, b],
        i,
        2 * rnd() - 1,
        2 * rnd() - 1,
        2 * rnd() - 1,
        sc,
        sc * (0.2 + rnd()),
        sc * (0.2 + rnd()),
        q[0],
        q[1],
        q[2],
        q[3],
        0.2 + 0.8 * rnd(),
        rnd(),
        rnd(),
        rnd(),
      );
    }
    return GpuSplatSource.fromExt(d, a, b, count);
  }

  function texture(format: GPUTextureFormat) {
    return d.createTexture({
      size: [W, H],
      format,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.TEXTURE_BINDING,
    });
  }

  function clear(t: GPUTexture) {
    const enc = d.createCommandEncoder();
    enc
      .beginRenderPass({
        colorAttachments: [
          {
            view: t.createView(),
            loadOp: "clear",
            storeOp: "store",
            clearValue: [0.1, 0.2, 0.3, 1],
          },
        ],
      })
      .end();
    d.queue.submit([enc.finish()]);
  }

  // Pixels of `t` as floats in [0, 1] (8-bit) or as stored (half float).
  async function read(t: GPUTexture): Promise<Float32Array> {
    const half = t.format === "rgba16float";
    const px = half ? 8 : 4;
    const bytesPerRow = 256 * Math.ceil((W * px) / 256);
    const buf = d.createBuffer({
      size: bytesPerRow * H,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = d.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: t }, { buffer: buf, bytesPerRow }, [
      W,
      H,
    ]);
    d.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(W * H * 4);
    const range = buf.getMappedRange();
    const bytes = new Uint8Array(range);
    const halves = new Uint16Array(range);
    for (let y = 0; y < H; y++) {
      for (let i = 0; i < W * 4; i++) {
        out[y * W * 4 + i] = half
          ? THREE.DataUtils.fromHalfFloat(halves[(y * bytesPerRow) / 2 + i])
          : bytes[y * bytesPerRow + i] / 255;
      }
    }
    buf.unmap();
    buf.destroy();
    return out;
  }

  const camera = new THREE.PerspectiveCamera(60, W / H, 0.05, 100);
  camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
  camera.updateProjectionMatrix();
  camera.position.set(0.3, 0.2, 2.6);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();

  async function renderWith(
    options: WgpuSplatRendererOptions,
    color: GPUTexture,
    target?: { depth: GPUTexture },
    frames = 3,
    check?: (splats: WgpuSplatRenderer) => void,
  ) {
    const fakeTarget = target
      ? ({
          texture: { colorSpace: THREE.LinearSRGBColorSpace },
          depthTexture: {},
          samples: 0,
        } as unknown as THREE.RenderTarget)
      : undefined;
    const renderer = {
      backend: {
        isWebGPUBackend: true,
        device: d,
        context: {
          getCurrentTexture: () => color,
        } as unknown as GPUCanvasContext,
        get: (r: object) =>
          fakeTarget && r === fakeTarget.texture
            ? { texture: color }
            : fakeTarget && r === fakeTarget.depthTexture
              ? { texture: target?.depth }
              : undefined,
      },
    };
    const splats = new WgpuSplatRenderer(renderer as never, {
      alwaysGenerate: true,
      ...options,
    });
    splats.add(cloud(20000));
    // Later frames have the pair buffers sized from a readback.
    for (let frame = 0; frame < frames; frame++) {
      clear(color);
      splats.render(camera, fakeTarget);
      await d.queue.onSubmittedWorkDone();
      await new Promise((r) => setTimeout(r, 5));
    }
    const px = await read(color);
    check?.(splats);
    splats.dispose();
    return px;
  }

  function compare(a: Float32Array, b: Float32Array) {
    let sum = 0;
    let max = 0;
    for (let i = 0; i < a.length; i++) {
      if (i % 4 === 3) continue;
      const diff = Math.abs(a[i] - b[i]);
      sum += diff;
      max = Math.max(max, diff);
    }
    return { mean: (sum / (a.length * 3)) * 4, max };
  }

  it("matches the quad draw on an 8-bit canvas", async () => {
    const canvas = texture("bgra8unorm");
    const hw = await renderWith({ rasterizer: "hardware" }, canvas);
    const tiles = await renderWith({ rasterizer: "tiles" }, canvas);
    const { mean, max } = compare(hw, tiles);
    if (process.env.TILES_LOG) console.log("8-bit", mean * 255, max * 255);
    // The quads round to 8 bits at every blend; the tiles once.
    expect(mean).toBeLessThan(1 / 255);
    expect(max).toBeLessThan(8 / 255);
    // Not a blank image either way.
    const lit = hw.filter((v, i) => i % 4 === 0 && v > 0.35).length;
    expect(lit).toBeGreaterThan(W * H * 0.1);
    canvas.destroy();
  });

  it("projectOnce: the projector's quads draw the vertex projection's image", async () => {
    for (const format of ["bgra8unorm", "rgba16float"] as const) {
      const color = texture(format);
      const vertex = await renderWith({ projectOnce: false }, color);
      const once = await renderWith({ projectOnce: true }, color);
      const { mean, max } = compare(vertex, once);
      if (process.env.TILES_LOG)
        console.log("projectOnce", format, mean * 255, max * 255);
      // The same maths in a compute stage: bit-identical on Dawn/Metal;
      // elsewhere at most an ulp's rounding.
      expect(max).toBeLessThan(format === "bgra8unorm" ? 1.5 / 255 : 1e-3);
      expect(mean).toBeLessThan(0.01 / 255);
      color.destroy();
    }
  });

  it("auto: times both paths and draws the same image", async () => {
    const canvas = texture("bgra8unorm");
    const hw = await renderWith({ rasterizer: "hardware" }, canvas);
    const timed = d.features.has("timestamp-query");
    const auto = await renderWith(
      { rasterizer: "auto" },
      canvas,
      undefined,
      120,
      (splats) => {
        const { auto, rasterizer } = splats.stats;
        if (!timed) {
          // Without timestamps it keeps to the quad draw.
          expect(rasterizer).toBe("hardware");
          return;
        }
        expect(auto?.probes).toBeGreaterThan(0);
        expect(auto?.hardwareMs).toBeGreaterThan(0);
        expect(auto?.tilesMs).toBeGreaterThan(0);
        expect(rasterizer).toBe(auto?.probing ? rasterizer : auto?.path);
      },
    );
    const { mean, max } = compare(hw, auto);
    expect(mean).toBeLessThan(1 / 255);
    expect(max).toBeLessThan(8 / 255);
    canvas.destroy();
  }, 60000);

  it("tests depth and matches in a float target", async () => {
    const color = texture("rgba16float");
    const depth = d.createTexture({
      size: [W, H],
      format: "depth16unorm",
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST,
    });
    // Depth 0 (nothing passes) on the left half, 1 on the right.
    const depths = new Uint16Array(W * H);
    for (let y = 0; y < H; y++)
      for (let x = W / 2; x < W; x++) depths[y * W + x] = 0xffff;
    d.queue.writeTexture({ texture: depth }, depths, { bytesPerRow: W * 2 }, [
      W,
      H,
    ]);
    const hw = await renderWith({ rasterizer: "hardware" }, color, { depth });
    const tiles = await renderWith({ rasterizer: "tiles" }, color, { depth });
    const { mean, max } = compare(hw, tiles);
    if (process.env.TILES_LOG) console.log("float", mean * 255, max * 255);
    expect(mean).toBeLessThan(0.25 / 255);
    expect(max).toBeLessThan(4 / 255);
    // The left half stays the clear colour in both.
    for (const px of [hw, tiles]) {
      const i = (H / 2) * W * 4 + 8;
      expect(px[i]).toBeCloseTo(0.1, 2);
      expect(px[i + 2]).toBeCloseTo(0.3, 2);
    }
    color.destroy();
    depth.destroy();
  });

  it("matches through the sRGB layer (srgbBlend)", async () => {
    const color = texture("rgba16float");
    const depth = d.createTexture({
      size: [W, H],
      format: "depth16unorm",
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST,
    });
    d.queue.writeTexture(
      { texture: depth },
      new Uint16Array(W * H).fill(0xffff),
      { bytesPerRow: W * 2 },
      [W, H],
    );
    const options = { srgbBlend: true } as const;
    const hw = await renderWith({ ...options, rasterizer: "hardware" }, color, {
      depth,
    });
    const tiles = await renderWith({ ...options, rasterizer: "tiles" }, color, {
      depth,
    });
    const { mean, max } = compare(hw, tiles);
    if (process.env.TILES_LOG) console.log("layer", mean * 255, max * 255);
    expect(mean).toBeLessThan(1 / 255);
    expect(max).toBeLessThan(8 / 255);
    color.destroy();
    depth.destroy();
  });
});

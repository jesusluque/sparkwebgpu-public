// Surfels (2DGS): a splat with exactly one zero scale drawn by its exact
// ray-splat intersection with Huang et al.'s screen filter
// (draw/splat_shape.slang surfelRadius2), against a CPU reference that
// intersects each pixel's ray with the disc's plane; and the same image on
// the vertex path, the projector (projectOnce) and the tiles.

import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { decodeExtSplat, encodeExtSplat } from "../../src/utils";
import { GpuSplatSource } from "../../src/webgpu/GpuSplatSource";
import {
  WgpuSplatRenderer,
  type WgpuSplatRendererOptions,
} from "../../src/webgpu/WgpuSplatRenderer";
import { drawSplatShape } from "../../src/webgpu/generated/constants";
import generated from "../../src/webgpu/generated/draw/splat_draw";
import {
  createBindGroups,
  createReflectedRenderPipeline,
} from "../../src/webgpu/renderPipeline";
import { UniformWriter } from "../../src/webgpu/uniforms";
import { device, storage } from "./device";

const { DRAW_EXT, DRAW_2DGS } = drawSplatShape;
const SIZE = 96;
const FOV_Y = Math.PI / 3;
const NEAR = 0.1;
const FAR = 100;
const MAX_STD_DEV = Math.sqrt(8);
const MIN_ALPHA = 0.5 / 255;
// The f16 usd-athc writes for a surfel's zero scale (ln -65504: exp 0).
const ZERO_LN_SCALE = 0xfbff;

function perspective() {
  const f = 1 / Math.tan(FOV_Y / 2);
  return [
    [f, 0, 0, 0],
    [0, f, 0, 0],
    [0, 0, FAR / (NEAR - FAR), -1],
    [0, 0, (NEAR * FAR) / (NEAR - FAR), 0],
  ];
}

// An ext splat with its third scale exactly 0 (as usd-athc writes it).
function encodeSurfel(
  arrays: [Uint32Array, Uint32Array],
  i: number,
  center: number[],
  su: number,
  sv: number,
  q: number[],
  rgba: number[],
) {
  encodeExtSplat(
    arrays,
    i,
    center[0],
    center[1],
    center[2],
    su,
    sv,
    1,
    q[0],
    q[1],
    q[2],
    q[3],
    rgba[3],
    rgba[0],
    rgba[1],
    rgba[2],
  );
  const b = arrays[1];
  b[4 * i + 2] = (b[4 * i + 2] & 0xffff) | (ZERO_LN_SCALE << 16);
}

describe.skipIf(!device)("surfels", () => {
  const d = device as GPUDevice;
  const rp = createReflectedRenderPipeline(d, generated, {
    vertex: "splatVertex",
    fragment: "splatFragment",
    targets: [{ format: "rgba32float" }],
  });

  // One surfel at render-space origin, seen through renderToView (view
  // quaternion `view`, the centre at `viewCenter`), into an rgba32float
  // target without blending: the fragment's (rgb, alpha).
  async function renderOne(
    words: Uint32Array,
    view: THREE.Quaternion,
    viewCenter: THREE.Vector3,
    flags: number,
  ): Promise<Float32Array> {
    const p = perspective();
    const params = UniformWriter.for(generated).setAll({
      proj0: p[0],
      proj1: p[1],
      proj2: p[2],
      proj3: p[3],
      renderToViewQuat: [view.x, view.y, view.z, view.w],
      renderToViewPos: [...viewCenter.toArray(), 0],
      renderToViewBasis0: [1, 0, 0, 0],
      renderToViewBasis1: [0, 1, 0, 0],
      renderToViewBasis2: [0, 0, 1, 0],
      renderWidth: SIZE,
      renderHeight: SIZE,
      maxStdDev: MAX_STD_DEV,
      minPixelRadius: 0,
      maxPixelRadius: 512,
      minAlpha: MIN_ALPHA,
      blurAmount: 0.3,
      preBlurAmount: 0,
      focalDistance: 0,
      apertureAngle: 0,
      clipXY: 1.4,
      focalAdjustment: 1,
      falloff: 1,
      flags,
    });
    const uniform = d.createBuffer({
      size: params.data.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    d.queue.writeBuffer(uniform, 0, params.data);
    const [group] = createBindGroups(d, rp, {
      ordering: storage(new Uint32Array([0])),
      splats: storage(words),
      params: uniform,
    });
    const target = d.createTexture({
      size: [SIZE, SIZE],
      format: "rgba32float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const bytesPerRow = SIZE * 16;
    const readback = d.createBuffer({
      size: bytesPerRow * SIZE,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = d.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [
        {
          view: target.createView(),
          loadOp: "clear",
          storeOp: "store",
          clearValue: [0, 0, 0, 0],
        },
      ],
    });
    pass.setPipeline(rp.pipeline);
    pass.setBindGroup(0, group);
    pass.draw(4, 1);
    pass.end();
    enc.copyTextureToBuffer(
      { texture: target },
      { buffer: readback, bytesPerRow },
      [SIZE, SIZE],
    );
    d.queue.submit([enc.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const pixels = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    readback.destroy();
    target.destroy();
    uniform.destroy();
    return pixels;
  }

  // The CPU reference: each pixel centre's ray met with the disc's plane,
  // the hit in the disc's sigmas, Huang's filter min(|uv|^2, 2 |d|^2),
  // and the draw's cuts (maxStdDev, minAlpha). `z2` too, to leave out the
  // pixels on the cut's edge, where either side of a rounding is right.
  function reference(
    su: number,
    sv: number,
    alpha: number,
    view: THREE.Quaternion,
    c: THREE.Vector3,
  ) {
    const f = (0.5 * SIZE) / Math.tan(FOV_Y / 2);
    const eu = new THREE.Vector3(1, 0, 0).applyQuaternion(view);
    const ev = new THREE.Vector3(0, 1, 0).applyQuaternion(view);
    const n = new THREE.Vector3(0, 0, 1).applyQuaternion(view);
    const cx = (f * c.x) / -c.z;
    const cy = (f * c.y) / -c.z;
    const out = new Float64Array(SIZE * SIZE);
    const z2s = new Float64Array(SIZE * SIZE);
    const dir = new THREE.Vector3();
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const qx = x + 0.5 - SIZE / 2;
        const qy = SIZE / 2 - (y + 0.5);
        const dx = qx - cx;
        const dy = qy - cy;
        let z2 = 2 * (dx * dx + dy * dy);
        dir.set(qx / f, qy / f, -1);
        const t = c.dot(n) / dir.dot(n);
        if (Number.isFinite(t) && t > 0) {
          const hit = dir.clone().multiplyScalar(t).sub(c);
          const u = hit.dot(eu) / su;
          const v = hit.dot(ev) / sv;
          z2 = Math.min(z2, u * u + v * v);
        }
        const a = alpha * Math.exp(-0.5 * z2);
        out[y * SIZE + x] = z2 > MAX_STD_DEV ** 2 || a < MIN_ALPHA ? 0 : a;
        z2s[y * SIZE + x] = z2;
      }
    }
    return { alpha: out, z2: z2s };
  }

  // The view rotation that turns the disc's normal `angle` from the eye
  // (about an axis across the line of sight), spun about the normal so its
  // axes line up with nothing.
  function tilt(angle: number, center: THREE.Vector3) {
    const toEye = center.clone().negate().normalize();
    const across = new THREE.Vector3(0.3, 1, 0.1).cross(toEye).normalize();
    const normal = toEye.clone().applyAxisAngle(across, angle);
    const spin = new THREE.Quaternion().setFromAxisAngle(
      new THREE.Vector3(0, 0, 1),
      0.7,
    );
    return new THREE.Quaternion()
      .setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal)
      .multiply(spin);
  }

  for (const degrees of [0, 60, 85, 89.9]) {
    it(`matches the CPU ray-splat reference at ${degrees} degrees`, async () => {
      const a = new Uint32Array(4);
      const b = new Uint32Array(4);
      encodeSurfel(
        [a, b],
        0,
        [0, 0, 0],
        0.22,
        0.13,
        [0, 0, 0, 1],
        [1, 0.5, 0.25, 0.9],
      );
      const decoded = decodeExtSplat([a, b], 0);
      expect(decoded.scales.z).toBe(0);
      const words = new Uint32Array(8);
      words.set(a, 0);
      words.set(b, 4);
      const center = new THREE.Vector3(0.35, -0.2, -2.5);
      const view = tilt((degrees * Math.PI) / 180, center);
      const px = await renderOne(words, view, center, DRAW_EXT | DRAW_2DGS);
      const ref = reference(
        decoded.scales.x,
        decoded.scales.y,
        decoded.opacity,
        view,
        center,
      );
      let worst = 0;
      let covered = 0;
      for (let i = 0; i < SIZE * SIZE; i++) {
        const gpu = px[4 * i + 3];
        if (gpu > 0) covered += gpu;
        // The cut's edge (stdDev^2 = 8): either side of a rounding.
        if (Math.abs(ref.z2[i] - MAX_STD_DEV ** 2) < 1e-3) continue;
        worst = Math.max(worst, Math.abs(gpu - ref.alpha[i]));
        if (gpu > 0) expect(px[4 * i]).toBeCloseTo(1, 3);
      }
      if (process.env.SURFEL_LOG)
        console.log(degrees, "worst", worst, "coverage", covered);
      // Measured on Dawn/Metal: <= 1.3e-5 (89.9 degrees).
      expect(worst).toBeLessThan(1e-4);
      // Edge-on the filter keeps a line of a pixel or two on screen.
      expect(covered).toBeGreaterThan(degrees > 89 ? 2 : 10);
    });
  }

  it("draws a disc as a 3D splat without DRAW_2DGS (surfels: 'ewa')", async () => {
    const a = new Uint32Array(4);
    const b = new Uint32Array(4);
    encodeSurfel(
      [a, b],
      0,
      [0, 0, 0],
      0.22,
      0.13,
      [0, 0, 0, 1],
      [1, 1, 1, 0.9],
    );
    const words = new Uint32Array(8);
    words.set(a, 0);
    words.set(b, 4);
    const center = new THREE.Vector3(0, 0, -2.5);
    const view = tilt(1, center);
    const ewa = await renderOne(words, view, center, DRAW_EXT);
    const exact = await renderOne(words, view, center, DRAW_EXT | DRAW_2DGS);
    let diff = 0;
    let lit = 0;
    for (let i = 0; i < SIZE * SIZE; i++) {
      diff = Math.max(diff, Math.abs(ewa[4 * i + 3] - exact[4 * i + 3]));
      if (ewa[4 * i + 3] > 0) lit++;
    }
    expect(lit).toBeGreaterThan(100);
    expect(diff).toBeGreaterThan(1e-3);
  });

  it("draws an additive surfel: its light, covering nothing", async () => {
    const a = new Uint32Array(4);
    const b = new Uint32Array(4);
    encodeSurfel(
      [a, b],
      0,
      [0, 0, 0],
      0.22,
      0.13,
      [0, 0, 0, 1],
      [1, 0.5, 0.25, 0.9],
    );
    // generate's GEN_ADDITIVE mark: alpha word's high half 1.0.
    a[3] = (a[3] & 0xffff) | (0x3c00 << 16);
    const words = new Uint32Array(8);
    words.set(a, 0);
    words.set(b, 4);
    const center = new THREE.Vector3(0, 0, -2.5);
    const px = await renderOne(
      words,
      tilt(1, center),
      center,
      DRAW_EXT | DRAW_2DGS,
    );
    let lit = 0;
    for (let i = 0; i < SIZE * SIZE; i++) {
      if (px[4 * i] > 0) {
        lit++;
        expect(px[4 * i + 3]).toBe(0);
      }
    }
    expect(lit).toBeGreaterThan(50);
  });

  // A mixed cloud: surfels at every angle among 3D splats.
  const W = 160;
  const H = 96;
  function mixed(count: number, surfelsOnly = false) {
    const a = new Uint32Array(count * 4);
    const b = new Uint32Array(count * 4);
    let s = 11;
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
      const center = [2 * rnd() - 1, 2 * rnd() - 1, 2 * rnd() - 1];
      const rgba = [rnd(), rnd(), rnd(), 0.2 + 0.8 * rnd()];
      if (surfelsOnly || i % 2 === 0) {
        encodeSurfel([a, b], i, center, sc, sc * (0.2 + rnd()), q, rgba);
      } else {
        encodeExtSplat(
          [a, b],
          i,
          center[0],
          center[1],
          center[2],
          sc,
          sc * (0.2 + rnd()),
          sc * (0.2 + rnd()),
          q[0],
          q[1],
          q[2],
          q[3],
          rgba[3],
          rgba[0],
          rgba[1],
          rgba[2],
        );
      }
    }
    return GpuSplatSource.fromExt(d, a, b, count);
  }

  const camera = new THREE.PerspectiveCamera(60, W / H, 0.05, 100);
  camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
  camera.updateProjectionMatrix();
  camera.position.set(0.3, 0.2, 2.6);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();

  async function renderMixed(
    options: WgpuSplatRendererOptions,
    stats?: (s: WgpuSplatRenderer["stats"]) => void,
    count = 6000,
    surfelsOnly = false,
  ): Promise<Float32Array> {
    const color = d.createTexture({
      size: [W, H],
      format: "rgba16float",
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.TEXTURE_BINDING,
    });
    const fakeTarget = {
      texture: { colorSpace: THREE.LinearSRGBColorSpace },
      samples: 0,
    } as unknown as THREE.RenderTarget;
    const renderer = {
      backend: {
        isWebGPUBackend: true,
        device: d,
        context: {
          getCurrentTexture: () => color,
        } as unknown as GPUCanvasContext,
        get: (r: object) =>
          r === fakeTarget.texture ? { texture: color } : undefined,
      },
    };
    const splats = new WgpuSplatRenderer(renderer as never, {
      alwaysGenerate: true,
      ...options,
    });
    splats.add(mixed(count, surfelsOnly));
    for (let frame = 0; frame < 3; frame++) {
      const enc = d.createCommandEncoder();
      enc
        .beginRenderPass({
          colorAttachments: [
            {
              view: color.createView(),
              loadOp: "clear",
              storeOp: "store",
              clearValue: [0.1, 0.2, 0.3, 1],
            },
          ],
        })
        .end();
      d.queue.submit([enc.finish()]);
      splats.render(camera, fakeTarget);
      await d.queue.onSubmittedWorkDone();
      await new Promise((r) => setTimeout(r, 5));
    }
    stats?.(splats.stats);
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
    const out = new Float32Array(W * H * 4);
    for (let y = 0; y < H; y++)
      for (let i = 0; i < W * 4; i++)
        out[y * W * 4 + i] = THREE.DataUtils.fromHalfFloat(
          halves[(y * bytesPerRow) / 2 + i],
        );
    buf.unmap();
    buf.destroy();
    splats.dispose();
    color.destroy();
    return out;
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

  it("draws a mixed cloud the same through the projector (projectOnce)", async () => {
    const vertex = await renderMixed({ projectOnce: false }, (s) =>
      expect(s.projected).toBe(false),
    );
    const once = await renderMixed({ projectOnce: true }, (s) =>
      expect(s.projected).toBe(true),
    );
    const ewa = await renderMixed({ surfels: "ewa" });
    const forced = await renderMixed({ surfels: "ewa", enable2DGS: true });
    const { mean, max } = compare(vertex, once);
    if (process.env.SURFEL_LOG)
      console.log("projectOnce", mean * 255, max * 255);
    expect(max).toBeLessThan(1e-3);
    // enable2DGS forces the same path; "ewa" draws the discs otherwise.
    expect(compare(vertex, forced).max).toBe(0);
    expect(compare(vertex, ewa).mean).toBeGreaterThan(1e-4);
  });

  it("tiles draw surfels as the quads do", async () => {
    // A few surfels, mostly apart: each pixel blends once or twice, so the
    // quads' half-float blend rounds about as the tiles' single store.
    const hw = await renderMixed(
      { rasterizer: "hardware" },
      undefined,
      60,
      true,
    );
    const tiles = await renderMixed(
      { rasterizer: "tiles" },
      (s) => expect(s.rasterizer).toBe("tiles"),
      60,
      true,
    );
    const sparse = compare(hw, tiles);
    // A dense mixed cloud: the quads round to half floats at every blend.
    const dense = compare(
      await renderMixed({ rasterizer: "hardware" }),
      await renderMixed({ rasterizer: "tiles" }),
    );
    if (process.env.SURFEL_LOG)
      console.log(
        "tiles sparse",
        sparse.mean * 255,
        sparse.max * 255,
        "dense",
        dense.mean * 255,
        dense.max * 255,
      );
    expect(sparse.max).toBeLessThan(0.5 / 255);
    // As tiles.test.ts's float target.
    expect(dense.mean).toBeLessThan(0.25 / 255);
    expect(dense.max).toBeLessThan(4 / 255);
  });
});

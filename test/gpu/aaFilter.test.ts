// The box anti-alias filter (aaFilter "box", splat_shape.slang's
// DRAW_AA_BOX): each fragment's alpha is the mean of the splat's Gaussian
// over the pixel, so a splat's summed coverage is its mass at any size and
// position (no ripple as a splat under a pixel moves), and a large splat
// draws as the Gaussian path does; the same image on every path (quads,
// projectOnce, tiles) and for surfels.

import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { encodeExtSplat } from "../../src/utils";
import { GpuSplatSource } from "../../src/webgpu/GpuSplatSource";
import {
  WgpuSplatRenderer,
  type WgpuSplatRendererOptions,
} from "../../src/webgpu/WgpuSplatRenderer";
import { device } from "./device";

const W = 160;
const H = 96;
const Z = 4;

type Splat = {
  x: number;
  y: number;
  sigmaPx: number;
  opacity: number;
  disc?: boolean;
  // A needle: sigmaPx along its axis at `angle` (radians), minorPx across.
  minorPx?: number;
  angle?: number;
};

describe.skipIf(!device)("aaFilter box", () => {
  const d = device as GPUDevice;
  const camera = new THREE.PerspectiveCamera(60, W / H, 0.05, 100);
  camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();
  const focal = H / 2 / Math.tan(Math.PI / 6);

  // Splats at pixel offsets (x, y) from the screen centre, facing the eye;
  // their mass alpha x 2 pi sigma^2 x the projection's stretch.
  function cloud(list: Splat[]) {
    const a = new Uint32Array(list.length * 4);
    const b = new Uint32Array(list.length * 4);
    let mass = 0;
    list.forEach((s, i) => {
      const x = (s.x * Z) / focal;
      const y = (s.y * Z) / focal;
      const scale = (s.sigmaPx * Z) / focal;
      const minor = ((s.minorPx ?? s.sigmaPx) * Z) / focal;
      const half = (s.angle ?? 0) / 2;
      // A disc facing the eye projects without the 3D stretch.
      mass +=
        s.opacity *
        2 *
        Math.PI *
        s.sigmaPx *
        (s.minorPx ?? s.sigmaPx) *
        (s.disc ? 1 : Math.sqrt(1 + (x / Z) ** 2 + (y / Z) ** 2));
      encodeExtSplat(
        [a, b],
        i,
        x,
        y,
        -Z,
        scale,
        minor,
        s.disc ? 0 : minor,
        0,
        0,
        Math.sin(half),
        Math.cos(half),
        s.opacity,
        1,
        1,
        1,
      );
    });
    return { source: GpuSplatSource.fromExt(d, a, b, list.length), mass };
  }

  // The frame's alpha channel over a clear of 0.
  async function render(
    list: Splat[],
    options: WgpuSplatRendererOptions,
  ): Promise<{ alpha: Float32Array; drawn: number; mass: number }> {
    const color = d.createTexture({
      size: [W, H],
      format: "rgba16float",
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.STORAGE_BINDING,
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
    const { source, mass } = cloud(list);
    splats.add(source);
    for (let frame = 0; frame < 2; frame++) {
      const enc = d.createCommandEncoder();
      enc
        .beginRenderPass({
          colorAttachments: [
            {
              view: color.createView(),
              loadOp: "clear",
              storeOp: "store",
              clearValue: [0, 0, 0, 0],
            },
          ],
        })
        .end();
      d.queue.submit([enc.finish()]);
      splats.render(camera, fakeTarget);
      await d.queue.onSubmittedWorkDone();
    }
    const bytesPerRow = W * 8;
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
    const px = new Uint16Array(buf.getMappedRange());
    const alpha = new Float32Array(W * H);
    let drawn = 0;
    for (let i = 0; i < W * H; i++) {
      alpha[i] = THREE.DataUtils.fromHalfFloat(px[4 * i + 3]);
      drawn += alpha[i];
    }
    buf.unmap();
    buf.destroy();
    splats.dispose();
    color.destroy();
    return { alpha, drawn, mass };
  }

  // Isolated splats on a grid 12 px apart.
  function grid(sigmaPx: number, opacity: number, disc = false): Splat[] {
    const list: Splat[] = [];
    let s = 3;
    const rnd = () => {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      return s / 2 ** 32;
    };
    for (let y = -36; y <= 36; y += 12)
      for (let x = -66; x <= 66; x += 12)
        list.push({ x: x + rnd(), y: y + rnd(), sigmaPx, opacity, disc });
    return list;
  }

  for (const [sigmaPx, opacity] of [
    [0.15, 0.6],
    [0.4, 0.5],
    [1, 0.3],
  ]) {
    it(`keeps the mass of splats of ${sigmaPx} px`, async () => {
      const box = await render(grid(sigmaPx, opacity), { aaFilter: "box" });
      const disc = await render(grid(sigmaPx, opacity, true), {
        aaFilter: "box",
      });
      if (process.env.AA_LOG)
        console.log(sigmaPx, box.drawn / box.mass, disc.drawn / disc.mass);
      // The cut at maxStdDev (sqrt 8) leaves 1.8% of a Gaussian's mass out,
      // as on the Gaussian path; under a pixel the box holds it.
      for (const r of [box, disc]) {
        expect(r.drawn / r.mass).toBeLessThan(1.005);
        expect(r.drawn / r.mass).toBeGreaterThan(0.975);
      }
    });
  }

  it("has no ripple as a splat under a pixel moves", async () => {
    const sums = { box: [] as number[], gaussian: [] as number[] };
    for (let k = 0; k < 8; k++) {
      const list = grid(0.1, 0.8).map((s) => ({ ...s, x: s.x + k / 8 }));
      sums.box.push((await render(list, { aaFilter: "box" })).drawn);
      sums.gaussian.push((await render(list, { blurAmount: 1 / 12 })).drawn);
    }
    const spread = (v: number[]) =>
      (Math.max(...v) - Math.min(...v)) /
      (v.reduce((a, b) => a + b) / v.length);
    if (process.env.AA_LOG)
      console.log(
        "ripple box",
        spread(sums.box),
        "gaussian 1/12",
        spread(sums.gaussian),
      );
    expect(spread(sums.box)).toBeLessThan(0.01);
    expect(spread(sums.gaussian)).toBeGreaterThan(0.05);
  });

  // Needles under a pixel or two (a wire's splats) at an angle: the box
  // turns towards their axes (boxFrame); their coverage still sums to
  // their mass as they move.
  for (const [major, angle] of [
    [0.4, 0.5],
    [0.8, 0.5],
    [1.5, 0.7],
  ]) {
    it(`has little ripple for needles of ${major} px`, async () => {
      const sums: number[] = [];
      const gs: number[] = [];
      for (let k = 0; k < 8; k++) {
        const list = grid(major, 0.8).map((s) => ({
          ...s,
          x: s.x + k / 8,
          y: s.y + k / 16,
          minorPx: 0.05,
          angle,
        }));
        const r = await render(list, { aaFilter: "box" });
        sums.push(r.drawn / r.mass);
        gs.push((await render(list, {})).drawn / r.mass);
      }
      const spread = (v: number[]) => Math.max(...v) - Math.min(...v);
      if (process.env.AA_LOG)
        console.log(
          "needle",
          major,
          "box",
          Math.min(...sums),
          spread(sums),
          "gaussian",
          Math.min(...gs),
          spread(gs),
        );
      expect(spread(sums)).toBeLessThan(0.03);
    });
  }

  it("draws a large splat as the Gaussian path", async () => {
    const list = [{ x: 0.3, y: 0.2, sigmaPx: 8, opacity: 0.7 }];
    const box = await render(list, { aaFilter: "box" });
    const g = await render(list, { blurAmount: 1 / 12 });
    let worst = 0;
    for (let i = 0; i < W * H; i++)
      worst = Math.max(worst, Math.abs(box.alpha[i] - g.alpha[i]));
    expect(worst).toBeLessThan(2e-3);
  });

  it("draws the same on quads, projectOnce and tiles", async () => {
    const list = [
      ...grid(0.3, 0.6),
      ...grid(2, 0.4).map((s) => ({ ...s, x: s.x + 5, y: s.y + 4 })),
    ];
    const quads = await render(list, { aaFilter: "box" });
    for (const o of [{ projectOnce: true }, { rasterizer: "tiles" as const }]) {
      const other = await render(list, { aaFilter: "box", ...o });
      let worst = 0;
      for (let i = 0; i < W * H; i++)
        worst = Math.max(worst, Math.abs(quads.alpha[i] - other.alpha[i]));
      if (process.env.AA_LOG) {
        const gq = await render(list, {});
        const go = await render(list, o);
        let gw = 0;
        for (let i = 0; i < W * H; i++)
          gw = Math.max(gw, Math.abs(gq.alpha[i] - go.alpha[i]));
        console.log(JSON.stringify(o), "box", worst, "gaussian", gw);
      }
      // Tiles blend in float, quads in the half-float target: ~7e-3 apart
      // on the Gaussian path too.
      expect(worst, JSON.stringify(o)).toBeLessThan(1e-2);
    }
  });
});

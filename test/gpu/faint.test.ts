// Faint splats under a pixel (splat_shape.slang's faint path, DRAW_FAINT):
// the anti-alias blur takes their alpha under minAlpha; with
// faintSplats "keep" the frame still holds their mass (alpha x area), with
// "drop" they vanish (athc_measure faint models the same on the CPU).

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
const COUNT = 3000;
const Z = 4;

describe.skipIf(!device)("faint splats", () => {
  const d = device as GPUDevice;
  const camera = new THREE.PerspectiveCamera(60, W / H, 0.05, 100);
  camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();
  const focal = H / 2 / Math.tan(Math.PI / 6);

  // Isotropic splats of `sigmaPx` pixels (on the axis), apart, on a plane
  // facing the eye; and their summed mass on screen, alpha x 2 pi sigma^2
  // with the projection's stretch off the axis (det of the 2D covariance:
  // sigma^4 (1 + tx^2 + ty^2), t = x / z).
  function cloud(sigmaPx: number, opacity: number) {
    const a = new Uint32Array(COUNT * 4);
    const b = new Uint32Array(COUNT * 4);
    let s = 5;
    const rnd = () => {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      return s / 2 ** 32;
    };
    const scale = (sigmaPx * Z) / focal;
    const halfW = ((W / 2 - 4) * Z) / focal;
    const halfH = ((H / 2 - 4) * Z) / focal;
    let mass = 0;
    for (let i = 0; i < COUNT; i++) {
      const x = (2 * rnd() - 1) * halfW;
      const y = (2 * rnd() - 1) * halfH;
      mass +=
        opacity *
        2 *
        Math.PI *
        sigmaPx *
        sigmaPx *
        Math.sqrt(1 + (x / Z) ** 2 + (y / Z) ** 2);
      encodeExtSplat(
        [a, b],
        i,
        x,
        y,
        -Z,
        scale,
        scale,
        scale,
        0,
        0,
        0,
        1,
        opacity,
        1,
        1,
        1,
      );
    }
    return { source: GpuSplatSource.fromExt(d, a, b, COUNT), mass };
  }

  // The frame's summed coverage (alpha channel over a clear of 0).
  async function coverage(
    sigmaPx: number,
    opacity: number,
    options: WgpuSplatRendererOptions,
  ): Promise<{ drawn: number; mass: number }> {
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
    const { source, mass } = cloud(sigmaPx, opacity);
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
    let sum = 0;
    for (let i = 3; i < px.length; i += 4)
      sum += THREE.DataUtils.fromHalfFloat(px[i]);
    buf.unmap();
    buf.destroy();
    splats.dispose();
    color.destroy();
    return { drawn: sum, mass };
  }

  // Splats of 0.05 px at alpha 0.25 take a blurred alpha just over minAlpha
  // (drawn, but the fragments' cut keeps ~5-18% of their mass, more off the
  // axis where the projection stretches them); of 0.1 px at
  // 0.05 one under it (dropped). Splats of 1 px are not faint: the same
  // either way, and what the harness gets for a plain splat (overlaps).
  for (const [sigmaPx, opacity, dropped] of [
    [0.05, 0.25, 0.3],
    [0.1, 0.05, 0.1],
    [1, 0.25, 1],
  ]) {
    it(`keeps the mass of splats of ${sigmaPx} px at alpha ${opacity}`, async () => {
      const keep = await coverage(sigmaPx, opacity, {});
      const drop = await coverage(sigmaPx, opacity, { faintSplats: "drop" });
      const kept = keep.drawn / keep.mass;
      const left = drop.drawn / drop.mass;
      if (process.env.FAINT_LOG)
        console.log(sigmaPx, opacity, "keep", kept, "drop", left);
      if (dropped < 1) {
        // Stochastic past the blur's floor: ~1500 of 3000 drawn, ~3% noise.
        expect(Math.abs(kept - 1)).toBeLessThan(0.08);
        expect(left).toBeLessThan(dropped);
      } else {
        expect(kept).toBe(left);
      }
    });
  }
});

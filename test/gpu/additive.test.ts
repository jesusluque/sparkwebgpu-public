// Additive splats (SplatMesh.additive / WgpuSplatMesh.additive, generate's
// GEN_ADDITIVE): their light is added and nothing behind them is covered,
// in the quad draw and in the tile rasterizer. A dark splat behind a lit one
// shows through it unchanged; the lit one's own light is the same as drawn
// normally (c x alpha), only the cover is gone.

import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { encodeExtSplat } from "../../src/utils";
import {
  GpuSplatSource,
  WgpuSplatRenderer,
  type WgpuSplatRendererOptions,
} from "../../src/webgpu/WgpuSplatRenderer";
import { device } from "./device";

const W = 64;
const H = 64;
const B = [0.1, 0.2, 0.3];

describe.skipIf(!device)("additive splats", () => {
  const d = device as GPUDevice;

  /** One round splat at z, radius r, of colour rgb and opacity a. */
  function splat(z: number, r: number, rgb: number[], a: number) {
    const ea = new Uint32Array(4);
    const eb = new Uint32Array(4);
    encodeExtSplat(
      [ea, eb],
      0,
      0,
      0,
      z,
      r,
      r,
      r * 0.1,
      0,
      0,
      0,
      1,
      a,
      rgb[0],
      rgb[1],
      rgb[2],
    );
    return GpuSplatSource.fromExt(d, ea, eb, 1);
  }

  const color = d?.createTexture({
    size: [W, H],
    format: "rgba16float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });

  async function centre(): Promise<number[]> {
    const bytesPerRow = 256 * Math.ceil((W * 8) / 256);
    const buf = d.createBuffer({
      size: bytesPerRow * H,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = d.createCommandEncoder();
    enc.copyTextureToBuffer(
      { texture: color as GPUTexture },
      { buffer: buf, bytesPerRow },
      [W, H],
    );
    d.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const halves = new Uint16Array(buf.getMappedRange());
    const at = (H / 2) * (bytesPerRow / 2) + (W / 2) * 4;
    const out = [0, 1, 2].map((c) =>
      THREE.DataUtils.fromHalfFloat(halves[at + c]),
    );
    buf.unmap();
    buf.destroy();
    return out;
  }

  const camera = new THREE.PerspectiveCamera(50, W / H, 0.05, 100);
  camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
  camera.position.set(0, 0, 3);
  camera.lookAt(0, 0, 0);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();

  /** The centre pixel of the scene: `back` (normal), `front` (additive or not). */
  async function render(
    options: WgpuSplatRendererOptions,
    parts: { back?: boolean; front?: boolean; additive?: boolean },
  ) {
    const renderer = {
      backend: {
        isWebGPUBackend: true,
        device: d,
        context: {
          getCurrentTexture: () => color,
        } as unknown as GPUCanvasContext,
        get: () => undefined,
      },
    };
    const splats = new WgpuSplatRenderer(renderer as never, {
      alwaysGenerate: true,
      hdr: true,
      ...options,
    });
    if (parts.back) splats.add(splat(-0.5, 0.6, [0.05, 0.4, 0.05], 0.9));
    if (parts.front) {
      const mesh = splats.add(splat(0.5, 0.3, [0.9, 0.5, 0.1], 0.8));
      mesh.additive = parts.additive ?? false;
    }
    for (let f = 0; f < 3; f++) {
      const enc = d.createCommandEncoder();
      enc
        .beginRenderPass({
          colorAttachments: [
            {
              view: (color as GPUTexture).createView(),
              loadOp: "clear",
              storeOp: "store",
              clearValue: [...B, 1],
            },
          ],
        })
        .end();
      d.queue.submit([enc.finish()]);
      splats.render(camera);
      await d.queue.onSubmittedWorkDone();
    }
    const px = await centre();
    splats.dispose();
    return px;
  }

  for (const rasterizer of ["hardware", "tiles"] as const) {
    it(`adds light and covers nothing (${rasterizer})`, async () => {
      const o: WgpuSplatRendererOptions = { rasterizer };
      const normal = await render(o, { front: true });
      const added = await render(o, { front: true, additive: true });
      // Drawn normally: B (1 - a) + c a; additive: B + c a.
      const c = [0.9, 0.5, 0.1];
      const a = (normal[0] - B[0]) / (c[0] - B[0]);
      expect(a).toBeGreaterThan(0.3);
      for (let k = 0; k < 3; k++)
        expect(added[k]).toBeCloseTo(B[k] + c[k] * a, 2);
      // A splat behind shows through unchanged, plus the light.
      const back = await render(o, { back: true });
      const both = await render(o, { back: true, front: true, additive: true });
      for (let k = 0; k < 3; k++)
        expect(both[k]).toBeCloseTo(back[k] + c[k] * a, 2);
      // Not additive, the front one covers it.
      const covered = await render(o, { back: true, front: true });
      expect(covered[1]).toBeLessThan(both[1] - 0.05);
    });
  }
});

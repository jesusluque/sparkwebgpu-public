// Additive splats (SplatMesh.additive / WgpuSplatMesh.additive, generate's
// GEN_ADDITIVE), athenea's light catcher: drawn as a layer of their own
// (the quad draw; the tile rasterizer hands such a frame to it), blended
// among themselves as any splats, the layer's light added to the frame
// with nothing covered, and the other splats drawn over it.

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

  const FRONT = { z: 0.5, r: 0.3, c: [0.9, 0.5, 0.1], a: 0.8 };
  const BACK = { z: -0.5, r: 0.6, c: [0.05, 0.4, 0.05], a: 0.9 };
  type Part = typeof FRONT & { additive?: boolean };

  /** The centre pixel of the scene drawn from `parts`. */
  async function render(options: WgpuSplatRendererOptions, parts: Part[]) {
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
    for (const p of parts) {
      const mesh = splats.add(splat(p.z, p.r, p.c, p.a));
      mesh.additive = p.additive ?? false;
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

  // The tile rasterizer hands a frame with additive splats to the quads.
  for (const rasterizer of ["hardware", "tiles"] as const) {
    it(`adds its light, blends with its own kind, is covered by the rest (${rasterizer})`, async () => {
      const o: WgpuSplatRendererOptions = { rasterizer };
      const add = (p: Part) => ({ ...p, additive: true });
      // Each one's coverage at the centre, from its normal draw over B.
      const coverage = async (p: Part) => {
        const px = await render(o, [p]);
        return (px[0] - B[0]) / (p.c[0] - B[0]);
      };
      const a1 = await coverage(FRONT);
      const a2 = await coverage(BACK);
      expect(a1).toBeGreaterThan(0.3);
      expect(a2).toBeGreaterThan(0.3);
      // Alone: B + c a (nothing covered).
      const alone = await render(o, [add(FRONT)]);
      for (let k = 0; k < 3; k++)
        expect(alone[k]).toBeCloseTo(B[k] + FRONT.c[k] * a1, 2);
      // Two additive ones blend among themselves (athenea's light catcher:
      // the front covers the back), then add: B + c1 a1 + (1 - a1) c2 a2.
      const pair = await render(o, [add(FRONT), add(BACK)]);
      for (let k = 0; k < 3; k++)
        expect(pair[k]).toBeCloseTo(
          B[k] + FRONT.c[k] * a1 + (1 - a1) * BACK.c[k] * a2,
          2,
        );
      // A normal splat covers the layer: front over (B + c2 a2).
      const covered = await render(o, [FRONT, add(BACK)]);
      for (let k = 0; k < 3; k++)
        expect(covered[k]).toBeCloseTo(
          (B[k] + BACK.c[k] * a2) * (1 - a1) + FRONT.c[k] * a1,
          2,
        );
    });
  }
});

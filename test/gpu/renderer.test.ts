// WgpuSplatRenderer end to end on Dawn: a stand-in for WebGPURenderer gives
// it the device and an offscreen "canvas" texture.

import * as THREE from "three";
import { describe, expect, it, vi } from "vitest";
import { encodeExtSplat } from "../../src/utils";
import {
  GpuSplatSource,
  WgpuSplatRenderer,
} from "../../src/webgpu/WgpuSplatRenderer";
import { device, readBack } from "./device";

const W = 96;
const H = 64;

describe.skipIf(!device)("WgpuSplatRenderer", () => {
  const d = device as GPUDevice;
  const canvas = d.createTexture({
    size: [W, H],
    format: "bgra8unorm",
    usage:
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.COPY_DST,
  });
  const fakeRenderer = {
    backend: {
      isWebGPUBackend: true,
      device: d,
      context: {
        getCurrentTexture: () => canvas,
      } as unknown as GPUCanvasContext,
      get: () => undefined,
    },
  };

  async function readCanvas(texture: GPUTexture = canvas): Promise<Uint8Array> {
    const bytesPerRow = 256 * Math.ceil((W * 4) / 256);
    const buf = d.createBuffer({
      size: bytesPerRow * H,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = d.createCommandEncoder();
    enc.copyTextureToBuffer({ texture }, { buffer: buf, bytesPerRow }, [W, H]);
    d.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buf.getMappedRange());
    const out = new Uint8Array(W * H * 4);
    for (let y = 0; y < H; y++)
      out.set(
        src.subarray(y * bytesPerRow, y * bytesPerRow + W * 4),
        y * W * 4,
      );
    buf.unmap();
    buf.destroy();
    return out;
  }

  function clearCanvas() {
    const enc = d.createCommandEncoder();
    enc
      .beginRenderPass({
        colorAttachments: [
          {
            view: canvas.createView(),
            loadOp: "clear",
            storeOp: "store",
            clearValue: [0, 0, 0, 1],
          },
        ],
      })
      .end();
    d.queue.submit([enc.finish()]);
  }

  it.each([false, true])(
    "draws a splat sphere in front of a three.js camera (flat: %s)",
    async (flat) => {
      const count = 5000;
      const a = new Uint32Array(count * 4);
      const b = new Uint32Array(count * 4);
      for (let i = 0; i < count; i++) {
        // Fibonacci sphere.
        const u = 1 - (2 * (i + 0.5)) / count;
        const t = i * Math.PI * (3 - Math.sqrt(5));
        const r = Math.sqrt(1 - u * u);
        const [x, y, z] = [r * Math.cos(t), u, r * Math.sin(t)];
        // Flat splats tangent to the sphere, as in examples/webgpu.
        const q = flat
          ? new THREE.Quaternion().setFromUnitVectors(
              new THREE.Vector3(0, 0, 1),
              new THREE.Vector3(x, y, z),
            )
          : new THREE.Quaternion();
        const sz = flat ? 0.01 : 0.05;
        encodeExtSplat(
          [a, b],
          i,
          x,
          y,
          z,
          0.05,
          0.05,
          sz,
          q.x,
          q.y,
          q.z,
          q.w,
          0.9,
          1,
          0.2,
          0.2,
        );
      }
      const splats = new WgpuSplatRenderer(fakeRenderer as never, {
        depthTest: false,
      });
      splats.add(GpuSplatSource.fromExt(d, a, b, count));

      const camera = new THREE.PerspectiveCamera(60, W / H, 0.05, 100);
      camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
      camera.updateProjectionMatrix();
      camera.position.set(0, 0, 4);
      camera.lookAt(0, 0, 0);
      camera.updateMatrixWorld();

      // Frame 1 generates and starts the sort readback; frame 2 draws.
      for (let frame = 0; frame < 3; frame++) {
        clearCanvas();
        splats.render(camera);
        await d.queue.onSubmittedWorkDone();
        await new Promise((r) => setTimeout(r, 10));
      }
      const px = await readCanvas();
      let lit = 0;
      for (let i = 0; i < W * H; i++) if (px[i * 4 + 2] > 40) lit++; // red (bgra)
      const centre = ((H / 2) * W + W / 2) * 4;
      if (process.env.ASCII) {
        // biome-ignore lint/suspicious/noExplicitAny: debugging internals
        const sp = splats as any;
        const metric = new Float32Array(await readBack(sp.metric));
        const ordering = new Uint32Array(await readBack(sp.ordering));
        const acc = new Uint32Array(await readBack(sp.accumulator));
        const finite = metric.slice(0, count).filter(Number.isFinite).length;
        console.log(
          "drawCount",
          sp.drawCount,
          "finite",
          finite,
          "metric0..4",
          JSON.stringify(Array.from(metric.slice(0, 5))),
        );
        console.log(
          "ordering0..9",
          JSON.stringify(Array.from(ordering.slice(0, 10))),
          "max",
          Math.max(...ordering.slice(0, sp.drawCount)),
        );
        const zeroSplats = Array.from(
          { length: count },
          (_, i) => acc[8 * i] | acc[8 * i + 1] | acc[8 * i + 2],
        ).filter((v) => v === 0).length;
        console.log("accumulator zero centers", zeroSplats);
      }
      if (process.env.ASCII) {
        for (let y = 0; y < H; y += 2) {
          let row = "";
          for (let x = 0; x < W; x++)
            row += px[(y * W + x) * 4 + 2] > 40 ? "#" : ".";
          console.log(row);
        }
      }
      expect(lit).toBeGreaterThan(W * H * 0.08);
      expect(px[centre + 2]).toBeGreaterThan(100);
      expect(px[0 + 2]).toBeLessThan(10); // corner stays clear
      // Three frames from the same camera: one generate, then redraws.
      expect(splats.stats.generated).toBe(1);
      const content = splats.contentVersion;
      camera.position.x += 0.1;
      camera.updateMatrixWorld();
      splats.render(camera);
      expect(splats.stats.generated).toBe(2);
      // A camera move is not a content change (ProgressiveAccumulator).
      expect(splats.contentVersion).toBe(content);
      // A sub-pixel jitter with the unjittered projection named: no generate.
      const base = camera.projectionMatrix.clone();
      camera.userData.unjitteredProjectionMatrix = base;
      camera.projectionMatrix.elements[8] += 0.01;
      splats.render(camera);
      expect(splats.stats.generated).toBe(2);
      camera.projectionMatrix.copy(base);
      camera.userData.unjitteredProjectionMatrix = undefined;
      splats.markDirty();
      splats.render(camera);
      expect(splats.stats.generated).toBe(3);
      expect(splats.contentVersion).toBe(content + 1);

      // minSortIntervalMs: a moved camera regenerates but draws the last
      // order; the next still frame sorts.
      // biome-ignore lint/suspicious/noExplicitAny: counting a private call
      const sorts = vi.spyOn(splats as any, "encodeSort");
      let deferred = 0;
      splats.onSortDeferred = () => deferred++;
      splats.options.minSortIntervalMs = 1e9;
      camera.position.x += 0.1;
      camera.updateMatrixWorld();
      splats.render(camera);
      expect([splats.stats.generated, sorts.mock.calls.length]).toEqual([4, 0]);
      expect([splats.sortStale, deferred]).toEqual([true, 1]);
      splats.render(camera);
      expect([splats.stats.generated, sorts.mock.calls.length]).toEqual([4, 1]);
      expect(splats.sortStale).toBe(false);
      splats.dispose();
    },
  );

  // The CPU sort draws a frame behind, in the order of an earlier camera's
  // metric: generate must not cull for that camera, or splats coming into
  // view are missing from the order.
  it("draws splats coming into view with the CPU sort", async () => {
    const a = new Uint32Array(4);
    const b = new Uint32Array(4);
    encodeExtSplat([a, b], 0, 0, 0, 0, 0.5, 0.5, 0.5, 0, 0, 0, 1, 1, 1, 1, 1);
    const splats = new WgpuSplatRenderer(fakeRenderer as never, {
      depthTest: false,
      sort: "cpu",
    });
    splats.add(GpuSplatSource.fromExt(d, a, b, 1));
    const camera = new THREE.PerspectiveCamera(60, W / H, 0.05, 100);
    camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
    camera.updateProjectionMatrix();
    camera.position.set(0, 0, 4);
    const look = async (x: number) => {
      camera.lookAt(x, 0, 0);
      camera.updateMatrixWorld();
      clearCanvas();
      splats.render(camera);
      await d.queue.onSubmittedWorkDone();
      await new Promise((r) => setTimeout(r, 10));
    };
    // Looking away until the sort settles, then at the splat.
    for (let frame = 0; frame < 3; frame++) await look(100);
    await look(0);
    const px = await readCanvas();
    const centre = ((H / 2) * W + W / 2) * 4;
    expect(px[centre + 2]).toBeGreaterThan(100);
    splats.dispose();
  });

  // A multisampled RenderTarget passed to render(): three resolves its
  // colour into target.texture's single-sample texture, which the splats
  // draw over, tested against a single-sample copy of its multisampled depth.
  it("draws into a multisampled render target", async () => {
    const color = d.createTexture({
      size: [W, H],
      format: "rgba8unorm",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const depth = d.createTexture({
      size: [W, H],
      format: "depth24plus",
      sampleCount: 4,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    const target = new THREE.RenderTarget(W, H, {
      samples: 4,
      depthTexture: new THREE.DepthTexture(W, H),
    });
    const textures = new Map<object, object>([
      [target.texture, { texture: color }],
      [target.depthTexture as THREE.DepthTexture, { texture: depth }],
    ]);
    const renderer = {
      backend: { ...fakeRenderer.backend, get: (o: object) => textures.get(o) },
    };
    const a = new Uint32Array(4);
    const b = new Uint32Array(4);
    encodeExtSplat([a, b], 0, 0, 0, 0, 0.5, 0.5, 0.5, 0, 0, 0, 1, 1, 1, 1, 1);
    const splats = new WgpuSplatRenderer(renderer as never);
    splats.add(GpuSplatSource.fromExt(d, a, b, 1));
    const camera = new THREE.PerspectiveCamera(60, W / H, 0.05, 100);
    camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
    camera.updateProjectionMatrix();
    camera.position.set(0, 0, 4);
    camera.updateMatrixWorld();
    // The scene's depth: far (1), or in front of the splat (0).
    const centre = async (sceneDepth: number) => {
      const enc = d.createCommandEncoder();
      enc
        .beginRenderPass({
          colorAttachments: [
            {
              view: color.createView(),
              loadOp: "clear",
              storeOp: "store",
              clearValue: [0, 0, 0, 1],
            },
          ],
        })
        .end();
      enc
        .beginRenderPass({
          colorAttachments: [],
          depthStencilAttachment: {
            view: depth.createView(),
            depthLoadOp: "clear",
            depthClearValue: sceneDepth,
            depthStoreOp: "store",
          },
        })
        .end();
      d.queue.submit([enc.finish()]);
      splats.markDirty();
      splats.render(camera, target);
      const px = await readCanvas(color);
      return px[((H / 2) * W + W / 2) * 4];
    };
    expect(await centre(1)).toBeGreaterThan(100);
    expect(await centre(0)).toBe(0);
    splats.dispose();
    color.destroy();
    depth.destroy();
  });
});

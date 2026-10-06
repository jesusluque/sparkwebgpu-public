// Plugins through WgpuSplatRenderer on Dawn: the preset kernel variants that
// tools/slang-build compiles from slang/plugins/presets.json, picked by a
// PluginHost per mesh (generate) and per frame (draw, tiles).
//
// The default kernels' ISplatColour and IBlendTerm are the identity on top of
// Spark's maths, so identity plugin settings must render bit for bit what no
// host renders; and a tint must render what the same factor as a recolor does.

import * as THREE from "three";
import { describe, expect, it } from "vitest";
import * as d from "../../src/dyno";
import { encodeExtSplat } from "../../src/utils";
import {
  GpuSplatSource,
  WgpuSplatRenderer,
  type WgpuSplatRendererOptions,
} from "../../src/webgpu/WgpuSplatRenderer";
import {
  PluginHost,
  dynoPlugin,
  fadePlugin,
  tintPlugin,
} from "../../src/webgpu/plugins";
import { device } from "./device";

const W = 128;
const H = 96;

describe.skipIf(!device)("plugins", () => {
  const dev = device as GPUDevice;

  function cloud(count: number, seed: number, dx: number) {
    const a = new Uint32Array(count * 4);
    const b = new Uint32Array(count * 4);
    let s = seed;
    const rnd = () => {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      return s / 2 ** 32;
    };
    for (let i = 0; i < count; i++) {
      const q = new THREE.Quaternion(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5, 0.5)
        .normalize()
        .toArray() as number[];
      const sc = 0.02 + 0.08 * rnd();
      encodeExtSplat(
        [a, b],
        i,
        rnd() - 0.5 + dx,
        2 * rnd() - 1,
        2 * rnd() - 1,
        sc,
        sc * (0.2 + rnd()),
        sc * (0.2 + rnd()),
        q[0],
        q[1],
        q[2],
        q[3],
        0.3 + 0.7 * rnd(),
        rnd(),
        rnd(),
        rnd(),
      );
    }
    return GpuSplatSource.fromExt(dev, a, b, count);
  }

  const canvas = dev.createTexture({
    size: [W, H],
    format: "rgba8unorm",
    usage:
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.TEXTURE_BINDING,
  });
  const fakeRenderer = {
    backend: {
      isWebGPUBackend: true,
      device: dev,
      context: {
        getCurrentTexture: () => canvas,
      } as unknown as GPUCanvasContext,
      get: () => undefined,
    },
  };

  const camera = new THREE.PerspectiveCamera(60, W / H, 0.05, 100);
  camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
  camera.updateProjectionMatrix();
  camera.position.set(0.2, 0.3, 3);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();

  async function read(): Promise<Uint8Array> {
    const bytesPerRow = 256 * Math.ceil((W * 4) / 256);
    const buf = dev.createBuffer({
      size: bytesPerRow * H,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = dev.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: canvas }, { buffer: buf, bytesPerRow }, [
      W,
      H,
    ]);
    dev.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buf.getMappedRange());
    const out = new Uint8Array(W * H * 4);
    for (let y = 0; y < H; y++) {
      out.set(
        src.subarray(y * bytesPerRow, y * bytesPerRow + W * 4),
        y * W * 4,
      );
    }
    buf.unmap();
    buf.destroy();
    return out;
  }

  function clear() {
    const enc = dev.createCommandEncoder();
    enc
      .beginRenderPass({
        colorAttachments: [
          {
            view: canvas.createView(),
            loadOp: "clear",
            storeOp: "store",
            clearValue: [0.1, 0.2, 0.3, 1],
          },
        ],
      })
      .end();
    dev.queue.submit([enc.finish()]);
  }

  /**
   * Two clouds (left, right) rendered after `setup` returns; the tile path
   * needs a frame or two to size its pair buffers.
   */
  async function render(
    options: WgpuSplatRendererOptions,
    setup?: (
      splats: WgpuSplatRenderer,
      meshes: ReturnType<WgpuSplatRenderer["add"]>[],
    ) => Promise<void> | void,
  ) {
    const splats = new WgpuSplatRenderer(fakeRenderer as never, {
      depthTest: false,
      alwaysGenerate: true,
      ...options,
    });
    const meshes = [
      splats.add(cloud(6000, 3, -0.6)),
      splats.add(cloud(6000, 11, 0.6)),
    ];
    await setup?.(splats, meshes);
    for (let frame = 0; frame < 3; frame++) {
      clear();
      splats.render(camera);
      await dev.queue.onSubmittedWorkDone();
      await new Promise((r) => setTimeout(r, 5));
    }
    const px = await read();
    const stats = { ...splats.stats };
    splats.plugins?.detach();
    splats.dispose();
    return { px, stats };
  }

  function same(a: Uint8Array, b: Uint8Array) {
    let differ = 0;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) differ++;
    return differ;
  }

  for (const rasterizer of ["hardware", "tiles"] as const) {
    it(`renders identity plugin settings as without plugins (${rasterizer})`, async () => {
      const plain = await render({ rasterizer });
      const empty = await render({ rasterizer }, (splats) => {
        new PluginHost({ capabilities: splats.capabilities }).attach(splats);
      });
      const identity = await render({ rasterizer }, async (splats) => {
        const host = new PluginHost({ capabilities: splats.capabilities });
        host.register(tintPlugin([1, 1, 1, 1]));
        host.register(fadePlugin({ fog: 0, opacity: 1 }));
        host.attach(splats);
        await host.ready();
        expect(host.generateKernel(splats.meshes[0])?.key).toBe("example.tint");
        expect(host.pixelKernels(dev)?.key).toBe("example.fade");
      });
      expect(plain.stats.rasterizer).toBe(rasterizer);
      expect(identity.stats.rasterizer).toBe(rasterizer);
      // Something drawn, and exactly the same.
      expect(same(plain.px, new Uint8Array(plain.px.length))).toBeGreaterThan(
        W * H,
      );
      expect(same(plain.px, empty.px)).toBe(0);
      expect(same(plain.px, identity.px)).toBe(0);
    });
  }

  it("tints per asset, as a recolor would", async () => {
    const tinted = await render({}, async (splats, meshes) => {
      const host = new PluginHost({ capabilities: splats.capabilities });
      const tint = tintPlugin([0.5, 0.75, 1, 1]);
      host.register(tint).attach(splats);
      host.disable(tint.id, meshes[1]);
      await host.ready();
    });
    const recolored = await render({}, (_, meshes) => {
      meshes[0].recolor.set(0.5, 0.75, 1, 1);
    });
    const plain = await render({});
    expect(same(tinted.px, recolored.px)).toBe(0);
    expect(same(tinted.px, plain.px)).toBeGreaterThan(W * H * 0.1);
  });

  it("runs dyno plugins in the tinted generate variant", async () => {
    const offset = d.dynoVec3(new THREE.Vector3(0, 0.3, 0));
    const shift = d.dynoBlock(
      { gsplat: d.Gsplat },
      { gsplat: d.Gsplat },
      ({ gsplat }) => {
        const { center } = d.splitGsplat(gsplat as never).outputs;
        return {
          gsplat: d.combineGsplat({
            gsplat: gsplat as never,
            center: d.add(center, offset),
          }),
        };
      },
    );
    const viaPlugins = await render({}, async (splats) => {
      const host = new PluginHost({ capabilities: splats.capabilities });
      host
        .register(tintPlugin([0.5, 0.5, 0.5, 1]))
        .register(dynoPlugin("shift", { objectModifiers: [shift as never] }))
        .attach(splats);
      await host.ready();
    });
    const direct = await render({}, (_, meshes) => {
      for (const m of meshes) {
        m.recolor.set(0.5, 0.5, 0.5, 1);
        m.dyno = { objectModifiers: [shift as never] };
      }
    });
    expect(same(viaPlugins.px, direct.px)).toBe(0);
  });

  for (const rasterizer of ["hardware", "tiles"] as const) {
    it(`blends per pixel with a blend plugin (${rasterizer})`, async () => {
      const clearColor = [26, 51, 77, 255];
      const hidden = await render({ rasterizer }, async (splats) => {
        const host = new PluginHost({ capabilities: splats.capabilities });
        host.register(fadePlugin({ opacity: 0 })).attach(splats);
        await host.ready();
      });
      for (let i = 0; i < W * H * 4; i++) {
        expect(Math.abs(hidden.px[i] - clearColor[i % 4])).toBeLessThanOrEqual(
          1,
        );
      }
      // Full fog at any depth (power 0): every covered pixel turns white.
      const fogged = await render({ rasterizer }, async (splats) => {
        const host = new PluginHost({ capabilities: splats.capabilities });
        host
          .register(fadePlugin({ fog: 1, power: 0, color: [1, 1, 1] }))
          .attach(splats);
        await host.ready();
      });
      const plain = await render({ rasterizer });
      let covered = 0;
      for (let i = 0; i < W * H; i++) {
        const k = i * 4;
        if (plain.px[k] === 26 && plain.px[k + 1] === 51) continue;
        covered++;
        // White over the clear colour: each channel is clear + a (255 - clear)
        // for one coverage a.
        const a = (c: number) =>
          (fogged.px[k + c] - clearColor[c]) / (255 - clearColor[c]);
        expect(a(0)).toBeGreaterThanOrEqual(-0.01);
        expect(Math.abs(a(0) - a(2))).toBeLessThan(0.05);
      }
      expect(covered).toBeGreaterThan(W * H * 0.1);
    });
  }
});

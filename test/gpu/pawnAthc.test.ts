// athenea's relightable (TX) clouds converted from USD by rust/build-lod's
// usd-athc: the .athc v3 (gzip sections) it writes goes through the existing
// WASM decoder with its transfer resident (112 f16 a splat, athenea's layout),
// and athenea's own decoders (slang/athenea_adapter) read it on the GPU.
//
// tx_cloud.athc is the converter's output for the tiny synthetic
// tx_cloud.usdc. The real pawn (publish-r2/sparkwebgpu/pawn/, not in git) is
// checked when it is there: PAWN_ATHC=/path/to/body-full.athc.

import { existsSync, readFileSync } from "node:fs";
import * as THREE from "three";
import { describe, expect, it, vi } from "vitest";
import { unpackAthcNormal, unpackAthcPbr } from "../../src/athc";
import type { AttribValues } from "../../src/defines";
import { GpuSplatSource } from "../../src/webgpu/GpuSplatSource";
import { KernelRegistry } from "../../src/webgpu/KernelRegistry";
import { WgpuSplatRenderer } from "../../src/webgpu/WgpuSplatRenderer";
import {
  ATTRIB_NONE,
  AttribPool,
  attribWords,
} from "../../src/webgpu/attributes/schema";
import athcTest from "../../src/webgpu/generated/tests/athc_adapter";
import { UniformWriter } from "../../src/webgpu/uniforms";
import { device, readBack, storage } from "./device";

const wasm = await vi.importActual<typeof import("spark-rs")>("spark-rs");
wasm.initSync({
  module: readFileSync(
    new URL("../../rust/spark-rs/pkg/spark_rs_bg.wasm", import.meta.url),
  ),
});

const TX = new Uint8Array(
  readFileSync(new URL("../fixtures/athc/tx_cloud.athc", import.meta.url)),
);
const PAWN = process.env.PAWN_ATHC;

type Decoded = {
  numSplats: number;
  packed: Uint32Array;
  splatEncoding: {
    lodOpacity: boolean;
    rgbMin: number;
    rgbMax: number;
    lnScaleMin: number;
    lnScaleMax: number;
  };
  attribSpecs?: AttribValues["specs"];
  attribColumns?: AttribValues["values"];
};

function decode(bytes: Uint8Array): Decoded {
  const decoder = wasm.decode_to_packedsplats(
    undefined,
    "cloud.athc",
    undefined,
    undefined,
    undefined,
    undefined,
  );
  decoder.push(bytes);
  return decoder.finish() as Decoded;
}

/// The v3 header's count, transfer and bounds (docs/docs/athc-v3.md).
function v3Header(bytes: Uint8Array) {
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  expect(v.getUint32(0, true)).toBe(0x33485441); // "ATH3"
  const f = (at: number) =>
    [0, 1, 2].map((k) => v.getFloat32(at + 4 * k, true));
  return {
    count: v.getUint32(12, true),
    transferCount: v.getUint32(40, true),
    boundsMin: f(80),
    boundsMax: f(92),
  };
}

const pool = (d: Decoded) =>
  AttribPool.fromValues({
    count: d.numSplats,
    specs: d.attribSpecs ?? [],
    values: d.attribColumns ?? [],
  });

describe("usd-athc output (WASM)", () => {
  it("decodes the converted TX cloud with its transfer resident", () => {
    const d = decode(TX);
    expect(d.attribSpecs?.map((s) => [s.name, s.format, s.components])).toEqual(
      [
        ["normalOct", "u32", 1],
        ["pbr", "u32", 1],
        ["transfer", "f16", 112],
        ["shadowBits", "u32", 8],
        ["athcGroup", "u32", 2],
      ],
    );
    const p = pool(d);
    // The splats follow the merged nodes, in Morton order: as
    // tx_cloud.usda wrote them, direct (0.5 at 1), indirect, field from 1.0.
    const h = v3Header(TX);
    expect(h.count).toBe(3);
    const splats = [0, 1, 2].map((k) => d.numSplats - 3 + k);
    for (const i of splats) expect(p.getAttribute("transfer", i)[1]).toBe(0.5);
    const firsts = splats
      .map((i) => p.getAttribute("transfer", i))
      .map((t) => [t[0], t[16], t[64]]);
    expect(firsts).toContainEqual([0, 0, 1]);
    for (const i of splats) {
      const t = p.getAttribute("transfer", i);
      expect(t[64]).toBeGreaterThanOrEqual(1);
      expect(t.length).toBe(112);
    }
    const metals = splats.map(
      (i) => unpackAthcPbr(p.getAttribute("pbr", i)[0]).metallic,
    );
    expect(metals.sort()).toEqual([0, 0, 1]);
    const bits = splats.map((i) => p.getAttribute("shadowBits", i)[0]);
    expect(bits.sort()).toEqual([0, 1431655765, 0xffffffff]);
  });
});

describe.skipIf(!device)("usd-athc output on the GPU", () => {
  const d = device as GPUDevice;
  const registry = new KernelRegistry(d);

  /// Every `step`-th element's attributes: a whole TX pawn's pool is past
  /// the default 128 MB storage binding (the pager splits it in pages).
  function subset(dec: Decoded, step: number): Decoded {
    const n = Math.ceil(dec.numSplats / step);
    const specs = dec.attribSpecs ?? [];
    const columns = (dec.attribColumns ?? []).map((col, k) => {
      // A .athc's streams come as their packed words (spec.packed).
      const s = specs[k];
      const c = s.packed ? attribWords(s as never) : s.components;
      const out = s.packed ? new Uint32Array(n * c) : new Float64Array(n * c);
      for (let i = 0; i < n; i++)
        out.set(col.subarray(i * step * c, (i * step + 1) * c), i * c);
      return out;
    });
    return { ...dec, numSplats: n, attribColumns: columns };
  }

  async function adapter(dec: Decoded, step: number) {
    const p = pool(dec);
    const { words } = p.pack();
    const id = (name: string) => {
      const k = p.id(name);
      return k < 0 ? ATTRIB_NONE : k;
    };
    const transfer = p.column("transfer")?.spec.components ?? 0;
    const shadow = p.column("shadowBits")?.spec.components ?? 0;
    const params = UniformWriter.for(athcTest).setAll({
      count: dec.numSplats,
      normalOct: id("normalOct"),
      emission: id("emission"),
      pbr: id("pbr"),
      lobes: id("lobes"),
      transfer: id("transfer"),
      shadowBits: id("shadowBits"),
      transferCount: transfer,
      shadowWords: shadow,
    });
    const out = storage(dec.numSplats * 28 * 4);
    registry.get(athcTest, "athcDecode").run({
      grid: [dec.numSplats],
      buffers: { pool: storage(words), decoded: out },
      uniforms: params.data,
    });
    const got = new Float32Array(await readBack(out));
    const bits = new Uint32Array(got.buffer);
    let checked = 0;
    for (let i = 0; i < dec.numSplats; i += step) {
      const o = i * 28;
      const n = unpackAthcNormal(p.getAttribute("normalOct", i)[0]);
      for (let c = 0; c < 3; c++) expect(got[o + c]).toBeCloseTo(n[c], 5);
      const m = unpackAthcPbr(p.getAttribute("pbr", i)[0]);
      expect(got[o + 7]).toBeCloseTo(m.roughness, 6);
      const t = p.getAttribute("transfer", i);
      expect(got[o + 23]).toBe(t[0]);
      expect(got[o + 24]).toBe(t[transfer - 1]);
      expect(bits[o + 25]).toBe(p.getAttribute("shadowBits", i)[0]);
      checked++;
    }
    return { transfer, shadow, checked, words: words.byteLength };
  }

  it("reads the converted TX cloud through athenea's decoders", async () => {
    const r = await adapter(decode(TX), 1);
    expect([r.transfer, r.shadow]).toEqual([112, 8]);
  });

  it.skipIf(!PAWN || !existsSync(PAWN))(
    "loads and draws the converted pawn",
    async () => {
      const bytes = new Uint8Array(readFileSync(PAWN as string));
      const t0 = performance.now();
      const dec = decode(bytes);
      const decodeMs = performance.now() - t0;
      const header = v3Header(bytes);
      const count = header.count;
      const r = await adapter(subset(dec, 31), 7);
      console.log(
        `pawn: ${dec.numSplats} nodes (${count} splats), transfer ${r.transfer}, shadow words ${r.shadow}, ` +
          `pool ${(r.words / 1e6).toFixed(1)} MB, decode ${decodeMs.toFixed(0)} ms, ${r.checked} checked`,
      );
      expect(r.transfer).toBe(header.transferCount);
      // Draw its splats (the finest level) from in front.
      const W = 96;
      const H = 96;
      const canvas = d.createTexture({
        size: [W, H],
        format: "bgra8unorm",
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT |
          GPUTextureUsage.COPY_SRC |
          GPUTextureUsage.COPY_DST,
      });
      const fake = {
        backend: {
          isWebGPUBackend: true,
          device: d,
          context: {
            getCurrentTexture: () => canvas,
          } as unknown as GPUCanvasContext,
          get: () => undefined,
        },
      };
      const e = dec.splatEncoding;
      const merged = dec.numSplats - count;
      const source = GpuSplatSource.fromPacked(
        d,
        dec.packed.subarray(merged * 4),
        count,
        {
          encoding: [e.rgbMin, e.rgbMax, e.lnScaleMin, e.lnScaleMax],
          lodOpacity: e.lodOpacity,
        },
      );
      const splats = new WgpuSplatRenderer(fake as never, { depthTest: false });
      splats.add(source);
      const [lo, hi] = [header.boundsMin, header.boundsMax];
      const centre = new THREE.Vector3(
        (lo[0] + hi[0]) / 2,
        (lo[1] + hi[1]) / 2,
        (lo[2] + hi[2]) / 2,
      );
      const size = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
      const camera = new THREE.PerspectiveCamera(40, W / H, 0.001, 10);
      camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
      camera.updateProjectionMatrix();
      camera.position.copy(centre).add(new THREE.Vector3(0, 0, size * 2));
      camera.lookAt(centre);
      camera.updateMatrixWorld();
      for (let frame = 0; frame < 3; frame++) {
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
        splats.render(camera);
        await d.queue.onSubmittedWorkDone();
        await new Promise((res) => setTimeout(res, 10));
      }
      const bytesPerRow = 256 * Math.ceil((W * 4) / 256);
      const buf = d.createBuffer({
        size: bytesPerRow * H,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const enc = d.createCommandEncoder();
      enc.copyTextureToBuffer(
        { texture: canvas },
        { buffer: buf, bytesPerRow },
        [W, H],
      );
      d.queue.submit([enc.finish()]);
      await buf.mapAsync(GPUMapMode.READ);
      const px = new Uint8Array(buf.getMappedRange());
      let lit = 0;
      for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++) {
          const o = y * bytesPerRow + x * 4;
          if (px[o] + px[o + 1] + px[o + 2] > 30) lit++;
        }
      buf.unmap();
      console.log(`pawn: ${lit} of ${W * H} pixels lit`);
      expect(lit).toBeGreaterThan(W * H * 0.05);
      expect(lit).toBeLessThan(W * H);
    },
    600_000,
  );
});

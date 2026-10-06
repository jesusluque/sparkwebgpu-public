// athenea's .athc on the WebGPU path: the real WASM decoder (rust/spark-lib
// athc.rs) on files athenea wrote (two_cards) or that mirror its tests
// (every_stream), the paged ATHV pages against the whole-file decode, the
// streams through slang/athenea_adapter (athenea's own decoders, vendored)
// against the CPU, and two_cards drawn by WgpuSplatRenderer.

import { readFileSync } from "node:fs";
import * as THREE from "three";
import { describe, expect, it, vi } from "vitest";
import {
  ATHV_KEEP_LINEAR,
  athvHead,
  unpackAthcLobes,
  unpackAthcNormal,
  unpackAthcPbr,
  unpackRgb9e5,
} from "../../src/athc";
import type { AttribValues } from "../../src/defines";
import { GpuSplatSource } from "../../src/webgpu/GpuSplatSource";
import { KernelRegistry } from "../../src/webgpu/KernelRegistry";
import { WgpuSplatRenderer } from "../../src/webgpu/WgpuSplatRenderer";
import { ATTRIB_NONE, AttribPool } from "../../src/webgpu/attributes/schema";
import athcTest from "../../src/webgpu/generated/tests/athc_adapter";
import { UniformWriter } from "../../src/webgpu/uniforms";
import { device, readBack, storage } from "./device";

// test/unit/setup.ts stubs the wasm package; these tests need the real one.
const wasm = await vi.importActual<typeof import("spark-rs")>("spark-rs");
wasm.initSync({
  module: readFileSync(
    new URL("../../rust/spark-rs/pkg/spark_rs_bg.wasm", import.meta.url),
  ),
});

const fixture = (name: string) =>
  new Uint8Array(
    readFileSync(new URL(`../fixtures/athc/${name}`, import.meta.url)),
  );
const TWO_CARDS = fixture("two_cards.athc");
const EVERY = fixture("every_stream.athc");

type Decoded = {
  numSplats: number;
  packed: Uint32Array;
  lodTree?: Uint32Array;
  splatEncoding: {
    lodOpacity: boolean;
    rgbMin: number;
    rgbMax: number;
    lnScaleMin: number;
    lnScaleMax: number;
  };
  attribSpecs?: AttribValues["specs"];
  attribColumns?: AttribValues["values"];
  fileType: string;
};

function decode(bytes: Uint8Array, pathName?: string): Decoded {
  const decoder = wasm.decode_to_packedsplats(
    undefined,
    pathName,
    undefined,
    undefined,
    undefined,
    undefined,
  );
  decoder.push(bytes);
  return decoder.finish() as Decoded;
}

function attribs(d: Decoded): AttribPool {
  return AttribPool.fromValues({
    count: d.numSplats,
    specs: d.attribSpecs ?? [],
    values: d.attribColumns ?? [],
  });
}

describe(".athc decode (WASM)", () => {
  it("reads two_cards as a LoD tree with its normals", () => {
    const layout = wasm.athc_layout(TWO_CARDS, TWO_CARDS.length);
    expect(layout.header.count).toBe(1352);
    expect(layout.header.flags).toBe(1);
    expect(layout.levelsEnd).toBeLessThanOrEqual(layout.chunks[0].offset);
    const d = decode(TWO_CARDS, "two_cards.athc");
    expect(d.fileType).toBe("athc");
    expect(d.splatEncoding.lodOpacity).toBe(true);
    const tree = wasm.athc_merged_pages(
      TWO_CARDS.subarray(0, layout.levelsEnd),
      TWO_CARDS.length,
    );
    expect(d.numSplats).toBe(tree.merged + 1352);
    expect(d.lodTree?.length).toBe(d.numSplats * 4);
    // Root: three children from 1 (level 1's groups).
    expect(d.lodTree?.[2]).toBe(3);
    expect(d.lodTree?.[3]).toBe(1);
    expect(d.attribSpecs?.map((s) => s.name)).toEqual([
      "normalOct",
      "athcGroup",
    ]);
    // athcGroup: the root covers every finest group, a splat its own.
    const pool = attribs(d);
    expect(pool.getAttribute("athcGroup", 0)).toEqual([0, 527]);
    const last = d.numSplats - 1;
    const [lo, hi] = pool.getAttribute("athcGroup", last);
    expect(hi - lo).toBe(1);
  });

  it("decodes paged ATHV pages as the whole file", () => {
    const whole = decode(TWO_CARDS);
    const layout = wasm.athc_layout(TWO_CARDS, TWO_CARDS.length);
    const { pages, merged, splatBase } = wasm.athc_merged_pages(
      TWO_CARDS.subarray(0, layout.levelsEnd),
      TWO_CARDS.length,
    );
    expect(pages.length).toBe(1);
    expect(splatBase).toBe(65536);
    const m = decode(pages[0]);
    expect(m.numSplats).toBe(merged);
    expect(Array.from(m.packed.subarray(0, merged * 4))).toEqual(
      Array.from(whole.packed.subarray(0, merged * 4)),
    );
    const { offset, count } = layout.chunks[0];
    const bytes = count * layout.elementBytes;
    const page = new Uint8Array(160 + bytes);
    page.set(athvHead(TWO_CARDS.subarray(0, 136), splatBase, count));
    page.set(TWO_CARDS.subarray(offset, offset + bytes), 160);
    const s = decode(page);
    expect(s.numSplats).toBe(1352);
    expect(Array.from(s.packed.subarray(0, 1352 * 4))).toEqual(
      Array.from(whole.packed.subarray(merged * 4, (merged + 1352) * 4)),
    );
    expect(Array.from(s.attribColumns?.[0] ?? [])).toEqual(
      Array.from(whole.attribColumns?.[0].subarray(merged) ?? []),
    );
  });

  it("keeps a linear cloud's colours linear when the ATHV head asks", () => {
    const layout = wasm.athc_layout(EVERY, EVERY.length);
    expect(layout.header.flags & 2).toBe(2);
    const { offset, count } = layout.chunks[0];
    const bytes = count * layout.elementBytes;
    const page = (flags: number) => {
      const out = new Uint8Array(160 + bytes);
      out.set(athvHead(EVERY.subarray(0, 136), 65536, count, flags));
      out.set(EVERY.subarray(offset, offset + bytes), 160);
      return out;
    };
    // every_stream's splats are 0.25, 0.5, 1 linear: red as a byte.
    const red = (d: Decoded) => d.packed[0] & 0xff;
    expect(red(decode(page(0)))).toBe(Math.round(0.5371 * 255));
    expect(red(decode(page(ATHV_KEEP_LINEAR)))).toBe(Math.round(0.25 * 255));
  });

  it("carries every stream athenea's test writes", () => {
    const d = decode(EVERY);
    expect(d.attribSpecs?.map((s) => [s.name, s.format, s.components])).toEqual(
      [
        ["normalOct", "u32", 1],
        ["emission", "u32", 1],
        ["pbr", "u32", 1],
        ["lobes", "u32", 3],
        ["transfer", "f16", 112],
        ["shadowBits", "u32", 8],
        ["athcGroup", "u32", 2],
      ],
    );
    const pool = attribs(d);
    const i = d.numSplats - 1;
    expect(pool.getAttribute("pbr", i)).toEqual([0x00ff8040]);
    expect(pool.getAttribute("transfer", i).every((t) => t === 1)).toBe(true);
    expect(unpackAthcPbr(0x00ff8040)).toMatchObject({
      metallic: 0x40 / 255,
      roughness: 0x80 / 255,
      transmission: 1,
    });
  });
});

describe.skipIf(!device)(".athc streams through slang/athenea_adapter", () => {
  const registry = new KernelRegistry(device as GPUDevice);

  it.each([
    ["every_stream.athc", EVERY],
    ["two_cards.athc", TWO_CARDS],
  ])("decodes %s's streams on the GPU as the CPU does", async (_, bytes) => {
    const d = decode(bytes);
    const pool = attribs(d);
    const { words } = pool.pack();
    const id = (name: string) => {
      const k = pool.id(name);
      return k < 0 ? ATTRIB_NONE : k;
    };
    const transfer = pool.column("transfer")?.spec.components ?? 0;
    const shadow = pool.column("shadowBits")?.spec.components ?? 0;
    const params = UniformWriter.for(athcTest).setAll({
      count: d.numSplats,
      normalOct: id("normalOct"),
      emission: id("emission"),
      pbr: id("pbr"),
      lobes: id("lobes"),
      transfer: id("transfer"),
      shadowBits: id("shadowBits"),
      transferCount: transfer,
      shadowWords: shadow,
    });
    const out = storage(d.numSplats * 28 * 4);
    registry.get(athcTest, "athcDecode").run({
      grid: [d.numSplats],
      buffers: { pool: storage(words), decoded: out },
      uniforms: params.data,
    });
    const got = new Float32Array(await readBack(out));
    const gotBits = new Uint32Array(got.buffer);
    for (let i = 0; i < d.numSplats; i += 7) {
      const o = i * 28;
      const n =
        id("normalOct") === ATTRIB_NONE
          ? [0, 0, 1]
          : unpackAthcNormal(pool.getAttribute("normalOct", i)[0]);
      for (let c = 0; c < 3; c++) expect(got[o + c]).toBeCloseTo(n[c], 5);
      if (id("emission") !== ATTRIB_NONE) {
        const e = unpackRgb9e5(pool.getAttribute("emission", i)[0]);
        for (let c = 0; c < 3; c++) expect(got[o + 3 + c]).toBeCloseTo(e[c], 6);
      }
      if (id("pbr") !== ATTRIB_NONE) {
        const m = unpackAthcPbr(pool.getAttribute("pbr", i)[0]);
        expect(got[o + 6]).toBeCloseTo(m.metallic, 6);
        expect(got[o + 7]).toBeCloseTo(m.roughness, 6);
        expect(got[o + 8]).toBeCloseTo(m.transmission, 6);
      } else {
        expect([got[o + 6], got[o + 7]]).toEqual([0, 1]);
      }
      if (id("lobes") !== ATTRIB_NONE) {
        const [w0, w1, w2] = pool.getAttribute("lobes", i);
        const l = unpackAthcLobes(w0, w1, w2);
        expect(got[o + 10]).toBeCloseTo(l.specularWeight, 6);
        expect(got[o + 14]).toBeCloseTo(l.specularIor, 6);
        expect(got[o + 17]).toBeCloseTo(l.coatIor, 6);
        expect(got[o + 18]).toBe(l.coatDarkening);
        expect(got[o + 22]).toBeCloseTo(l.sheenRoughness, 6);
      } else {
        // plainLobes(): weight 1, index 1.5.
        expect([got[o + 10], got[o + 14]]).toEqual([1, 1.5]);
      }
      if (transfer) {
        const t = pool.getAttribute("transfer", i);
        expect(got[o + 23]).toBe(t[0]);
        expect(got[o + 24]).toBe(t[transfer - 1]);
      }
      expect(gotBits[o + 25]).toBe(
        shadow ? pool.getAttribute("shadowBits", i)[0] : 0xffffffff,
      );
    }
  });
});

const W = 96;
const H = 64;

describe.skipIf(!device)(".athc drawn by WgpuSplatRenderer", () => {
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

  async function readCanvas(): Promise<Uint8Array> {
    const bytesPerRow = 256 * Math.ceil((W * 4) / 256);
    const buf = d.createBuffer({
      size: bytesPerRow * H,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = d.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: canvas }, { buffer: buf, bytesPerRow }, [
      W,
      H,
    ]);
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

  it("draws two_cards' splats where its bounds are", async () => {
    const decoded = decode(TWO_CARDS);
    const layout = wasm.athc_layout(TWO_CARDS, TWO_CARDS.length);
    const merged = decoded.numSplats - 1352;
    const e = decoded.splatEncoding;
    const source = GpuSplatSource.fromPacked(
      d,
      decoded.packed.subarray(merged * 4),
      1352,
      {
        encoding: [e.rgbMin, e.rgbMax, e.lnScaleMin, e.lnScaleMax],
        lodOpacity: e.lodOpacity,
      },
    );
    const splats = new WgpuSplatRenderer(fakeRenderer as never, {
      depthTest: false,
    });
    splats.add(source);
    const [lo, hi] = [layout.header.boundsMin, layout.header.boundsMax];
    const centre = new THREE.Vector3(
      (lo[0] + hi[0]) / 2,
      (lo[1] + hi[1]) / 2,
      (lo[2] + hi[2]) / 2,
    );
    const size = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    const camera = new THREE.PerspectiveCamera(60, W / H, 0.01, 100);
    camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
    camera.updateProjectionMatrix();
    // The cards face +z (mesh2splat's test scene): look at them from there.
    camera.position.copy(centre).add(new THREE.Vector3(0, 0, size * 1.5));
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
      await new Promise((r) => setTimeout(r, 10));
    }
    const px = await readCanvas();
    let lit = 0;
    for (let i = 0; i < W * H; i++) {
      if (px[i * 4] + px[i * 4 + 1] + px[i * 4 + 2] > 30) lit++;
    }
    // The cards cover a good part of the view, not all of it.
    expect(lit).toBeGreaterThan(W * H * 0.05);
    expect(lit).toBeLessThan(W * H);
  });
});

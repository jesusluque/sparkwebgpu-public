// The compact path over band pages (thread CN, compactPages.ts) on Dawn,
// with test/fixtures/athc/compact_pages.athc (a small paged tree with its
// compact tier: rust athc_compact.rs write_compact_fixture):
//   - streamed cold, it settles to the cut and the image of the same path
//     with every page resident;
//   - under a budget smaller than the tree it evicts, and its cut stays in
//     what is resident;
//   - the paged draw against the resident compact cloud (encodeCompact, its
//     8192-gaussian chunks) holding the same cut's values.

import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import * as THREE from "three";
import { describe, expect, it, vi } from "vitest";
import type { AthcV3Layout } from "../../src/athc";
import {
  CompactRelitPath,
  encodeCompact,
} from "../../src/webgpu/athenea/compact";
import {
  COMPACT_PLANE_WIDTHS,
  CompactBandStreamer,
  type CompactPiece,
  type LodCall,
  compactPlaneOffset,
  compactTablesOf,
} from "../../src/webgpu/athenea/compactPages";
import { halfToFloat } from "../../src/webgpu/athenea/cpcaJoint";
import { makeLodCall } from "../unit/compactLod";
import { compare, readImage } from "./compactHarness";
import { wideDevice } from "./device";

const wasm = (await vi.importActual(
  "../../rust/spark-rs/pkg/spark_rs.js",
)) as typeof import("spark-rs");
wasm.initSync({
  module: readFileSync(
    new URL("../../rust/spark-rs/pkg/spark_rs_bg.wasm", import.meta.url),
  ),
});

const FILE = new URL("../fixtures/athc/compact_pages.athc", import.meta.url)
  .pathname;

/** src/worker.ts's LoD calls, in-process. */
export const lodInProcess: LodCall = makeLodCall(wasm);

function piece(io: { ranges: number; bytes: number }): CompactPiece {
  const bytes = new Uint8Array(readFileSync(FILE));
  const prefix = bytes.slice(0, wasm.athc_prefix_bytes(bytes.slice(0, 256)));
  return {
    name: "fixture",
    layout: wasm.athc_layout(prefix, bytes.length) as AthcV3Layout,
    prefix,
    read: async (offset, n) => {
      io.ranges += 1;
      io.bytes += n;
      return bytes.slice(offset, offset + n);
    },
  };
}

const gunzip = async (b: Uint8Array) => new Uint8Array(gunzipSync(b));

/** A small dome: a sky brighter above, a warm ground. */
function dome() {
  const W = 64;
  const H = 32;
  const data = new Float32Array(W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const up = 1 - y / H;
      const k = 4 * (y * W + x);
      data.set([0.3 + up, 0.4 + up, 0.6 + 1.5 * up, 1], k);
    }
  }
  return { width: W, height: H, data };
}

function camera(W: number, H: number) {
  const c = new THREE.PerspectiveCamera(50, W / H, 0.05, 100);
  c.coordinateSystem = THREE.WebGPUCoordinateSystem;
  c.up.set(0, 0, 1);
  c.position.set(9, -6, 1.5);
  c.lookAt(new THREE.Vector3(14, 0, 0));
  c.updateProjectionMatrix();
  c.updateMatrixWorld();
  return c;
}

describe.skipIf(!wideDevice)("the compact path over band pages", () => {
  const d = wideDevice as GPUDevice;
  const W = 320;
  const H = 200;
  const sky = dome();

  async function draw(path: CompactRelitPath, cam: THREE.PerspectiveCamera) {
    const tex = d.createTexture({
      size: [W, H],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const enc = d.createCommandEncoder();
    path.render(enc, cam, tex.createView(), "rgba16float", W, H, {
      clear: { r: 0, g: 0, b: 0, a: 0 },
    });
    path.registry.submit(enc.finish());
    await d.queue.onSubmittedWorkDone();
    const image = await readImage(d, tex);
    tex.destroy();
    return image;
  }

  async function streamed(slots: number, fit = false, tau = 0.05) {
    const io = { ranges: 0, bytes: 0 };
    const p = piece(io);
    const s = new CompactBandStreamer(d, [p], {
      slots,
      slotNodes: 512,
      tau,
      tauToFit: fit,
      gunzip,
      lod: lodInProcess,
    });
    await s.init();
    const path = new CompactRelitPath(d, { hdri: sky, stableOrder: true });
    path.setPagedCloud(s.pool);
    const cam = camera(W, H);
    let frames = 0;
    for (;;) {
      const asked = s.stats.pagesRequested;
      await s.update(cam, H, path);
      frames += 1;
      if (!s.busy && s.stats.pagesRequested === asked) break;
      await s.idle();
      expect(frames).toBeLessThan(50);
    }
    return { s, path, cam, io, p, frames };
  }

  const keysOf = (s: CompactBandStreamer) =>
    [...s.drawList]
      .map((i) => s.nodeOf(i))
      .map((n) => `${n?.page}:${n?.node}`)
      .sort();

  it("settles to the resident tree's cut and image", async () => {
    const a = await streamed(8);
    const total = a.p.layout.pages?.length ?? 0;
    expect(total).toBe(8);
    expect(a.s.stats.cut).toBeGreaterThan(100);
    // Only what the view needs came, each run of pages one Range.
    expect(a.s.stats.residentPages).toBeLessThanOrEqual(total);
    expect(a.io.ranges).toBe(a.s.stats.ranges);
    const imageA = await draw(a.path, a.cam);

    const b = await streamed(8);
    await b.s.loadAll();
    await b.s.update(b.cam, H, b.path);
    expect(b.s.stats.residentPages).toBe(total);
    expect(keysOf(b.s)).toEqual(keysOf(a.s));
    const imageB = await draw(b.path, b.cam);
    const c = compare(imageA, imageB);
    expect(c.relMSE).toBeLessThan(1e-6);
    // Something was drawn.
    let lit = 0;
    for (let k = 0; k < imageA.length; k += 4) lit += imageA[k + 3] > 0 ? 1 : 0;
    expect(lit).toBeGreaterThan(200);

    // The same cut as a resident compact cloud (encodeCompact, its chunks):
    // only the quantization boxes differ.
    const p = a.p;
    const decoder = wasm.decode_to_extsplats(undefined, "cloud.athc");
    decoder.push(new Uint8Array(readFileSync(FILE)));
    const dd = decoder.finish() as {
      ext0: Uint32Array;
      ext1: Uint32Array;
      attribSpecs: { name: string }[];
      attribColumns: Uint32Array[];
    };
    const col = (name: string) =>
      dd.attribColumns[dd.attribSpecs.findIndex((x) => x.name === name)];
    const tables = compactTablesOf(p);
    const nodes = [...a.s.drawList].map(
      (i) => a.s.nodeOf(i) as { page: number; node: number },
    );
    const n = nodes.length;
    const rows = p.layout.pages ?? [];
    const k4 = p.layout.sections.findIndex((x) => (x.id as string) === "CMPC");
    const raws = new Map<number, Uint8Array>();
    const rawOf = async (page: number) => {
      let r = raws.get(page);
      if (!r) {
        const span = p.layout.blocks[page].spans[k4];
        r = await gunzip(await p.read(span.offset, span.stored));
        raws.set(page, r);
      }
      return r;
    };
    const pick = (src: Uint32Array, per: number) => {
      const out = new Uint32Array(n * per);
      nodes.forEach(({ page, node }, k) => {
        const i = rows[page].first + node;
        out.set(src.subarray(per * i, per * i + per), per * k);
      });
      return out;
    };
    const q8 = new Uint32Array(n * 7);
    const shadowBits = new Uint32Array(n * 8);
    for (const [k, { page, node }] of nodes.entries()) {
      const raw = await rawOf(page);
      const count = rows[page].count;
      const at = (plane: number) =>
        compactPlaneOffset(plane, count) + node * COMPACT_PLANE_WIDTHS[plane];
      q8[7 * k + 6] = raw[at(10)];
      for (let b = 0; b < 3; b++) {
        const w = new DataView(raw.buffer, raw.byteOffset).getUint32(
          at(11 + b),
          true,
        );
        for (let j = 0; j < 8; j++) {
          const q4 = ((w >>> (4 * j)) << 28) >> 28;
          const v = Math.round((q4 * 127) / 7) & 0xff;
          q8[7 * k + 2 * b + (j >> 2)] |= v << (8 * (j & 3));
        }
      }
      const e = raw[at(14)] | (raw[at(14) + 1] << 8);
      shadowBits.set(tables.codebook.subarray(8 * e, 8 * e + 8), 8 * k);
    }
    const basis = new Float32Array(tables.basis.length * 2);
    for (let w = 0; w < tables.basis.length; w++) {
      basis[2 * w] = halfToFloat(tables.basis[w] & 0xffff);
      basis[2 * w + 1] = halfToFloat(tables.basis[w] >>> 16);
    }
    const cloud = encodeCompact({
      count: n,
      ext0: pick(dd.ext0, 4),
      ext1: pick(dd.ext1, 4),
      normalOct: pick(col("normalOct"), 1),
      pbr: pick(col("pbr"), 1),
      lobes: pick(col("lobes"), 3),
      curvature: pick(col("curvature"), 2),
      shadowBits,
      q8,
      basis,
      clusters: tables.clusters,
      weightBits: 4,
    });
    const resident = new CompactRelitPath(d, { hdri: sky, stableOrder: true });
    resident.setCloud(cloud);
    const imageR = await draw(resident, a.cam);
    {
      // Splat by splat: the relit colour, and which are drawn.
      a.path.setDrawList(null);
      const pc = await a.path.colours(a.cam, W, H);
      a.path.setDrawList(a.s.drawList);
      const rc = await resident.colours(a.cam, W, H);
      const worst: Record<string, number> = {};
      const list = [...a.s.drawList];
      for (let k = 0; k < n; k++) {
        const i = list[k];
        for (const [name, arr, arr2] of [
          ["colour", pc.colours, rc.colours],
          ["shape", pc.shapes, rc.shapes],
          ["place", pc.places, rc.places],
        ] as const) {
          for (let c = 0; c < 4; c++) {
            const d = Math.abs(arr[4 * i + c] - arr2[4 * k + c]);
            const key = `${name}${c}`;
            if (!(d <= (worst[key] ?? 0))) worst[key] = d;
          }
        }
        if (k < 3 && process.env.CN_DEBUG)
          process.stderr.write(
            `CN ${k} ${[...pc.colours.subarray(4 * i, 4 * i + 4)]} | ${[...rc.colours.subarray(4 * k, 4 * k + 4)]} || ${[...pc.shapes.subarray(4 * i, 4 * i + 4)]} | ${[...rc.shapes.subarray(4 * k, 4 * k + 4)]}\n`,
          );
      }
      let bad = 0;
      for (let k = 0; k < n; k++) {
        const i = list[k];
        const d =
          Math.abs(pc.places[4 * i] - rc.places[4 * k]) +
          Math.abs(pc.places[4 * i + 1] - rc.places[4 * k + 1]);
        if (d > 1) {
          if (bad < 6 && process.env.CN_DEBUG)
            process.stderr.write(
              `CN bad ${k} node ${JSON.stringify(a.s.nodeOf(i))} p ${[...pc.places.subarray(4 * i, 4 * i + 3)]} r ${[...rc.places.subarray(4 * k, 4 * k + 3)]} ps ${[...pc.shapes.subarray(4 * i, 4 * i + 4)]} rs ${[...rc.shapes.subarray(4 * k, 4 * k + 4)]}\n`,
            );
          bad++;
        }
      }
      process.stderr.write(`CN bad ${bad} of ${n}\n`);
      process.stderr.write(`CN worst ${JSON.stringify(worst)}\n`);
      for (const c of [0, 1, 2]) expect(worst[`colour${c}`]).toBeLessThan(1e-3);
      expect(bad).toBeLessThan(0.02 * n);
    }
    const r = compare(imageA, imageR);
    expect(r.relMSE).toBeLessThan(2e-3);
    expect(Math.abs(r.ratio - 1)).toBeLessThan(0.01);
    process.stderr.write(
      `COMPACT_PAGES_DRAW ${JSON.stringify({ cut: n, frames: a.frames, ranges: a.io.ranges, bytes: a.io.bytes, pagedVsResidentTree: c, pagedVsResidentCloud: r })}\n`,
    );
    for (const x of [a, b]) {
      x.path.dispose();
      x.s.dispose();
    }
    resident.dispose();
  });

  it("evicts finest first under a budget, the cut within what is resident", async () => {
    const a = await streamed(3, true);
    expect(a.s.stats.residentPages).toBeLessThanOrEqual(3);
    expect(a.s.stats.tau).toBeGreaterThan(0.05);
    // Every drawn index is in a slot holding a page.
    for (const i of a.s.drawList) expect(a.s.nodeOf(i)).not.toBeNull();
    // A jump to a view that needs other pages: evictions, still in budget.
    a.cam.position.set(30, -4, 1);
    a.cam.lookAt(new THREE.Vector3(34, 0, 0));
    a.cam.updateMatrixWorld();
    for (let k = 0; k < 20; k++) {
      await a.s.update(a.cam, H, a.path);
      await a.s.idle();
    }
    expect(a.s.stats.residentPages).toBeLessThanOrEqual(3);
    for (const i of a.s.drawList) expect(a.s.nodeOf(i)).not.toBeNull();
    const image = await draw(a.path, a.cam);
    expect(image.some((v) => v > 0)).toBe(true);
    a.path.dispose();
    a.s.dispose();
  });
});

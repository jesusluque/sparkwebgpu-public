// Several CPCA bases in one pool (thread CX: athenea's car with Part_2 and
// Infinite_Black on bases of their own). The pool keeps each distinct basis
// once, back to back, and a page's head names its piece's first cluster
// (compactBases, headWord10); the projections and the bins are by that
// global cluster. Here the fixture's pages drawn two ways must be the same,
// bit for bit:
//   two pieces (pages 0..3 and 4..7) on one basis of 512 clusters U ++ V,
//   the second's nodes pointing at its second half (the cluster's 9th bit);
//   the same two pieces on bases U and V of 256 each.

import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import * as THREE from "three";
import { describe, expect, it, vi } from "vitest";
import type { AthcV3Layout } from "../../src/athc";
import { CompactRelitPath } from "../../src/webgpu/athenea/compact";
import {
  COMPACT_PLANE_WIDTHS,
  CompactPagePool,
  type CompactPieceTables,
  compactBases,
  compactPlaneOffset,
  parseCompactTables,
} from "../../src/webgpu/athenea/compactPages";
import { readImage } from "./compactHarness";
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
const CLUSTER_WORDS = 504;
const P_CLUSTER = 10;
const P_SHADOW = 14;
void COMPACT_PLANE_WIDTHS;

function dome() {
  const W = 64;
  const H = 32;
  const data = new Float32Array(W * H * 4);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const up = 1 - y / H;
      data.set([0.3 + up, 0.4 + up, 0.6 + 1.5 * up, 1], 4 * (y * W + x));
    }
  return { width: W, height: H, data };
}

describe.skipIf(!wideDevice)("several bases in one pool", () => {
  const d = wideDevice as GPUDevice;
  const W = 320;
  const H = 200;

  it("two bases draw as one basis holding both", async () => {
    const bytes = new Uint8Array(readFileSync(FILE));
    const prefix = bytes.slice(0, wasm.athc_prefix_bytes(bytes.slice(0, 256)));
    const layout = wasm.athc_layout(prefix, bytes.length) as AthcV3Layout;
    const t = (layout as unknown as { compactTable: number[] }).compactTable;
    const own = parseCompactTables(prefix.subarray(t[0], t[0] + t[1]));
    const rows = layout.pages ?? [];
    const k4 = layout.sections.findIndex((x) => (x.id as string) === "CMPC");
    const raws = rows.map((_, page) => {
      const span = layout.blocks[page].spans[k4];
      return new Uint8Array(
        gunzipSync(bytes.subarray(span.offset, span.offset + span.stored)),
      );
    });
    // U: the fixture's clusters tiled to 256; V: the same shifted by one
    // (another table, the same values a node of cluster c reads at c + 1).
    const c = own.clusters;
    const tile = (shift: number) => {
      const out = new Uint32Array(256 * CLUSTER_WORDS);
      for (let k = 0; k < 256; k++) {
        const from = ((k + shift) % c) * CLUSTER_WORDS;
        out.set(
          own.basis.subarray(from, from + CLUSTER_WORDS),
          k * CLUSTER_WORDS,
        );
      }
      return out;
    };
    const U = tile(0);
    const V = tile(1);
    const UV = new Uint32Array(U.length + V.length);
    UV.set(U);
    UV.set(V, U.length);
    const split = 4;
    expect(rows.length).toBeGreaterThan(split);

    // Config one: one piece, basis UV, pages from `split` on at cluster + 256.
    const one: CompactPieceTables = { ...own, basis: UV, clusters: 512 };
    const rawsOne = raws.map((r, page) => {
      if (page < split) return r;
      const n = rows[page].count;
      const out = r.slice();
      const at = compactPlaneOffset(P_SHADOW, n);
      for (let i = 0; i < n; i++) {
        const v = out[at + 2 * i] | (out[at + 2 * i + 1] << 8);
        expect(v).toBeLessThan(0x8000);
        out[at + 2 * i + 1] = ((v | 0x8000) >> 8) & 0xff;
      }
      return out;
    });
    // Config two: two pieces on U and V (the same codebook and materials).
    const pu: CompactPieceTables = { ...own, basis: U, clusters: 256 };
    const pv: CompactPieceTables = { ...own, basis: V, clusters: 256 };
    expect(compactBases([pu, pv]).clusterBase).toEqual([0, 256]);
    expect(compactBases([one]).bases).toBe(1);
    void P_CLUSTER;

    const render = async (
      pieces: CompactPieceTables[],
      pieceOf: (page: number) => number,
      rawOf: (page: number) => Uint8Array,
    ) => {
      const pool = new CompactPagePool(d, pieces, rows.length, 512);
      const list: number[] = [];
      rows.forEach((row, page) => {
        pool.writePage(
          page,
          pieceOf(page),
          page,
          row.count,
          rawOf(page),
          row.compact,
        );
        for (let i = 0; i < row.count; i++) list.push(page * 512 + i);
      });
      // No faint dither (its seed is the node's place in its piece).
      const path = new CompactRelitPath(d, {
        hdri: dome(),
        stableOrder: true,
        faint: false,
      });
      path.setPagedCloud(pool);
      path.setDrawList(Uint32Array.from(list));
      const cam = new THREE.PerspectiveCamera(50, W / H, 0.05, 100);
      cam.up.set(0, 0, 1);
      cam.position.set(9, -6, 1.5);
      cam.lookAt(new THREE.Vector3(14, 0, 0));
      cam.updateProjectionMatrix();
      cam.updateMatrixWorld();
      const tex = d.createTexture({
        size: [W, H],
        format: "rgba16float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      for (let f = 0; f < 3; f++) {
        const enc = d.createCommandEncoder();
        path.render(enc, cam, tex.createView(), "rgba16float", W, H, {
          clear: { r: 0, g: 0, b: 0, a: 0 },
        });
        path.registry.submit(enc.finish());
        await d.queue.onSubmittedWorkDone();
      }
      const image = await readImage(d, tex);
      const { colours } = await path.colours(cam, W, H);
      tex.destroy();
      path.dispose();
      pool.buffer.destroy();
      return { image, colours, list };
    };
    // The same two pieces both ways (a page's faint and sheet dither seeds are
    // its place in its piece): one basis UV for both, or U and V.
    const a = await render(
      [one, { ...one }],
      (p) => (p < split ? 0 : 1),
      (p) => rawsOne[p],
    );
    const b = await render(
      [pu, pv],
      (p) => (p < split ? 0 : 1),
      (p) => raws[p],
    );
    let lit = 0;
    let worst = 0;
    for (let k = 0; k < a.image.length; k++) {
      worst = Math.max(worst, Math.abs(a.image[k] - b.image[k]));
      if (k % 4 === 3 && a.image[k] > 0) lit += 1;
    }
    expect(lit).toBeGreaterThan(200);
    expect(worst).toBe(0);
    // The listed (written) nodes: a slot past its page's count reads zeros
    // under its page's head, a cluster of no one.
    let worstC = 0;
    for (const i of a.list)
      for (let c = 0; c < 4; c++)
        worstC = Math.max(
          worstC,
          Math.abs(a.colours[4 * i + c] - b.colours[4 * i + c]),
        );
    expect(worstC).toBe(0);
  });
});

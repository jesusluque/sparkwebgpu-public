// Thread CX: athenea's 5120 close-up level (car5120) converted to the compact
// format (car-data/convert8.sh: CMPC v4 runs, against an asset table) parses
// through the viewer's CPU path: the WASM layout (athc_layout), the piece's
// tables resolved against the asset's .cmpa, every page's runs container
// (traversal member gunzipped, every run inflated, unpackRunsContainer as
// CompactBandStreamer lands it) and the WASM traversal over all pages
// resident: a cut at a far and a near view, every key a node of a page.
//
// Opt-in, real data (outside git, never published):
//   CLOSEUP_TRIAL=<dir with car.cmpa and car_*.athc> [CLOSEUP_FILES=a.athc,b.athc]
//   npx vitest run --dir test/unit test/unit/athcCloseUpTrial.test.ts
// With CLOSEUP_PAIR=flat.athc,hier.athc (in that dir): the same piece packed
// flat and under a synthetic hierarchy over its roots (usd-athc
// --synthetic-fanout) cuts the same nodes (by position) at every view and
// tau, and the load and traversal times of both.

import {
  closeSync,
  existsSync,
  openSync,
  readFileSync,
  readSync,
  readdirSync,
  statSync,
} from "node:fs";
import { gunzipSync } from "node:zlib";
import * as THREE from "three";
import { describe, expect, it, vi } from "vitest";
import type { AthcPageRow, AthcV3Layout } from "../../src/athc";
import {
  parseRunsContainer,
  unpackRunsContainer,
} from "../../src/webgpu/athenea/compactNodes";
import {
  compactChildPages,
  parseCompactAsset,
  parseCompactTables,
  resolveCompactTables,
} from "../../src/webgpu/athenea/compactPages";
import { makeLodCall } from "./compactLod";

vi.mock("spark-rs", async () => {
  const { readFileSync } = await import("node:fs");
  const actual = await vi.importActual<Record<string, unknown>>("spark-rs");
  (actual.initSync as (o: { module: Buffer }) => void)({
    module: readFileSync(
      new URL("../../rust/spark-rs/pkg/spark_rs_bg.wasm", import.meta.url),
    ),
  });
  return { ...actual, default: async () => ({}) };
});

// biome-ignore lint/suspicious/noExplicitAny: the WASM bindings
const wasm = (await import("spark-rs")) as unknown as Record<string, any>;
const lod = makeLodCall(wasm);

const SLOT = 3072;
const DIR = process.env.CLOSEUP_TRIAL ?? "";
type Row = AthcPageRow & { compact: number[] };

const ON = DIR !== "" && existsSync(DIR);

describe.skipIf(!ON)("close-up trial (car5120)", () => {
  // The describe's body runs even when skipped: nothing read without data.
  const pair = process.env.CLOSEUP_PAIR?.split(",") ?? [];
  const files = ON
    ? (
        process.env.CLOSEUP_FILES?.split(",") ??
        (pair.length ? [] : readdirSync(DIR).filter((f) => f.endsWith(".athc")))
      ).sort()
    : [];
  const asset = ON
    ? parseCompactAsset(new Uint8Array(readFileSync(`${DIR}/car.cmpa`)))
    : undefined;
  it.skipIf(files.length > 0 || pair.length > 0)(
    "needs CLOSEUP_TRIAL",
    () => {},
  );

  for (const file of files) {
    it(`${file} parses and cuts`, async () => {
      const t = await openTree(`${DIR}/${file}`, asset);
      try {
        const cuts: Record<string, number> = {};
        for (const v of viewsOf(t.lo, t.hi, [
          ["far", 6, 393, 852, 3],
          ["near", 1.2, 1920, 1080, 1],
        ])) {
          const keys = await cutOf(t, v);
          expect(keys.length).toBeGreaterThan(0);
          for (const i of keys) {
            const page = Math.floor(i / SLOT);
            expect(page).toBeLessThan(t.rows.length);
            expect(i % SLOT).toBeLessThan(t.rows[page].count);
          }
          cuts[v.name] = keys.length;
        }
        console.log(
          `CLOSEUP_TRIAL ${JSON.stringify({ file, bytes: t.bytes, prefixBytes: t.prefixBytes, pages: t.rows.length, nodes: t.nodes, largestPage: t.largest, runs: t.runs, cmpcStored: t.stored, cuts })}`,
        );
      } finally {
        wasm.dispose_lod_tree(t.lodId);
      }
    }, 600_000);
  }

  it.skipIf(pair.length !== 2)(
    "a synthetic hierarchy cuts as the flat forest",
    async () => {
      const [a, b] = await Promise.all(
        pair.map((f) => openTree(`${DIR}/${f}`, asset)),
      );
      try {
        const views = viewsOf(a.lo, a.hi, [
          ["far", 12, 393, 852, 3],
          ["far tau 1", 6, 393, 852, 1],
          ["mid", 3, 1920, 1080, 2],
          ["near", 1.2, 1920, 1080, 1],
          ["close", 0.5, 1920, 1080, 0.5],
          ["very far", 60, 393, 852, 4],
        ]);
        const times = { a: 0, b: 0 };
        const sizes: Record<string, number> = {};
        for (const v of views) {
          let t0 = performance.now();
          const ka = await cutOf(a, v);
          times.a += performance.now() - t0;
          t0 = performance.now();
          const kb = await cutOf(b, v);
          times.b += performance.now() - t0;
          expect(kb.length, v.name).toBe(ka.length);
          // The same nodes: each of b's at a position of a's, nearest first
          // within a page box's code, then within an f16 ulp (a merge's centre
          // is f16 in the traversal's planes: ~2 mm at 2-4 m).
          const extent = Math.max(...a.hi.map((h, k) => h - a.lo[k]));
          let left = ka.map((i) => a.position(i));
          let rest = kb.map((i) => b.position(i));
          for (const tol of [2e-4 * extent, Math.max(2e-4 * extent, 4e-3)]) {
            const cellOf = (p: number[]) => p.map((x) => Math.floor(x / tol));
            const grid = new Map<string, number[]>();
            left.forEach((p, j) => {
              const key = cellOf(p).join(",");
              grid.set(key, [...(grid.get(key) ?? []), j]);
            });
            const used = new Set<number>();
            const unmatched: number[][] = [];
            for (const p of rest) {
              const [x, y, z] = cellOf(p);
              let best = -1;
              let bestD = tol;
              for (let dx = -1; dx <= 1; dx++)
                for (let dy = -1; dy <= 1; dy++)
                  for (let dz = -1; dz <= 1; dz++) {
                    for (const j of grid.get(`${x + dx},${y + dy},${z + dz}`) ??
                      []) {
                      if (used.has(j)) continue;
                      const q = left[j];
                      const d = Math.hypot(
                        q[0] - p[0],
                        q[1] - p[1],
                        q[2] - p[2],
                      );
                      if (d <= bestD) {
                        bestD = d;
                        best = j;
                      }
                    }
                  }
              if (best >= 0) used.add(best);
              else unmatched.push(p);
            }
            left = left.filter((_, j) => !used.has(j));
            rest = unmatched;
          }
          expect(rest.length, v.name).toBe(0);
          expect(left.length, v.name).toBe(0);
          sizes[v.name] = ka.length;
        }
        console.log(
          `CLOSEUP_PAIR ${JSON.stringify({ pair, cuts: sizes, loadMs: [a.loadMs, b.loadMs], traverseMs: [times.a, times.b], pages: [a.rows.length, b.rows.length], nodes: [a.nodes, b.nodes], page0: [a.rows[0].count, b.rows[0].count] })}`,
        );
      } finally {
        wasm.dispose_lod_tree(a.lodId);
        wasm.dispose_lod_tree(b.lodId);
      }
    },
    600_000,
  );
});

interface Tree {
  lodId: number;
  rows: Row[];
  bytes: number;
  prefixBytes: number;
  nodes: number;
  largest: number;
  runs: number;
  stored: number;
  lo: number[];
  hi: number[];
  loadMs: number;
  /** A node's position (its page box's codes decoded). */
  position: (key: number) => number[];
}

/** A file through the viewer's CPU path, every page in the WASM tree. */
async function openTree(
  path: string,
  asset: ReturnType<typeof parseCompactAsset> | undefined,
): Promise<Tree> {
  const bytes = statSync(path).size;
  const fd = openSync(path, "r");
  const read = (offset: number, n: number) => {
    const out = new Uint8Array(n);
    readSync(fd, out, 0, n, offset);
    return out;
  };
  try {
    const t0 = performance.now();
    const prefixBytes = wasm.athc_prefix_bytes(read(0, 256)) as number;
    const prefix = read(0, prefixBytes);
    const layout = wasm.athc_layout(prefix, bytes) as AthcV3Layout;
    const rows = (layout.pages ?? []) as Row[];
    expect(rows.length).toBeGreaterThan(0);
    const k4 = layout.sections.findIndex((s) => (s.id as string) === "CMPC");
    expect(k4).toBeGreaterThanOrEqual(0);
    const t = (layout as unknown as { compactTable: number[] }).compactTable;
    const tables = resolveCompactTables(
      parseCompactTables(prefix.slice(t[0], t[0] + t[1])),
      asset,
    );
    expect(tables.basis.length).toBeGreaterThan(0);
    const { lodId } = (await lod("newLodTree", {
      capacity: rows.length * SLOT,
      slot: SLOT,
      compact: true,
    })) as { lodId: number };
    let nodes = 0;
    let runs = 0;
    let stored = 0;
    let largest = 0;
    const lo = [1e9, 1e9, 1e9];
    const hi = [-1e9, -1e9, -1e9];
    const ranges: Record<string, unknown>[] = [];
    const inputs: { input: Uint8Array; box: Float32Array; n: number }[] = [];
    for (let page = 0; page < rows.length; page++) {
      const n = rows[page].count;
      expect(n).toBeGreaterThan(0);
      expect(n).toBeLessThanOrEqual(SLOT);
      largest = Math.max(largest, n);
      const span = layout.blocks[page].spans[k4];
      stored += span.stored;
      const c = read(span.offset, span.stored);
      const h = parseRunsContainer(c);
      expect(h.n).toBe(n);
      const trav = new Uint8Array(
        gunzipSync(c.subarray(h.travAt, h.travAt + h.travBytes)),
      );
      const u = unpackRunsContainer(c, h, trav, true);
      expect(u.planes.length).toBe(h.runs);
      for (const p of compactChildPages(u.raw as Uint8Array, n, page)) {
        expect(p).toBeGreaterThan(page);
        expect(p).toBeLessThan(rows.length);
      }
      const box = new Float32Array(
        Uint32Array.from(rows[page].compact.slice(0, 6)).buffer,
      );
      for (let k = 0; k < 3; k++) {
        lo[k] = Math.min(lo[k], box[k]);
        hi[k] = Math.max(hi[k], box[3 + k]);
      }
      ranges.push({
        lodId,
        pageBase: page * SLOT,
        chunkBase: page * 65536,
        count: n,
        input: u.input,
        box,
      });
      inputs.push({ input: u.input, box, n });
      nodes += n;
      runs += h.runs;
    }
    await lod("updateCompactLodTrees", { ranges });
    const loadMs = performance.now() - t0;
    const position = (key: number) => {
      const { input, box, n } = inputs[Math.floor(key / SLOT)];
      const i = key % SLOT;
      const v = new DataView(input.buffer, input.byteOffset);
      return [0, 1, 2].map(
        (k) =>
          box[k] +
          v.getUint16(2 * k * n + 2 * i, true) *
            ((box[3 + k] - box[k]) / 65535),
      );
    };
    return {
      lodId,
      rows,
      bytes,
      prefixBytes,
      nodes,
      largest,
      runs,
      stored,
      lo,
      hi,
      loadMs,
      position,
    };
  } finally {
    closeSync(fd);
  }
}

interface View {
  name: string;
  camera: THREE.PerspectiveCamera;
  H: number;
  tau: number;
}

/** Views round a box: [name, distance in radii, W, H, tau]. */
function viewsOf(
  lo: number[],
  hi: number[],
  specs: [string, number, number, number, number][],
): View[] {
  const c = lo.map((v, k) => (v + hi[k]) / 2);
  const r = Math.hypot(...lo.map((v, k) => hi[k] - v)) / 2;
  return specs.map(([name, d, W, H, tau]) => {
    const camera = new THREE.PerspectiveCamera(40, W / H, 0.001, 1000);
    camera.up.set(0, 0, 1);
    camera.position.fromArray(c.map((v, k) => v + d * r * [0.6, -0.7, 0.4][k]));
    camera.lookAt(new THREE.Vector3().fromArray(c));
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld();
    return { name, camera, H, tau };
  });
}

/** The cut's keys (page * SLOT + place), every page resident. */
async function cutOf(t: Tree, v: View) {
  const pixel = (2 * Math.tan((v.camera.fov * Math.PI) / 360)) / v.H;
  const { keyIndices } = (await lod("traverseLodTrees", {
    maxSplats: t.rows.length * SLOT,
    pixelScaleLimit: pixel,
    instances: {
      piece: {
        lodId: t.lodId,
        rootPage: 0,
        viewToObjectCols: v.camera.matrixWorld.elements.slice(),
        lodScale: 1,
        behindFoveate: 1,
        coneFov0: 0,
        coneFov: 0,
        coneFoveate: 1,
      },
    },
    errorLimit: pixel * v.tau,
  })) as {
    keyIndices: Record<string, { numSplats: number; indices: Uint32Array }>;
  };
  const k = keyIndices.piece;
  return Array.from(k.indices.subarray(0, k.numSplats));
}

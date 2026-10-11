// The compact tier's traversal words, v1 (thread CN: the LoD tree's 4 words
// and LODE's 2, 24 bytes a node, 24 in the WASM traversal) against v2 (thread
// CQ: 16 bytes a node in the file, 14 in the traversal): the same cut, node
// for node, through the WASM traversal (spark-rs lod_tree.rs), with every
// page resident and with the pages a view needs (WgpuBandPager planBand), at
// athenea's far camera (393x695), near (1920x1080) and the phone at the near
// camera (393x695), tau 0.5, 1 and 2. Also: the pages each page's children
// are in (the pager's dependencies) are the same.
//
// Always: CN's fixture (test/fixtures/athc/compact_pages.v1.athc) against its
// rebase (compact_pages.athc). Opt-in, real data (outside git, never
// published):
//   COMPACT_WORDS_V1=<CN's cn-data> COMPACT_WORDS_V2=<the same pieces after
//   athc-convert --compact-asset; car.cmpa beside them>
//   npx vitest run --dir test/unit test/unit/compactWords.test.ts
// prints COMPACT_WORDS {json} lines: per view the cut, the needed pages and
// nodes, and the bytes before and after (download, GPU, CPU).

import { closeSync, existsSync, openSync, readSync, statSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import * as THREE from "three";
import { describe, expect, it, vi } from "vitest";
import type { AthcPageRow, AthcV3Layout } from "../../src/athc";
import { planBand } from "../../src/webgpu/WgpuBandPager";
import {
  COMPACT_LOD_NODE_BYTES,
  type CompactAssetTables,
  compactChildPages,
  compactSlotLayout,
  compactTraversalInput,
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
const V1_GPU = 40;
const V1_NODE = 64;
const V2_NODE = 56;
const PIECES = [
  "Hood",
  "Front_Fender",
  "Headlights_Cover",
  "Front_Logo",
  "F_Grill",
  "Door_001",
];

type Row = AthcPageRow & { tier4Bytes: number; compact: number[] };

interface Piece {
  name: string;
  layout: AthcV3Layout;
  rows: Row[];
  prefixBytes: number;
  tableBlob: Uint8Array;
  /** Each page's raw CMPC section. */
  page: (p: number) => Uint8Array;
  close: () => void;
}

function openPiece(path: string, name: string): Piece {
  const fd = openSync(path, "r");
  const size = statSync(path).size;
  const read = (offset: number, bytes: number) => {
    const out = new Uint8Array(bytes);
    readSync(fd, out, 0, bytes, offset);
    return out;
  };
  const prefixBytes = wasm.athc_prefix_bytes(read(0, 256)) as number;
  const prefix = read(0, prefixBytes);
  const layout = wasm.athc_layout(prefix, size) as AthcV3Layout;
  const k4 = layout.sections.findIndex((s) => (s.id as string) === "CMPC");
  const gz = layout.sections[k4].compression === 1;
  const t = (layout as unknown as { compactTable: number[] }).compactTable;
  return {
    name,
    layout,
    rows: (layout.pages ?? []) as Row[],
    prefixBytes,
    tableBlob: prefix.slice(t[0], t[0] + t[1]),
    page: (p) => {
      const span = layout.blocks[p].spans[k4];
      const stored = read(span.offset, span.stored);
      return gz ? new Uint8Array(gunzipSync(stored)) : stored;
    },
    close: () => closeSync(fd),
  };
}

// athenea's /World/Camera and CameraFar (pieces/t4/cameras.usda), Z-up,
// horizontal field 39.6 degrees (test/gpu/compactPages.test.ts).
const DIR_Z = [0.6080817, -0.778309, 0.1564344];
const NEAR = [4.378188, -5.853825, 1.676328];
const FAR = [13.4994, -17.5284, 4.0228];
function cameraAt(t: number, W: number, H: number) {
  const aspect = W / H;
  const vfov =
    (2 * Math.atan(Math.tan((39.6 * Math.PI) / 360) / aspect) * 180) / Math.PI;
  const camera = new THREE.PerspectiveCamera(vfov, aspect, 0.05, 100);
  camera.up.set(0, 0, 1);
  const p = FAR.map((f, k) => f + (NEAR[k] - f) * t);
  camera.position.fromArray(p);
  camera.lookAt(
    new THREE.Vector3().fromArray(p.map((v, k) => v - DIR_Z[k] * 7.2)),
  );
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();
  return camera;
}
interface View {
  name: string;
  W: number;
  H: number;
  camera: THREE.PerspectiveCamera;
}
const ATHENEA_VIEWS: View[] = [
  { name: "far 393x695", W: 393, H: 695, camera: cameraAt(0, 393, 695) },
  {
    name: "near 1920x1080",
    W: 1920,
    H: 1080,
    camera: cameraAt(1, 1920, 1080),
  },
  {
    name: "phone 393x695 (near camera)",
    W: 393,
    H: 695,
    camera: cameraAt(1, 393, 695),
  },
];

/** Far, near and phone views of a box: at 20, 1.5 and 1.5 radii. */
function boxViews(lo: number[], hi: number[]): View[] {
  const c = lo.map((v, k) => (v + hi[k]) / 2);
  const r = Math.hypot(...lo.map((v, k) => hi[k] - v)) / 2;
  const at = (d: number, W: number, H: number) => {
    const camera = new THREE.PerspectiveCamera(50, W / H, 0.001, 1000);
    camera.position.fromArray(c.map((v, k) => v + d * r * [0.6, -0.7, 0.4][k]));
    camera.lookAt(new THREE.Vector3().fromArray(c));
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld();
    return camera;
  };
  return [
    { name: "far 393x695", W: 393, H: 695, camera: at(20, 393, 695) },
    { name: "near 1920x1080", W: 1920, H: 1080, camera: at(1.5, 1920, 1080) },
    { name: "phone 393x695", W: 393, H: 695, camera: at(1.5, 393, 695) },
  ];
}

/** v1 words of a page: the LoD tree's 4, LODE's 2 a node. */
function v1Words(raw: Uint8Array, n: number) {
  const w = (from: number, words: number) =>
    new Uint32Array(
      raw.buffer.slice(
        raw.byteOffset + from,
        raw.byteOffset + from + 4 * words,
      ),
    );
  return { tree: w(V1_GPU * n, 4 * n), lode: w((V1_GPU + 16) * n, 2 * n) };
}

/** The pages a v1 page's nodes have children in (CN's streamer). */
function v1ChildPages(tree: Uint32Array, n: number, page: number) {
  const kids = new Set<number>();
  for (let i = 0; i < n; i++) {
    if ((tree[4 * i + 2] & 0xffff) === 0) continue;
    const c = tree[4 * i + 3] >>> 16;
    if (c !== page) kids.add(c);
  }
  return [...kids].sort((a, b) => a - b);
}

/**
 * One tree per piece over `resident` pages (page p in pool slot p), v1 words
 * (a full tree) or v2 (a compact one); the cut of every piece in one call,
 * as `piece:page:place` keys in the traversal's order.
 */
async function cut(
  pieces: Piece[],
  raws: Map<string, Map<number, Uint8Array>>,
  resident: Map<string, Set<number>>,
  v2: boolean,
  camera: THREE.PerspectiveCamera,
  H: number,
  tau: number,
) {
  const instances: Record<string, unknown> = {};
  const lodIds: number[] = [];
  for (const p of pieces) {
    const { lodId } = (await lod("newLodTree", {
      capacity: p.rows.length * SLOT,
      slot: SLOT,
      compact: v2,
    })) as { lodId: number };
    lodIds.push(lodId);
    const ranges = [...(resident.get(p.name) ?? [])].map((page) => {
      const raw = raws.get(p.name)?.get(page) as Uint8Array;
      const n = p.rows[page].count;
      const base = {
        lodId,
        pageBase: page * SLOT,
        chunkBase: page * 65536,
        count: n,
      };
      if (v2) {
        return {
          ...base,
          input: compactTraversalInput(raw, n),
          box: new Float32Array(
            Uint32Array.from(p.rows[page].compact.slice(0, 6)).buffer,
          ),
        };
      }
      const { tree, lode } = v1Words(raw, n);
      return { ...base, lodTreeData: tree, lodErrorData: lode };
    });
    await lod(v2 ? "updateCompactLodTrees" : "updateLodTrees", { ranges });
    instances[p.name] = {
      lodId,
      rootPage: 0,
      viewToObjectCols: camera.matrixWorld.elements.slice(),
      lodScale: 1,
      behindFoveate: 1,
      coneFov0: 0,
      coneFov: 0,
      coneFoveate: 1,
    };
  }
  const pixel = (2 * Math.tan((camera.fov * Math.PI) / 360)) / H;
  const { keyIndices } = (await lod("traverseLodTrees", {
    maxSplats: pieces.reduce((n, p) => n + p.rows.length * SLOT, 0),
    pixelScaleLimit: pixel,
    instances,
    errorLimit: pixel * tau,
  })) as {
    keyIndices: Record<string, { numSplats: number; indices: Uint32Array }>;
  };
  for (const id of lodIds) wasm.dispose_lod_tree(id);
  const keys: string[] = [];
  for (const p of pieces) {
    const k = keyIndices[p.name];
    for (const i of k.indices.subarray(0, k.numSplats)) {
      keys.push(`${p.name}:${Math.floor(i / SLOT)}:${i % SLOT}`);
    }
  }
  return keys;
}

/** GPU table words of a pool over these pieces' tables (CompactPagePool). */
function tableBytes(
  tables: {
    basis: Uint32Array;
    codebook: Uint32Array;
    materials: Uint32Array;
  }[],
) {
  let words = 0;
  const seen = new Set<Uint32Array>();
  const push = (w: Uint32Array) => {
    if (seen.has(w)) return;
    seen.add(w);
    words = Math.ceil(words / 16) * 16 + w.length;
  };
  push(tables[0].basis);
  for (const t of tables) {
    push(t.codebook);
    push(t.materials);
  }
  return Math.ceil(words / 16) * 16 * 4;
}

/** v1 tables blob: the same words as v2's for the arrays. */
function v1Tables(blob: Uint8Array) {
  const w = new Uint32Array(blob.buffer.slice(blob.byteOffset));
  return {
    codebook: w.slice(w[11], w[11] + 8 * w[5]),
    materials: w.slice(w[12], w[12] + 4 * w[6]),
    basis: w.slice(w[13], w[13] + w[14]),
  };
}

async function prove(
  label: string,
  before: Piece[],
  after: Piece[],
  asset: CompactAssetTables | undefined,
  assetBytes: number,
  views: View[],
) {
  // Every page of both, held for the whole run (the real data: ~70 MB raw).
  const raws1 = new Map<string, Map<number, Uint8Array>>();
  const raws2 = new Map<string, Map<number, Uint8Array>>();
  const all = new Map<string, Set<number>>();
  for (const [k, p] of before.entries()) {
    const q = after[k];
    expect(q.rows.length).toBe(p.rows.length);
    const m1 = new Map<number, Uint8Array>();
    const m2 = new Map<number, Uint8Array>();
    for (let page = 0; page < p.rows.length; page++) {
      const n = p.rows[page].count;
      expect(q.rows[page].count).toBe(n);
      const a = p.page(page);
      const b = q.page(page);
      expect(a.length).toBe(V1_NODE * n);
      expect(b.length).toBe(V2_NODE * n);
      // The pager's dependencies.
      expect(compactChildPages(b, n, page).sort((x, y) => x - y)).toEqual(
        v1ChildPages(v1Words(a, n).tree, n, page),
      );
      m1.set(page, a);
      m2.set(page, b);
    }
    raws1.set(p.name, m1);
    raws2.set(p.name, m2);
    all.set(p.name, new Set(m1.keys()));
  }
  const slotBytes = compactSlotLayout(SLOT).slotWords * 4;
  const gpuTables1 = tableBytes(before.map((p) => v1Tables(p.tableBlob)));
  const gpuTables2 = tableBytes(
    after.map((p) =>
      resolveCompactTables(parseCompactTables(p.tableBlob), asset),
    ),
  );
  const prefix1 = before.reduce((n, p) => n + p.prefixBytes, 0);
  const prefix2 = after.reduce((n, p) => n + p.prefixBytes, 0) + assetBytes;
  const out: Record<string, unknown>[] = [];
  for (const v of views) {
    const camera = v.camera;
    const eye = camera.position.toArray();
    for (const tau of [0.5, 1, 2]) {
      const pixel = (2 * Math.tan((camera.fov * Math.PI) / 360)) / v.H;
      // What the view needs (the streamer settles to it, cold, with room).
      const plan = planBand(
        before.map((p) => ({
          key: p.name,
          rows: p.rows,
          pageBytes: p.rows.map((r) => r.tier4Bytes),
          eye,
          limit: pixel * tau,
          resident: () => false,
          inFlight: () => false,
          children: () => undefined,
        })),
        1 << 30,
      );
      const needed = plan.needed as Map<string, Set<number>>;
      for (const [which, resident] of [
        ["all pages", all],
        ["needed pages", needed],
      ] as const) {
        const a = await cut(before, raws1, resident, false, camera, v.H, tau);
        const b = await cut(after, raws2, resident, true, camera, v.H, tau);
        const setA = new Set(a);
        const onlyB = b.filter((k) => !setA.has(k)).length;
        const setB = new Set(b);
        const onlyA = a.filter((k) => !setB.has(k)).length;
        expect(b.length, `${label} ${v.name} tau ${tau} ${which}`).toBe(
          a.length,
        );
        expect(onlyA + onlyB, `${label} ${v.name} tau ${tau} ${which}`).toBe(0);
        const sameOrder = a.every((k, i) => b[i] === k);
        if (which === "needed pages") {
          let pages = 0;
          let nodes = 0;
          let tier4a = 0;
          let tier4b = 0;
          for (const [k, p] of before.entries()) {
            for (const page of needed.get(p.name) ?? []) {
              pages += 1;
              nodes += p.rows[page].count;
              tier4a += p.rows[page].tier4Bytes;
              tier4b += after[k].rows[page].tier4Bytes;
            }
          }
          out.push({
            set: label,
            view: v.name,
            tau,
            cut: a.length,
            sameCut: true,
            sameOrder,
            neededPages: pages,
            neededNodes: nodes,
            download: {
              before: prefix1 + tier4a,
              after: prefix2 + tier4b,
              tablesBefore: prefix1,
              tablesAfter: prefix2,
              tier4Before: tier4a,
              tier4After: tier4b,
            },
            gpu: {
              before: gpuTables1 + pages * slotBytes,
              after: gpuTables2 + pages * slotBytes,
              tablesBefore: gpuTables1,
              tablesAfter: gpuTables2,
            },
            cpu: {
              before: 24 * pages * SLOT,
              after: COMPACT_LOD_NODE_BYTES * pages * SLOT,
            },
          });
        }
      }
    }
  }
  for (const r of out)
    process.stderr.write(`COMPACT_WORDS ${JSON.stringify(r)}\n`);
  return out;
}

describe("compact traversal words, v1 against v2", () => {
  it("CN's fixture: the same cut", async () => {
    const dir = new URL("../fixtures/athc/", import.meta.url).pathname;
    const a = openPiece(`${dir}compact_pages.v1.athc`, "fixture");
    const b = openPiece(`${dir}compact_pages.athc`, "fixture");
    // The fixture's views: its own box, near and far.
    const h = a.layout.header as unknown as {
      boundsMin: number[];
      boundsMax: number[];
    };
    const rows = await prove(
      "fixture",
      [a],
      [b],
      undefined,
      0,
      boxViews(h.boundsMin, h.boundsMax),
    );
    // The views reach into the tree: the near cut is finer than the root.
    expect((rows[3].cut as number) > (rows[0].cut as number)).toBe(true);
    expect(rows.length).toBe(9);
    a.close();
    b.close();
  });

  const V1 = process.env.COMPACT_WORDS_V1 ?? "";
  const V2 = process.env.COMPACT_WORDS_V2 ?? "";
  it.skipIf(!V1 || !V2)(
    "athenea's t48 pieces (cn-data): the same cut",
    async () => {
      const before = PIECES.map((p) => openPiece(`${V1}/${p}.athc`, p));
      const after = PIECES.map((p) => openPiece(`${V2}/${p}.athc`, p));
      const t = parseCompactTables(after[0].tableBlob);
      let asset: CompactAssetTables | undefined;
      let assetBytes = 0;
      if (t.assetFlags) {
        const path = `${V2}/${t.assetName}`;
        expect(existsSync(path)).toBe(true);
        const fd = openSync(path, "r");
        const bytes = new Uint8Array(statSync(path).size);
        readSync(fd, bytes, 0, bytes.length, 0);
        closeSync(fd);
        asset = parseCompactAsset(bytes);
        assetBytes = bytes.length;
      }
      const nodes = before.reduce((n, p) => n + p.layout.header.count, 0);
      const tier4 = (ps: Piece[]) =>
        ps.reduce(
          (n, p) => n + p.rows.reduce((m, r) => m + r.tier4Bytes, 0),
          0,
        );
      process.stderr.write(
        `COMPACT_WORDS_FILES ${JSON.stringify({
          nodes,
          tier4Before: tier4(before),
          tier4After: tier4(after),
          tier4PerNodeBefore: +(tier4(before) / nodes).toFixed(2),
          tier4PerNodeAfter: +(tier4(after) / nodes).toFixed(2),
          prefixBefore: before.reduce((n, p) => n + p.prefixBytes, 0),
          prefixAfter: after.reduce((n, p) => n + p.prefixBytes, 0),
          assetBytes,
        })}\n`,
      );
      await prove("t48", before, after, asset, assetBytes, ATHENEA_VIEWS);
      for (const p of [...before, ...after]) p.close();
    },
    1_800_000,
  );
});

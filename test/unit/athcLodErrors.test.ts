// The LoD error cut (rust athc_lod_cut.rs, the v3 section LODE) through the
// browser's paths, on the real WASM: the bounds of each merged node reach
// the decoded splats (`extra.lodError`) whole and paged, and the traversal
// (lod_tree.rs traverse_lod_trees with an error limit) draws the cut athenea's
// lod_tree_cut.slang defines: P(n) <= tau < P(parent), every leaf once; with
// back-face culling, wholly back-facing nodes go with their subtrees.

import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";

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

vi.mock("../../src/SplatWorker", async () => {
  const wasm = (await import("spark-rs")) as unknown as Record<
    string,
    // biome-ignore lint/suspicious/noExplicitAny: the WASM bindings
    (...args: any[]) => any
  >;
  const handlers: Record<string, (args: Record<string, unknown>) => unknown> = {
    athc3MergedPages: ({ tables, levels }) => {
      const { pages, headers, ...tree } = wasm.athc3_merged_pages(
        tables,
        levels,
      );
      return { tree, pages, headers };
    },
  };
  const worker = {
    call: async (name: string, args: Record<string, unknown>) =>
      handlers[name](args),
  };
  return {
    SplatWorker: class {},
    workerPool: {
      withWorker: async <T>(fn: (w: typeof worker) => Promise<T>) => fn(worker),
    },
  };
});

const { fetchAthcPage, openAthc } = await import("../../src/athc");
// biome-ignore lint/suspicious/noExplicitAny: the WASM bindings
const wasm = (await import("spark-rs")) as unknown as Record<string, any>;

// hood_t16.athc with its levels' coverage and error bounds:
// athc-convert hood_t16.athc hood_t16.lode.athc --coverage --lod-errors --gzip
const BYTES = new Uint8Array(
  readFileSync(new URL("../fixtures/athc/hood_t16.lode.athc", import.meta.url)),
);

function decode(bytes: Uint8Array) {
  const d = wasm.decode_to_packedsplats(
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
  );
  d.push(bytes);
  return d.finish() as {
    numSplats: number;
    lodTree: Uint32Array;
    lodError?: Uint32Array;
  };
}

function half(bits: number) {
  const e = (bits >> 10) & 31;
  const m = bits & 1023;
  const s = bits & 0x8000 ? -1 : 1;
  if (e === 31) return m ? Number.NaN : s * Number.POSITIVE_INFINITY;
  return s * (e === 0 ? m * 2 ** -24 : (1 + m / 1024) * 2 ** (e - 15));
}

type Node = {
  c: number[];
  eps: number;
  rho: number;
  start: number;
  count: number;
};

function nodes(tree: Uint32Array, err: Uint32Array, n: number): Node[] {
  return Array.from({ length: n }, (_, k) => ({
    c: [
      half(tree[k * 4] & 0xffff),
      half(tree[k * 4] >>> 16),
      half(tree[k * 4 + 1] & 0xffff),
    ],
    eps: half(err[k * 2] & 0xffff),
    rho: half(err[k * 2] >>> 16),
    count: tree[k * 4 + 2] & 0xffff,
    start: tree[k * 4 + 3],
  }));
}

const PX = (2 * Math.tan(Math.PI / 6)) / 852; // 60 degrees over 852 px

function traverse(
  lodId: number,
  eye: number[],
  tau: number | undefined,
  backCull: boolean,
) {
  const m = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, eye[0], eye[1], eye[2], 1];
  const r = wasm.traverse_lod_trees(
    10_000_000,
    PX,
    undefined,
    new Uint32Array([lodId]),
    new Uint32Array([0xffffffff]),
    new Float32Array(m),
    new Float32Array([1]),
    new Float32Array([1]), // no foveation: the cut alone
    new Float32Array([1]),
    new Float32Array([0]),
    new Float32Array([0]),
    tau === undefined ? undefined : PX * tau,
    backCull,
  ) as {
    culledCount: number;
    instanceIndices: { numSplats: number; indices: Uint32Array }[];
  };
  const { numSplats, indices } = r.instanceIndices[0];
  return {
    cut: Array.from(indices.slice(0, numSplats)),
    culled: r.culledCount,
  };
}

describe(".athc LoD error cut (LODE)", () => {
  const whole = decode(BYTES);
  const n = whole.numSplats;

  it("reaches the decoded tree whole and paged", async () => {
    const err = whole.lodError as Uint32Array;
    expect(err.length).toBe(n * 2);
    const paging = await openAthc({ fileBytes: BYTES, keepLinear: true });
    const merged = paging.tree.merged;
    let withBounds = 0;
    for (let k = 0; k < merged; ++k) if (err[k * 2] !== 0) ++withBounds;
    expect(withBounds).toBe(merged);
    for (let k = merged * 2; k < n * 2; ++k) expect(err[k]).toBe(0);
    const page = await fetchAthcPage(paging, 0, { fileBytes: BYTES });
    const flags = new DataView(page.buffer, page.byteOffset).getUint32(
      152,
      true,
    );
    expect(flags & 4).toBe(4);
    const m = decode(page);
    expect(
      Array.from((m.lodError as Uint32Array).subarray(0, merged * 2)),
    ).toEqual(Array.from(err.subarray(0, merged * 2)));
  });

  it("cuts where P(n) <= tau < P(parent), each leaf once", () => {
    const { lodId } = wasm.init_lod_tree(n, whole.lodTree, whole.lodError);
    const all = nodes(whole.lodTree, whole.lodError as Uint32Array, n);
    const parent = new Int32Array(n).fill(-1);
    for (let k = 0; k < n; ++k) {
      for (let c = all[k].start; c < all[k].start + all[k].count; ++c) {
        parent[c] = k;
      }
    }
    const root = all[0].c;
    const P = (k: number, eye: number[]) => {
      const d = Math.hypot(...all[k].c.map((v, i) => v - eye[i]));
      return all[k].eps / Math.max(d - all[k].rho, 1e-3) / PX;
    };
    for (const [dist, tau] of [
      [0.5, 1],
      [1.5, 0.5],
      [3, 2],
    ]) {
      const eye = [root[0], root[1] + dist * 0.3, root[2] + dist];
      const { cut } = traverse(lodId, eye, tau, false);
      const size = traverse(lodId, eye, undefined, false).cut;
      expect(cut.length).toBeGreaterThan(0);
      const leafOwners = new Map<number, number>();
      for (const k of cut) {
        if (all[k].count > 0) expect(P(k, eye)).toBeLessThanOrEqual(tau * 1.01);
        if (parent[k] >= 0)
          expect(P(parent[k], eye)).toBeGreaterThan(tau * 0.99);
      }
      // Each splat under exactly one drawn node.
      const drawn = new Set(cut);
      for (let s = 0; s < n; ++s) {
        if (all[s].count > 0) continue;
        let owners = 0;
        for (let a = s; a >= 0; a = parent[a]) if (drawn.has(a)) ++owners;
        leafOwners.set(owners, (leafOwners.get(owners) ?? 0) + 1);
      }
      expect([...leafOwners.keys()]).toEqual([1]);
      // A different cut from the size cut's.
      expect(cut.sort().join()).not.toBe(size.sort().join());
    }
    wasm.dispose_lod_tree(lodId);
  });

  it("drops wholly back-facing nodes with their subtrees", () => {
    const { lodId } = wasm.init_lod_tree(n, whole.lodTree, whole.lodError);
    const err = whole.lodError as Uint32Array;
    // The hood's mean facing, from its nodes' cone axes.
    const mean = [0, 0, 0];
    for (let k = 0; k < n; ++k) {
      if (err[k * 2] === 0 || (err[k * 2 + 1] & 0xffff) === 0x7c00) continue;
      const b = err[k * 2 + 1] >>> 16;
      const f = [((b & 255) / 255) * 2 - 1, ((b >> 8) / 255) * 2 - 1];
      const v = [f[0], f[1], 1 - Math.abs(f[0]) - Math.abs(f[1])];
      const t = Math.max(-v[2], 0);
      v[0] += v[0] >= 0 ? -t : t;
      v[1] += v[1] >= 0 ? -t : t;
      for (let i = 0; i < 3; ++i) mean[i] += v[i];
    }
    const l = Math.hypot(...mean);
    const root = nodes(whole.lodTree, err, 1)[0].c;
    const eyeAt = (s: number) => root.map((v, i) => v + (s * 2 * mean[i]) / l);
    const front = traverse(lodId, eyeAt(1), 1, true);
    const back = traverse(lodId, eyeAt(-1), 1, true);
    const backPlain = traverse(lodId, eyeAt(-1), 1, false);
    expect(front.culled).toBe(0);
    expect(back.culled).toBeGreaterThan(0);
    expect(back.cut.length).toBeLessThan(backPlain.cut.length);
    wasm.dispose_lod_tree(lodId);
  });
});

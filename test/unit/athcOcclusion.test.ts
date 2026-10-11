// athenea's occlusion masks (`decimate --occlusion-mask`, rust athc_lod_cut.rs
// `occlusion_hides`, the .athc section OCCL and its codebook) through the
// WASM decoder and traversal: a node is left out with its subtree when the
// eye is outside the asset's box and the four cells toward it are closed.

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

// biome-ignore lint/suspicious/noExplicitAny: the WASM bindings
const wasm = (await import("spark-rs")) as unknown as Record<string, any>;

// usd-athc --tree pieces/t27/tree_Door_001.usdc door_occlusion.athc
//   --transfer none --no-shadow --no-curvature --lod-errors --closed --gzip
const BYTES = new Uint8Array(
  readFileSync(
    new URL("../fixtures/athc/door_occlusion.athc", import.meta.url),
  ),
);

function half(bits: number) {
  const e = (bits >> 10) & 31;
  const m = bits & 1023;
  const s = bits & 0x8000 ? -1 : 1;
  return s * (e === 0 ? m * 2 ** -24 : (1 + m / 1024) * 2 ** (e - 15));
}

// The rule, as athc_lod_cut.rs occlusion_hides.
function hides(
  codebook: Uint32Array,
  stored: number,
  b: Float32Array,
  c: number[],
  eye: number[],
) {
  if (stored === 0) return false;
  if ([0, 1, 2].every((k) => eye[k] >= b[k] && eye[k] <= b[3 + k]))
    return false;
  const d = eye.map((v, k) => v - c[k]);
  const l1 = Math.abs(d[0]) + Math.abs(d[1]) + Math.abs(d[2]);
  const n = d.map((v) => v / l1);
  const sg = (v: number) => (v >= 0 ? 1 : -1);
  const p =
    n[2] >= 0
      ? [n[0], n[1]]
      : [(1 - Math.abs(n[1])) * sg(n[0]), (1 - Math.abs(n[0])) * sg(n[1])];
  const at = p.map((v) => (v * 0.5 + 0.5) * 16 - 0.5);
  const lo = at.map(Math.floor);
  const f = at.map((v, k) => v - lo[k]);
  for (let k = 0; k < 4; ++k) {
    const kx = k & 1;
    const ky = k >> 1;
    const w = (kx ? f[0] : 1 - f[0]) * (ky ? f[1] : 1 - f[1]);
    if (w <= 0) continue;
    const cx = Math.min(15, Math.max(0, lo[0] + kx));
    const cy = Math.min(15, Math.max(0, lo[1] + ky));
    const bit = cy * 16 + cx;
    if ((codebook[(stored - 1) * 8 + (bit >> 5)] >>> (bit & 31)) & 1)
      return false;
  }
  return true;
}

describe(".athc occlusion masks (OCCL)", () => {
  const d = wasm.decode_to_packedsplats(
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
  );
  d.push(BYTES);
  const r = d.finish() as {
    numSplats: number;
    lodTree: Uint32Array;
    lodError: Uint32Array;
    lodOcclusion: Uint32Array;
    lodOcclusionCodebook: Uint32Array;
    lodOcclusionBounds: Float32Array;
  };

  it("decodes the indices, the codebook and the box", () => {
    expect(r.lodOcclusion.length).toBe(r.numSplats);
    expect(r.lodOcclusionCodebook.length).toBe(256 * 8);
    expect(r.lodOcclusionBounds.length).toBe(6);
    expect(r.lodOcclusion.filter((v) => v !== 0).length).toBeGreaterThan(
      r.numSplats * 0.9,
    );
  });

  it("leaves out exactly the subtrees the masks hide", () => {
    const n = r.numSplats;
    const { lodId } = wasm.init_lod_tree(
      n,
      r.lodTree,
      r.lodError,
      r.lodOcclusion,
    );
    wasm.set_lod_occlusion_codebook(
      lodId,
      r.lodOcclusionCodebook,
      r.lodOcclusionBounds,
    );
    const parent = new Int32Array(n).fill(-1);
    const centre: number[][] = [];
    for (let k = 0; k < n; ++k) {
      const t = r.lodTree;
      centre.push([
        half(t[k * 4] & 0xffff),
        half(t[k * 4] >>> 16),
        half(t[k * 4 + 1] & 0xffff),
      ]);
      const start = t[k * 4 + 3];
      for (let c = start; c < start + (t[k * 4 + 2] & 0xffff); ++c)
        parent[c] = k;
    }
    const b = r.lodOcclusionBounds;
    const mid = [0, 1, 2].map((k) => (b[k] + b[3 + k]) / 2);
    const run = (eye: number[], cull: boolean) => {
      const m = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, eye[0], eye[1], eye[2], 1];
      const out = wasm.traverse_lod_trees(
        10_000_000,
        1e-4,
        undefined,
        new Uint32Array([lodId]),
        new Uint32Array([0xffffffff]),
        new Float32Array(m),
        new Float32Array([1]),
        new Float32Array([1]),
        new Float32Array([1]),
        new Float32Array([0]),
        new Float32Array([0]),
        undefined,
        undefined,
        cull,
      );
      const { numSplats, indices } = out.instanceIndices[0];
      return {
        cut: Array.from(indices.slice(0, numSplats)) as number[],
        occluded: out.occludedCount as number,
      };
    };
    let total = 0;
    for (const dir of [
      [0, -1, 0.2],
      [0, 1, 0.2],
      [1, 0, 0],
      [-1, 0, 0],
      [0, 0, -1],
      [0.3, 0.3, 1],
    ]) {
      const eye = mid.map((v, k) => v + dir[k] * 3);
      const plain = run(eye, false);
      const culled = run(eye, true);
      // The plain cut less every node with itself or an ancestor hidden.
      const expected = plain.cut.filter((k) => {
        for (let a = k; a >= 0; a = parent[a]) {
          if (
            hides(r.lodOcclusionCodebook, r.lodOcclusion[a], b, centre[a], eye)
          )
            return false;
        }
        return true;
      });
      expect(culled.cut.sort()).toEqual(expected.sort());
      total += culled.occluded;
    }
    expect(total).toBeGreaterThan(0);
    // Inside the asset's box nothing is culled.
    expect(run(mid, true).occluded).toBe(0);
    wasm.dispose_lod_tree(lodId);
  });

  it("reads the codebook from a paged file's tables", () => {
    const prefixBytes = wasm.athc_prefix_bytes(BYTES.subarray(0, 4096));
    const layout = wasm.athc_layout(
      BYTES.subarray(0, prefixBytes),
      BYTES.length,
    );
    expect(layout.occlusionCodebook.length).toBe(2048);
    expect(layout.sections.map((s: { id: string }) => s.id)).toContain("OCCL");
  });
});

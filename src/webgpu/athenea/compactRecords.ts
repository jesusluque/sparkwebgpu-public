import { atheneaAdapterCompact as K } from "../generated/constants";
import type { CompactPieceTables } from "./compactPages";
// A node record of the node pool (thread CT) packed from a CMPC v4 run's
// inflated planes: no three.js, no WebGPU, so the run decoder's worker
// (runWorker.ts, thread CU) imports it too.

/**
 * Bytes a node in a CMPC section, by plane (rust athc_compact PLANE_WIDTHS):
 * v3 (thread CS) appends the cone word (4), the occlusion index (2) and 2
 * spare to v2's 31 planes, at the same offsets.
 */
export const COMPACT_PLANE_WIDTHS = [
  2, 2, 2, 3, 1, 3, 4, 3, 3, 2, 1, 4, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
] as const;

/** Plane k's first byte in a section of n nodes. */
export function compactPlaneOffset(k: number, n: number) {
  let at = 0;
  for (let j = 0; j < k; j++) at += COMPACT_PLANE_WIDTHS[j];
  return at * n;
}

/** Words a node record: the GPU's 40 bytes, then its origin word. */
export const NODE_RECORD_WORDS = 11;

/**
 * The GPU planes (compactPages GPU_PLANES' order, plane k of a CMPC section)
 * and their byte offset in a record: the 4-byte fields first, every 2-byte
 * field at an even byte, so no read crosses more than one word boundary.
 */
export const RECORD_FIELDS: [name: string, plane: number, at: number][] = [
  ["rotation", 6, 0],
  ["weights", 11, 4],
  ["weights1", 12, 8],
  ["weights2", 13, 12],
  ["posX", 0, 16],
  ["posY", 1, 18],
  ["posZ", 2, 20],
  ["material", 9, 22],
  ["shadow", 14, 24],
  ["albedo", 3, 26],
  ["scale", 5, 29],
  ["normal", 7, 32],
  ["curvature", 8, 35],
  ["opacity", 4, 38],
  ["cluster", 10, 39],
];

/** A run's planes (inflated): plane k of a run of m nodes at prefix(k) x m. */
export function runPlaneOffsets(m: number) {
  const offs: number[] = [];
  let at = 0;
  for (let k = 0; k < 15; k++) {
    offs.push(at);
    at += COMPACT_PLANE_WIDTHS[k] * m;
  }
  return offs;
}

/**
 * Records of nodes `locals` (places within run r's nodes) from the run's
 * inflated planes (m nodes), packed in order into `out` from record
 * `first`; `page` their head-table page, `base` the run's first place.
 */
export function packRunRecords(
  planes: Uint8Array,
  m: number,
  locals: ArrayLike<number>,
  page: number,
  base: number,
  out: Uint8Array,
  first: number | ArrayLike<number>,
) {
  const offs = runPlaneOffsets(m);
  const R = NODE_RECORD_WORDS * 4;
  const F = RECORD_FIELDS.length;
  // Each field's source plane start, width, and offset in the record.
  const src = new Int32Array(F);
  const wid = new Int32Array(F);
  const dst = new Int32Array(F);
  for (let f = 0; f < F; f++) {
    const [, plane, off] = RECORD_FIELDS[f];
    src[f] = offs[plane];
    wid[f] = COMPACT_PLANE_WIDTHS[plane];
    dst[f] = off;
  }
  const at0 = typeof first === "number" ? first : -1;
  const pos = typeof first === "number" ? null : first;
  for (let k = 0; k < locals.length; k++) {
    const j = locals[k];
    const at = (pos ? pos[k] : at0 + k) * R;
    for (let f = 0; f < F; f++) {
      const w = wid[f];
      const from = src[f] + j * w;
      const to = at + dst[f];
      out[to] = planes[from];
      if (w > 1) {
        out[to + 1] = planes[from + 1];
        if (w > 2) {
          out[to + 2] = planes[from + 2];
          if (w > 3) out[to + 3] = planes[from + 3];
        }
      }
    }
    const origin = (page + (base + j) * 65536) >>> 0;
    out[at + 40] = origin & 0xff;
    out[at + 41] = (origin >>> 8) & 0xff;
    out[at + 42] = (origin >>> 16) & 0xff;
    out[at + 43] = (origin >>> 24) & 0xff;
  }
}

/**
 * The pieces' bases in one table (thread CX): each distinct basis once, in
 * the order the pieces name them, and each piece's first cluster in it (the
 * shader adds it to the node's cluster; the projections and the bins are by
 * that global cluster). One basis: its own table, every base 0 (as before).
 */
export function compactBases(pieces: CompactPieceTables[]) {
  const distinct: { basis: Uint32Array; clusters: number; base: number }[] = [];
  let clusters = 0;
  const clusterBase = pieces.map((p) => {
    let d = distinct.find(
      (x) =>
        x.clusters === p.clusters &&
        (x.basis === p.basis || sameBasisWords(x.basis, p.basis)),
    );
    if (!d) {
      d = { basis: p.basis, clusters: p.clusters, base: clusters };
      distinct.push(d);
      clusters += p.clusters;
    }
    return d.base;
  });
  if (distinct.length > 1) {
    if (distinct.some((d) => d.clusters % 256 !== 0)) {
      throw new Error(
        "compact bases: several, each of a multiple of 256 clusters",
      );
    }
    if (clusters > COMPACT_MAX_CLUSTERS) {
      throw new Error(
        `compact bases: ${clusters} clusters (${COMPACT_MAX_CLUSTERS} at most)`,
      );
    }
  }
  const table =
    distinct.length === 1
      ? distinct[0].basis
      : (() => {
          const t = new Uint32Array(
            distinct.reduce((n, d) => n + d.basis.length, 0),
          );
          let at = 0;
          for (const d of distinct) {
            t.set(d.basis, at);
            at += d.basis.length;
          }
          return t;
        })();
  return { table, clusters, clusterBase, bases: distinct.length };
}

/**
 * A page head's word 10: its piece's materials (words) and, in the top 8
 * bits, its basis's first cluster / 256 (thread CX; 0 with one basis).
 */
export function headWord10(b: { materials: number; clusterBase: number }) {
  if (b.materials >= 1 << 24) {
    throw new Error("compact: materials past 2^24 words");
  }
  return (b.materials | ((b.clusterBase / 256) << 24)) >>> 0;
}

/** Clusters a pool's bases may hold together (compact.slang kCompactBins). */
export const COMPACT_MAX_CLUSTERS = 2048;

function sameBasisWords(a: Uint32Array, b: Uint32Array) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/** Wall-clock ms (the same in a worker and the page: the load grid's stamps). */
export function gridClockMs() {
  return performance.timeOrigin + performance.now();
}

/** A load grid stamp (6 bits, kCompactGridTicks a second) for wall-clock ms `t`. */
export function gridTick(t: number) {
  return Math.floor((t * K.kCompactGridTicks) / 1000) & 63;
}

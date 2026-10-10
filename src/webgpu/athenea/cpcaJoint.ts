// athenea's joint CPCA of the TX transfer (athenea-research proposal 090,
// inbox/q97: cpca_joint_C256_K8.{bin,json}), kept on the GPU as one cluster
// id and 3 x K int8 weights a gaussian, rebuilt a batch at a time by
// slang/athenea_adapter/cpca_joint.slang for the relight pass
// (atheneaRelightPlugin({ transferResident: "cpca8" })).
//
// The basis (athenea's, read-only): C clusters fitted on the Corvette's
// base_tx.usdc, each with, for the three blocks of the transfer
// (transferDirect 16, transferIndirect 48, transferReflected 48: our
// TXDI, TXIN, TXFD, the 112 values of `transfer` in that order), a mean,
// K orthonormal directions and K scales:
//   encode  z = concat(x_b / energyNorm[b]); c = argmin_j |z - centroid_j|^2;
//           per block w = dir[c]^T (x - mean[c]); q = clamp(round(w / scale[c] x 127), -127, 127)
//   decode  x = mean[c] + dir[c] (q x scale[c] / 127)
//
// The record (`transferQ8`, 7 u32 words, 28 bytes: athenea counts 25):
//   words 2b, 2b + 1   block b's K = 8 weights, int8, weight k in byte k & 3
//                      of word 2b + (k >> 2)
//   word 6             the cluster (0 .. C - 1)
// The GPU table (`packJointBasis`, f32): a cluster every 1008 floats, its
// blocks one after the other, each its mean (D) and then its K directions
// already scaled by scale / 127 (D each), so a value is mean + sum q_k col_k.

import type { AttributeColumn, AttributeSpec } from "../attributes/schema";

export const TRANSFER_Q8 = "transferQ8";
/** Words a record (cpca_joint.slang kJointRecordWords). */
export const Q8_RECORD_WORDS = 7;
/** The transfer's blocks: values each, in the transfer's order. */
export const JOINT_BLOCKS = [16, 48, 48] as const;
/** Weights a block (the record holds 8). */
export const JOINT_K = 8;
/** Floats a cluster in the GPU table (cpca_joint.slang kJointClusterFloats). */
export const JOINT_CLUSTER_FLOATS = JOINT_BLOCKS.reduce(
  (n, d) => n + d * (JOINT_K + 1),
  0,
);
/** Where block b starts in a cluster of the GPU table, floats. */
export const JOINT_BLOCK_FLOATS = [0, 16 * 9, 16 * 9 + 48 * 9];

export interface JointCpcaBasis {
  clusters: number;
  k: number;
  /** [C][112]: the three blocks each over its energyNorm. */
  centroids: Float32Array;
  energyNorm: Float32Array;
  /** Per block: [C][D] means, [C][D][K] directions, [C][K] scales. */
  mean: Float32Array[];
  dir: Float32Array[];
  scale: Float32Array[];
}

interface BasisJson {
  C: number;
  K: number;
  dims: Record<string, number>;
  arrays: {
    name: string;
    shape: number[];
    offset_bytes: number;
    size_bytes: number;
  }[];
}

/** athenea's basis from its .json and .bin (little-endian f32 arrays). */
export function parseJointCpca(
  json: BasisJson,
  bin: Uint8Array,
): JointCpcaBasis {
  if (json.K !== JOINT_K)
    throw new Error(`cpca8: K ${json.K}, the record holds ${JOINT_K}`);
  if (json.C > 65536) throw new Error(`cpca8: ${json.C} clusters`);
  const names = ["transferDirect", "transferIndirect", "transferReflected"];
  names.forEach((n, b) => {
    if (json.dims[n] !== JOINT_BLOCKS[b]) {
      throw new Error(`cpca8: ${n} of ${json.dims[n]} values`);
    }
  });
  const array = (name: string) => {
    const a = json.arrays.find((x) => x.name === name);
    if (!a) throw new Error(`cpca8: no ${name}`);
    const out = new Float32Array(a.size_bytes / 4);
    new Uint8Array(out.buffer).set(
      bin.subarray(a.offset_bytes, a.offset_bytes + a.size_bytes),
    );
    return out;
  };
  return {
    clusters: json.C,
    k: json.K,
    centroids: array("centroids_joint"),
    energyNorm: array("energy_norm"),
    mean: names.map((n) => array(`${n}_mean`)),
    dir: names.map((n) => array(`${n}_basis`)),
    scale: names.map((n) => array(`${n}_scale`)),
  };
}

/** The GPU table: JOINT_CLUSTER_FLOATS a cluster (see the head of the file). */
export function packJointBasis(basis: JointCpcaBasis): Float32Array {
  const C = basis.clusters;
  const K = basis.k;
  const out = new Float32Array(C * JOINT_CLUSTER_FLOATS);
  for (let c = 0; c < C; c++) {
    for (let b = 0; b < 3; b++) {
      const D = JOINT_BLOCKS[b];
      const at = c * JOINT_CLUSTER_FLOATS + JOINT_BLOCK_FLOATS[b];
      out.set(basis.mean[b].subarray(c * D, (c + 1) * D), at);
      for (let k = 0; k < K; k++) {
        const s = Math.fround(basis.scale[b][c * K + k] / 127);
        for (let d = 0; d < D; d++) {
          out[at + (1 + k) * D + d] = basis.dir[b][(c * D + d) * K + k] * s;
        }
      }
    }
  }
  return out;
}

/** The attribute spec of the records. */
export function q8Spec(): AttributeSpec {
  return {
    name: TRANSFER_Q8,
    format: "u32",
    components: Q8_RECORD_WORDS,
    lodMerge: "first",
  };
}

/**
 * Encodes `n` transfers of 112 values (`row(i, out)` fills out with row
 * i's values) into records: athenea's encode (the head of the file).
 */
export function encodeJointCpca(
  basis: JointCpcaBasis,
  n: number,
  row: (i: number, out: Float32Array) => void,
): { records: Uint32Array; clusters: Uint32Array } {
  const C = basis.clusters;
  const K = basis.k;
  const records = new Uint32Array(n * Q8_RECORD_WORDS);
  const clusters = new Uint32Array(C);
  const cent = basis.centroids;
  const norm2 = new Float64Array(C);
  for (let c = 0; c < C; c++) {
    let s = 0;
    for (let v = 0; v < 112; v++) s += cent[c * 112 + v] ** 2;
    norm2[c] = s;
  }
  const inv = [0, 1, 2].map((b) => 1 / basis.energyNorm[b]);
  const x = new Float32Array(112);
  const z = new Float64Array(112);
  for (let i = 0; i < n; i++) {
    row(i, x);
    for (let v = 0; v < 112; v++)
      z[v] = x[v] * inv[v < 16 ? 0 : v < 64 ? 1 : 2];
    let best = 0;
    let bestD = Number.POSITIVE_INFINITY;
    for (let c = 0; c < C; c++) {
      const o = c * 112;
      let dot = 0;
      for (let v = 0; v < 112; v++) dot += z[v] * cent[o + v];
      const d = norm2[c] - 2 * dot;
      if (d < bestD) {
        bestD = d;
        best = c;
      }
    }
    clusters[best]++;
    const r = i * Q8_RECORD_WORDS;
    records[r + 6] = best;
    let first = 0;
    for (let b = 0; b < 3; b++) {
      const D = JOINT_BLOCKS[b];
      const mean = basis.mean[b];
      const dir = basis.dir[b];
      for (let k = 0; k < K; k++) {
        let w = 0;
        for (let d = 0; d < D; d++) {
          w +=
            dir[(best * D + d) * K + k] * (x[first + d] - mean[best * D + d]);
        }
        const s = basis.scale[b][best * K + k];
        const q =
          s > 0 ? Math.max(-127, Math.min(127, Math.round((w / s) * 127))) : 0;
        records[r + 2 * b + (k >> 2)] |= (q & 0xff) << (8 * (k & 3));
      }
      first += D;
    }
  }
  return { records, clusters };
}

/** Weight k of block b of record r, as the int8 it is. */
export function jointWeight(
  records: Uint32Array,
  r: number,
  b: number,
  k: number,
) {
  const w = records[r * Q8_RECORD_WORDS + 2 * b + (k >> 2)];
  return ((w >>> (8 * (k & 3))) << 24) >> 24;
}

/**
 * cpca_joint.slang's rebuild of record r on the CPU: the first `values`
 * (16, 64 or 112) values, as f32 (`halves`: the f16 bits the GPU writes).
 */
export function decodeJointCpca(
  table: Float32Array,
  records: Uint32Array,
  r: number,
  values: number,
): { floats: Float32Array; halves: Uint16Array } {
  const f = Math.fround;
  const floats = new Float32Array(values);
  const halves = new Uint16Array(values);
  const c = records[r * Q8_RECORD_WORDS + 6];
  let first = 0;
  for (let b = 0; b < 3 && first < values; b++) {
    const D = JOINT_BLOCKS[b];
    const at = c * JOINT_CLUSTER_FLOATS + JOINT_BLOCK_FLOATS[b];
    for (let d = 0; d < D; d++) {
      let acc = table[at + d];
      for (let k = 0; k < JOINT_K; k++) {
        acc = f(
          acc + f(jointWeight(records, r, b, k) * table[at + (1 + k) * D + d]),
        );
      }
      floats[first + d] = acc;
      halves[first + d] = floatToHalf(acc);
    }
    first += D;
  }
  return { floats, halves };
}

/** The attribute column of the records. */
export function q8Column(records: Uint32Array): AttributeColumn {
  return { spec: q8Spec(), words: records };
}

/** What a pool keeps beside `transferQ8` (AttribPool.cpcaJoint). */
export interface JointCpcaResident {
  basis: JointCpcaBasis;
  /** packJointBasis(basis), made once. */
  table: Float32Array;
  /** Values of the transfer rebuilt: 16, 64 or 112 (whole blocks). */
  values: number;
}

const F16 = (
  globalThis as unknown as { Float16Array?: Float32ArrayConstructor }
).Float16Array;

/** A float to the nearest half (ties even). */
export function floatToHalf(v: number): number {
  if (F16) {
    const a = new F16(1);
    a[0] = v;
    return new Uint16Array(a.buffer)[0];
  }
  const f32 = new Float32Array([v]);
  const x = new Uint32Array(f32.buffer)[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  const e = exp - 127 + 15;
  if (e >= 31) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - e;
    let h = mant >>> shift;
    const rem = mant & ((1 << shift) - 1);
    const half = 1 << (shift - 1);
    if (rem > half || (rem === half && h & 1)) h += 1;
    return sign | h;
  }
  let h = (e << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && h & 1)) h += 1;
  return sign | h;
}

/** A half's value. */
export function halfToFloat(h: number) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >>> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? Number.NaN : s * Number.POSITIVE_INFINITY;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

/**
 * A basis of athenea's shape (C clusters, K 8) with SYNTHETIC values, for a
 * cloud that carries no 112-value transfer to encode (the published sets'
 * t16): the costs of the real path, not its image. Each cluster's direct
 * mean is an open sky's constant term (an ambient body), every other mean
 * zero, its directions small.
 */
export function syntheticJointBasis(C = 256, seed = 1): JointCpcaBasis {
  let s = seed >>> 0;
  const rand = () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32 - 0.5;
  };
  const mean = JOINT_BLOCKS.map((D) => new Float32Array(C * D));
  for (let c = 0; c < C; c++) mean[0][c * 16] = 0.28209479177387814;
  return {
    clusters: C,
    k: JOINT_K,
    centroids: new Float32Array(C * 112),
    energyNorm: Float32Array.from([1, 1, 1]),
    mean,
    dir: JOINT_BLOCKS.map((D) =>
      Float32Array.from({ length: C * D * JOINT_K }, () => rand() * 0.02),
    ),
    scale: JOINT_BLOCKS.map(() => new Float32Array(C * JOINT_K).fill(1)),
  };
}

/**
 * SYNTHETIC records for `n` gaussians: clusters by runs of 512 (as a cloud's
 * neighbours share a material), weights pseudo-random in [-32, 32].
 */
export function syntheticJointRecords(
  n: number,
  clusters = 256,
  seed = 7,
): Uint32Array {
  const records = new Uint32Array(n * Q8_RECORD_WORDS);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    const r = i * Q8_RECORD_WORDS;
    for (let w = 0; w < 6; w++) {
      let word = 0;
      for (let b = 0; b < 4; b++) {
        s = (Math.imul(s, 1103515245) + 12345) >>> 0;
        word |= ((((s >>> 16) % 65) - 32) & 0xff) << (8 * b);
      }
      records[r + w] = word >>> 0;
    }
    records[r + 6] = (Math.imul((i >>> 9) + 1, 2654435761) >>> 0) % clusters;
  }
  return records;
}

/**
 * Puts athenea's joint CPCA on `pool` in place of its `transfer`: the
 * records (`transferQ8`) and the basis the relight rebuilds them with
 * (atheneaRelightPlugin({ transferResident: "cpca8" })).
 */
export function attachJointCpca(
  pool: {
    count: number;
    removeAttribute(name: string): void;
    setColumn(col: AttributeColumn): void;
    cpcaJoint?: JointCpcaResident;
  },
  resident: JointCpcaResident,
  records: Uint32Array,
) {
  if (records.length !== pool.count * Q8_RECORD_WORDS) {
    throw new Error(
      `cpca8: ${records.length / Q8_RECORD_WORDS} records for ${pool.count} splats`,
    );
  }
  pool.removeAttribute("transfer");
  pool.setColumn(q8Column(records));
  pool.cpcaJoint = resident;
}

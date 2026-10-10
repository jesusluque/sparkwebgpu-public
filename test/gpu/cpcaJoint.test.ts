// athenea's joint CPCA of the transfer (src/webgpu/athenea/cpcaJoint.ts,
// slang/athenea_adapter/cpca_joint.slang): the GPU rebuild of the int8
// records against the CPU's, on a synthetic basis (always), and, opt-in, on
// athenea's own basis and its sample of the Corvette's transfer
// (athenea-research inbox/q97, read-only):
//   CPCA8_DIR=.../inbox/q97/data npx vitest run --config vitest.gpu.config.ts test/gpu/cpcaJoint.test.ts
// which encodes the sample's real transfer (526 134 gaussians of
// base_tx.usdc), checks the clusters against athenea's labels, measures the
// relative energy error per block (athenea's weighting) and rebuilds every
// record on the GPU. Prints CPCA8_SAMPLE {json}.

import { readFileSync } from "node:fs";
import { inflateRawSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import { KernelRegistry } from "../../src/webgpu/KernelRegistry";
import {
  JOINT_BLOCKS,
  type JointCpcaBasis,
  Q8_RECORD_WORDS,
  decodeJointCpca,
  encodeJointCpca,
  halfToFloat,
  packJointBasis,
  parseJointCpca,
  q8Column,
  syntheticJointBasis,
} from "../../src/webgpu/athenea/cpcaJoint";
import { AttribPool } from "../../src/webgpu/attributes/schema";
import jointModule from "../../src/webgpu/generated/athenea_adapter/cpca_joint";
import {
  createReadback,
  createStorage,
  readAndDestroy,
  upload,
} from "../../src/webgpu/gpuBuffers";
import { UniformWriter } from "../../src/webgpu/uniforms";
import { device } from "./device";

const DIR = process.env.CPCA8_DIR ?? "";
const ROW = 65535 * 256;
const grid = (n: number): [number, number] =>
  n <= ROW ? [n, 1] : [ROW, Math.ceil(n / ROW)];

/** Rebuilds every record of `records` on the GPU: the halves, 56 words a row. */
async function gpuDecode(
  d: GPUDevice,
  registry: KernelRegistry,
  table: Float32Array,
  records: Uint32Array,
  n: number,
  batch = 65536,
): Promise<Uint32Array> {
  const pool = new AttribPool(n);
  pool.setColumn(q8Column(records));
  const { words, layout } = pool.pack();
  const attribute = layout.specs.findIndex((s) => s.name === "transferQ8");
  const poolBuffer = upload(d, words, "cpca8 pool");
  const basis = upload(
    d,
    new Uint32Array(table.buffer, table.byteOffset, table.length),
    "cpca8 basis",
  );
  const list = createStorage(d, 16, "cpca8 list");
  const scratch = createStorage(d, batch * 56 * 4, "cpca8 scratch");
  const out = new Uint32Array(n * 56);
  const params = UniformWriter.for(jointModule, "jointParams");
  for (let first = 0; first < n; first += batch) {
    const m = Math.min(batch, n - first);
    registry.get(jointModule, "atheneaJointDecode").run({
      grid: grid(m),
      buffers: {
        jointPool: poolBuffer,
        jointBasis: basis,
        jointList: list,
        jointOut: scratch,
      },
      uniforms: params.setAll({
        count: n,
        listCount: 0,
        listBase: 0,
        threadFirst: first,
        batch: m,
        attribute,
        scratchBase: 0,
        scratchWords: 56,
        blocks: 3,
      }).data,
    });
    const staging = createReadback(d, m * 56 * 4, "cpca8 readback");
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(scratch, 0, staging, 0, m * 56 * 4);
    d.queue.submit([enc.finish()]);
    out.set(new Uint32Array(await readAndDestroy(staging)), first * 56);
  }
  for (const b of [poolBuffer, basis, list, scratch]) b.destroy();
  return out;
}

/** GPU halves against the CPU rebuild: exact, within one ulp, worse. */
function compareHalves(
  table: Float32Array,
  records: Uint32Array,
  gpu: Uint32Array,
  n: number,
) {
  const g16 = new Uint16Array(gpu.buffer);
  let exact = 0;
  let ulp = 0;
  let worse = 0;
  for (let r = 0; r < n; r++) {
    const { halves } = decodeJointCpca(table, records, r, 112);
    for (let v = 0; v < 112; v++) {
      const a = halves[v];
      const b = g16[r * 112 + v];
      if (a === b) exact++;
      else if (
        (Math.abs(a - b) === 1 && (a & 0x8000) === (b & 0x8000)) ||
        // A sum that cancels to about 0: a fused product moves its last bits.
        Math.abs(halfToFloat(a) - halfToFloat(b)) < 2 ** -20
      )
        ulp++;
      else worse++;
    }
  }
  return { exact, ulp, worse };
}

function syntheticBasis(C: number, seed: number): JointCpcaBasis {
  let s = seed >>> 0;
  const rand = () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32 - 0.5;
  };
  const fill = (n: number, k = 1) =>
    Float32Array.from({ length: n }, () => rand() * k);
  return {
    clusters: C,
    k: 8,
    centroids: fill(C * 112),
    energyNorm: Float32Array.from([1, 1, 1]),
    mean: JOINT_BLOCKS.map((D) => fill(C * D)),
    dir: JOINT_BLOCKS.map((D) => fill(C * D * 8, 0.5)),
    scale: JOINT_BLOCKS.map(() => fill(C * 8, 4).map(Math.abs)),
  };
}

describe.skipIf(!device)("joint CPCA (int8) rebuilt on the GPU", () => {
  const d = device as GPUDevice;
  const registry = new KernelRegistry(d);

  it("rebuilds synthetic records as the CPU does", async () => {
    const basis = syntheticBasis(37, 7);
    const table = packJointBasis(basis);
    const n = 70000; // past one batch of 65 536
    const records = new Uint32Array(n * Q8_RECORD_WORDS);
    let s = 99;
    for (let r = 0; r < n; r++) {
      for (let w = 0; w < 6; w++) {
        s = (Math.imul(s, 1103515245) + 12345) >>> 0;
        records[r * Q8_RECORD_WORDS + w] = s;
      }
      records[r * Q8_RECORD_WORDS + 6] = r % basis.clusters;
    }
    const gpu = await gpuDecode(d, registry, table, records, n);
    const c = compareHalves(table, records, gpu, n);
    expect(c.worse).toBe(0);
    expect(c.exact / (n * 112)).toBeGreaterThan(0.99);
  });

  it("projects each cluster on the light as the CPU does", async () => {
    const basis = syntheticJointBasis(256, 5);
    // Directions large enough to matter.
    for (const d of basis.dir) for (let k = 0; k < d.length; k++) d[k] *= 50;
    const table = packJointBasis(basis);
    const envShValues = new Float32Array(16 * 4).map((_, k) =>
      Math.sin(k * 0.7),
    );
    const sun = [0.3, 0.8, 0.5];
    const n = Math.hypot(...sun);
    const envSunValues = new Float32Array([
      sun[0] / n,
      sun[1] / n,
      sun[2] / n,
      0.01,
      2,
      2,
      2,
      0,
    ]);
    const words = (f: Float32Array) =>
      new Uint32Array(f.buffer, f.byteOffset, f.length);
    const tableBase = 64;
    const out = createStorage(
      d,
      (tableBase + 256 * 144) * 4,
      "cpca8 projections",
    );
    const empty = createStorage(d, 16, "empty");
    registry.get(jointModule, "atheneaJointProject").run({
      grid: [256],
      buffers: {
        jointPool: empty,
        jointBasis: upload(d, words(table), "cpca8 basis"),
        jointList: empty,
        jointOut: out,
        envSh: upload(d, words(envShValues), "env sh"),
        envSun: upload(d, words(envSunValues), "env sun"),
      },
      uniforms: UniformWriter.for(jointModule, "jointParams").setAll({
        count: 256,
        dome: 0,
        tableBase,
      }).data,
    });
    const staging = createReadback(d, out.size, "projections readback");
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(out, 0, staging, 0, out.size);
    d.queue.submit([enc.finish()]);
    const got = new Float32Array(await readAndDestroy(staging));
    // The CPU: the same sums over the packed table.
    const Y = (k: number, x: number, y: number, z: number) =>
      [
        0.28209479177387814,
        -0.4886025119029199 * y,
        0.4886025119029199 * z,
        -0.4886025119029199 * x,
        1.0925484305920792 * x * y,
        -1.0925484305920792 * y * z,
        0.31539156525252005 * (2 * z * z - x * x - y * y),
        -1.0925484305920792 * x * z,
        0.5462742152960396 * (x * x - y * y),
        -0.5900435899266435 * y * (3 * x * x - y * y),
        2.890611442640554 * x * y * z,
        -0.4570457994644658 * y * (4 * z * z - x * x - y * y),
        0.3731763325901154 * z * (2 * z * z - 3 * x * x - 3 * y * y),
        -0.4570457994644658 * x * (4 * z * z - x * x - y * y),
        1.445305721320277 * z * (x * x - y * y),
        -0.5900435899266435 * x * (x * x - 3 * y * y),
      ][k];
    let worst = 0;
    for (let c = 0; c < 256; c++) {
      const at = c * 1008;
      for (let j = 0; j < 9; j++) {
        for (let ch = 0; ch < 3; ch++) {
          let body = 0;
          for (let k = 0; k < 16; k++)
            body += envShValues[4 * k + ch] * table[at + j * 16 + k];
          let gathered = 0;
          let along = 0;
          for (let k = 0; k < 16; k++) {
            const v = table[at + 144 + j * 48 + 3 * k + ch];
            gathered += envShValues[4 * k + ch] * v;
            along +=
              Y(k, envSunValues[0], envSunValues[1], envSunValues[2]) * v;
          }
          const white = table[at + 144 + j * 48 + ch];
          const w = tableBase + c * 144 + j * 4 + ch;
          for (const [term, want] of [body, gathered, along, white].entries()) {
            const e =
              Math.abs(got[w + 36 * term] - want) / (Math.abs(want) + 1e-3);
            worst = Math.max(worst, e);
          }
        }
      }
    }
    expect(worst).toBeLessThan(1e-4);
  });

  it("encodes and decodes on the CPU within the quantization step", () => {
    const basis = syntheticBasis(16, 3);
    const table = packJointBasis(basis);
    // Transfers made from the clusters' own space: mean + a few weights.
    const n = 512;
    const truth: Float32Array[] = [];
    for (let i = 0; i < n; i++) {
      const c = i % basis.clusters;
      const x = new Float32Array(112);
      let first = 0;
      JOINT_BLOCKS.forEach((D, b) => {
        for (let dd = 0; dd < D; dd++) {
          let v = basis.mean[b][c * D + dd];
          for (let k = 0; k < 8; k++) {
            const w = basis.scale[b][c * 8 + k] * Math.sin(i + k + b);
            v += basis.dir[b][(c * D + dd) * 8 + k] * w;
          }
          x[first + dd] = v;
        }
        first += D;
      });
      truth.push(x);
    }
    const { records } = encodeJointCpca(basis, n, (i, out) =>
      out.set(truth[i]),
    );
    // The directions are not orthonormal here: only the decode's
    // arithmetic is checked, against the weights the record holds.
    for (let r = 0; r < n; r += 37) {
      const { floats } = decodeJointCpca(table, records, r, 112);
      const c = records[r * Q8_RECORD_WORDS + 6];
      expect(c).toBeLessThan(basis.clusters);
      expect(floats.every(Number.isFinite)).toBe(true);
    }
  });

  it.skipIf(!DIR)(
    "athenea's basis on its sample of the Corvette's transfer",
    async () => {
      const json = JSON.parse(
        readFileSync(`${DIR}/cpca_joint_C256_K8.json`, "utf8"),
      );
      const bin = new Uint8Array(readFileSync(`${DIR}/cpca_joint_C256_K8.bin`));
      const basis = parseJointCpca(json, bin);
      const table = packJointBasis(basis);
      const sample = readNpz(`${DIR}/sample.npz`, [
        "transferDirect",
        "transferIndirect",
        "transferReflected",
        "crypto",
      ]);
      const labels = readNpz(`${DIR}/cpca_joint_C256_K8.npz`, [
        "labels_sample",
      ]).labels_sample;
      const blocks = [
        sample.transferDirect,
        sample.transferIndirect,
        sample.transferReflected,
      ].map((a) => a.data as Float32Array);
      const n = blocks[0].length / 16;
      const t0 = performance.now();
      const { records } = encodeJointCpca(basis, n, (i, out) => {
        out.set(blocks[0].subarray(i * 16, i * 16 + 16), 0);
        out.set(blocks[1].subarray(i * 48, i * 48 + 48), 16);
        out.set(blocks[2].subarray(i * 48, i * 48 + 48), 64);
      });
      const encodeMs = performance.now() - t0;
      let same = 0;
      for (let i = 0; i < n; i++)
        if (records[i * Q8_RECORD_WORDS + 6] === Number(labels.data[i])) same++;

      // athenea's weights (cpca2.py): pieces past 2000 in the cloud were
      // sampled at 7 %.
      const crypto = sample.crypto.data;
      const counts = new Map<number, number>();
      for (let i = 0; i < n; i++)
        counts.set(Number(crypto[i]), (counts.get(Number(crypto[i])) ?? 0) + 1);
      const weight = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const k = counts.get(Number(crypto[i])) as number;
        const full = k < 2000 ? k : k / 0.07;
        weight[i] = full >= 2000 ? 1 / 0.07 : 1;
      }
      const gpu = await gpuDecode(d, registry, table, records, n);
      const g16 = new Uint16Array(gpu.buffer);
      // Energy error per block: sum w |x - x^|^2 / sum w |x|^2, of the
      // GPU's halves (what the relight reads) and of the f32 rebuild.
      const err = JOINT_BLOCKS.map(() => ({ e: 0, e32: 0, r: 0 }));
      for (let i = 0; i < n; i++) {
        const { floats } = decodeJointCpca(table, records, i, 112);
        let first = 0;
        JOINT_BLOCKS.forEach((D, b) => {
          for (let v = 0; v < D; v++) {
            const x = blocks[b][i * D + v];
            const h = halfToFloat(g16[i * 112 + first + v]);
            err[b].e += weight[i] * (x - h) ** 2;
            err[b].e32 += weight[i] * (x - floats[first + v]) ** 2;
            err[b].r += weight[i] * x * x;
          }
          first += D;
        });
      }
      const c = compareHalves(table, records, gpu, n);
      const result = {
        gaussians: n,
        encodeMs: Math.round(encodeMs),
        sameClusterAsAthenea: +(same / n).toFixed(4),
        relEnergyError: Object.fromEntries(
          ["direct", "indirect", "reflected"].map((k, b) => [
            k,
            {
              gpuHalves: +(err[b].e / err[b].r).toExponential(3),
              f32: +(err[b].e32 / err[b].r).toExponential(3),
            },
          ]),
        ),
        gpuVsCpuHalves: c,
      };
      process.stderr.write(`CPCA8_SAMPLE ${JSON.stringify(result)}\n`);
      expect(c.worse).toBe(0);
      expect(same / n).toBeGreaterThan(0.98);
      // athenea's measured error for this basis (cpca2.txt "id conjunto
      // C=256 K=8 8b", todo): 0.0006 / 0.0038 / 0.0087.
      expect(err[2].e / err[2].r).toBeLessThan(0.0087 * 1.2);
    },
    600_000,
  );
});

type Npy = { data: ArrayLike<number | bigint>; shape: number[] };

/** Arrays `names` of a .npz (zip, stored or deflated), as typed arrays. */
function readNpz(path: string, names: string[]): Record<string, Npy> {
  const z = readFileSync(path);
  let eocd = z.length - 22;
  while (eocd >= 0 && z.readUInt32LE(eocd) !== 0x06054b50) eocd--;
  const entries = z.readUInt16LE(eocd + 10);
  let at = z.readUInt32LE(eocd + 16);
  const out: Record<string, Npy> = {};
  for (let e = 0; e < entries; e++) {
    const method = z.readUInt16LE(at + 10);
    const size = z.readUInt32LE(at + 20);
    const nameLen = z.readUInt16LE(at + 28);
    const extraLen = z.readUInt16LE(at + 30);
    const commentLen = z.readUInt16LE(at + 32);
    const local = z.readUInt32LE(at + 42);
    const name = z
      .toString("utf8", at + 46, at + 46 + nameLen)
      .replace(/\.npy$/, "");
    at += 46 + nameLen + extraLen + commentLen;
    if (!names.includes(name)) continue;
    const start =
      local + 30 + z.readUInt16LE(local + 26) + z.readUInt16LE(local + 28);
    const raw = z.subarray(start, start + size);
    const npy = method === 8 ? inflateRawSync(raw) : raw;
    out[name] = parseNpy(Buffer.from(npy));
  }
  for (const n of names) if (!out[n]) throw new Error(`${path}: no ${n}`);
  return out;
}

function parseNpy(b: Buffer): Npy {
  const major = b[6];
  const headerLen = major === 1 ? b.readUInt16LE(8) : b.readUInt32LE(8);
  const start = (major === 1 ? 10 : 12) + headerLen;
  const header = b.toString("latin1", major === 1 ? 10 : 12, start);
  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1] ?? "";
  const shape = (/'shape':\s*\(([^)]*)\)/.exec(header)?.[1] ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number);
  const body = new Uint8Array(b.subarray(start)).slice().buffer;
  const kinds: Record<
    string,
    (buf: ArrayBuffer) => ArrayLike<number | bigint>
  > = {
    "<f4": (x) => new Float32Array(x),
    "<f8": (x) => new Float64Array(x),
    "<i4": (x) => new Int32Array(x),
    "<u4": (x) => new Uint32Array(x),
    "<i8": (x) => new BigInt64Array(x),
    "<u8": (x) => new BigUint64Array(x),
    "|u1": (x) => new Uint8Array(x),
  };
  const make = kinds[descr];
  if (!make) throw new Error(`npy dtype ${descr}`);
  return { data: make(body), shape };
}

// The compact relit path on the Corvette (thread CB, research/iphone-compact.md):
// resident bytes by category, GPU ms by pass, and accuracy against today's
// relight (atheneaRelightPlugin, cpca8), on BY's cuts of athenea's lights
// bake (research/iphone-cost.md §7) with athenea's basis.
//
// Opt-in (assets outside git):
//   IPHONE_COMPACT=cut.athc[:cut-vq.athc][,...] CPCA8_DIR=<athenea q97 data>
//   [IPHONE_SIZES=393x852,1920x1080] [IPHONE_FRAMES=30] [IPHONE_HDR=x_1k.hdr]
//   [IPHONE_ACCURACY=0] npx vitest run --dir test/gpu test/gpu/iphoneCompact.test.ts
// cut-vq.athc: the same cut through athc-convert --shadow-vq (the shadowBits
// dictionary). Prints IPHONE_COMPACT_* {json} lines.

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import * as THREE from "three";
import { HDRLoader } from "three/examples/jsm/loaders/HDRLoader.js";
import { describe, expect, it, vi } from "vitest";
import {
  compactToSource,
  encodeCompact,
} from "../../src/webgpu/athenea/compact";
import {
  Q8_RECORD_WORDS,
  encodeJointCpca,
  halfToFloat,
  packJointBasis,
  parseJointCpca,
} from "../../src/webgpu/athenea/cpcaJoint";
import {
  type Streams,
  compactRun,
  compare,
  placeCamera,
  referenceRun,
  tracker,
} from "./compactHarness";
import { wideDevice } from "./device";

const FILES = (process.env.IPHONE_COMPACT ?? "").split(",").filter(Boolean);
const SIZES = (process.env.IPHONE_SIZES ?? "393x852,1920x1080")
  .split(",")
  .map((s) => s.split("x").map(Number) as [number, number]);
const FRAMES = Number(process.env.IPHONE_FRAMES ?? 30);
const ACCURACY = process.env.IPHONE_ACCURACY !== "0";
const TIMING = process.env.IPHONE_TIMING !== "0";
const DIR = process.env.CPCA8_DIR ?? "";
const HDR =
  process.env.IPHONE_HDR ??
  new URL(
    "../../../publish-r2/sparkwebgpu/corvette-lights-v8-light/hdri/golden_gate_hills_1k.hdr",
    import.meta.url,
  ).pathname;
const TARGET = new THREE.Vector3(0, -0.252, 0.553);
const CAMERA = new THREE.Vector3(4.378, -5.854, 1.676);
const HFOV = 39.6;

type Decoded = {
  numSplats: number;
  ext0: Uint32Array;
  ext1: Uint32Array;
  attribSpecs: { name: string; format: string; components: number }[];
  attribColumns: (Float64Array | Uint32Array)[];
};

async function load(file: string) {
  const wasm = await vi.importActual<typeof import("spark-rs")>("spark-rs");
  wasm.initSync({
    module: readFileSync(
      new URL("../../rust/spark-rs/pkg/spark_rs_bg.wasm", import.meta.url),
    ),
  });
  const bytes = new Uint8Array(readFileSync(file));
  const decoder = wasm.decode_to_extsplats(undefined, "cloud.athc");
  decoder.push(bytes);
  const d = decoder.finish() as Decoded;
  const layout = wasm.athc_layout(bytes, bytes.length) as {
    header: { count: number };
  };
  const all = d.numSplats;
  const n = layout.header.count;
  const first = all - n;
  const col = (name: string, words: number) => {
    const k = d.attribSpecs.findIndex((s) => s.name === name);
    if (k < 0) return null;
    const c = d.attribColumns[k] as Uint32Array;
    return c.slice(first * words, all * words);
  };
  return {
    n,
    ext0: d.ext0.slice(first * 4, all * 4),
    ext1: d.ext1.slice(first * 4, all * 4),
    normalOct: col("normalOct", 1) as Uint32Array,
    pbr: col("pbr", 1) as Uint32Array,
    lobes: col("lobes", 3) as Uint32Array,
    curvature: col("curvature", 2),
    shadowBits: col("shadowBits", 8) as Uint32Array,
    transfer: col("transfer", 56) as Uint32Array,
  };
}

describe.skipIf(!wideDevice || FILES.length === 0)(
  "the compact relit Corvette",
  () => {
    const d = wideDevice as GPUDevice;
    it.each(FILES)(
      "%s",
      async (spec) => {
        const [file, vqFile] = spec.split(":");
        const name = file.split("/").slice(-1)[0];
        const hdr = new HDRLoader()
          .setDataType(THREE.FloatType)
          .parse(readFileSync(HDR).buffer as ArrayBuffer) as {
          width: number;
          height: number;
          data: Float32Array;
        };
        const sky = { width: hdr.width, height: hdr.height, data: hdr.data };
        const raw = await load(file);
        const n = raw.n;
        const json = JSON.parse(
          readFileSync(`${DIR}/cpca_joint_C256_K8.json`, "utf8"),
        );
        const basis = parseJointCpca(
          json,
          new Uint8Array(readFileSync(`${DIR}/cpca_joint_C256_K8.bin`)),
        );
        const levels = Number(process.env.IPHONE_WEIGHT_LEVELS ?? 127);
        // The int8 records (today's cpca8), and the compact path's at
        // `levels` (127 the same; 7: 4-bit weights on the int8 grid).
        const records = (l: number) => {
          const cache = `${file}.q8${l === 127 ? "" : `-${l}`}`;
          if (existsSync(cache)) {
            const b = readFileSync(cache);
            return new Uint32Array(
              b.buffer.slice(b.byteOffset, b.byteOffset + b.length),
            );
          }
          const t16 = new Uint16Array(
            raw.transfer.buffer,
            raw.transfer.byteOffset,
            n * 112,
          );
          const lut = new Float32Array(65536);
          for (let h = 0; h < 65536; h++) lut[h] = halfToFloat(h);
          const r = encodeJointCpca(
            basis,
            n,
            (i, out) => {
              for (let v = 0; v < 112; v++) out[v] = lut[t16[i * 112 + v]];
            },
            l,
          ).records;
          writeFileSync(cache, new Uint8Array(r.buffer));
          return r;
        };
        const q8 = records(127);
        const q8w = levels === 127 ? q8 : records(levels);
        expect(q8.length).toBe(n * Q8_RECORD_WORDS);
        const rawStreams: Streams = { ...raw, q8 };
        let vqStreams: Streams = rawStreams;
        if (vqFile) {
          const vq = await load(vqFile);
          expect(vq.n).toBe(n);
          vqStreams = { ...rawStreams, shadowBits: vq.shadowBits };
        }
        const table = packJointBasis(basis);
        const t0 = performance.now();
        const cloud = encodeCompact({
          count: n,
          ...vqStreams,
          q8: q8w,
          weightBits: levels === 7 ? 4 : 8,
          basis: table,
          clusters: basis.clusters,
          normalBits: Number(process.env.IPHONE_NORMAL_BITS ?? 12),
        });
        const encodeMs = Math.round(performance.now() - t0);
        const rawDictionary = vqFile
          ? encodeCompact({
              count: n,
              ...rawStreams,
              basis: table,
              clusters: basis.clusters,
            }).layout
          : null;
        process.stderr.write(
          `IPHONE_COMPACT_FORMAT ${JSON.stringify({
            file: name,
            gaussians: n,
            encodeMs,
            materials: cloud.layout.materials,
            pieces: cloud.layout.pieces,
            codebookEntries: cloud.layout.codebookEntries,
            rawCodebookEntries: rawDictionary?.codebookEntries,
            rawCodebookBytes: rawDictionary?.bytes.codebooks,
            bytes: cloud.layout.bytes,
            bytesPerGaussian: +(cloud.layout.bytes.total / n).toFixed(2),
          })}\n`,
        );

        const [W0, H0] = SIZES[0];
        if (ACCURACY) {
          const { camera, place } = placeCamera(
            W0,
            H0,
            TARGET,
            CAMERA,
            HFOV,
            2.2,
          );
          const run = {
            W: W0,
            H: H0,
            sky,
            camera,
            place,
            colours: true,
            image: true,
          };
          const refRaw = await referenceRun(d, rawStreams, n, basis, run);
          const refVq = vqFile
            ? await referenceRun(d, vqStreams, n, basis, run)
            : refRaw;
          const deq = compactToSource(cloud, q8w);
          const refDeq = await referenceRun(
            d,
            { ...deq, curvature: rawStreams.curvature ? deq.curvature : null },
            n,
            basis,
            run,
          );
          if (process.env.IPHONE_ABLATE) {
            // Which plane's quantization costs what: the dequantized cloud
            // with one field put back as it was.
            const fields: Record<string, (s: Streams) => void> = {
              position: (s) => {
                for (let i = 0; i < n; i++)
                  for (let k = 0; k < 3; k++)
                    s.ext0[4 * i + k] = vqStreams.ext0[4 * i + k];
              },
              opacity: (s) => {
                for (let i = 0; i < n; i++)
                  s.ext0[4 * i + 3] = vqStreams.ext0[4 * i + 3];
              },
              albedo: (s) => {
                for (let i = 0; i < n; i++) {
                  s.ext1[4 * i] = vqStreams.ext1[4 * i];
                  s.ext1[4 * i + 1] =
                    (s.ext1[4 * i + 1] & 0xffff0000) |
                    (vqStreams.ext1[4 * i + 1] & 0xffff);
                }
              },
              scale: (s) => {
                for (let i = 0; i < n; i++) {
                  s.ext1[4 * i + 1] =
                    (s.ext1[4 * i + 1] & 0xffff) |
                    (vqStreams.ext1[4 * i + 1] & 0xffff0000);
                  s.ext1[4 * i + 2] = vqStreams.ext1[4 * i + 2];
                }
              },
              normal: (s) => s.normalOct.set(vqStreams.normalOct),
              curvature: (s) =>
                (s.curvature as Uint32Array).set(
                  vqStreams.curvature as Uint32Array,
                ),
            };
            const ablation: Record<string, unknown> = {};
            for (const [field, put] of Object.entries(fields)) {
              const s0 = compactToSource(cloud, q8w);
              const s: Streams = { ...s0, curvature: s0.curvature };
              put(s);
              const r = await referenceRun(d, s, n, basis, run);
              ablation[field] = {
                colours: compare(
                  r.colours as Float32Array,
                  refVq.colours as Float32Array,
                ),
                image: compare(
                  r.image as Float32Array,
                  refVq.image as Float32Array,
                ),
              };
            }
            process.stderr.write(
              `IPHONE_COMPACT_ABLATE ${JSON.stringify({ file: name, ablation })}\n`,
            );
          }
          const cullBacks = process.env.IPHONE_CULL_BACKS === "1";
          const got = await compactRun(d, cloud, {
            ...run,
            path: { cullBacks },
          });
          const c = (
            a: { colours: Float32Array | null },
            b: { colours: Float32Array | null },
          ) => compare(a.colours as Float32Array, b.colours as Float32Array);
          const im = (
            a: { image: Float32Array | null },
            b: { image: Float32Array | null },
          ) => compare(a.image as Float32Array, b.image as Float32Array);
          if (process.env.IPHONE_DUMP) {
            for (const [k, v] of Object.entries({ refRaw, refVq, refDeq, got }))
              writeFileSync(
                `${process.env.IPHONE_DUMP}/${name}-${k}.f32`,
                v.image as Float32Array,
              );
          }
          // The draw's record keeps the colour as RGB9E5: its rounding alone.
          const rgb9e5 = (c: Float32Array) => {
            const out = new Float32Array(c.length);
            for (let k = 0; k < c.length; k += 4) {
              const v = [0, 1, 2].map((j) =>
                Math.min(Math.max(c[k + j], 0), 65408),
              );
              const largest = Math.max(...v);
              if (!(largest >= 5.9604645e-8)) continue;
              let e = Math.max(-16, Math.floor(Math.log2(largest))) + 16;
              let step = 2 ** (e - 24);
              if (Math.floor(largest / step + 0.5) >= 512) {
                e += 1;
                step *= 2;
              }
              for (let j = 0; j < 3; j++)
                out[k + j] =
                  Math.min(Math.floor(v[j] / step + 0.5), 511) * step;
            }
            return out;
          };
          process.stderr.write(
            `IPHONE_COMPACT_RGB9E5 ${JSON.stringify(compare(rgb9e5(got.colours as Float32Array), got.colours as Float32Array))}\n`,
          );
          process.stderr.write(
            `IPHONE_COMPACT_ACCURACY ${JSON.stringify({
              file: name,
              size: `${W0}x${H0}`,
              drawn: got.drawn,
              colours: {
                shadowVqVsRaw: vqFile ? c(refVq, refRaw) : null,
                pathVsTodayOnSameValues: c(got, refDeq),
                quantizationVsVq: c(refDeq, refVq),
                compactVsTodayRaw: c(got, refRaw),
              },
              images: {
                shadowVqVsRaw: vqFile ? im(refVq, refRaw) : null,
                pathVsTodayOnSameValues: im(got, refDeq),
                quantizationVsVq: im(refDeq, refVq),
                compactVsTodayRaw: im(got, refRaw),
              },
            })}\n`,
          );
        }

        if (TIMING) {
          for (const [W, H] of SIZES) {
            const { camera, place } = placeCamera(
              W,
              H,
              TARGET,
              CAMERA,
              HFOV,
              2.2,
            );
            const run = {
              W,
              H,
              sky,
              camera,
              place,
              timing: true,
              frames: FRAMES,
            };
            const track = tracker(d);
            try {
              const today =
                process.env.IPHONE_TODAY_TIMING === "0"
                  ? { timing: null, memory: null }
                  : await referenceRun(d, vqStreams, n, basis, run, track);
              const compact = await compactRun(
                d,
                cloud,
                {
                  ...run,
                  fit: true,
                  path: { cullBacks: process.env.IPHONE_CULL_BACKS === "1" },
                },
                track,
              );
              process.stderr.write(
                `IPHONE_COMPACT_COST ${JSON.stringify({
                  file: name,
                  gaussians: n,
                  size: `${W}x${H}`,
                  drawn: compact.drawn,
                  sortCapacity: compact.capacity,
                  compact: { gpuMs: compact.timing, memory: compact.memory },
                  todayCpca8: { gpuMs: today.timing, memory: today.memory },
                })}\n`,
              );
            } finally {
              track.restore();
            }
          }
        }
      },
      3_600_000,
    );
  },
);

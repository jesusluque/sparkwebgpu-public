// CPCA of the transfer (rust athc_cpca.rs, .athc v3 encoding 3) against the
// file it came from, through athenea's own relitSplat on Dawn
// (slang/tests/athenea_relight.slang referenceRelight, athenea's buffer
// layout): every element (merged nodes included) of both files with the same
// streams but the transfer, under a real HDRI (AtheneaSky, the full image)
// and a sun, the relit rgb compared in float.
//
// Opt-in (real assets, outside git):
//   CPCA_PAIRS=orig.athc:cpca.athc[,orig2:cpca2] [CPCA_HDR=x.hdr] [CPCA_MAX=N]
//   npx vitest run --config vitest.gpu.config.ts test/gpu/athcCpcaRelight.test.ts

import { readFileSync } from "node:fs";
import * as THREE from "three";
import { HDRLoader } from "three/examples/jsm/loaders/HDRLoader.js";
import { describe, expect, it, vi } from "vitest";
import { KernelRegistry } from "../../src/webgpu/KernelRegistry";
import { AtheneaSky } from "../../src/webgpu/athenea/AtheneaSky";
import {
  LIGHT_SHADOW,
  distantMatrix,
  packLightRecords,
} from "../../src/webgpu/athenea/lights";
import { AttribPool } from "../../src/webgpu/attributes/schema";
import refModule from "../../src/webgpu/generated/tests/athenea_relight";
import {
  createReadback,
  createStorage,
  readAndDestroy,
  upload,
} from "../../src/webgpu/gpuBuffers";
import { UniformWriter } from "../../src/webgpu/uniforms";
import { wideDevice } from "./device";

// The real WASM (test/unit/setup.ts stubs it), initialized up front.
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

const PAIRS = (process.env.CPCA_PAIRS ?? "")
  .split(",")
  .filter(Boolean)
  .map((p) => p.split(":") as [string, string]);
const HDR =
  process.env.CPCA_HDR ??
  new URL(
    "../../../publish-r2/sparkwebgpu/hdri-4k/golden_gate_hills_4k.hdr",
    import.meta.url,
  ).pathname;
const MAX = Number(process.env.CPCA_MAX ?? 262144);
const NONE = 0xffffffff;

type Decoded = {
  numSplats: number;
  ext0?: Uint32Array;
  ext1?: Uint32Array;
  attribSpecs: { name: string; format: string; components: number }[];
  attribColumns: (Float64Array | Uint32Array)[];
};

const wasm = (await import("spark-rs")) as unknown as {
  default?: (opts?: unknown) => Promise<unknown>;
  decode_to_packedsplats: (...a: unknown[]) => {
    push(b: Uint8Array): void;
    finish(): Decoded;
  };
  decode_to_extsplats: (...a: unknown[]) => {
    push(b: Uint8Array): void;
    finish(): Decoded;
  };
};

function decode(path: string) {
  const bytes = new Uint8Array(readFileSync(path));
  const packed = wasm.decode_to_packedsplats(
    undefined,
    path,
    undefined,
    undefined,
    undefined,
    undefined,
  );
  packed.push(bytes);
  const p = packed.finish();
  const ext = wasm.decode_to_extsplats(
    undefined,
    path,
    undefined,
    undefined,
    undefined,
  );
  ext.push(bytes);
  const e = ext.finish();
  expect(e.numSplats).toBe(p.numSplats);
  const pool = AttribPool.fromValues({
    count: p.numSplats,
    specs: p.attribSpecs as never,
    values: p.attribColumns,
  });
  return { n: p.numSplats, ext: [e.ext0, e.ext1] as Uint32Array[], pool };
}

describe.skipIf(!wideDevice || PAIRS.length === 0)(
  "CPCA transfer through athenea's relitSplat",
  () => {
    const d = wideDevice as GPUDevice;
    const registry = new KernelRegistry(d);

    async function read(buffer: GPUBuffer) {
      const staging = createReadback(d, buffer.size, "cpca readback");
      const enc = d.createCommandEncoder();
      enc.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
      d.queue.submit([enc.finish()]);
      return readAndDestroy(staging);
    }

    it.each(PAIRS)(
      "relights %s and its CPCA alike",
      async (origPath, cpcaPath) => {
        const hdr = new HDRLoader()
          .setDataType(THREE.FloatType)
          .parse(readFileSync(HDR).buffer as ArrayBuffer) as {
          width: number;
          height: number;
          data: Float32Array;
        };
        const sky = new AtheneaSky(d, registry);
        sky.set({
          image: { width: hdr.width, height: hdr.height, data: hdr.data },
        });
        sky.prepareNow();

        const a = decode(origPath);
        const b = decode(cpcaPath);
        expect(b.n).toBe(a.n);
        // Every element, or an even sample of them (merged nodes and splats).
        const n = Math.min(a.n, MAX);
        const pick = (i: number) => Math.floor((i * a.n) / n);
        const spec = (name: string) => a.pool.column(name)?.spec;
        const transferCount = spec("transfer")?.components ?? 0;
        const cells = spec("shadowBits")?.components ?? 0;
        expect(transferCount).toBeGreaterThan(0);
        const tw = Math.ceil(transferCount / 2);

        // The geometry (ext, interleaved a b a splat) and its bounds.
        const src = new Uint32Array(n * 8);
        const f = new Float32Array(a.ext[0].buffer);
        const lo = [1e30, 1e30, 1e30];
        const hi = [-1e30, -1e30, -1e30];
        for (let i = 0; i < n; i++) {
          const j = pick(i);
          src.set(a.ext[0].subarray(4 * j, 4 * j + 4), 8 * i);
          src.set(a.ext[1].subarray(4 * j, 4 * j + 4), 8 * i + 4);
          for (let c = 0; c < 3; c++) {
            lo[c] = Math.min(lo[c], f[4 * j + c]);
            hi[c] = Math.max(hi[c], f[4 * j + c]);
          }
        }
        const centre = new THREE.Vector3(
          (lo[0] + hi[0]) / 2,
          (lo[1] + hi[1]) / 2,
          (lo[2] + hi[2]) / 2,
        );
        const radius = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
        const eye = centre
          .clone()
          .add(
            new THREE.Vector3(0.6, 0.5, 1).normalize().multiplyScalar(radius),
          );

        // athenea's streams, one after the other, from a pool.
        const at = {
          pbr: 0,
          lobes: n,
          normals: 4 * n,
          emission: 5 * n,
          transfer: 6 * n,
          cells: 6 * n + n * tw,
          curvature: 6 * n + n * tw + n * cells,
        };
        const hasCurvature = !!spec("curvature");
        function native(pool: AttribPool) {
          const words = new Uint32Array(at.curvature + 3 * n);
          const { words: packed, layout } = pool.pack();
          const header = packed[2];
          const stride = packed[0];
          const off = (name: string) => {
            const k = layout.specs.findIndex((s) => s.name === name);
            return k < 0 ? -1 : packed[4 + 4 * k];
          };
          const o = {
            pbr: off("pbr"),
            lobes: off("lobes"),
            normals: off("normalOct"),
            emission: off("emission"),
            transfer: off("transfer"),
            cells: off("shadowBits"),
            curvature: off("curvature"),
          };
          const cf = new Float32Array(words.buffer);
          for (let i = 0; i < n; i++) {
            const base = header + pick(i) * stride;
            if (o.pbr >= 0) words[at.pbr + i] = packed[base + o.pbr];
            if (o.lobes >= 0)
              for (let k = 0; k < 3; k++)
                words[at.lobes + 3 * i + k] = packed[base + o.lobes + k];
            if (o.normals >= 0)
              words[at.normals + i] = packed[base + o.normals];
            if (o.emission >= 0)
              words[at.emission + i] = packed[base + o.emission];
            for (let k = 0; k < tw; k++)
              words[at.transfer + i * tw + k] = packed[base + o.transfer + k];
            for (let k = 0; k < cells; k++)
              words[at.cells + i * cells + k] = packed[base + o.cells + k];
            if (o.curvature >= 0) {
              const w0 = packed[base + o.curvature];
              const w1 = packed[base + o.curvature + 1];
              const h = [w0 & 0xffff, w0 >>> 16, w1 & 0xffff];
              h.forEach((v, c) => {
                cf[at.curvature + 3 * i + c] = halfToFloat(v);
              });
            }
          }
          return words;
        }

        const records = packLightRecords([
          sky.record(),
          {
            kind: "distant",
            flags: 2 | LIGHT_SHADOW,
            sizeX: (0.53 * Math.PI) / 180,
            colour: [3, 3, 3],
            matrix: distantMatrix(new THREE.Vector3(0.3, 0.8, 0.5)),
          },
        ]);
        const lights = upload(d, records, "cpca lights");
        const envOfLight = upload(d, new Uint32Array([0, NONE]), "envOfLight");
        const srcBuffer = upload(d, src, "cpca src");

        async function relit(pool: AttribPool) {
          const nat = upload(d, native(pool), "native streams");
          const out = createStorage(d, n * 16, "cpca out");
          registry.get(refModule, "referenceRelight").run({
            grid: [n],
            buffers: {
              src: srcBuffer,
              native: nat,
              lights,
              envSh: sky.envSh,
              envSun: sky.envSun,
              envTexels: sky.envTexels as GPUBuffer,
              envOfLight,
              outColour: out,
              iesRecords: createStorage(d, 32, "ies records"),
              iesValues: createStorage(d, 16, "ies values"),
            },
            uniforms: UniformWriter.for(refModule).setAll({
              count: n,
              transferCount,
              transferWords: tw,
              shadowBits: cells,
              pbrAt: spec("pbr") ? at.pbr : NONE,
              lobesAt: spec("lobes") ? at.lobes : NONE,
              normalsAt: spec("normalOct") ? at.normals : NONE,
              emissionAt: spec("emission") ? at.emission : NONE,
              transferAt: at.transfer,
              cellsAt: cells ? at.cells : NONE,
              lightCount: 2,
              envLights: 1,
              envBaseSide: sky.baseSide,
              transferIndirect: transferCount >= 36 ? 1 : 0,
              litBody: 0,
              linearCloud: 1,
              ior: 1.5,
              curvature: 0,
              perSplatCurvature: hasCurvature ? 1 : 0,
              curvatureAt: at.curvature,
              w0: [1, 0, 0, 0],
              w1: [0, 1, 0, 0],
              w2: [0, 0, 1, 0],
              eyeCloud: [eye.x, eye.y, eye.z, 1],
              eyeWorld: [eye.x, eye.y, eye.z, 1],
            }).data,
          });
          const got = new Float32Array(await read(out));
          nat.destroy();
          out.destroy();
          return got;
        }

        const want = await relit(a.pool);
        const got = await relit(b.pool);
        let e2 = 0;
        let r2 = 0;
        let bad = 0;
        const rel: number[] = [];
        for (let i = 0; i < n; i++) {
          let ei = 0;
          let ri = 0;
          for (let c = 0; c < 3; c++) {
            const w = want[4 * i + c];
            const g = got[4 * i + c];
            if (!Number.isFinite(w) || !Number.isFinite(g)) {
              bad++;
              continue;
            }
            ei += (g - w) ** 2;
            ri += w * w;
          }
          e2 += ei;
          r2 += ri;
          if (ri > 1e-8) rel.push(Math.sqrt(ei / ri));
        }
        rel.sort((x, y) => x - y);
        const q = (p: number) => rel[Math.floor(p * (rel.length - 1))];
        const result = {
          original: origPath,
          cpca: cpcaPath,
          hdr: HDR.split("/").pop(),
          elements: n,
          of: a.n,
          transferCount,
          relMSE: e2 / r2,
          p50: q(0.5),
          p99: q(0.99),
          max: rel[rel.length - 1],
          nonFinite: bad,
        };
        process.stderr.write(`CPCA_RELIT ${JSON.stringify(result)}\n`);
        expect(bad).toBe(0);
        expect(e2 / r2).toBeLessThan(1e-4);
        lights.destroy();
        envOfLight.destroy();
        srcBuffer.destroy();
        sky.destroy();
      },
      600_000,
    );
  },
);

function halfToFloat(h: number) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >>> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? Number.NaN : s * Number.POSITIVE_INFINITY;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

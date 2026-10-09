// The TX transfer kept on the GPU as clustered PCA (relight plugin
// transferResident "cpca": the pool's transferCpca and its cluster table,
// rebuilt per batch by slang/athenea_adapter/cpca.slang) against the WASM
// decoder's halves (transferResident "words"), through the whole relight
// pass: every element of a CPCA .athc (merged nodes and splats), relit
// under a real HDRI and a sun, with the kept terms (viewless) and the eye's
// pass, over every splat and over a LoD list, in batches.
//
// Opt-in (real assets, outside git):
//   CPCA_RESIDENT=a.athc[,b.athc] [CPCA_HDR=x.hdr] [CPCA_BATCH=65536] [CPCA_FRAMES=8]
//   npx vitest run --config vitest.gpu.config.ts test/gpu/athcCpcaResident.test.ts
// Prints CPCA_RESIDENT {json}: relMSE of the relit colour and of the kept
// terms, the transfer's bytes as words and as clusters, and the frame time
// of a relight each frame (the eye moving) both ways.

import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import * as THREE from "three";
import { HDRLoader } from "three/examples/jsm/loaders/HDRLoader.js";
import { describe, expect, it, vi } from "vitest";
import {
  GpuSplatSource,
  WgpuSplatRenderer,
} from "../../src/webgpu/WgpuSplatRenderer";
import {
  TRANSFER_CPCA,
  athcCpcaResident,
} from "../../src/webgpu/athenea/cpcaResident";
import { atheneaRelightPlugin } from "../../src/webgpu/athenea/relightPlugin";
import { AttribPool, attribWords } from "../../src/webgpu/attributes/schema";
import { createReadback, readAndDestroy } from "../../src/webgpu/gpuBuffers";
import { PluginHost } from "../../src/webgpu/plugins";
import { wideDevice } from "./device";

const FILES = (process.env.CPCA_RESIDENT ?? "").split(",").filter(Boolean);
const HDR =
  process.env.CPCA_HDR ??
  new URL(
    "../../../publish-r2/sparkwebgpu/hdri-4k/golden_gate_hills_4k.hdr",
    import.meta.url,
  ).pathname;
const BATCH = Number(process.env.CPCA_BATCH ?? 65536);
const FRAMES = Number(process.env.CPCA_FRAMES ?? 8);
const STRIDE = 5; // relight.slang kRelightStride (uint4s a splat)

type Decoded = {
  numSplats: number;
  ext0: Uint32Array;
  ext1: Uint32Array;
  attribSpecs: {
    name: string;
    format: string;
    components: number;
    packed?: boolean;
  }[];
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
  const layout = wasm.athc_layout(bytes, bytes.length) as never;
  return { bytes, d, layout };
}

describe.skipIf(!wideDevice || FILES.length === 0)(
  "the transfer resident as clusters, through the relight pass",
  () => {
    const d = wideDevice as GPUDevice;
    const SIZE = 256;
    const color = d.createTexture({
      size: [SIZE, SIZE],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const target = {
      texture: { colorSpace: THREE.LinearSRGBColorSpace },
      samples: 0,
    } as unknown as THREE.RenderTarget;
    const fakeRenderer = {
      backend: {
        isWebGPUBackend: true,
        device: d,
        context: {} as GPUCanvasContext,
        get: (r: object) =>
          r === target.texture ? { texture: color } : undefined,
      },
    };

    async function read(buffer: GPUBuffer) {
      const staging = createReadback(d, buffer.size, "cpca readback");
      const enc = d.createCommandEncoder();
      enc.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
      d.queue.submit([enc.finish()]);
      return new Float32Array(await readAndDestroy(staging));
    }

    it.each(FILES)(
      "relights %s alike from its words and from its clusters",
      async (file) => {
        const hdr = new HDRLoader()
          .setDataType(THREE.FloatType)
          .parse(readFileSync(HDR).buffer as ArrayBuffer) as {
          width: number;
          height: number;
          data: Float32Array;
        };
        const sky = { width: hdr.width, height: hdr.height, data: hdr.data };
        const { bytes, d: dec, layout } = await load(file);
        const n = dec.numSplats;
        const words = AttribPool.fromValues({
          count: n,
          specs: dec.attribSpecs as never,
          values: dec.attribColumns,
        });
        const transfer = words.column("transfer");
        expect(transfer).toBeTruthy();
        const values = transfer?.spec.components ?? 0;
        const t0 = performance.now();
        const { table, column } = await athcCpcaResident(bytes, layout, {
          rows: n,
          values,
          transfer: transfer?.words,
          gunzip: async (b) => new Uint8Array(gunzipSync(b)),
        });
        const buildMs = performance.now() - t0;
        // The records rebuild the decoder's halves (CPU, a sample).
        const tw = attribWords(transfer?.spec as never);
        let mismatched = 0;
        let checked = 0;
        for (let r = 0; r < n; r += Math.max(1, Math.floor(n / 20000))) {
          const got = table.rebuild(column.words, r);
          const want = (transfer?.words as Uint32Array).subarray(
            r * tw,
            (r + 1) * tw,
          );
          for (let k = 0; k < tw; k++) if (got[k] !== want[k]) mismatched++;
          checked += tw;
        }
        const clusters = AttribPool.fromValues({
          count: n,
          specs: dec.attribSpecs as never,
          values: dec.attribColumns,
        });
        clusters.removeAttribute("transfer");
        clusters.setColumn(column);
        clusters.cpca = table;

        // The cloud's bounds and an eye on a sphere around it.
        const f = new Float32Array(dec.ext0.buffer, dec.ext0.byteOffset);
        const lo = [1e30, 1e30, 1e30];
        const hi = [-1e30, -1e30, -1e30];
        for (let i = 0; i < n; i++) {
          for (let c = 0; c < 3; c++) {
            lo[c] = Math.min(lo[c], f[4 * i + c]);
            hi[c] = Math.max(hi[c], f[4 * i + c]);
          }
        }
        const centre = new THREE.Vector3(
          (lo[0] + hi[0]) / 2,
          (lo[1] + hi[1]) / 2,
          (lo[2] + hi[2]) / 2,
        );
        const radius = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
        const camera = new THREE.PerspectiveCamera(
          50,
          1,
          radius / 100,
          radius * 10,
        );
        camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
        const place = (a: number) => {
          camera.position
            .set(Math.cos(a), 0.5, Math.sin(a))
            .normalize()
            .multiplyScalar(1.5 * radius)
            .add(centre);
          camera.lookAt(centre);
          camera.updateProjectionMatrix();
          camera.updateMatrixWorld();
        };

        async function run(
          pool: AttribPool,
          resident: "words" | "cpca",
          list: Uint32Array | null,
        ) {
          const source = GpuSplatSource.fromExt(d, dec.ext0, dec.ext1, n);
          source.attribs = pool;
          const splats = new WgpuSplatRenderer(fakeRenderer as never, {
            depthTest: false,
            alwaysGenerate: true,
          });
          const host = new PluginHost({
            capabilities: splats.capabilities,
            tier: 2,
          });
          const relight = atheneaRelightPlugin({
            hdri: sky,
            sun: { direction: [0.3, 0.8, 0.5], intensity: 3 },
            transferResident: resident,
            cpcaBatch: BATCH,
          });
          host.register(relight).attach(splats);
          const mesh = splats.add(source);
          if (list) splats.setLodIndices(mesh, list);
          relight.setIor(mesh, 1.5);
          await host.ready();
          place(0.4);
          splats.render(camera, target);
          await d.queue.onSubmittedWorkDone();
          const relit = relight.relitBufferOf(mesh) as GPUBuffer;
          const out = await read(relit);
          // A relight each frame: the eye moves.
          const times: number[] = [];
          for (let k = 0; k < FRAMES; k++) {
            place(0.4 + 0.01 * (k + 1));
            const s = performance.now();
            splats.render(camera, target);
            await d.queue.onSubmittedWorkDone();
            times.push(performance.now() - s);
          }
          times.sort((a, b) => a - b);
          const stats = { ...relight.stats };
          host.detach();
          splats.dispose();
          return {
            out,
            frameMs: times[Math.floor(times.length / 2)],
            stats,
          };
        }

        function compare(
          a: Float32Array,
          b: Float32Array,
          field: number,
          rows: Iterable<number>,
        ) {
          let e2 = 0;
          let r2 = 0;
          let bad = 0;
          let same = 0;
          let count = 0;
          const u = new Uint32Array(a.buffer);
          const v = new Uint32Array(b.buffer);
          for (const i of rows) {
            const at = (i * STRIDE + field) * 4;
            count++;
            let exact = true;
            for (let c = 0; c < 4; c++)
              if (u[at + c] !== v[at + c]) exact = false;
            if (exact) same++;
            if (field === 0) {
              for (let c = 0; c < 3; c++) {
                const w = a[at + c];
                const g = b[at + c];
                if (!Number.isFinite(w) || !Number.isFinite(g)) {
                  bad++;
                  continue;
                }
                e2 += (g - w) ** 2;
                r2 += w * w;
              }
            } else {
              // The kept terms: halves (front rgb, coupling rgb, sun share).
              const h = (x: number) => {
                const s = x & 0x8000 ? -1 : 1;
                const e = (x >>> 10) & 0x1f;
                const m = x & 0x3ff;
                return e === 0
                  ? s * m * 2 ** -24
                  : s * (1 + m / 1024) * 2 ** (e - 15);
              };
              for (let c = 0; c < 4; c++) {
                for (const sh of [0, 16]) {
                  if (c === 3 && sh === 16) continue;
                  const w = h((u[at + c] >>> sh) & 0xffff);
                  const g = h((v[at + c] >>> sh) & 0xffff);
                  e2 += (g - w) ** 2;
                  r2 += w * w;
                }
              }
            }
          }
          return {
            relMSE: r2 > 0 ? e2 / r2 : 0,
            identical: same / count,
            nonFinite: bad,
          };
        }

        const all = {
          [Symbol.iterator]: function* () {
            for (let i = 0; i < n; i++) yield i;
          },
        };
        const a = await run(words, "words", null);
        const b = await run(clusters, "cpca", null);
        // A LoD list: every third splat, from the far end.
        const list = new Uint32Array(Math.floor(n / 3));
        for (let k = 0; k < list.length; k++) list[k] = n - 1 - 3 * k;
        const al = await run(words, "words", list);
        const bl = await run(clusters, "cpca", list);
        const transferBytes = n * tw * 4;
        const cpcaBytes = n * table.recordWords * 4 + table.bytes;
        const result = {
          file: file.split("/").slice(-2).join("/"),
          splats: n,
          values,
          sections: table.sections.map(
            (s) => `${s.id}:${s.cpca ? `cpca M${s.coeffs}` : "plain"}`,
          ),
          entries: table.entries,
          buildMs: Math.round(buildMs),
          cpuMismatchedHalves: mismatched,
          cpuChecked: checked,
          relit: compare(a.out, b.out, 0, all),
          kept: compare(a.out, b.out, 1, all),
          listRelit: compare(al.out, bl.out, 0, list),
          transferMB: +(transferBytes / 1e6).toFixed(2),
          cpcaRecordsMB: +((n * table.recordWords * 4) / 1e6).toFixed(2),
          cpcaTableMB: +(table.bytes / 1e6).toFixed(3),
          cpcaVsWords: +(cpcaBytes / transferBytes).toFixed(3),
          bytesPerSplat: { words: tw * 4, cpca: table.recordWords * 4 },
          frameMs: {
            words: a.frameMs,
            cpca: b.frameMs,
            listWords: al.frameMs,
            listCpca: bl.frameMs,
          },
          batches: b.stats.cpca,
        };
        process.stderr.write(`CPCA_RESIDENT ${JSON.stringify(result)}\n`);
        expect(mismatched).toBe(0);
        expect(result.relit.nonFinite).toBe(0);
        expect(result.relit.relMSE).toBeLessThan(1e-6);
        expect(result.kept.relMSE).toBeLessThan(1e-6);
        expect(result.listRelit.relMSE).toBeLessThan(1e-6);
      },
      1_200_000,
    );
  },
);

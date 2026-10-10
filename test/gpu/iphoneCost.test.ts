// Cost of the relit Corvette on WebGPU (Dawn on this Mac's Metal), as a
// proxy for an iPhone (thread BY, research/iphone-cost.md): GPU frame time
// and resident GPU memory by category for each way of keeping the TX
// transfer:
//   words16   the published sets' t16 transfer (its 16 direct values, halves)
//   words112  the whole TX transfer as halves (224 B a gaussian)
//   cpca      BP's GPU-resident file CPCA (.athc v3 encoding 3; a -pc file)
//   cpca8     athenea's joint CPCA (proposal 090): 1 cluster + 24 int8 a
//             gaussian and a shared 1 MB basis (src/webgpu/athenea/cpcaJoint.ts)
// Each file is a cut of the Corvette (athc-convert --keep-splats N --error);
// only its leaves are loaded (the gaussians the cut keeps, every one drawn
// and relit: the LoD levels above them are counted apart).
//
// Opt-in (assets outside git):
//   IPHONE_COST=cut.athc[:cut-pc.athc][,...] CPCA8_DIR=<athenea q97 data>
//   [IPHONE_SIZES=393x852,1920x1080] [IPHONE_MODES=words16,words112,cpca,cpca8]
//   [IPHONE_FRAMES=30] [IPHONE_HDR=x_1k.hdr] [IPHONE_BATCH=65536]
//   npx vitest run --dir test/gpu test/gpu/iphoneCost.test.ts
// Prints IPHONE_COST {json} a file, mode and size.

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import * as THREE from "three";
import { HDRLoader } from "three/examples/jsm/loaders/HDRLoader.js";
import { describe, expect, it, vi } from "vitest";
import { GpuSplatSource } from "../../src/webgpu/GpuSplatSource";
import { WgpuSplatRenderer } from "../../src/webgpu/WgpuSplatRenderer";
import {
  JOINT_BLOCKS,
  Q8_RECORD_WORDS,
  TRANSFER_Q8,
  decodeJointCpca,
  encodeJointCpca,
  halfToFloat,
  packJointBasis,
  parseJointCpca,
  q8Column,
} from "../../src/webgpu/athenea/cpcaJoint";
import {
  TRANSFER_CPCA,
  athcCpcaResident,
} from "../../src/webgpu/athenea/cpcaResident";
import {
  type TransferResident,
  atheneaRelightPlugin,
} from "../../src/webgpu/athenea/relightPlugin";
import { AttribPool, attribWords } from "../../src/webgpu/attributes/schema";
import { createReadback, readAndDestroy } from "../../src/webgpu/gpuBuffers";
import { PluginHost } from "../../src/webgpu/plugins";
import { wideDevice } from "./device";

const FILES = (process.env.IPHONE_COST ?? "").split(",").filter(Boolean);
const SIZES = (process.env.IPHONE_SIZES ?? "393x852,1920x1080")
  .split(",")
  .map((s) => s.split("x").map(Number) as [number, number]);
const MODES = (process.env.IPHONE_MODES ?? "words16,words112,cpca,cpca8").split(
  ",",
);
const FRAMES = Number(process.env.IPHONE_FRAMES ?? 30);
const BATCH = Number(process.env.IPHONE_BATCH ?? 65536);
const DIR = process.env.CPCA8_DIR ?? "";
/** The renderer's rasterizer ("hardware" by default, "tiles", "auto"). */
const RASTER = process.env.IPHONE_RASTER ?? "";
const HDR =
  process.env.IPHONE_HDR ??
  new URL(
    "../../../publish-r2/sparkwebgpu/corvette-lights-v8-light/hdri/golden_gate_hills_1k.hdr",
    import.meta.url,
  ).pathname;

// athenea's /World/Camera in its Z-up stage (the clouds' frame), and the
// light page's default distance (2.2 x).
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

async function wasmModule() {
  const wasm = await vi.importActual<typeof import("spark-rs")>("spark-rs");
  wasm.initSync({
    module: readFileSync(
      new URL("../../rust/spark-rs/pkg/spark_rs_bg.wasm", import.meta.url),
    ),
  });
  return wasm;
}

async function load(file: string) {
  const wasm = await wasmModule();
  const bytes = new Uint8Array(readFileSync(file));
  const decoder = wasm.decode_to_extsplats(undefined, "cloud.athc");
  decoder.push(bytes);
  const d = decoder.finish() as Decoded;
  const layout = wasm.athc_layout(bytes, bytes.length) as {
    header: { count: number };
  };
  return { bytes, d, layout };
}

/** Rows first .. first + n - 1 of a pool. */
function slicePool(pool: AttribPool, first: number, n: number) {
  const out = new AttribPool(n);
  for (const c of pool.columns) {
    const w = attribWords(c.spec);
    out.setColumn({
      spec: { ...c.spec },
      words: c.words.slice(first * w, (first + n) * w),
    });
  }
  return out;
}

/** GPU buffers and textures made while `fn` runs, alive at its end. */
function tracker(d: GPUDevice) {
  const live = new Map<object, { label: string; bytes: number }>();
  const createBuffer = d.createBuffer.bind(d);
  const createTexture = d.createTexture.bind(d);
  d.createBuffer = (desc: GPUBufferDescriptor) => {
    const b = createBuffer(desc);
    live.set(b, { label: desc.label ?? "", bytes: desc.size });
    const destroy = b.destroy.bind(b);
    b.destroy = () => {
      live.delete(b);
      destroy();
    };
    return b;
  };
  d.createTexture = (desc: GPUTextureDescriptor) => {
    const t = createTexture(desc);
    const size = desc.size as number[];
    const px = (size[0] ?? 1) * (size[1] ?? 1) * (size[2] ?? 1);
    const fmt = String(desc.format);
    const bpp = /32float/.test(fmt)
      ? 4 * (fmt.startsWith("rgba") ? 4 : fmt.startsWith("rg") ? 2 : 1)
      : /16float/.test(fmt)
        ? 2 * (fmt.startsWith("rgba") ? 4 : fmt.startsWith("rg") ? 2 : 1)
        : 4;
    live.set(t, {
      label: `texture ${desc.label ?? fmt}`,
      bytes: px * bpp * (desc.sampleCount ?? 1),
    });
    const destroy = t.destroy.bind(t);
    t.destroy = () => {
      live.delete(t);
      destroy();
    };
    return t;
  };
  return {
    snapshot() {
      const by = new Map<string, number>();
      for (const { label, bytes } of live.values())
        by.set(label, (by.get(label) ?? 0) + bytes);
      return Object.fromEntries(
        [...by].sort((a, b) => b[1] - a[1]).map(([k, v]) => [k, v]),
      );
    },
    reset() {
      live.clear();
    },
    restore() {
      d.createBuffer = createBuffer;
      d.createTexture = createTexture;
    },
  };
}

describe.skipIf(!wideDevice || FILES.length === 0)(
  "the relit Corvette's cost by transfer layout",
  () => {
    const d = wideDevice as GPUDevice;

    it.each(FILES)(
      "%s",
      async (spec) => {
        const [file, pcFile] = spec.split(":");
        const hdr = new HDRLoader()
          .setDataType(THREE.FloatType)
          .parse(readFileSync(HDR).buffer as ArrayBuffer) as {
          width: number;
          height: number;
          data: Float32Array;
        };
        const sky = { width: hdr.width, height: hdr.height, data: hdr.data };
        const { bytes, d: dec, layout } = await load(file);
        const all = dec.numSplats;
        const leaves = layout.header.count;
        const first = all - leaves;
        const full = AttribPool.fromValues({
          count: all,
          specs: dec.attribSpecs as never,
          values: dec.attribColumns,
        });
        const words112 = slicePool(full, first, leaves);
        const transfer = words112.column("transfer");
        expect(transfer?.spec.components).toBe(112);
        const ext0 = dec.ext0.slice(first * 4, all * 4);
        const ext1 = dec.ext1.slice(first * 4, all * 4);

        // t16: the first 16 values (8 words) of each row.
        const words16 = slicePool(full, first, leaves);
        {
          const t = words16.column("transfer");
          const src = t?.words as Uint32Array;
          const w = new Uint32Array(leaves * 8);
          for (let i = 0; i < leaves; i++)
            w.set(src.subarray(i * 56, i * 56 + 8), i * 8);
          words16.removeAttribute("transfer");
          words16.setColumn({
            spec: { ...(t?.spec as never), components: 16 },
            words: w,
          });
        }

        // athenea's joint CPCA of the leaves' transfer (cached beside the file).
        const json = JSON.parse(
          readFileSync(`${DIR}/cpca_joint_C256_K8.json`, "utf8"),
        );
        const basis = parseJointCpca(
          json,
          new Uint8Array(readFileSync(`${DIR}/cpca_joint_C256_K8.bin`)),
        );
        const table = packJointBasis(basis);
        const tw = transfer?.words as Uint32Array;
        const t16 = new Uint16Array(tw.buffer, tw.byteOffset, leaves * 112);
        const lut = new Float32Array(65536);
        for (let h = 0; h < 65536; h++) lut[h] = halfToFloat(h);
        const cache = `${file}.q8`;
        let records: Uint32Array;
        let encodeMs = 0;
        if (existsSync(cache)) {
          const b = readFileSync(cache);
          records = new Uint32Array(
            b.buffer.slice(b.byteOffset, b.byteOffset + b.length),
          );
        } else {
          const t0 = performance.now();
          records = encodeJointCpca(basis, leaves, (i, out) => {
            for (let v = 0; v < 112; v++) out[v] = lut[t16[i * 112 + v]];
          }).records;
          encodeMs = performance.now() - t0;
          writeFileSync(cache, new Uint8Array(records.buffer));
        }
        expect(records.length).toBe(leaves * Q8_RECORD_WORDS);
        // The reconstruction against the decoded transfer (its halves).
        const err = JOINT_BLOCKS.map(() => ({ e: 0, r: 0 }));
        const step = Math.max(1, Math.floor(leaves / 100000));
        for (let i = 0; i < leaves; i += step) {
          const { floats } = decodeJointCpca(table, records, i, 112);
          let at = 0;
          JOINT_BLOCKS.forEach((D, b) => {
            for (let v = 0; v < D; v++) {
              const x = lut[t16[i * 112 + at + v]];
              err[b].e += (x - floats[at + v]) ** 2;
              err[b].r += x * x;
            }
            at += D;
          });
        }
        const cpca8 = slicePool(full, first, leaves);
        cpca8.removeAttribute("transfer");
        cpca8.setColumn(q8Column(records));
        cpca8.cpcaJoint = { basis, table, values: 112 };

        // BP's file CPCA (a -pc file of the same cut), its leaves' records.
        let cpca: AttribPool | null = null;
        if (pcFile) {
          const pc = await load(pcFile);
          expect(pc.d.numSplats).toBe(all);
          const { table: pcTable, column } = await athcCpcaResident(
            pc.bytes,
            pc.layout as never,
            {
              rows: all,
              values: 112,
              transfer: full.column("transfer")?.words as Uint32Array,
              gunzip: async (b) => new Uint8Array(gunzipSync(b)),
            },
          );
          cpca = slicePool(full, first, leaves);
          cpca.removeAttribute("transfer");
          const rw = attribWords(column.spec);
          cpca.setColumn({
            spec: column.spec,
            words: column.words.slice(first * rw, all * rw),
          });
          cpca.cpca = pcTable;
        }
        void bytes;

        const pools: Record<string, [AttribPool | null, TransferResident]> = {
          words16: [words16, "words"],
          words112: [words112, "words"],
          cpca: [cpca, "cpca"],
          cpca8: [cpca8, "cpca8"],
        };
        const track = tracker(d);
        let wantColours = false;
        try {
          const colourOf: Record<string, Float32Array> = {};
          for (const [W, H] of SIZES) {
            for (const mode of MODES) {
              const [pool, resident] = pools[mode] ?? [null, "words"];
              if (!pool) continue;
              track.reset();
              wantColours = W === SIZES[0][0] && H === SIZES[0][1];
              const {
                _memory: memory,
                _colours: colours,
                ...result
              } = await run(pool, resident, W, H);
              const out = {
                file: file.split("/").slice(-1)[0],
                gaussians: leaves,
                lodLevelsAbove: first,
                mode,
                size: `${W}x${H}`,
                batch: BATCH,
                rasterizer: RASTER || "hardware",
                ...result,
                memory,
                ...(mode === "cpca8"
                  ? {
                      encodeMs: Math.round(encodeMs),
                      relEnergyErrorVsDecoded: {
                        direct: +(err[0].e / err[0].r).toExponential(3),
                        indirect: +(err[1].e / err[1].r).toExponential(3),
                        reflected: +(err[2].e / err[2].r).toExponential(3),
                      },
                    }
                  : {}),
                poolColumns: Object.fromEntries(
                  pool.columns.map((c) => [
                    c.spec.name,
                    attribWords(c.spec) * 4,
                  ]),
                ),
                cpcaTableBytes: mode === "cpca" ? pool.cpca?.bytes : undefined,
              };
              if (colours) colourOf[mode] = colours;
              process.stderr.write(`IPHONE_COST ${JSON.stringify(out)}\n`);
            }
          }
          // Per-splat relit colour against the whole transfer as halves.
          const ref = colourOf.words112;
          if (ref) {
            const vs: Record<string, unknown> = {};
            for (const [mode, c] of Object.entries(colourOf)) {
              if (mode === "words112") continue;
              let e = 0;
              let r = 0;
              let gotSum = 0;
              let refSum = 0;
              for (let k = 0; k < ref.length; k++) {
                if (!Number.isFinite(ref[k]) || !Number.isFinite(c[k]))
                  continue;
                e += (c[k] - ref[k]) ** 2;
                r += ref[k] ** 2;
                gotSum += c[k];
                refSum += ref[k];
              }
              vs[mode] = {
                relMSE: +(e / r).toExponential(3),
                ratio: +(gotSum / refSum).toFixed(4),
              };
            }
            process.stderr.write(
              `IPHONE_RELIT ${JSON.stringify({ file: file.split("/").slice(-1)[0], gaussians: leaves, vsWords112: vs })}\n`,
            );
          }
        } finally {
          track.restore();
        }

        async function run(
          pool: AttribPool,
          resident: TransferResident,
          W: number,
          H: number,
        ) {
          const color = d.createTexture({
            size: [W, H],
            format: "rgba16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
            label: "target",
          });
          const target = {
            texture: { colorSpace: THREE.LinearSRGBColorSpace },
            samples: 0,
            width: W,
            height: H,
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
          const source = GpuSplatSource.fromExt(d, ext0, ext1, leaves);
          source.attribs = pool;
          const splats = new WgpuSplatRenderer(fakeRenderer as never, {
            depthTest: false,
            alwaysGenerate: true,
            ...(RASTER ? { rasterizer: RASTER as never } : {}),
          });
          const host = new PluginHost({
            capabilities: splats.capabilities,
            tier: 2,
          });
          const relight = atheneaRelightPlugin({
            hdri: sky,
            transferResident: resident,
            cpcaBatch: BATCH,
          });
          host.register(relight).attach(splats);
          const mesh = splats.add(source);
          await host.ready();
          const aspect = W / H;
          const vfov =
            (2 * Math.atan(Math.tan((HFOV * Math.PI) / 360) / aspect) * 180) /
            Math.PI;
          const camera = new THREE.PerspectiveCamera(vfov, aspect, 0.05, 100);
          camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
          camera.up.set(0, 0, 1);
          const away = CAMERA.clone().sub(TARGET).multiplyScalar(2.2);
          const place = (deg: number) => {
            camera.position
              .copy(away)
              .applyAxisAngle(new THREE.Vector3(0, 0, 1), (deg * Math.PI) / 180)
              .add(TARGET);
            camera.lookAt(TARGET);
            camera.updateProjectionMatrix();
            camera.updateMatrixWorld();
          };
          const frames = async (moving: boolean) => {
            const ms: number[] = [];
            for (let k = 0; k < FRAMES; k++) {
              if (moving) place(0.2 * (k + 1));
              const s = performance.now();
              splats.render(camera, target);
              await d.queue.onSubmittedWorkDone();
              ms.push(performance.now() - s);
            }
            ms.sort((a, b) => a - b);
            return +ms[Math.floor(ms.length / 2)].toFixed(2);
          };
          place(0);
          for (let k = 0; k < 6; k++) {
            splats.render(camera, target);
            await d.queue.onSubmittedWorkDone();
          }
          // The relit colours of the first view (stride 5 uint4 a splat).
          let colours: Float32Array | null = null;
          if (wantColours) {
            const relit = relight.relitBufferOf(mesh) as GPUBuffer;
            const staging = createReadback(d, relit.size, "relit readback");
            const enc = d.createCommandEncoder();
            enc.copyBufferToBuffer(relit, 0, staging, 0, relit.size);
            d.queue.submit([enc.finish()]);
            const all = new Float32Array(await readAndDestroy(staging));
            colours = new Float32Array(leaves * 3);
            for (let i = 0; i < leaves; i++)
              for (let c = 0; c < 3; c++) colours[3 * i + c] = all[i * 20 + c];
          }
          const relitBefore = relight.stats.relit;
          const moving = await frames(true);
          const relitFrames = relight.stats.relit - relitBefore;
          const stillBefore = relight.stats.relit;
          const still = await frames(false);
          const stillRelit = relight.stats.relit - stillBefore;
          // GPU time per pass (timestamps), relighting and still: the
          // median of each label over the frames read back.
          splats.options.profile = true;
          splats.render(camera, target);
          await d.queue.onSubmittedWorkDone();
          const profiler = (
            splats as unknown as {
              profiler: {
                onFrame: ((f: Record<string, number>) => void) | null;
              } | null;
            }
          ).profiler;
          const timed = async (moving: boolean) => {
            const got: Record<string, number>[] = [];
            if (profiler) profiler.onFrame = (f) => got.push({ ...f });
            await frames(moving);
            // The last readbacks land after their frames.
            for (let k = 0; k < 4; k++) {
              await d.queue.onSubmittedWorkDone();
              await new Promise((r) => setTimeout(r, 5));
            }
            if (profiler) profiler.onFrame = null;
            const labels = new Set(got.flatMap((f) => Object.keys(f)));
            const out: Record<string, number> = {};
            for (const l of labels) {
              const v = got.map((f) => f[l] ?? 0).sort((a, b) => a - b);
              out[l] = v[Math.floor(v.length / 2)];
            }
            out.frames = got.length;
            return out;
          };
          const gpuMoving = await timed(true);
          const gpuStill = await timed(false);
          splats.options.profile = false;
          const round = (o: Record<string, number>) =>
            Object.fromEntries(
              Object.entries(o).map(([k, v]) => [k, +v.toFixed(3)]),
            );
          const result = {
            wallMs: { relighting: moving, still },
            relitFrames: { relighting: relitFrames, still: stillRelit },
            cpcaBatches: relight.stats.cpca,
            gpuMs: { relighting: round(gpuMoving), still: round(gpuStill) },
          };
          const snapshot = track.snapshot();
          host.detach();
          splats.dispose();
          source.dispose?.();
          color.destroy();
          return { ...result, _memory: snapshot, _colours: colours };
        }
      },
      3_600_000,
    );
  },
);

void TRANSFER_CPCA;
void TRANSFER_Q8;

// The compact relit path over athenea's error-band pages (thread CN,
// src/webgpu/athenea/compactPages.ts): the pieces' tier 4 alone streamed
// into pool slots by the page table, the WASM traversal over what is
// resident, the cut drawn by the compact path. Measured against
//   - the same path with every page resident (the cut should be the same
//     set and the image the same);
//   - today's relight (WgpuSplatRenderer + atheneaRelightPlugin) on the same
//     cut decoded from tiers 1-3 (cpca8 with the same basis, and the full
//     transfer).
//
// Opt-in (assets outside git):
//   COMPACT_PAGES=<dir of <piece>.athc written with usd-athc --tree ... --pages
//   --compact-basis> CPCA8_DIR=<athenea q97 data> [COMPACT_PAGES_PIECES=Hood,...]
//   [COMPACT_PAGES_SLOTS=574] [COMPACT_PAGES_TODAY=0] [COMPACT_PAGES_SWEEP=6]
//   npx vitest run --dir test/gpu test/gpu/compactPages.test.ts
// Prints COMPACT_PAGES_* {json} lines.

import {
  closeSync,
  fstatSync,
  openSync,
  readFileSync,
  readSync,
  writeFileSync,
} from "node:fs";
import { gunzipSync } from "node:zlib";
import * as THREE from "three";
import { HDRLoader } from "three/examples/jsm/loaders/HDRLoader.js";
import { describe, expect, it, vi } from "vitest";
import type { AthcV3Layout } from "../../src/athc";
import { CompactRelitPath } from "../../src/webgpu/athenea/compact";
import {
  COMPACT_LOD_NODE_BYTES,
  CompactBandStreamer,
  type CompactPiece,
  type LodCall,
  parseCompactAsset,
  parseCompactTables,
} from "../../src/webgpu/athenea/compactPages";
import {
  encodeJointCpca,
  halfToFloat,
  parseJointCpca,
} from "../../src/webgpu/athenea/cpcaJoint";
import { makeLodCall } from "../unit/compactLod";
import {
  type Streams,
  compare,
  readImage,
  referenceRun,
  tracker,
} from "./compactHarness";
import { wideDevice } from "./device";

const DIR = process.env.COMPACT_PAGES ?? "";
const PIECES = (
  process.env.COMPACT_PAGES_PIECES ??
  "Hood,Front_Fender,Headlights_Cover,Front_Logo,F_Grill,Door_001"
).split(",");
const CPCA_DIR = process.env.CPCA8_DIR ?? "";
const TODAY = process.env.COMPACT_PAGES_TODAY !== "0";
const SWEEP = Number(process.env.COMPACT_PAGES_SWEEP ?? 6);
const TAU = Number(process.env.COMPACT_PAGES_TAU ?? 1);
const HDR =
  process.env.COMPACT_PAGES_HDR ??
  new URL(
    "../../../publish-r2/sparkwebgpu/corvette-lights-v8-light/hdri/golden_gate_hills_1k.hdr",
    import.meta.url,
  ).pathname;
const DUMP = process.env.COMPACT_PAGES_DUMP ?? "";

// athenea's /World/Camera and CameraFar (pieces/t4/cameras.usda) in her
// Z-up stage; horizontal field 39.6 degrees (tools/band-pages/run.mjs).
const DIR_Z = [0.6080817, -0.778309, 0.1564344];
const NEAR = [4.378188, -5.853825, 1.676328];
const FAR = [13.4994, -17.5284, 4.0228];
function cameraAt(t: number, W: number, H: number) {
  const aspect = W / H;
  const vfov =
    (2 * Math.atan(Math.tan((39.6 * Math.PI) / 360) / aspect) * 180) / Math.PI;
  const camera = new THREE.PerspectiveCamera(vfov, aspect, 0.05, 100);
  camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
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

// This worktree's WASM (layouts with the compact tier), in-process.
const wasm = (await vi.importActual(
  "../../rust/spark-rs/pkg/spark_rs.js",
)) as typeof import("spark-rs");
wasm.initSync({
  module: readFileSync(
    new URL("../../rust/spark-rs/pkg/spark_rs_bg.wasm", import.meta.url),
  ),
});

/** The loader worker's LoD calls (src/worker.ts), in-process. */
const lodCall: LodCall = makeLodCall(wasm);

interface Io {
  ranges: number;
  bytes: number;
}

function openPiece(name: string, io: Io): CompactPiece & { fileBytes: number } {
  const path = `${DIR}/${name}.athc`;
  const fd = openSync(path, "r");
  const size = fstatSync(fd).size;
  const read = (offset: number, bytes: number) => {
    const out = new Uint8Array(bytes);
    readSync(fd, out, 0, bytes, offset);
    return out;
  };
  const head = read(0, 256);
  const prefixBytes = wasm.athc_prefix_bytes(head) as number;
  const prefix = read(0, prefixBytes);
  const layout = wasm.athc_layout(prefix, size) as AthcV3Layout;
  return {
    name,
    layout,
    prefix,
    fileBytes: size,
    read: async (offset, bytes) => {
      io.ranges += 1;
      io.bytes += bytes;
      return read(offset, bytes);
    },
    close: () => closeSync(fd),
  } as CompactPiece & { fileBytes: number };
}

const gunzip = async (b: Uint8Array) => new Uint8Array(gunzipSync(b));

describe.skipIf(!wideDevice || !DIR)("the compact path over band pages", () => {
  const d = wideDevice as GPUDevice;
  it(`${PIECES.join(",")}`, async () => {
    const hdr = new HDRLoader()
      .setDataType(THREE.FloatType)
      .parse(readFileSync(HDR).buffer as ArrayBuffer) as {
      width: number;
      height: number;
      data: Float32Array;
    };
    const sky = { width: hdr.width, height: hdr.height, data: hdr.data };
    const io: Io = { ranges: 0, bytes: 0 };
    const pieces = PIECES.map((p) => openPiece(p, io));
    // The asset table the pieces share (thread CQ), beside them.
    const shared = parseCompactTables(
      pieces[0].prefix.subarray(
        (pieces[0].layout as unknown as { compactTable: number[] })
          .compactTable[0],
        (pieces[0].layout as unknown as { compactTable: number[] })
          .compactTable[0] +
          (pieces[0].layout as unknown as { compactTable: number[] })
            .compactTable[1],
      ),
    );
    const asset = shared.assetFlags
      ? parseCompactAsset(
          new Uint8Array(readFileSync(`${DIR}/${shared.assetName}`)),
        )
      : undefined;
    const totalPages = pieces.reduce(
      (n, p) => n + (p.layout.pages?.length ?? 0),
      0,
    );
    const totalNodes = pieces.reduce((n, p) => n + p.layout.header.count, 0);
    const tier4 = pieces.reduce(
      (n, p) =>
        n +
        (p.layout.pages ?? []).reduce(
          (m, r) =>
            m + ((r as unknown as { tier4Bytes: number }).tier4Bytes ?? 0),
          0,
        ),
      0,
    );
    const prefixBytes = pieces.reduce((n, p) => n + p.prefix.length, 0);
    const slots = Number(process.env.COMPACT_PAGES_SLOTS ?? totalPages);
    process.stderr.write(
      `COMPACT_PAGES_FILES ${JSON.stringify({
        pieces: PIECES,
        pages: totalPages,
        nodes: totalNodes,
        fileBytes: pieces.reduce((n, p) => n + p.fileBytes, 0),
        tier4Bytes: tier4,
        tier4BytesPerNode: +(tier4 / totalNodes).toFixed(2),
        prefixBytes,
        slots,
      })}\n`,
    );

    const target = (W: number, H: number) =>
      d.createTexture({
        size: [W, H],
        format: "rgba16float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        label: "target",
      });
    const frame = (
      path: CompactRelitPath,
      camera: THREE.PerspectiveCamera,
      tex: GPUTexture,
    ) => {
      const enc = d.createCommandEncoder();
      path.render(
        enc,
        camera,
        tex.createView(),
        "rgba16float",
        tex.width,
        tex.height,
        {
          clear: { r: 0, g: 0, b: 0, a: 0 },
        },
      );
      path.registry.submit(enc.finish());
    };
    /** Frames until no page is coming and none is asked for. */
    const settle = async (
      s: CompactBandStreamer,
      path: CompactRelitPath,
      camera: THREE.PerspectiveCamera,
      tex: GPUTexture,
    ) => {
      const before = { ...s.stats, ioRanges: io.ranges, ioBytes: io.bytes };
      const t0 = performance.now();
      let frames = 0;
      let firstMs = -1;
      for (;;) {
        const asked = s.stats.pagesRequested;
        await s.update(camera, tex.height, path);
        if (
          s.stats.cut > (path.capacity || 0) ||
          path.capacity > 2 * s.stats.cut + 4096
        ) {
          path.fitCapacity(Math.max(s.stats.cut, 1), 0.1);
        }
        frame(path, camera, tex);
        await d.queue.onSubmittedWorkDone();
        if (firstMs < 0 && s.stats.cut > 0) firstMs = performance.now() - t0;
        frames += 1;
        if (!s.busy && s.stats.pagesRequested === asked) break;
        await s.idle();
        if (frames > 400) throw new Error("did not settle");
      }
      const a = s.stats;
      return {
        ms: +(performance.now() - t0).toFixed(1),
        firstFrameMs: +firstMs.toFixed(1),
        frames,
        pages: a.pagesRequested - before.pagesRequested,
        ranges: io.ranges - before.ioRanges,
        bytes: io.bytes - before.ioBytes,
        evicted: a.evicted - before.evicted,
      };
    };
    const cutKeys = (s: CompactBandStreamer) => {
      const keys: string[] = [];
      for (const i of s.drawList) {
        const n = s.nodeOf(i);
        if (n) keys.push(`${n.piece}:${n.page}:${n.node}`);
      }
      return keys.sort();
    };

    // A: paged, cold.
    const track = tracker(d);
    const pathA = new CompactRelitPath(d, { hdri: sky, stableOrder: true });
    const A = new CompactBandStreamer(d, pieces, {
      slots,
      tau: TAU,
      tauToFit: process.env.COMPACT_PAGES_FIT === "1",
      nodeOrder: true,
      gunzip,
      lod: lodCall,
      asset,
    });
    await A.init();
    pathA.setPagedCloud(A.pool);
    // B: every page resident (loaded once), the same traversal.
    const views: {
      name: string;
      t: number;
      W: number;
      H: number;
      compare?: boolean;
    }[] = [{ name: "far 393x695 (cold)", t: 0, W: 393, H: 695, compare: true }];
    for (let k = 1; k <= SWEEP; k++) {
      views.push({
        name: `sweep ${k}/${SWEEP} 1920x1080`,
        t: k / SWEEP,
        W: 1920,
        H: 1080,
        compare: k === SWEEP,
      });
    }
    // The phone at the near camera: what an iPhone in portrait keeps there.
    views.push({
      name: "near 393x695 (phone)",
      t: 1,
      W: 393,
      H: 695,
      compare: true,
    });
    views.push({ name: "far 393x695 (back)", t: 0, W: 393, H: 695 });
    const results: Record<string, unknown>[] = [];
    const cuts: Record<string, { keys: string[]; image: Float32Array }> = {};
    for (const v of views) {
      const camera = cameraAt(v.t, v.W, v.H);
      const tex = target(v.W, v.H);
      const st = await settle(A, pathA, camera, tex);
      const capA = pathA.capacity;
      const drawnA = await pathA.drawnCount();
      process.stderr.write(`CN A ${v.name} capacity ${capA} drawn ${drawnA}\n`);
      const memory = track.snapshot();
      const image = await readImage(d, tex);
      const s = A.stats;
      const resident = A.residentBytes();
      const slotBytes = A.pool.slotWords * 4;
      const sortBytes = Object.entries(memory)
        .filter(([k]) => /sort/.test(k))
        .reduce((n, [, b]) => n + (b as number), 0);
      const usedGpu = resident.gpuTables + s.residentPages * slotBytes;
      const row = {
        view: v.name,
        tau: +s.tau.toFixed(3),
        cut: s.cut,
        neededPages: s.neededPages,
        neededNodes: s.neededNodes,
        residentPages: s.residentPages,
        residentNodes: s.residentNodes,
        residentOverCut: +(s.residentNodes / Math.max(1, s.cut)).toFixed(2),
        settle: st,
        bytes: {
          // The slots holding pages (a pool sized to them), the tables.
          usedSlots: s.residentPages * slotBytes,
          tables: resident.gpuTables,
          sort: sortBytes,
          gpuUsed: usedGpu + sortBytes,
          perResidentNodeGpu: +(usedGpu / Math.max(1, s.residentNodes)).toFixed(
            2,
          ),
          perResidentNodeCpu: COMPACT_LOD_NODE_BYTES,
          poolAllocated: resident.gpuSlots + resident.gpuTables,
          all: memory,
        },
      };
      results.push(row);
      process.stderr.write(`COMPACT_PAGES_VIEW ${JSON.stringify(row)}\n`);
      if (v.compare) cuts[v.name] = { keys: cutKeys(A), image };
      if (DUMP)
        writeFileSync(`${DUMP}/paged-${v.t.toFixed(2)}-${v.W}.f32`, image);
      tex.destroy();
    }
    track.restore();
    expect(results.length).toBe(views.length);

    // B: every page resident.
    const pathB = new CompactRelitPath(d, { hdri: sky, stableOrder: true });
    const B = new CompactBandStreamer(d, pieces, {
      slots: totalPages,
      tau: TAU,
      nodeOrder: true,
      gunzip,
      lod: lodCall,
      asset,
    });
    await B.init();
    pathB.setPagedCloud(B.pool);
    await B.loadAll();
    for (const v of views.filter((x) => x.compare)) {
      const camera = cameraAt(v.t, v.W, v.H);
      const tex = target(v.W, v.H);
      await B.update(camera, v.H, pathB);
      expect(B.stats.residentPages).toBe(totalPages);
      pathB.fitCapacity(B.stats.cut, 0.1);
      frame(pathB, camera, tex);
      await d.queue.onSubmittedWorkDone();
      const image = await readImage(d, tex);
      if (process.env.CN_DEBUG) {
        pathB.touchLight();
        frame(pathB, camera, tex);
        await d.queue.onSubmittedWorkDone();
        const again = await readImage(d, tex);
        await B.update(cameraAt(1, v.W, v.H), v.H, pathB);
        frame(pathB, cameraAt(1, v.W, v.H), tex);
        await B.update(camera, v.H, pathB);
        frame(pathB, camera, tex);
        await d.queue.onSubmittedWorkDone();
        const after = await readImage(d, tex);
        process.stderr.write(
          `CN B repeat ${v.name} ${JSON.stringify(compare(again, image))} after-other-view ${JSON.stringify(compare(after, image))} vsA-after ${JSON.stringify(compare(cuts[v.name].image, after))}\n`,
        );
      }
      const keys = cutKeys(B);
      if (process.env.CN_DEBUG) {
        // Per node: A's and B's relit colour, footprint and place.
        await A.update(camera, v.H, pathA);
        const cA = { list: [...A.drawList] };
        pathA.setDrawList(null);
        const ca = await pathA.colours(camera, v.W, v.H);
        pathA.setDrawList(A.drawList);
        pathB.setDrawList(null);
        const cb = await pathB.colours(camera, v.W, v.H);
        pathB.setDrawList(B.drawList);
        const where = new Map<string, number>();
        for (const i of B.drawList) {
          const n = B.nodeOf(i);
          if (n) where.set(`${n.piece}:${n.page}:${n.node}`, i);
        }
        const worst: Record<string, number> = {};
        let shown = 0;
        for (const i of cA.list) {
          const n = A.nodeOf(i);
          if (!n) continue;
          const j = where.get(`${n.piece}:${n.page}:${n.node}`) as number;
          for (const [name, x, y] of [
            ["colour", ca.colours, cb.colours],
            ["shape", ca.shapes, cb.shapes],
            ["place", ca.places, cb.places],
          ] as const) {
            for (let c = 0; c < 4; c++) {
              const dd = Math.abs(x[4 * i + c] - y[4 * j + c]);
              if (dd > (worst[name + c] ?? 0)) worst[name + c] = dd;
              if (name !== "colour" && dd > 0.5 && shown < 8) {
                shown++;
                process.stderr.write(
                  `CN diff ${name} ${JSON.stringify(n)} i ${i} j ${j} a ${[...x.subarray(4 * i, 4 * i + 4)]} b ${[...y.subarray(4 * j, 4 * j + 4)]} ca ${[...ca.colours.subarray(4 * i, 4 * i + 4)]}\n`,
                );
              }
            }
          }
        }
        process.stderr.write(`CN worst ${v.name} ${JSON.stringify(worst)}\n`);
      }
      const a = cuts[v.name];
      const setA = new Set(a.keys);
      const onlyB = keys.filter((k) => !setA.has(k)).length;
      const setB = new Set(keys);
      const onlyA = a.keys.filter((k) => !setB.has(k)).length;
      const out = {
        view: v.name,
        capacityB: pathB.capacity,
        drawnB: await pathB.drawnCount(),
        cutPaged: a.keys.length,
        cutResident: keys.length,
        onlyPaged: onlyA,
        onlyResident: onlyB,
        image: compare(a.image, image),
      };
      process.stderr.write(
        `COMPACT_PAGES_VS_RESIDENT ${JSON.stringify(out)}\n`,
      );
      if (DUMP)
        writeFileSync(`${DUMP}/resident-${v.t.toFixed(2)}-${v.W}.f32`, image);
      cuts[v.name] = { ...a, image: a.image };
      (cuts[v.name] as unknown as { cutB: string[] }).cutB = keys;
      tex.destroy();
    }
    pathA.dispose();
    A.dispose();
    pathB.dispose();
    B.dispose();

    // C: today's relight on the same cut, decoded from tiers 1-3.
    if (TODAY && CPCA_DIR) {
      const json = JSON.parse(
        readFileSync(`${CPCA_DIR}/cpca_joint_C256_K8.json`, "utf8"),
      );
      const basis = parseJointCpca(
        json,
        new Uint8Array(readFileSync(`${CPCA_DIR}/cpca_joint_C256_K8.bin`)),
      );
      const want = views.filter((x) => x.compare);
      // The cut's nodes, by piece, as file indices (page order).
      const picks = want.map((v) =>
        cuts[v.name].keys.map((k) => k.split(":").map(Number)),
      );
      const gathered = want.map(() => ({
        ext0: [] as number[],
        ext1: [] as number[],
        normalOct: [] as number[],
        pbr: [] as number[],
        lobes: [] as number[],
        curvature: [] as number[],
        shadowBits: [] as number[],
        transfer: [] as number[],
      }));
      pieces.forEach((p, pi) => {
        const bytes = new Uint8Array(readFileSync(`${DIR}/${p.name}.athc`));
        const decoder = wasm.decode_to_extsplats(undefined, "cloud.athc");
        decoder.push(bytes);
        const dd = decoder.finish() as {
          numSplats: number;
          ext0: Uint32Array;
          ext1: Uint32Array;
          attribSpecs: { name: string; components: number }[];
          attribColumns: Uint32Array[];
        };
        const col = (name: string) => {
          const k = dd.attribSpecs.findIndex((s) => s.name === name);
          return k < 0 ? null : (dd.attribColumns[k] as Uint32Array);
        };
        const cols = {
          normalOct: [col("normalOct"), 1],
          pbr: [col("pbr"), 1],
          lobes: [col("lobes"), 3],
          curvature: [col("curvature"), 2],
          shadowBits: [col("shadowBits"), 8],
          transfer: [col("transfer"), 56],
        } as const;
        const rows = p.layout.pages ?? [];
        want.forEach((_, vi) => {
          const g = gathered[vi];
          for (const [piece, page, node] of picks[vi]) {
            if (piece !== pi) continue;
            const i = rows[page].first + node;
            for (let k = 0; k < 4; k++) {
              g.ext0.push(dd.ext0[4 * i + k]);
              g.ext1.push(dd.ext1[4 * i + k]);
            }
            for (const [name, [c, per]] of Object.entries(cols)) {
              const out = g[name as keyof typeof g];
              for (let k = 0; k < per; k++) out.push(c ? c[per * i + k] : 0);
            }
          }
        });
      });
      for (const [vi, v] of want.entries()) {
        const g = gathered[vi];
        const n = g.ext0.length / 4;
        const transfer = new Uint32Array(g.transfer);
        const t16 = new Uint16Array(transfer.buffer);
        const lut = new Float32Array(65536);
        for (let h = 0; h < 65536; h++) lut[h] = halfToFloat(h);
        const q8 = encodeJointCpca(basis, n, (i, out) => {
          for (let k = 0; k < 112; k++) out[k] = lut[t16[i * 112 + k]];
        }).records;
        const s: Streams = {
          ext0: new Uint32Array(g.ext0),
          ext1: new Uint32Array(g.ext1),
          normalOct: new Uint32Array(g.normalOct),
          pbr: new Uint32Array(g.pbr),
          lobes: new Uint32Array(g.lobes),
          curvature: new Uint32Array(g.curvature),
          shadowBits: new Uint32Array(g.shadowBits),
          q8,
          transfer,
        };
        const camera = cameraAt(v.t, v.W, v.H);
        const run = {
          W: v.W,
          H: v.H,
          sky,
          camera,
          place: () => {},
          image: true,
          lodOpacity: true,
        };
        const cpca8 = await referenceRun(
          d,
          { ...s, transfer: undefined },
          n,
          basis,
          run,
        );
        const words = await referenceRun(d, s, n, basis, {
          ...run,
          transferResident: "words",
        });
        const paged = cuts[v.name].image;
        const out = {
          view: v.name,
          nodes: n,
          compactVsTodayCpca8: compare(paged, cpca8.image as Float32Array),
          compactVsTodayWords: compare(paged, words.image as Float32Array),
          todayCpca8VsWords: compare(
            cpca8.image as Float32Array,
            words.image as Float32Array,
          ),
        };
        process.stderr.write(`COMPACT_PAGES_VS_TODAY ${JSON.stringify(out)}\n`);
        if (DUMP) {
          writeFileSync(
            `${DUMP}/today-cpca8-${v.t.toFixed(2)}-${v.W}.f32`,
            cpca8.image as Float32Array,
          );
          writeFileSync(
            `${DUMP}/today-words-${v.t.toFixed(2)}-${v.W}.f32`,
            words.image as Float32Array,
          );
        }
      }
    }
    for (const p of pieces) (p as unknown as { close(): void }).close();
  }, 3_600_000);
});

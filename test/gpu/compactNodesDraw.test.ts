// The node pool (thread CT, compactNodes.ts) on Dawn against the page pool
// (CompactPagePool), with test/fixtures/athc/compact_pages.athc:
//   - streamed to the same view, the same cut, node for node, and the same
//     image bit for bit (stable order on both sides);
//   - from a small start the pool grows by copy and keeps its records;
//   - a camera jump under a page budget (evictions, tau fitted): still the
//     page pool's cut and image, the records of pages that left reused.

import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import * as THREE from "three";
import { describe, expect, it, vi } from "vitest";
import type { AthcV3Layout } from "../../src/athc";
import { CompactRelitPath } from "../../src/webgpu/athenea/compact";
import type { CompactNodePool } from "../../src/webgpu/athenea/compactNodes";
import {
  CompactBandStreamer,
  type CompactPiece,
  type LodCall,
} from "../../src/webgpu/athenea/compactPages";
import { makeLodCall } from "../unit/compactLod";
import { readImage } from "./compactHarness";
import { wideDevice } from "./device";

const wasm = (await vi.importActual(
  "../../rust/spark-rs/pkg/spark_rs.js",
)) as typeof import("spark-rs");
wasm.initSync({
  module: readFileSync(
    new URL("../../rust/spark-rs/pkg/spark_rs_bg.wasm", import.meta.url),
  ),
});

const FILE = new URL("../fixtures/athc/compact_pages.athc", import.meta.url)
  .pathname;
/** The same tree as CMPC v4 (athc-convert --compact-runs). */
const RUNS = new URL(
  "../fixtures/athc/compact_pages.runs.athc",
  import.meta.url,
).pathname;
const lod: LodCall = makeLodCall(wasm);

function piece(file = FILE, io?: { ranges: number }): CompactPiece {
  const bytes = new Uint8Array(readFileSync(file));
  const prefix = bytes.slice(0, wasm.athc_prefix_bytes(bytes.slice(0, 256)));
  return {
    name: "fixture",
    layout: wasm.athc_layout(prefix, bytes.length) as AthcV3Layout,
    prefix,
    read: async (offset, n) => {
      if (io) io.ranges += 1;
      return bytes.slice(offset, offset + n);
    },
  };
}

const gunzip = async (b: Uint8Array) => new Uint8Array(gunzipSync(b));

function dome() {
  const W = 64;
  const H = 32;
  const data = new Float32Array(W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const up = 1 - y / H;
      data.set([0.3 + up, 0.4 + up, 0.6 + 1.5 * up, 1], 4 * (y * W + x));
    }
  }
  return { width: W, height: H, data };
}

function camera(W: number, H: number) {
  const c = new THREE.PerspectiveCamera(50, W / H, 0.05, 100);
  c.coordinateSystem = THREE.WebGPUCoordinateSystem;
  c.up.set(0, 0, 1);
  c.position.set(9, -6, 1.5);
  c.lookAt(new THREE.Vector3(14, 0, 0));
  c.updateProjectionMatrix();
  c.updateMatrixWorld();
  return c;
}

describe.skipIf(!wideDevice)("the node pool", () => {
  const d = wideDevice as GPUDevice;
  const W = 320;
  const H = 200;
  const sky = dome();

  async function draw(path: CompactRelitPath, cam: THREE.PerspectiveCamera) {
    const tex = d.createTexture({
      size: [W, H],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const enc = d.createCommandEncoder();
    path.render(enc, cam, tex.createView(), "rgba16float", W, H, {
      clear: { r: 0, g: 0, b: 0, a: 0 },
    });
    path.registry.submit(enc.finish());
    await d.queue.onSubmittedWorkDone();
    const image = await readImage(d, tex);
    tex.destroy();
    return image;
  }

  async function streamer(
    slots: number,
    nodes: boolean,
    fit: boolean,
    file = FILE,
    runBudget?: number,
    io?: { ranges: number },
  ) {
    const s = new CompactBandStreamer(d, [piece(file, io)], {
      slots,
      slotNodes: 512,
      tau: 0.05,
      tauToFit: fit,
      nodeOrder: true,
      gunzip,
      lod,
      ...(nodes
        ? {
            nodePool: {
              capacity: 64,
              decodedPages: 2,
              inflatedRuns: 4,
              ...(runBudget !== undefined ? { runBudget } : {}),
            },
          }
        : {}),
    });
    await s.init();
    const path = new CompactRelitPath(d, { hdri: sky, stableOrder: true });
    path.setPagedCloud(s.pool);
    return { s, path };
  }

  async function settle(
    x: { s: CompactBandStreamer; path: CompactRelitPath },
    cam: THREE.PerspectiveCamera,
  ) {
    // As the page pool's test: 20 updates, each after the fetches landed.
    for (let frames = 0; frames < 20; frames++) {
      await x.s.update(cam, H, x.path);
      await x.s.idle();
    }
    // One more: the cut over what landed last.
    await x.s.update(cam, H, x.path);
  }

  const keysOf = (s: CompactBandStreamer) =>
    [...s.drawList].map((i) => {
      const n = s.nodeOf(i);
      return `${n?.page}:${n?.node}`;
    });

  function same(a: Float32Array | Uint16Array, b: Float32Array | Uint16Array) {
    expect(a.length).toBe(b.length);
    let diff = 0;
    for (let k = 0; k < a.length; k++) if (a[k] !== b[k]) diff++;
    return diff;
  }

  it("draws the page pool's cut and image, bit for bit", async () => {
    const cam = camera(W, H);
    const pages = await streamer(8, false, false);
    const nodes = await streamer(8, true, false);
    await settle(pages, cam);
    await settle(nodes, cam);
    expect(nodes.s.stats.cut).toBeGreaterThan(100);
    expect(keysOf(nodes.s)).toEqual(keysOf(pages.s));
    // It grew from 64 by copy, and only the cut is held.
    const pool = nodes.s.pool as CompactNodePool;
    expect(pool.capacity).toBeGreaterThanOrEqual(nodes.s.stats.cut);
    expect(nodes.s.nodes?.used).toBeGreaterThanOrEqual(nodes.s.stats.cut);
    expect(nodes.s.stats.missing).toBe(0);
    const a = await draw(pages.path, cam);
    const b = await draw(nodes.path, cam);
    expect(same(a, b)).toBe(0);
    let lit = 0;
    for (let k = 0; k < a.length; k += 4) lit += a[k + 3] > 0 ? 1 : 0;
    expect(lit).toBeGreaterThan(200);
    const bytes = nodes.s.residentBytes();
    process.stderr.write(
      `CT_NODES ${JSON.stringify({ cut: nodes.s.stats.cut, capacity: pool.capacity, bytes, pageBytes: pages.s.residentBytes() })}\n`,
    );
    for (const x of [pages, nodes]) {
      x.path.dispose();
      x.s.dispose();
    }
  });

  it("reads CMPC v4 runs: the same cut and image, in both pools", async () => {
    const cam = camera(W, H);
    const pages = await streamer(8, false, false);
    const pagesRuns = await streamer(8, false, false, RUNS);
    const nodesRuns = await streamer(8, true, false, RUNS);
    for (const x of [pages, pagesRuns, nodesRuns]) await settle(x, cam);
    expect(nodesRuns.s.runs).not.toBeNull();
    expect(keysOf(pagesRuns.s)).toEqual(keysOf(pages.s));
    expect(keysOf(nodesRuns.s)).toEqual(keysOf(pages.s));
    const a = await draw(pages.path, cam);
    expect(same(a, await draw(pagesRuns.path, cam))).toBe(0);
    expect(same(a, await draw(nodesRuns.path, cam))).toBe(0);
    for (const x of [pages, pagesRuns, nodesRuns]) {
      x.path.dispose();
      x.s.dispose();
    }
  });

  it("drops the runs the cut cannot need and fetches them again", async () => {
    const cam = camera(W, H);
    const pages = await streamer(8, false, false);
    const io = { ranges: 0 };
    // No budget: only the cut's runs (its nodes, parents, children) kept.
    const nodes = await streamer(8, true, false, RUNS, 0, io);
    await settle(pages, cam);
    await settle(nodes, cam);
    const runs = nodes.s.runs as NonNullable<typeof nodes.s.runs>;
    const c = runs.census();
    expect(c.held).toBeLessThan(c.runs);
    expect(c.other).toBe(0);
    expect(keysOf(nodes.s)).toEqual(keysOf(pages.s));
    expect(same(await draw(pages.path, cam), await draw(nodes.path, cam))).toBe(
      0,
    );
    // A jump: the new cut's dropped runs come again by Range; it settles to
    // the page pool's image.
    cam.position.set(30, -4, 1);
    cam.lookAt(new THREE.Vector3(34, 0, 0));
    cam.updateMatrixWorld();
    await settle(pages, cam);
    // Fetches of runs land between updates: a few more.
    for (let k = 0; k < 10; k++) {
      await nodes.s.update(cam, H, nodes.path);
      await new Promise((r) => setTimeout(r, 5));
    }
    expect(nodes.s.stats.missing).toBe(0);
    expect(keysOf(nodes.s)).toEqual(keysOf(pages.s));
    expect(same(await draw(pages.path, cam), await draw(nodes.path, cam))).toBe(
      0,
    );
    process.stderr.write(
      `CT_RUNS ${JSON.stringify({ census: runs.census(), stats: runs.stats, runRanges: nodes.s.stats.runRanges })}\n`,
    );
    for (const x of [pages, nodes]) {
      x.path.dispose();
      x.s.dispose();
    }
  });

  it("follows a camera jump under a page budget as the page pool does", async () => {
    const cam = camera(W, H);
    const pages = await streamer(3, false, true);
    const nodes = await streamer(3, true, true);
    await settle(pages, cam);
    await settle(nodes, cam);
    expect(keysOf(nodes.s)).toEqual(keysOf(pages.s));
    const used0 = nodes.s.nodes?.stats.reused ?? 0;
    cam.position.set(30, -4, 1);
    cam.lookAt(new THREE.Vector3(34, 0, 0));
    cam.updateMatrixWorld();
    await settle(pages, cam);
    await settle(nodes, cam);
    expect(nodes.s.stats.evicted).toBeGreaterThan(0);
    expect(nodes.s.stats.residentPages).toBeLessThanOrEqual(3);
    expect(keysOf(nodes.s)).toEqual(keysOf(pages.s));
    expect(nodes.s.stats.missing).toBe(0);
    expect(same(await draw(pages.path, cam), await draw(nodes.path, cam))).toBe(
      0,
    );
    // And back: the records of the first view's nodes, still held, reused.
    cam.position.set(9, -6, 1.5);
    cam.lookAt(new THREE.Vector3(14, 0, 0));
    cam.updateMatrixWorld();
    await settle(pages, cam);
    await settle(nodes, cam);
    expect(keysOf(nodes.s)).toEqual(keysOf(pages.s));
    expect(nodes.s.nodes?.stats.reused ?? 0).toBeGreaterThan(used0);
    expect(same(await draw(pages.path, cam), await draw(nodes.path, cam))).toBe(
      0,
    );
    // The payload store holds the resident pages only.
    expect(nodes.s.payload?.pages).toBe(nodes.s.stats.residentPages);
    for (const x of [pages, nodes]) {
      x.path.dispose();
      x.s.dispose();
    }
  });
});

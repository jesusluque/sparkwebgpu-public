// THE COMPACT PATH OVER BAND PAGES (thread CN): athenea's error-band pages
// (a paged .athc tree, docs/docs/athc-v3.md "Error-band pages") drawn by the
// compact relit path (compact.ts) from their fourth tier alone.
//
// A paged tree written with `usd-athc --tree ... --pages --compact-basis`
// carries, per page, a `CMPC` section (tier 4, after tier 3, so consecutive
// pages are one Range): NODE_BYTES (64) a node as byte planes, the 40 the
// GPU keeps then the traversal's LoD tree words (4) and LODE words (2); the
// page's quantization box in its page-table row (words 17..29); and, among
// the file's tables, the piece's shadow dictionary, material table and
// athenea's joint CPCA basis in f16 (rust athc_compact.rs).
//
// Here:
//   - CompactPagePool: one GPU buffer, the pieces' tables (always resident)
//     then `slots` slots of `slotNodes` (3072) nodes, each the page's head
//     and its planes (compact.slang, slotNodes > 0);
//   - CompactBandStreamer: the page tables, the WASM traversal over the
//     resident pages (spark-rs lod_tree.rs, one shared pool tree, a slot a
//     page), the needed set and requests by WgpuBandPager's planBand, Range
//     reads of tier 4 alone, eviction finest first, and the draw list
//     (CompactRelitPath.setDrawList) of slot indices.

import * as THREE from "three";
import { SplatWorker } from "../../SplatWorker";
import {
  type AthcPageRow,
  type AthcV3Layout,
  readAthcLayout,
  readRange,
} from "../../athc";
import { BAND_NEAR, type BandMesh, planBand } from "../WgpuBandPager";
import type { CompactCloud, CompactLayout, CompactRelitPath } from "./compact";

/** Bytes a node in a CMPC section, by plane (rust athc_compact PLANE_WIDTHS). */
export const COMPACT_PLANE_WIDTHS = [
  2, 2, 2, 3, 1, 3, 4, 3, 3, 2, 1, 4, 4, 4, 2, 16, 8,
] as const;
export const COMPACT_NODE_BYTES = 64;
/** The planes the GPU keeps, by name (the rest are the traversal's). */
const GPU_PLANES = [
  "posX",
  "posY",
  "posZ",
  "albedo",
  "opacity",
  "scale",
  "rotation",
  "normal",
  "curvature",
  "material",
  "cluster",
  "weights",
  "weights1",
  "weights2",
  "shadow",
] as const;
const P_LOD_TREE = 15;
const P_LOD_ERROR = 16;
/** Words a slot head (compact.slang kCompactChunkWords, rounded to 16). */
const HEAD_WORDS = 12;
const SLOT_HEAD_WORDS = 16;
const TABLE_MAGIC = 0x54504d43; // "CMPT"

/** Byte offset of plane k in a section of n nodes. */
export function compactPlaneOffset(k: number, n: number) {
  let at = 0;
  for (let j = 0; j < k; j++) at += COMPACT_PLANE_WIDTHS[j];
  return at * n;
}

/** A piece's compact tables (rust athc_compact CompactTables). */
export interface CompactPieceTables {
  codebook: Uint32Array;
  materials: Uint32Array;
  basis: Uint32Array;
  clusters: number;
  normalBits: number;
  weightBits: number;
  linear: boolean;
  curvature: boolean;
}

/** The tables blob (CompactTables::to_bytes). */
export function parseCompactTables(bytes: Uint8Array): CompactPieceTables {
  const w = new Uint32Array(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
  if (w[0] !== TABLE_MAGIC || w[1] !== 1) {
    throw new Error("compact tables: not CMPT v1");
  }
  if (w[2] !== COMPACT_NODE_BYTES || w[3] !== COMPACT_PLANE_WIDTHS.length) {
    throw new Error("compact tables: another layout");
  }
  return {
    codebook: w.slice(w[11], w[11] + 8 * w[5]),
    materials: w.slice(w[12], w[12] + 4 * w[6]),
    basis: w.slice(w[13], w[13] + w[14]),
    clusters: w[7],
    normalBits: w[8],
    weightBits: w[9],
    linear: (w[10] & 1) !== 0,
    curvature: (w[10] & 2) !== 0,
  };
}

/** Where each GPU plane sits in a slot of `slotNodes` (words), and its size. */
export function compactSlotLayout(slotNodes: number) {
  const words: Record<string, number> = {};
  let at = SLOT_HEAD_WORDS;
  GPU_PLANES.forEach((name, k) => {
    words[name] = at;
    at = Math.ceil((at + (COMPACT_PLANE_WIDTHS[k] * slotNodes) / 4) / 16) * 16;
  });
  return { words, slotWords: at };
}

function sameWords(a: Uint32Array, b: Uint32Array) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/**
 * The pool: the pieces' tables then `slots` slots of `slotNodes`, in one
 * buffer the compact path reads (CompactRelitPath.setPagedCloud).
 */
export class CompactPagePool {
  readonly buffer: GPUBuffer;
  readonly slotWords: number;
  readonly slotsBase: number;
  readonly cloud: CompactCloud;
  /** Each piece's codebook and materials (words in the buffer). */
  readonly pieceBase: { codebook: number; materials: number }[] = [];
  readonly tableWords: number;
  private staging: Uint8Array;

  constructor(
    readonly device: GPUDevice,
    readonly pieces: CompactPieceTables[],
    readonly slots: number,
    readonly slotNodes = 3072,
  ) {
    if (!pieces.length) throw new Error("CompactPagePool: no piece");
    const first = pieces[0];
    for (const p of pieces) {
      if (
        p.clusters !== first.clusters ||
        p.normalBits !== first.normalBits ||
        p.weightBits !== first.weightBits ||
        p.linear !== first.linear ||
        !sameWords(p.basis, first.basis)
      ) {
        // One projection table a frame: the pieces share one basis.
        throw new Error("CompactPagePool: the pieces' bases or codes differ");
      }
    }
    // Tables: the basis once, then each piece's codebook and materials.
    const tables: number[] = [];
    const push = (w: Uint32Array) => {
      const at = Math.ceil(tables.length / 16) * 16;
      while (tables.length < at) tables.push(0);
      for (const v of w) tables.push(v);
      return at;
    };
    const basis = push(first.basis);
    for (const p of pieces) {
      this.pieceBase.push({
        codebook: push(p.codebook),
        materials: push(p.materials),
      });
    }
    this.tableWords = Math.ceil(tables.length / 16) * 16;
    const { words, slotWords } = compactSlotLayout(slotNodes);
    this.slotWords = slotWords;
    this.slotsBase = this.tableWords;
    const bytes = (this.tableWords + slots * slotWords) * 4;
    this.buffer = device.createBuffer({
      label: "compact pages",
      size: bytes,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(this.buffer, 0, new Uint32Array(tables));
    this.staging = new Uint8Array(slotWords * 4);
    const layout: CompactLayout & {
      paged: { slotNodes: number; slotWords: number; slotsBase: number };
    } = {
      count: slots * slotNodes,
      chunks: slots,
      pieces: pieces.length,
      materials: pieces.reduce((n, p) => n + p.materials.length / 4, 0),
      codebookEntries: pieces.reduce((n, p) => n + p.codebook.length / 8, 0),
      clusters: first.clusters,
      linear: first.linear,
      curvature: pieces.every((p) => p.curvature),
      normalBits: first.normalBits,
      weightBits: first.weightBits,
      words: {
        ...words,
        weightsStride: words.weights1 - words.weights,
        chunkTable: 0,
        pieceTable: 0,
        materials: 0,
        basis,
      },
      bytes: {
        tables: this.tableWords * 4,
        slots: slots * slotWords * 4,
        total: bytes,
      },
      paged: { slotNodes, slotWords, slotsBase: this.slotsBase },
    };
    this.cloud = { data: new Uint32Array(0), layout };
  }

  /**
   * Page `raw` (its CMPC section, n nodes) of piece `piece` into `slot`:
   * the head (its box, its piece's tables) and the GPU planes. Returns the
   * traversal's words (LoD tree, 4 a node; LODE, 2 a node).
   */
  writePage(
    slot: number,
    piece: number,
    n: number,
    raw: Uint8Array,
    head: ArrayLike<number>,
  ) {
    if (n > this.slotNodes) {
      throw new Error(
        `CompactPagePool: a page of ${n} in slots of ${this.slotNodes}`,
      );
    }
    if (raw.length !== n * COMPACT_NODE_BYTES) {
      throw new Error(`CompactPagePool: ${raw.length} bytes for ${n} nodes`);
    }
    const out = this.staging;
    out.fill(0);
    const w = new Uint32Array(out.buffer);
    for (let k = 0; k < HEAD_WORDS; k++) w[k] = head[k] >>> 0;
    w[9] = this.pieceBase[piece].codebook;
    w[10] = this.pieceBase[piece].materials;
    const words = (this.cloud.layout as CompactLayout).words;
    GPU_PLANES.forEach((name, k) => {
      const from = compactPlaneOffset(k, n);
      const len = COMPACT_PLANE_WIDTHS[k] * n;
      out.set(raw.subarray(from, from + len), words[name] * 4);
    });
    this.device.queue.writeBuffer(
      this.buffer,
      (this.slotsBase + slot * this.slotWords) * 4,
      out,
    );
    const words32 = (k: number, per: number) => {
      const from = compactPlaneOffset(k, n);
      return new Uint32Array(
        raw.buffer.slice(
          raw.byteOffset + from,
          raw.byteOffset + from + 4 * per * n,
        ),
      );
    };
    return {
      lodTree: words32(P_LOD_TREE, 4),
      lodError: words32(P_LOD_ERROR, 2),
    };
  }

  dispose() {
    this.buffer.destroy();
  }
}

/** The WASM LoD trees, as the loader worker's RPC (src/worker.ts) or in-process. */
export type LodCall = (
  name:
    | "newLodTree"
    | "newSharedLodTree"
    | "updateLodTrees"
    | "traverseLodTrees",
  args: Record<string, unknown>,
) => Promise<unknown>;

/** A paged tree with its compact tier, and how to read its bytes. */
export interface CompactPiece {
  name: string;
  layout: AthcV3Layout;
  /** The file's first bytes through its tables (athc_prefix_bytes). */
  prefix: Uint8Array;
  /** Bytes [offset, offset + bytes) of the file. */
  read: (offset: number, bytes: number) => Promise<Uint8Array>;
}

export interface CompactBandOptions {
  /** Pool slots (the residency budget, in pages). */
  slots: number;
  slotNodes?: number;
  /** The error cut, pixels. */
  tau?: number;
  /** Fit tau to the pool when the needed set does not (bandTauToFit). */
  tauToFit?: boolean;
  maxRangeBytes?: number;
  /** Fetches in flight. */
  fetchers?: number;
  /** gunzip a stored section (DecompressionStream, or node's zlib). */
  gunzip: (bytes: Uint8Array) => Promise<Uint8Array>;
  /** The WASM traversal. */
  lod: LodCall;
}

type Row = AthcPageRow & { tier4Bytes: number; compact: number[] };

interface PieceState {
  piece: CompactPiece;
  index: number;
  rows: Row[];
  /** CMPC's index in the section table. */
  section: number;
  compressed: boolean;
  pageBytes: number[];
  lodId: number;
  /** page -> slot, slot -> page. */
  slotOf: Map<number, number>;
  /** Pages a resident page's nodes have children in. */
  kids: Map<number, number[]>;
  inFlight: Set<number>;
}

/** The compact tables of a piece, from its prefix. */
export function compactTablesOf(piece: CompactPiece): CompactPieceTables {
  const t = (piece.layout as unknown as { compactTable?: number[] })
    .compactTable;
  if (!t?.[0]) throw new Error(`${piece.name}: no compact tier`);
  return parseCompactTables(piece.prefix.subarray(t[0], t[0] + t[1]));
}

/** Streams the pieces' band pages' tier 4 into a CompactPagePool. */
export class CompactBandStreamer {
  readonly pool: CompactPagePool;
  readonly pieces: PieceState[] = [];
  readonly slotNodes: number;
  private freeSlots: number[] = [];
  private slotOwner: ({ piece: number; page: number } | null)[];
  private poolLodId = -1;
  private landed: {
    piece: PieceState;
    page: number;
    raw: Uint8Array;
  }[] = [];
  private fetching = 0;
  private lastPlan: ReturnType<typeof planBand<PieceState>> | null = null;
  tau: number;
  /** The draw list of the last traversal (slot indices). */
  drawList = new Uint32Array(0);
  readonly stats = {
    pagesRequested: 0,
    ranges: 0,
    bytesRead: 0,
    evicted: 0,
    landed: 0,
    neededPages: 0,
    neededNodes: 0,
    residentPages: 0,
    residentNodes: 0,
    cut: 0,
    tau: 0,
    traverseMs: 0,
    starved: 0,
  };

  constructor(
    readonly device: GPUDevice,
    pieces: CompactPiece[],
    readonly options: CompactBandOptions,
  ) {
    this.slotNodes = options.slotNodes ?? 3072;
    this.tau = options.tau ?? 1;
    const tables = pieces.map(compactTablesOf);
    this.pool = new CompactPagePool(
      device,
      tables,
      options.slots,
      this.slotNodes,
    );
    this.slotOwner = new Array(options.slots).fill(null);
    for (let s = options.slots - 1; s >= 0; s--) this.freeSlots.push(s);
    pieces.forEach((piece, index) => {
      const layout = piece.layout;
      const section = layout.sections.findIndex(
        (s) => (s.id as string) === "CMPC",
      );
      if (section < 0) throw new Error(`${piece.name}: no CMPC section`);
      const rows = (layout.pages ?? []) as Row[];
      if (rows.some((r) => r.count > this.slotNodes)) {
        throw new Error(`${piece.name}: a page over ${this.slotNodes} nodes`);
      }
      this.pieces.push({
        piece,
        index,
        rows,
        section,
        compressed: layout.sections[section].compression === 1,
        pageBytes: layout.blocks.map((b) => b.spans[section].stored),
        lodId: -1,
        slotOf: new Map(),
        kids: new Map(),
        inFlight: new Set(),
      });
    });
  }

  /** The WASM trees: one pool tree, one shared tree a piece. */
  async init() {
    const { lodId } = (await this.options.lod("newLodTree", {
      capacity: this.options.slots * this.slotNodes,
      slot: this.slotNodes,
    })) as { lodId: number };
    this.poolLodId = lodId;
    for (const p of this.pieces) {
      p.lodId = (
        (await this.options.lod("newSharedLodTree", { lodId })) as {
          lodId: number;
        }
      ).lodId;
    }
  }

  /** Fetching or landed pages not yet in the pool. */
  get busy() {
    return this.fetching > 0 || this.landed.length > 0;
  }

  /**
   * One frame: the pages that landed go into the pool (evicting what is
   * not needed, finest first), the traversal cuts the resident tree, its
   * cut becomes `path`'s draw list, and the missing needed pages are asked
   * for. `width` x `height`: the target, for the cut's pixel.
   */
  async update(
    camera: THREE.PerspectiveCamera,
    height: number,
    path?: CompactRelitPath,
  ) {
    const lod = this.options.lod;
    // 1. Landed pages into slots.
    const ranges: Record<string, unknown>[] = [];
    const landed = this.landed;
    this.landed = [];
    for (const { piece, page, raw } of landed) {
      piece.inFlight.delete(page);
      if (piece.slotOf.has(page)) continue;
      const slot = this.takeSlot();
      if (slot === undefined) continue; // no room: dropped, asked again
      const row = piece.rows[page];
      const { lodTree, lodError } = this.pool.writePage(
        slot,
        piece.index,
        row.count,
        raw,
        row.compact,
      );
      piece.slotOf.set(page, slot);
      this.slotOwner[slot] = { piece: piece.index, page };
      const kids = new Set<number>();
      for (let i = 0; i < row.count; i++) {
        if ((lodTree[4 * i + 2] & 0xffff) === 0) continue;
        const c = lodTree[4 * i + 3] >>> 16;
        if (c !== page) kids.add(c);
      }
      piece.kids.set(page, [...kids]);
      ranges.push({
        lodId: piece.lodId,
        pageBase: slot * this.slotNodes,
        chunkBase: page * 65536,
        count: row.count,
        lodTreeData: lodTree,
        lodErrorData: lodError,
      });
      this.stats.landed += 1;
    }
    for (const r of this.evictedRanges) ranges.unshift(r);
    this.evictedRanges = [];
    if (ranges.length) await lod("updateLodTrees", { ranges });

    // 2. The cut over the resident pages.
    camera.updateMatrixWorld();
    const tanY = Math.tan((camera.fov * Math.PI) / 360);
    const pixel = (2 * tanY) / height;
    const eye = new THREE.Vector3().setFromMatrixPosition(camera.matrixWorld);
    if (this.options.tauToFit) {
      this.tau = Math.max(this.options.tau ?? 1, this.tauToFitNow(eye, pixel));
    }
    this.stats.tau = this.tau;
    const errorLimit = pixel * this.tau;
    const instances: Record<string, unknown> = {};
    for (const p of this.pieces) {
      const root = p.slotOf.get(0);
      if (root === undefined) continue;
      instances[p.piece.name] = {
        instanceId: p.piece.name,
        lodId: p.lodId,
        rootPage: root,
        viewToObjectCols: camera.matrixWorld.elements.slice(),
        lodScale: 1,
        behindFoveate: 1,
        coneFov0: 0,
        coneFov: 0,
        coneFoveate: 1,
      };
    }
    let list = new Uint32Array(0);
    if (Object.keys(instances).length) {
      const t0 = performance.now();
      const { keyIndices } = (await lod("traverseLodTrees", {
        maxSplats: this.options.slots * this.slotNodes,
        pixelScaleLimit: pixel,
        instances,
        traverseMode: "standard",
        errorLimit,
      })) as {
        keyIndices: Record<string, { numSplats: number; indices: Uint32Array }>;
      };
      this.stats.traverseMs = performance.now() - t0;
      const parts = Object.values(keyIndices).map((k) =>
        k.indices.subarray(0, k.numSplats),
      );
      list = new Uint32Array(parts.reduce((n, p) => n + p.length, 0));
      let at = 0;
      for (const part of parts) {
        list.set(part, at);
        at += part.length;
      }
    }
    this.drawList = list;
    this.stats.cut = list.length;
    path?.setDrawList(list);

    // 3. The needed set, requests, eviction order.
    const meshes: BandMesh<PieceState>[] = this.pieces.map((p) => ({
      key: p,
      rows: p.rows,
      pageBytes: p.pageBytes,
      eye: [eye.x, eye.y, eye.z],
      limit: errorLimit,
      resident: (page) => p.slotOf.has(page),
      inFlight: (page) => p.inFlight.has(page),
      children: (page) => p.kids.get(page),
    }));
    const plan = planBand(
      meshes,
      this.options.slots,
      this.options.maxRangeBytes ?? 8 << 20,
    );
    this.lastPlan = plan;
    const s = this.stats;
    s.neededPages = plan.neededPages;
    s.neededNodes = plan.neededNodes;
    let rp = 0;
    let rn = 0;
    for (const p of this.pieces) {
      for (const page of p.slotOf.keys()) {
        rp += 1;
        rn += p.rows[page].count;
      }
    }
    s.residentPages = rp;
    s.residentNodes = rn;
    const requested = plan.requests.reduce((n, r) => n + r.pages.length, 0);
    s.starved = Math.max(0, plan.neededPages - rp - requested);
    const fetchers = this.options.fetchers ?? 4;
    for (const r of plan.requests) {
      if (this.fetching >= fetchers) break;
      if (r.pages.some((page) => r.key.inFlight.has(page))) continue;
      this.fetchRun(r.key, r.pages);
    }
  }

  private evictedRanges: Record<string, unknown>[] = [];

  /** A free slot, else one evicted from the last plan's order. */
  private takeSlot(): number | undefined {
    const free = this.freeSlots.pop();
    if (free !== undefined) return free;
    const order = this.lastPlan?.evictable ?? [];
    while (order.length) {
      const { key, page } = order.shift() as { key: PieceState; page: number };
      const slot = key.slotOf.get(page);
      if (slot === undefined) continue;
      key.slotOf.delete(page);
      key.kids.delete(page);
      this.slotOwner[slot] = null;
      this.evictedRanges.push({
        lodId: key.lodId,
        pageBase: slot * this.slotNodes,
        chunkBase: page * 65536,
        count: key.rows[page].count,
        lodTreeData: undefined,
      });
      this.stats.evicted += 1;
      return slot;
    }
    return undefined;
  }

  private tauToFitNow(eye: THREE.Vector3, pixel: number) {
    const t: number[] = [];
    for (const p of this.pieces) {
      p.rows.forEach((row, page) => {
        if (page === 0) return;
        let d2 = 0;
        for (let k = 0; k < 3; k++) {
          const e = [eye.x, eye.y, eye.z][k];
          const v = Math.max(row.lo[k] - e, 0, e - row.hi[k]);
          d2 += v * v;
        }
        t.push(row.epsMax / Math.max(Math.sqrt(d2), BAND_NEAR) / pixel);
      });
    }
    const room = this.options.slots - this.pieces.length;
    if (t.length <= room) return 0;
    t.sort((a, b) => b - a);
    return t[Math.max(0, room)] * (1 + 1e-6);
  }

  private fetchRun(p: PieceState, pages: number[]) {
    for (const page of pages) p.inFlight.add(page);
    this.fetching += 1;
    const s = this.stats;
    s.pagesRequested += pages.length;
    s.ranges += 1;
    const blocks = p.piece.layout.blocks;
    const first = blocks[pages[0]].spans[p.section];
    const last = blocks[pages[pages.length - 1]].spans[p.section];
    const start = first.offset;
    const bytes = last.offset + last.stored - start;
    s.bytesRead += bytes;
    p.piece
      .read(start, bytes)
      .then(async (data) => {
        for (const page of pages) {
          const span = blocks[page].spans[p.section];
          const stored = data.subarray(
            span.offset - start,
            span.offset - start + span.stored,
          );
          const raw = p.compressed
            ? await this.options.gunzip(stored)
            : stored.slice();
          this.landed.push({ piece: p, page, raw });
        }
      })
      .catch((e) => {
        for (const page of pages) p.inFlight.delete(page);
        console.warn(e);
      })
      .finally(() => {
        this.fetching -= 1;
      });
  }

  /**
   * Every page of every piece asked for (consecutive pages in Ranges of at
   * most maxRangeBytes), for a reference with the whole tree resident: the
   * next update() puts them in the pool (which must hold them all).
   */
  async loadAll() {
    const max = this.options.maxRangeBytes ?? 8 << 20;
    for (const p of this.pieces) {
      let run: number[] = [];
      let bytes = 0;
      for (let page = 0; page < p.rows.length; page++) {
        if (run.length && bytes + p.pageBytes[page] > max) {
          this.fetchRun(p, run);
          run = [];
          bytes = 0;
        }
        run.push(page);
        bytes += p.pageBytes[page];
      }
      if (run.length) this.fetchRun(p, run);
    }
    await this.idle();
  }

  /** Waits for the fetches in flight. */
  async idle() {
    while (this.fetching > 0) await new Promise((r) => setTimeout(r, 1));
  }

  /** Resident bytes: the GPU pool (tables, slots), and the traversal's CPU nodes. */
  residentBytes() {
    return {
      gpuTables: this.pool.tableWords * 4,
      gpuSlots: this.options.slots * this.pool.slotWords * 4,
      /** spark-rs LodSplat: 24 bytes a pool node. */
      cpuTraversal: 24 * this.options.slots * this.slotNodes,
    };
  }

  /** The (piece, page, node) of slot index i, or null. */
  nodeOf(i: number) {
    const slot = Math.floor(i / this.slotNodes);
    const owner = this.slotOwner[slot];
    if (!owner) return null;
    return { ...owner, node: i - slot * this.slotNodes };
  }

  dispose() {
    this.pool.dispose();
  }
}

// --- in the browser ----------------------------------------------------------

/** A paged .athc with its compact tier, read over HTTP Range requests. */
export async function openCompactPiece(
  url: string,
  name = url,
): Promise<CompactPiece> {
  const { layout, prefix } = await readAthcLayout({ url });
  return {
    name,
    layout: layout as AthcV3Layout,
    prefix,
    read: async (offset, bytes) =>
      (await readRange({ url }, offset, bytes)).data,
  };
}

/** gunzip with the browser's DecompressionStream. */
export async function gunzipStream(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([bytes as BlobPart])
    .stream()
    .pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** The LoD calls on a loader worker of their own (its WASM keeps the trees). */
export function workerLodCall(worker = new SplatWorker()): LodCall {
  return (name, args) =>
    worker.call(name as never, args as never) as Promise<unknown>;
}

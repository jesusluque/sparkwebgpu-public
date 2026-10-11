// THE COMPACT PATH OVER BAND PAGES (thread CN): athenea's error-band pages
// (a paged .athc tree, docs/docs/athc-v3.md "Error-band pages") drawn by the
// compact relit path (compact.ts) from their fourth tier alone.
//
// A paged tree written with `usd-athc --tree ... --pages --compact-basis`
// carries, per page, a `CMPC` section (tier 4, after tier 3, so consecutive
// pages are one Range): NODE_BYTES (56) a node as byte planes, the 40 the
// GPU keeps then the traversal's 16 (v2, thread CQ: the centre as a
// correction to the compact position, the children, LODE's eps and rho);
// the page's quantization box in its page-table row (words 17..29); and,
// among the file's tables, the piece's shadow dictionary, material table
// and athenea's joint CPCA basis in f16 -- or a reference to the ASSET
// table (a `.cmpa` beside the pieces) that holds the basis once, and the
// dictionary and materials when the pieces share them (rust athc_compact.rs).
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
import {
  BAND_NEAR,
  type BandMesh,
  type Foveation,
  pageFoveate,
  planBand,
} from "../WgpuBandPager";
import type { CompactCloud, CompactLayout, CompactRelitPath } from "./compact";
import {
  CompactNodePool,
  NODE_RECORD_WORDS,
  NodeSlotAllocator,
  PAGE_PLACES,
  PagePayloadStore,
  packPageRecords,
} from "./compactNodes";

/**
 * Bytes a node in a CMPC section, by plane (rust athc_compact PLANE_WIDTHS):
 * v3 (thread CS) appends the cone word (4), the occlusion index (2) and 2
 * spare to v2's 31 planes, at the same offsets.
 */
export const COMPACT_PLANE_WIDTHS = [
  2, 2, 2, 3, 1, 3, 4, 3, 3, 2, 1, 4, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
] as const;
export const COMPACT_NODE_BYTES = 64;
export const COMPACT_NODE_BYTES_V2 = 56;
const COMPACT_PLANES_V2 = 31;
/** The GPU's bytes a node, the traversal's (rust GPU_NODE_BYTES, TRAVERSAL_NODE_BYTES). */
export const COMPACT_GPU_NODE_BYTES = 40;
export const COMPACT_TRAVERSAL_NODE_BYTES = 24;
export const COMPACT_TRAVERSAL_NODE_BYTES_V2 = 16;
/** Bytes a node of a compact LoD tree in the WASM traversal (CompactLodNode, v3). */
export const COMPACT_LOD_NODE_BYTES = 20;
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
/** The traversal's planes (one byte each): the children's page, their word. */
const P_CHILD_PAGE = 21;
const P_KIDS = 23;
/** Words a slot head (compact.slang kCompactChunkWords, rounded to 16). */
const HEAD_WORDS = 12;
const SLOT_HEAD_WORDS = 16;
const TABLE_MAGIC = 0x54504d43; // "CMPT"
const TABLE_VERSION = 3;
const ASSET_MAGIC = 0x41504d43; // "CMPA"
/** CMPT flags: the basis, the dictionary, the materials in the asset table. */
const TABLE_ASSET_BASIS = 4;
const TABLE_ASSET_CODEBOOK = 8;
const TABLE_ASSET_MATERIALS = 16;
/** v3: the occlusion codebook in the asset table. */
const TABLE_ASSET_OCCLUSION = 32;

/** Byte offset of plane k in a section of n nodes. */
export function compactPlaneOffset(k: number, n: number) {
  let at = 0;
  for (let j = 0; j < k; j++) at += COMPACT_PLANE_WIDTHS[j];
  return at * n;
}

/** A piece's compact tables (rust athc_compact CompactTables). */
export interface CompactPieceTables {
  /** CMPT 2 (no cones, no occlusion indices) or 3 (thread CS). */
  version: number;
  codebook: Uint32Array;
  materials: Uint32Array;
  basis: Uint32Array;
  clusters: number;
  normalBits: number;
  weightBits: number;
  linear: boolean;
  curvature: boolean;
  /**
   * What the asset table holds for the piece (TABLE_ASSET_* bits, 0: the
   * piece holds its own), its id and its file name beside the piece.
   */
  assetFlags: number;
  assetId: bigint;
  assetName: string;
}

/** An asset's shared compact tables (rust athc_compact CompactAsset, `.cmpa`). */
export interface CompactAssetTables {
  basis: Uint32Array;
  clusters: number;
  codebook: Uint32Array;
  materials: Uint32Array;
  /**
   * athenea's occlusion codebook (8 words an entry; empty: none) and the
   * asset's box (min xyz, max xyz), once per asset (thread CS).
   */
  occlusion: Uint32Array;
  occlusionBox: Float32Array;
  id: bigint;
  bytes: number;
}

function wordsOf(bytes: Uint8Array) {
  return new Uint32Array(
    bytes.buffer.slice(
      bytes.byteOffset,
      bytes.byteOffset + (bytes.byteLength & ~3),
    ),
  );
}

/** FNV-1a, 64 bits (rust athc_compact fnv1a64). */
function fnv1a64(bytes: Uint8Array) {
  let h = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  const mask = 0xffffffffffffffffn;
  for (let i = 0; i < bytes.length; i++) {
    h ^= BigInt(bytes[i]);
    h = (h * prime) & mask;
  }
  return h;
}

/** The tables blob (CompactTables::to_bytes, v2). */
export function parseCompactTables(bytes: Uint8Array): CompactPieceTables {
  const w = wordsOf(bytes);
  if (w[0] !== TABLE_MAGIC || (w[1] !== TABLE_VERSION && w[1] !== 2)) {
    throw new Error(
      `compact tables: not CMPT v2 or v${TABLE_VERSION} (a v1 file: athc-convert --compact-rebase)`,
    );
  }
  const v2 = w[1] === 2;
  if (
    w[2] !== (v2 ? COMPACT_NODE_BYTES_V2 : COMPACT_NODE_BYTES) ||
    w[3] !== (v2 ? COMPACT_PLANES_V2 : COMPACT_PLANE_WIDTHS.length)
  ) {
    throw new Error("compact tables: another layout");
  }
  const name = bytes.subarray(4 * w[18], 4 * w[18] + w[19]);
  return {
    version: w[1],
    codebook: w.slice(w[11], w[11] + 8 * w[5]),
    materials: w.slice(w[12], w[12] + 4 * w[6]),
    basis: w.slice(w[13], w[13] + w[14]),
    clusters: w[7],
    normalBits: w[8],
    weightBits: w[9],
    linear: (w[10] & 1) !== 0,
    curvature: (w[10] & 2) !== 0,
    assetFlags:
      w[10] &
      (TABLE_ASSET_BASIS |
        TABLE_ASSET_CODEBOOK |
        TABLE_ASSET_MATERIALS |
        TABLE_ASSET_OCCLUSION),
    assetId: BigInt(w[16]) | (BigInt(w[17]) << 32n),
    assetName: new TextDecoder().decode(name),
  };
}

/** An asset table (`.cmpa`, CompactAsset::to_bytes), its id checked. */
export function parseCompactAsset(bytes: Uint8Array): CompactAssetTables {
  const w = wordsOf(bytes);
  if (w[0] !== ASSET_MAGIC || w[1] !== 1) {
    throw new Error("compact asset table: not CMPA v1");
  }
  const total = w[2];
  if (w.length < total) throw new Error("compact asset table: too short");
  const zeroed = new Uint8Array(bytes.subarray(0, 4 * total));
  zeroed.fill(0, 44, 52);
  const id = BigInt(w[11]) | (BigInt(w[12]) << 32n);
  if (fnv1a64(zeroed) !== id) {
    throw new Error("compact asset table: its id does not match its bytes");
  }
  return {
    clusters: w[4],
    basis: w.slice(w[7], w[7] + w[8]),
    codebook: w.slice(w[9], w[9] + 8 * w[5]),
    materials: w.slice(w[10], w[10] + 4 * w[6]),
    occlusion:
      w[3] & 8 ? w.slice(w[13], w[13] + 8 * w[14]) : new Uint32Array(0),
    occlusionBox:
      w[3] & 8
        ? new Float32Array(
            w.slice(w[13] + 8 * w[14], w[13] + 8 * w[14] + 6).buffer,
          )
        : new Float32Array(6),
    id,
    bytes: 4 * total,
  };
}

/**
 * A piece's tables as the pool uses them: the asset's arrays (the same
 * objects, so the pool keeps them once) where the piece leaves them to it.
 */
export function resolveCompactTables(
  t: CompactPieceTables,
  asset?: CompactAssetTables,
): CompactPieceTables {
  if (!t.assetFlags) return t;
  if (!asset) {
    throw new Error(
      `compact tables: the piece needs asset table ${t.assetName}`,
    );
  }
  if (asset.id !== t.assetId) {
    throw new Error(
      `compact tables: asset table ${asset.id.toString(16)}, the piece wants ${t.assetId.toString(16)}`,
    );
  }
  return {
    ...t,
    basis: t.assetFlags & TABLE_ASSET_BASIS ? asset.basis : t.basis,
    codebook: t.assetFlags & TABLE_ASSET_CODEBOOK ? asset.codebook : t.codebook,
    materials:
      t.assetFlags & TABLE_ASSET_MATERIALS ? asset.materials : t.materials,
    assetFlags: t.assetFlags & TABLE_ASSET_OCCLUSION,
  };
}

/**
 * The traversal's input of a page (rust athc_compact traversal_input): its
 * position planes then the traversal's planes, 22 bytes a node (v2) or 30 (v3).
 */
export function compactTraversalInput(raw: Uint8Array, n: number) {
  const t =
    raw.length === n * COMPACT_NODE_BYTES_V2
      ? COMPACT_TRAVERSAL_NODE_BYTES_V2
      : COMPACT_TRAVERSAL_NODE_BYTES;
  const out = new Uint8Array((6 + t) * n);
  out.set(raw.subarray(0, 6 * n), 0);
  const at = COMPACT_GPU_NODE_BYTES * n;
  out.set(raw.subarray(at, at + t * n), 6 * n);
  return out;
}

/** The pages a page's nodes have children in (other than itself). */
export function compactChildPages(raw: Uint8Array, n: number, page: number) {
  const pages = new Set<number>();
  const pageAt = compactPlaneOffset(P_CHILD_PAGE, n);
  const kidsAt = compactPlaneOffset(P_KIDS, n);
  for (let i = 0; i < n; i++) {
    // The count: bits 12-23 of the children word (bytes 1 and 2).
    const count = (raw[kidsAt + n + i] >> 4) | (raw[kidsAt + 2 * n + i] << 4);
    if (count === 0) continue;
    const c = (page + (raw[pageAt + i] | (raw[pageAt + n + i] << 8))) & 0xffff;
    if (c !== page) pages.add(c);
  }
  return [...pages];
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
      if (p.assetFlags & ~TABLE_ASSET_OCCLUSION) {
        throw new Error(
          "CompactPagePool: resolve the pieces' asset tables first",
        );
      }
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
    // Tables: the basis once, then each piece's codebook and materials --
    // once each when the pieces share their asset's (the same arrays).
    const tables: number[] = [];
    const placed = new Map<Uint32Array, number>();
    const push = (w: Uint32Array) => {
      const known = placed.get(w);
      if (known !== undefined) return known;
      const at = Math.ceil(tables.length / 16) * 16;
      while (tables.length < at) tables.push(0);
      for (const v of w) tables.push(v);
      placed.set(w, at);
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
      materials: [...new Set(pieces.map((p) => p.materials))].reduce(
        (n, m) => n + m.length / 4,
        0,
      ),
      codebookEntries: [...new Set(pieces.map((p) => p.codebook))].reduce(
        (n, c) => n + c.length / 8,
        0,
      ),
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
   * traversal's input (compactTraversalInput: 22 bytes a node) and its box.
   */
  writePage(
    slot: number,
    piece: number,
    page: number,
    n: number,
    raw: Uint8Array,
    head: ArrayLike<number>,
  ) {
    if (n > this.slotNodes) {
      throw new Error(
        `CompactPagePool: a page of ${n} in slots of ${this.slotNodes}`,
      );
    }
    if (
      raw.length !== n * COMPACT_NODE_BYTES &&
      raw.length !== n * COMPACT_NODE_BYTES_V2
    ) {
      throw new Error(`CompactPagePool: ${raw.length} bytes for ${n} nodes`);
    }
    const out = this.staging;
    out.fill(0);
    const w = new Uint32Array(out.buffer);
    for (let k = 0; k < HEAD_WORDS; k++) w[k] = head[k] >>> 0;
    w[9] = this.pieceBase[piece].codebook;
    w[10] = this.pieceBase[piece].materials;
    // The faint dither's seed: the node's place in its piece, not its slot.
    w[11] = ((piece << 26) + page * 4096) >>> 0;
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
    return {
      input: compactTraversalInput(raw, n),
      box: new Float32Array(
        Uint32Array.from({ length: 6 }, (_, k) => head[k] >>> 0).buffer,
      ),
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
    | "updateCompactLodTrees"
    | "setLodOcclusionCodebook"
    | "traverseLodTrees"
    | "wasmMemory",
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
  /**
   * Spark's foveation (thread CS): nodes outside the view cut coarser, and
   * the pages planned with the same weight (pageFoveate). Default: full
   * detail within the camera's diagonal field + 8 degrees, coneFoveate 0.4
   * at 30 degrees more, behindFoveate 0.2 (WgpuLod's). null: none (CN's).
   */
  foveation?: Partial<Foveation> | null;
  /**
   * The trees' object-to-world (the car's frame: athenea's Z-up stage
   * turned to three's Y-up). The traversal and the planner take the eye in
   * the trees' own space (thread CS: CN's took the world's, the eye 90
   * degrees off about x for a Z-up car). Default identity.
   */
  frame?: THREE.Matrix4;
  /**
   * The back-face cull of closed parts (athenea:splat:closed, CMPC v3's
   * cones): a node or leaf whose whole cone faces away (n . v < -sin theta)
   * is not drawn. Default true (v2 pieces carry no cones: nothing culled).
   */
  backCull?: boolean;
  /**
   * athenea's occlusion masks (the asset table's codebook, CMPC v3's
   * indices): opt-in, default false.
   */
  occlusion?: boolean;
  maxRangeBytes?: number;
  /** Fetches in flight. */
  fetchers?: number;
  /**
   * The draw list in node order (piece, page, place) rather than the
   * traversal's: an image that does not depend on which slots the pages
   * landed in (for checks; costs a sort of the cut each traversal).
   */
  nodeOrder?: boolean;
  /** gunzip a stored section (DecompressionStream, or node's zlib). */
  gunzip: (bytes: Uint8Array) => Promise<Uint8Array>;
  /**
   * The asset table the pieces share (`openCompactAsset`, a `.cmpa`), for
   * pieces written with one (thread CQ): its basis held once.
   */
  asset?: CompactAssetTables;
  /** The WASM traversal. */
  lod: LodCall;
  /**
   * The node pool (thread CT, compactNodes.ts): the GPU holds the drawn
   * cut's records only (`capacity` to start, grown x1.5 up to
   * `maxCapacity`, the traversal's budget), the resident pages' payload
   * stays compressed on the CPU (`decodedPages` decompressed, an LRU).
   * Absent: whole pages in GPU slots (CompactPagePool).
   */
  nodePool?: {
    capacity?: number;
    maxCapacity?: number;
    decodedPages?: number;
    /** Synchronous gunzip (default fflate's). */
    gunzipSync?: (b: Uint8Array) => Uint8Array;
  };
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

/** The compact tables of a piece, from its prefix (the asset's where it shares them). */
export function compactTablesOf(
  piece: CompactPiece,
  asset?: CompactAssetTables,
): CompactPieceTables {
  const t = (piece.layout as unknown as { compactTable?: number[] })
    .compactTable;
  if (!t?.[0]) throw new Error(`${piece.name}: no compact tier`);
  const own = parseCompactTables(piece.prefix.subarray(t[0], t[0] + t[1]));
  return resolveCompactTables(own, asset);
}

/** Streams the pieces' band pages' tier 4 into a CompactPagePool. */
export class CompactBandStreamer {
  readonly pool: CompactPagePool | CompactNodePool;
  /** The node pool's slots and the pages' CPU payload (options.nodePool). */
  readonly nodes: NodeSlotAllocator | null = null;
  readonly payload: PagePayloadStore | null = null;
  /** Each piece's first page in the node pool's head table. */
  private pageBase: number[] = [];
  /** A head-table page's piece and page. */
  private pageOwner: { piece: number; page: number }[] = [];
  readonly pieces: PieceState[] = [];
  readonly slotNodes: number;
  private freeSlots: number[] = [];
  private slotOwner: ({ piece: number; page: number } | null)[];
  private poolLodId = -1;
  private landed: {
    piece: PieceState;
    page: number;
    raw: Uint8Array;
    /** The section as stored (gzip), when it was. */
    stored: Uint8Array | null;
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
    /** Nodes the last traversal culled from behind (closed parts), and by occlusion. */
    culled: 0,
    occluded: 0,
    tau: 0,
    traverseMs: 0,
    starved: 0,
    /** Node pool: nodes uploaded, their writes, gunzips and their ms, at the last update. */
    uploaded: 0,
    writes: 0,
    decodes: 0,
    decodeMs: 0,
    packMs: 0,
    /** Node pool: cut nodes with no payload (not drawn), at the last update. */
    missing: 0,
  };

  constructor(
    readonly device: GPUDevice,
    pieces: CompactPiece[],
    readonly options: CompactBandOptions,
  ) {
    this.slotNodes = options.slotNodes ?? 3072;
    this.tau = options.tau ?? 1;
    const tables = pieces.map((p) => compactTablesOf(p, options.asset));
    const np = options.nodePool;
    if (np) {
      const heads = pieces.map((p) =>
        ((p.layout.pages ?? []) as Row[]).map((r) => r.compact),
      );
      let g = 0;
      pieces.forEach((p, piece) => {
        this.pageBase.push(g);
        const n = p.layout.pages?.length ?? 0;
        for (let page = 0; page < n; page++) {
          this.pageOwner.push({ piece, page });
        }
        g += n;
      });
      const max = np.maxCapacity ?? options.slots * this.slotNodes;
      const pool = new CompactNodePool(
        device,
        tables,
        heads,
        np.capacity ?? 65536,
        max,
      );
      this.pool = pool;
      this.nodes = new NodeSlotAllocator(pool.capacity);
      this.payload = new PagePayloadStore(np.decodedPages ?? 8, np.gunzipSync);
    } else {
      this.pool = new CompactPagePool(
        device,
        tables,
        options.slots,
        this.slotNodes,
      );
    }
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
      compact: true,
    })) as { lodId: number };
    this.poolLodId = lodId;
    // athenea's occlusion codebook, once per asset (the WASM keeps it a
    // tree), and the asset's box its rule tests the eye against.
    const asset = this.options.asset;
    const occlusion = asset?.occlusion.length ? asset : null;
    for (const p of this.pieces) {
      p.lodId = (
        (await this.options.lod("newSharedLodTree", { lodId })) as {
          lodId: number;
        }
      ).lodId;
      if (occlusion) {
        await this.options.lod("setLodOcclusionCodebook", {
          lodId: p.lodId,
          // Copies: the worker's call transfers (detaches) its buffers.
          codebook: occlusion.occlusion.slice(),
          bounds: occlusion.occlusionBox.slice(),
        });
      }
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
    for (const { piece, page, raw, stored } of landed) {
      piece.inFlight.delete(page);
      if (piece.slotOf.has(page)) continue;
      const slot = this.takeSlot();
      if (slot === undefined) continue; // no room: dropped, asked again
      const row = piece.rows[page];
      const { input, box } =
        this.pool instanceof CompactPagePool
          ? this.pool.writePage(
              slot,
              piece.index,
              page,
              row.count,
              raw,
              row.compact,
            )
          : this.landNodes(piece, page, raw, stored);
      piece.slotOf.set(page, slot);
      this.slotOwner[slot] = { piece: piece.index, page };
      piece.kids.set(page, compactChildPages(raw, row.count, page));
      ranges.push({
        lodId: piece.lodId,
        pageBase: slot * this.slotNodes,
        chunkBase: page * 65536,
        count: row.count,
        input,
        box,
      });
      this.stats.landed += 1;
    }
    if (this.evictedSlots.size && this.nodes) {
      // Node pool: the records outlive their page (keyed by page and place).
      this.evictedSlots.clear();
    }
    if (this.evictedSlots.size) {
      // The pages just evicted are gone from the pool: out of the draw list
      // now, before a frame drawn while the traversal runs reads their slots.
      const keep = this.drawList.filter(
        (i) => !this.evictedSlots.has(Math.floor(i / this.slotNodes)),
      );
      this.evictedSlots.clear();
      this.drawList = keep;
      path?.setDrawList(keep);
    }
    for (const r of this.evictedRanges) ranges.unshift(r);
    this.evictedRanges = [];
    if (ranges.length) await lod("updateCompactLodTrees", { ranges });

    // 2. The cut over the resident pages.
    camera.updateMatrixWorld();
    const tanY = Math.tan((camera.fov * Math.PI) / 360);
    const pixel = (2 * tanY) / height;
    // The view in the trees' space.
    const view = this.options.frame
      ? this.options.frame.clone().invert().multiply(camera.matrixWorld)
      : camera.matrixWorld;
    const eye = new THREE.Vector3().setFromMatrixPosition(view);
    const fov = this.foveationFor(camera);
    const forward = new THREE.Vector3()
      .setFromMatrixColumn(view, 2)
      .negate()
      .normalize();
    const eyeArr = [eye.x, eye.y, eye.z];
    const fwdArr = [forward.x, forward.y, forward.z];
    const foveateFor = (p: PieceState) =>
      fov
        ? (page: number) => pageFoveate(p.rows[page], eyeArr, fwdArr, fov)
        : undefined;
    if (this.options.tauToFit) {
      const fit = Math.max(
        this.options.tau ?? 1,
        this.tauToFitNow(eye, pixel, foveateFor),
      );
      // Pruned (wanted pages): the set grows as tau falls, so tau comes
      // down by at most 1/1.25 a traversal (up at once: never past the pool).
      const next =
        this.wanted && fit < this.tau ? Math.max(fit, this.tau / 1.25) : fit;
      // Still coming down: the caller keeps traversing a still view.
      this.converging = next < this.tau * 0.999;
      this.tau = next;
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
        viewToObjectCols: view.elements.slice(),
        lodScale: 1,
        behindFoveate: fov ? fov.behindFoveate : 1,
        coneFov0: fov ? fov.coneFov0 : 0,
        coneFov: fov ? fov.coneFov : 0,
        coneFoveate: fov ? fov.coneFoveate : 1,
      };
    }
    let list = new Uint32Array(0);
    if (Object.keys(instances).length) {
      const t0 = performance.now();
      const { keyIndices, culledCount, occludedCount, chunks } = (await lod(
        "traverseLodTrees",
        {
          maxSplats:
            this.pool instanceof CompactNodePool
              ? this.pool.maxCapacity
              : this.options.slots * this.slotNodes,
          pixelScaleLimit: pixel,
          instances,
          traverseMode: "standard",
          errorLimit,
          // Negative: any size, pruned with the subtree (spark-rs).
          backCullMin: this.options.backCull === false ? undefined : -1,
          occlusionCull: this.options.occlusion === true,
        },
      )) as {
        keyIndices: Record<string, { numSplats: number; indices: Uint32Array }>;
        culledCount?: number;
        occludedCount?: number;
        chunks?: [number, number][];
      };
      this.stats.traverseMs = performance.now() - t0;
      // The pages the traversal asked for, by tree (spark-rs touched).
      if (this.pruning() && chunks) {
        const wanted = new Map<number, Set<number>>();
        for (const [lodId, chunk] of chunks) {
          const set = wanted.get(lodId) ?? new Set<number>();
          wanted.set(lodId, set);
          set.add(chunk);
        }
        this.wanted = wanted;
      } else {
        this.wanted = null;
      }
      this.stats.culled = culledCount ?? 0;
      this.stats.occluded = occludedCount ?? 0;
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
    if (this.options.nodeOrder) {
      // By node (piece, page, place), not by slot: the same order whatever
      // slots the pages landed in (equal sort keys keep it).
      const id = new Float64Array(list.length);
      for (let k = 0; k < list.length; k++) {
        const slot = Math.floor(list[k] / this.slotNodes);
        const o = this.slotOwner[slot] as { piece: number; page: number };
        id[k] =
          (o.piece * 65536 + o.page) * 65536 +
          (list[k] - slot * this.slotNodes);
      }
      const order = Array.from(list.keys()).sort((x, y) => id[x] - id[y]);
      list = Uint32Array.from(order, (k) => list[k]);
    }
    if (this.nodes) list = this.toNodeSlots(list, path);
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
      foveate: foveateFor(p),
      wanted: this.wantedOf(p),
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

  /** A landed page, node pool: its payload kept on the CPU, its traversal input. */
  private landNodes(
    piece: PieceState,
    page: number,
    raw: Uint8Array,
    stored: Uint8Array | null,
  ) {
    const row = piece.rows[page];
    (this.payload as PagePayloadStore).put(
      this.pageBase[piece.index] + page,
      row.count,
      stored,
      raw,
    );
    return {
      input: compactTraversalInput(raw, row.count),
      box: new Float32Array(
        Uint32Array.from({ length: 6 }, (_, k) => row.compact[k] >>> 0).buffer,
      ),
    };
  }

  /**
   * The traversal's cut (pool slot indices) to node slots: the nodes held
   * are reused, the others get a slot (the pool grown if the cut outgrew
   * it) and their records packed from their page's payload and written.
   * The list keeps the cut's order; a node with no payload is left out.
   */
  private toNodeSlots(list: Uint32Array, path?: CompactRelitPath) {
    const alloc = this.nodes as NodeSlotAllocator;
    const pool = this.pool as CompactNodePool;
    const store = this.payload as PagePayloadStore;
    const N = this.slotNodes;
    const t0 = performance.now();
    const d0 = store.stats.decodes;
    const ms0 = store.stats.decodeMs;
    if (list.length > alloc.capacity && pool.grow(list.length)) {
      alloc.grow(pool.capacity);
      path?.setPagedCloud(pool);
    }
    alloc.begin();
    const keys = new Float64Array(list.length);
    const out = new Uint32Array(list.length);
    const miss: number[] = [];
    for (let k = 0; k < list.length; k++) {
      const i = list[k];
      const slot = Math.floor(i / N);
      const o = this.slotOwner[slot] as { piece: number; page: number };
      const key =
        (this.pageBase[o.piece] + o.page) * PAGE_PLACES + (i - slot * N);
      keys[k] = key;
      const s = alloc.use(key);
      if (s < 0) miss.push(k);
      else out[k] = s;
    }
    // The misses page by page (one section each), in place order.
    miss.sort((a, b) => keys[a] - keys[b]);
    const got: number[] = [];
    const gotSlot: number[] = [];
    let missing = 0;
    for (const k of miss) {
      const s = alloc.allocate(keys[k]);
      if (s < 0) {
        missing += 1;
        continue;
      }
      out[k] = s;
      got.push(k);
      gotSlot.push(s);
    }
    // Packed page by page (one section each, in key order), then written
    // in slot order (runs of slots, one write each).
    const R = NODE_RECORD_WORDS * 4;
    const packed = new Uint8Array(got.length * R);
    const dropped = new Set<number>();
    let j = 0;
    while (j < got.length) {
      const g = Math.floor(keys[got[j]] / PAGE_PLACES);
      let e = j + 1;
      while (e < got.length && Math.floor(keys[got[e]] / PAGE_PLACES) === g) {
        e++;
      }
      const owner = this.pageOwner[g];
      const n = this.pieces[owner.piece].rows[owner.page].count;
      const raw = store.sectionOf(g);
      const places = new Uint32Array(e - j);
      for (let q = j; q < e; q++)
        places[q - j] = keys[got[q]] - g * PAGE_PLACES;
      if (raw) packPageRecords(raw, n, places, g, packed, j);
      else for (let q = j; q < e; q++) dropped.add(got[q]);
      j = e;
    }
    const order = got.map((_, q) => q).sort((a, b) => gotSlot[a] - gotSlot[b]);
    const records = new Uint8Array(order.length * R);
    const slots = new Uint32Array(order.length);
    order.forEach((q, k) => {
      records.set(packed.subarray(q * R, q * R + R), k * R);
      slots[k] = gotSlot[q];
    });
    this.stats.writes = order.length ? pool.writeRecords(slots, records) : 0;
    this.stats.uploaded = order.length;
    this.stats.decodes = store.stats.decodes - d0;
    this.stats.decodeMs = store.stats.decodeMs - ms0;
    let result = out;
    if (missing || dropped.size) {
      // No slot or no payload: not drawn (the cut outgrew maxCapacity).
      const failed = new Set(dropped);
      for (const k of dropped) alloc.release(out[k]);
      const placed = new Set(got);
      for (const k of miss) if (!placed.has(k)) failed.add(k);
      result = out.filter((_, k) => !failed.has(k));
      this.stats.missing = failed.size;
    } else {
      this.stats.missing = 0;
    }
    alloc.commit(result);
    this.stats.packMs = performance.now() - t0 - this.stats.decodeMs;
    return result;
  }

  private evictedRanges: Record<string, unknown>[] = [];
  private evictedSlots = new Set<number>();

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
      this.payload?.drop(this.pageBase[key.index] + page);
      this.slotOwner[slot] = null;
      this.evictedSlots.add(slot);
      this.evictedRanges.push({
        lodId: key.lodId,
        pageBase: slot * this.slotNodes,
        chunkBase: page * 65536,
        count: key.rows[page].count,
        input: undefined,
      });
      this.stats.evicted += 1;
      return slot;
    }
    return undefined;
  }

  /** The foveation for this camera (options.foveation), or null. */
  private foveationFor(camera: THREE.PerspectiveCamera): Foveation | null {
    const o = this.options.foveation;
    if (o === null) return null;
    const tanY = Math.tan((camera.fov * Math.PI) / 360);
    const diag =
      (2 *
        Math.atan(tanY * Math.sqrt(1 + camera.aspect * camera.aspect)) *
        180) /
      Math.PI;
    const coneFov0 = o?.coneFov0 ?? Math.min(diag + 8, 179);
    return {
      coneFov0,
      coneFov: o?.coneFov ?? Math.min(coneFov0 + 30, 180),
      coneFoveate: o?.coneFoveate ?? 0.4,
      behindFoveate: o?.behindFoveate ?? 0.2,
    };
  }

  /**
   * Whether the traversal prunes what the page table cannot see (the back
   * cull of closed parts, the occlusion masks): the needed set is then the
   * pages it asked for (planBand's `wanted`).
   */
  private pruning() {
    const v3 = this.pool.pieces.some((t) => t.version >= 3);
    const occlusion = (this.options.asset?.occlusion.length ?? 0) > 0;
    return (
      (v3 && this.options.backCull !== false) ||
      (v3 && occlusion && this.options.occlusion === true)
    );
  }

  /** tau fell at the last update (pruned planning): not settled yet. */
  converging = false;

  /** The pages of each tree the last traversal asked for (null: no pruning). */
  private wanted: Map<number, Set<number>> | null = null;

  private wantedOf(p: PieceState) {
    const w = this.wanted;
    if (!w) return undefined;
    const set = w.get(p.lodId);
    return (page: number) => page === 0 || (set?.has(page) ?? false);
  }

  private tauToFitNow(
    eye: THREE.Vector3,
    pixel: number,
    foveateFor: (p: PieceState) => ((page: number) => number) | undefined,
  ) {
    const t: number[] = [];
    for (const p of this.pieces) {
      const fv = foveateFor(p);
      const wanted = this.wantedOf(p);
      p.rows.forEach((row, page) => {
        if (wanted && !wanted(page)) return;
        if (page === 0) return;
        let d2 = 0;
        for (let k = 0; k < 3; k++) {
          const e = [eye.x, eye.y, eye.z][k];
          const v = Math.max(row.lo[k] - e, 0, e - row.hi[k]);
          d2 += v * v;
        }
        const f = fv ? fv(page) : 1;
        t.push((f * row.epsMax) / Math.max(Math.sqrt(d2), BAND_NEAR) / pixel);
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
          this.landed.push({
            piece: p,
            page,
            raw,
            // The node pool keeps the stored form (its own copy, not the Range's).
            stored: this.payload && p.compressed ? stored.slice() : null,
          });
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

  /**
   * Resident bytes: the GPU pool (tables, slots or the node pool's heads and
   * records), the traversal's CPU nodes, and the node pool's CPU payload
   * (compressed, decompressed).
   */
  residentBytes() {
    /** spark-rs CompactLodNode: 20 bytes a pool node (v3). */
    const cpuTraversal =
      COMPACT_LOD_NODE_BYTES * this.options.slots * this.slotNodes;
    if (this.pool instanceof CompactNodePool) {
      const b = this.pool.bytes();
      return {
        gpuTables: b.tables + b.heads,
        gpuSlots: b.records,
        gpuHeads: b.heads,
        gpuRecordsUsed: (this.nodes?.used ?? 0) * NODE_RECORD_WORDS * 4,
        cpuTraversal,
        cpuCompressed: this.payload?.compressedBytes ?? 0,
        cpuDecoded: this.payload?.decodedBytes ?? 0,
      };
    }
    return {
      gpuTables: this.pool.tableWords * 4,
      gpuSlots: this.options.slots * this.pool.slotWords * 4,
      cpuTraversal,
    };
  }

  /** The (piece, page, node) of draw-list entry i (a slot index, or a node slot), or null. */
  nodeOf(i: number) {
    if (this.nodes) {
      const key = this.nodes.keyIn(i);
      if (key < 0) return null;
      const g = Math.floor(key / PAGE_PLACES);
      return { ...this.pageOwner[g], node: key - g * PAGE_PLACES };
    }
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

/**
 * An asset table (`.cmpa`) the pieces share: by its URL, or a piece's
 * (`assetName` beside it).
 */
export async function openCompactAsset(
  url: string,
): Promise<CompactAssetTables> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return parseCompactAsset(new Uint8Array(await response.arrayBuffer()));
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

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
  ENCODING_COMPACT_RUNS,
  NODE_RECORD_WORDS,
  NodeSlotAllocator,
  PAGE_PLACES,
  PagePayloadStore,
  RUN_CULLED,
  RUN_CUT,
  RunPayloadStore,
  packPageRecords,
  packRunRecords,
  parseRunsContainer,
  unpackRunsContainer,
} from "./compactNodes";
import type { CutMsg, CutPart, FromCutter, ToCutter } from "./nodeWorker";
import NodeWorker from "./nodeWorker?worker&inline";

import { COMPACT_PLANE_WIDTHS, compactPlaneOffset } from "./compactRecords";

export { COMPACT_PLANE_WIDTHS };
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
export { compactPlaneOffset };

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
export function compactChildPages(
  raw: Uint8Array,
  n: number,
  page: number,
  /** Bytes a node before the traversal's planes: 40 (a section), 6 (a traversal input). */
  before = COMPACT_GPU_NODE_BYTES,
) {
  const pages = new Set<number>();
  const shift = (before - COMPACT_GPU_NODE_BYTES) * n;
  const pageAt = compactPlaneOffset(P_CHILD_PAGE, n) + shift;
  const kidsAt = compactPlaneOffset(P_KIDS, n) + shift;
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
    | "wasmMemory"
    | "setCompactQuant",
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
  /** The file's absolute URL (the node pool's worker reads runs by Range itself). */
  url?: string;
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
   * The traversal nodes' quantization (thread CT, spark-rs CompactQuant):
   * bits for the centre, eps, rho, theta and the cone's axis (0: exact).
   */
  quant?: number[];
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
    /**
     * CMPC v4 pieces: bytes of payload runs kept beyond those the cut may
     * need (its nodes, their parents and children); default unbounded.
     */
    runBudget?: number;
    /** Inflated runs kept (an LRU), default 256. */
    inflatedRuns?: number;
    /**
     * The runs kept whatever the budget: those of the cut's nodes, parents
     * and children ("neighbours", default), or none (an LRU by last use:
     * a node whose run went is drawn as its parent or children until its
     * run comes again by Range).
     */
    protect?: "neighbours" | "none";
    /** New records a update at most (the rest drawn as parent or children meanwhile). */
    uploadsPerUpdate?: number;
    /** Range requests for dropped runs in flight at most; default 12. */
    runFetchers?: number;
    /**
     * Decode the misses largest on screen first (thread CU), default true;
     * false: in node order (CT's).
     */
    priority?: boolean;
    /**
     * How far ahead the view is extrapolated for the prefetch of dropped
     * runs (thread CU), ms; default 2.5 updates (at most 600); 0 off.
     */
    aheadMs?: number;
    /**
     * CMPC v4: the node pool's CPU side in a worker (nodeWorker.ts, thread
     * CU), default true: slots, runs, inflates, packing and draw lists; the
     * main thread only writes the records and sets the lists it sends.
     */
    decoder?: boolean;
    /** Runs inflated an update at most (the frame's decode budget); default 256. */
    inflatesPerUpdate?: number;
  };
}

type Row = AthcPageRow & { tier4Bytes: number; compact: number[] };

/** A streamed traversal's cut (the node pool's worker): chunks of any instance. */
interface StreamedCut {
  parts?: CutPart[];
  culled?: Uint32Array[];
}

/** The traversal's neighbours of each cut node (the node pool). */
interface CutNeighbours {
  parents: Uint32Array;
  firstKids: Uint32Array;
  kidCounts: Uint32Array;
  /** Its angle on screen, foveated (thread CU): the decode order. */
  screens?: Float32Array;
}

interface PieceState {
  piece: CompactPiece;
  index: number;
  rows: Row[];
  /** CMPC's index in the section table. */
  section: number;
  compressed: boolean;
  /** CMPC v4: each page a container (runs). */
  runs: boolean;
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
  /** CMPC v4 pieces' payload runs (node pool). */
  readonly runs: RunPayloadStore | null = null;
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
    raw: Uint8Array | null;
    /** The section as stored (gzip), when it was. */
    stored: Uint8Array | null;
    /** CMPC v4: the traversal's input, the container, its file offset, its inflated runs. */
    input?: Uint8Array;
    container?: Uint8Array;
    fileAt?: number;
    planes?: Uint8Array[];
  }[] = [];
  private fetching = 0;
  private lastPlan: ReturnType<typeof planBand<PieceState>> | null = null;
  tau: number;
  /** The draw list of the last traversal (slot indices). */
  drawList: Uint32Array = new Uint32Array(0);
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
    /** The WASM traversal alone (the rest: the worker's round trip). */
    traverseWasmMs: 0,
    starved: 0,
    /** Node pool: nodes uploaded, their writes, gunzips and their ms, at the last update. */
    uploaded: 0,
    writes: 0,
    decodes: 0,
    decodeMs: 0,
    packMs: 0,
    /** Node pool: cut nodes with no payload at hand, at the last update. */
    missing: 0,
    /** Node pool: of those, whose run was dropped (fetched again), and left by the decode budget. */
    missDropped: 0,
    missDeferred: 0,
    /** Node pool: the missing nodes' share of the cut's screen area, and of the holes (not substituted). */
    missingArea: 0,
    holeArea: 0,
    holes: 0,
    /** Node pool: the view ahead's traversal ms, and the dropped runs it asked for. */
    aheadMs: 0,
    prefetchRuns: 0,
    /** The update's synchronous main-thread chunks (ms by phase) and the longest. */
    chunks: {} as Record<string, number>,
    maxChunk: 0,
    /** The worker's last cut's own ms, and its slices so far. */
    workerCutMs: 0,
    workerSlices: 0,
    sub: {} as Record<string, number>,
    /** The longest refresh and decoder reply handling (reset by the reader). */
    refreshMax: 0,
    onRunsMax: 0,
    /** Node pool, the run decoder: cut nodes whose records are on their way; the decoder's ms, inflates and records written, in all. */
    inFlight: 0,
    decoderMs: 0,
    decoderInflates: 0,
    uploadedAsync: 0,
    /** Node pool: of those, drawn as their parent or children. */
    substituted: 0,
    /** Node pool: cut nodes with no slot (past maxCapacity). */
    noSlot: 0,
    /** CMPC v4: Range requests for dropped runs, and their bytes. */
    runRanges: 0,
    runBytes: 0,
  };

  constructor(
    readonly device: GPUDevice,
    pieces: CompactPiece[],
    readonly options: CompactBandOptions,
  ) {
    this.slotNodes = options.slotNodes ?? 3072;
    this.tau = options.tau ?? 1;
    const tables = pieces.map((p) => compactTablesOf(p, options.asset));
    this.tablesV3 = tables.some((t) => t.version >= 3);
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
      const runsPieces = pieces.some((p) => {
        const k = p.layout.sections.findIndex(
          (x) => (x.id as string) === "CMPC",
        );
        return (
          (p.layout.sections[k] as { encoding?: number }).encoding ===
          ENCODING_COMPACT_RUNS
        );
      });
      const remote =
        runsPieces && np.decoder !== false && typeof Worker !== "undefined";
      if (!remote) {
        this.nodes = new NodeSlotAllocator(pool.capacity, this.slotNodes);
      }
      if (remote) {
        // The node pool's CPU side in a worker (thread CU, nodeWorker.ts).
        const w: Worker = new NodeWorker();
        w.onmessage = (e: MessageEvent<FromCutter>) => this.onCutter(e.data);
        this.remote = w;
        const pagePiece = new Uint16Array(this.pageOwner.length);
        this.pageOwner.forEach((o, g) => {
          pagePiece[g] = o.piece;
        });
        this.post({
          t: "init",
          capacity: pool.capacity,
          maxCapacity: max,
          slotNodes: this.slotNodes,
          runBudget: np.runBudget ?? Number.POSITIVE_INFINITY,
          inflatedRuns: np.inflatedRuns ?? 128,
          protect: np.protect !== "none",
          priority: np.priority !== false,
          pagePiece,
          // Runs fetched by the main thread (a worker's fetches waited behind its slices).
          urls: pieces.map(() => ""),
          runFetchers: np.runFetchers ?? 12,
        });
      } else if (
        pieces.some((p) => {
          const k = p.layout.sections.findIndex(
            (x) => (x.id as string) === "CMPC",
          );
          return (
            (p.layout.sections[k] as { encoding?: number }).encoding ===
            ENCODING_COMPACT_RUNS
          );
        })
      ) {
        this.runs = new RunPayloadStore(
          np.runBudget ?? Number.POSITIVE_INFINITY,
          np.inflatedRuns ?? 256,
        );
      } else {
        this.payload = new PagePayloadStore(
          np.decodedPages ?? 8,
          np.gunzipSync,
        );
      }
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
        runs:
          (layout.sections[section] as { encoding?: number }).encoding ===
          ENCODING_COMPACT_RUNS,
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
    if (this.options.quant) {
      await this.options.lod("setCompactQuant", { bits: this.options.quant });
    }
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
    // The asset's tables are on the GPU (the pool's buffer, copied on a
    // resize) and in the WASM: its arrays go (thread CU: ~0.9 MB held).
    this.assetOcclusion = occlusion !== null;
    if (asset) {
      (this.options as { asset?: CompactAssetTables }).asset = undefined;
    }
  }

  /** The asset table had an occlusion codebook (kept past init). */
  private assetOcclusion = false;

  /** Fetching or landed pages not yet in the pool. */
  get busy() {
    return (
      this.fetching > 0 ||
      this.landed.length > 0 ||
      // The node pool: runs coming again, or cut nodes not yet uploaded.
      this.runFetches > 0 ||
      this.wantedRuns.size > 0 ||
      this.unpacking > 0 ||
      this.remoteBusy ||
      this.remoteFetches > 0 ||
      this.stats.missing > this.stats.noSlot
    );
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
    // The main thread's synchronous chunks of this update (thread CU).
    let c0 = performance.now();
    const chunks: Record<string, number> = {};
    const chunk = (name: string) => {
      const t = performance.now();
      chunks[name] = (chunks[name] ?? 0) + (t - c0);
      c0 = t;
    };
    const resume = () => {
      c0 = performance.now();
    };
    // 1. Landed pages into slots.
    const ranges: Record<string, unknown>[] = [];
    const landed = this.landed;
    this.landed = [];
    for (const l of landed) {
      const { piece, page, raw } = l;
      piece.inFlight.delete(page);
      if (piece.slotOf.has(page)) continue;
      const slot = this.takeSlot();
      if (slot === undefined) {
        // No room: dropped, asked again.
        if (this.remote) {
          this.post({ t: "drop", g: this.pageBase[piece.index] + page });
        }
        continue;
      }
      const row = piece.rows[page];
      const { input, box } =
        this.pool instanceof CompactPagePool
          ? this.pool.writePage(
              slot,
              piece.index,
              page,
              row.count,
              raw as Uint8Array,
              row.compact,
            )
          : this.landNodes(piece, page, l);
      piece.slotOf.set(page, slot);
      this.slotOwner[slot] = { piece: piece.index, page };
      piece.kids.set(
        page,
        raw
          ? compactChildPages(raw, row.count, page)
          : compactChildPages(input, row.count, page, 6),
      );
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
    if (this.evictedSlots.size && (this.nodes || this.remote)) {
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
    chunk("land");
    if (ranges.length) await lod("updateCompactLodTrees", { ranges });
    resume();

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
    let list: Uint32Array = new Uint32Array(0);
    let near: CutNeighbours | undefined;
    if (Object.keys(instances).length) {
      const t0 = performance.now();
      chunk("plan");
      const traversal = (await lod("traverseLodTrees", {
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
        neighbours: !!this.runs || !!this.remote,
        // The budget never binds the cut (it fits the pool): any order.
        unordered: true,
        // The worker takes the cut in chunks (never whole in the WASM heap).
        stream: !!this.remote,
      })) as StreamedCut & {
        keyIndices: Record<
          string,
          {
            numSplats: number;
            indices: Uint32Array;
            parents?: Uint32Array;
            firstKids?: Uint32Array;
            kidCounts?: Uint32Array;
            culled?: Uint32Array;
            screens?: Uint32Array;
          }
        >;
        culledCount?: number;
        occludedCount?: number;
        chunks?: [number, number][];
        wasmMs?: number;
      };
      const { keyIndices, culledCount, occludedCount, chunks, wasmMs } =
        traversal;
      const streamed = traversal;
      this.stats.traverseMs = performance.now() - t0;
      resume();
      this.stats.traverseWasmMs = wasmMs ?? 0;
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
      if (this.remote) this.postCut(streamed as StreamedCut);
      const parts = (this.remote ? [] : Object.values(keyIndices)).map((k) =>
        k.indices.subarray(0, k.numSplats),
      );
      list = new Uint32Array(parts.reduce((n, p) => n + p.length, 0));
      let at = 0;
      for (const part of parts) {
        list.set(part, at);
        at += part.length;
      }
      if (this.runs) {
        const cat = (
          name: "parents" | "firstKids" | "kidCounts" | "screens",
        ) => {
          const a = new Uint32Array(list.length);
          let o = 0;
          for (const k of Object.values(keyIndices)) {
            a.set((k[name] as Uint32Array).subarray(0, k.numSplats), o);
            o += k.numSplats;
          }
          return a;
        };
        const hasScreens = Object.values(keyIndices).every((k) => k.screens);
        near = {
          parents: cat("parents"),
          firstKids: cat("firstKids"),
          kidCounts: cat("kidCounts"),
          // Each cut node's angle on screen (f32 bits; older WASM: none).
          screens: hasScreens
            ? new Float32Array(cat("screens").buffer)
            : undefined,
        };
        // The runs the cut may need (its nodes, their parents and
        // children), then the culled ones; the rest go first over budget.
        const runs = this.runs;
        runs.begin();
        const mark = (i: number, cls: number) => {
          const key = this.keyOfIndex(i);
          if (key < 0) return;
          const g = Math.floor(key / PAGE_PLACES);
          runs.mark(g, key - g * PAGE_PLACES, cls);
        };
        const protect = this.options.nodePool?.protect !== "none";
        for (let k = 0; protect && k < list.length; k++) {
          mark(list[k], RUN_CUT);
          if (near.parents[k] !== 0xffffffff) mark(near.parents[k], RUN_CUT);
          const f = near.firstKids[k];
          if (f !== 0xffffffff) {
            for (let c = 0; c < near.kidCounts[k]; c++) mark(f + c, RUN_CUT);
          }
        }
        for (const k of Object.values(keyIndices)) {
          for (const i of k.culled ?? []) mark(i, RUN_CULLED);
        }
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
      const before = list;
      list = Uint32Array.from(order, (k) => before[k]);
      if (near) {
        const n0 = near;
        near = {
          parents: Uint32Array.from(order, (k) => n0.parents[k]),
          firstKids: Uint32Array.from(order, (k) => n0.firstKids[k]),
          kidCounts: Uint32Array.from(order, (k) => n0.kidCounts[k]),
          screens: n0.screens
            ? Float32Array.from(order, (k) => (n0.screens as Float32Array)[k])
            : undefined,
        };
      }
    }
    if (path) this.lastPath = path;
    chunk("cut");
    if (this.nodes) list = this.toNodeSlots(list, path, near);
    // The new list at once (records written from now on are its own); the
    // worker's come with its records (onCutter).
    if (!this.remote) {
      this.drawList = list;
      this.stats.cut = list.length;
      path?.setDrawList(list);
    }
    chunk("slots");
    // The view a moment ahead (thread CU): its cut's dropped runs asked
    // for now, so they are here when the camera is.
    const ahead = this.runs || this.remote ? this.predictView(view) : null;
    if (ahead && Object.keys(instances).length) {
      await this.prefetchAhead(ahead, instances, pixel, errorLimit, resume);
      chunk("ahead");
    } else this.stats.prefetchRuns = 0;
    if (this.runs) {
      this.runs.trim();
      this.fetchRuns();
    }

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
    // The pool's room (thread CU): a page asked for only when a free or
    // evictable slot will take it. Without it a pool full of needed pages
    // and pages kept by hysteresis fetched the next pages again and again,
    // never landing them (busy for good: no settle after an orbit).
    let pendingPages = this.landed.length;
    for (const p of this.pieces) pendingPages += p.inFlight.size;
    const plan = planBand(
      meshes,
      this.options.slots,
      this.options.maxRangeBytes ?? 8 << 20,
      { free: this.freeSlots.length, pending: pendingPages },
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
    chunk("pages");
    this.stats.chunks = chunks;
    this.stats.maxChunk = Math.max(...Object.values(chunks));
  }

  /** A landed page, node pool: its payload kept on the CPU, its traversal input. */
  private landNodes(
    piece: PieceState,
    page: number,
    l: {
      raw: Uint8Array | null;
      stored: Uint8Array | null;
      input?: Uint8Array;
      container?: Uint8Array;
      fileAt?: number;
      planes?: Uint8Array[];
    },
  ) {
    const row = piece.rows[page];
    const g = this.pageBase[piece.index] + page;
    if (this.remote) {
      // Its runs are in the worker already.
    } else if (this.runs && l.container) {
      this.runs.put(g, l.container, l.fileAt ?? 0, l.planes);
    } else {
      (this.payload as PagePayloadStore).put(
        g,
        row.count,
        l.stored,
        l.raw as Uint8Array,
      );
    }
    return {
      input: l.input ?? compactTraversalInput(l.raw as Uint8Array, row.count),
      box: new Float32Array(
        Uint32Array.from({ length: 6 }, (_, k) => row.compact[k] >>> 0).buffer,
      ),
    };
  }

  /** The head-table page and place of pool index i (a resident page's node). */
  private keyOfIndex(i: number) {
    const N = this.slotNodes;
    const slot = Math.floor(i / N);
    const o = this.slotOwner[slot];
    if (!o) return -1;
    return (this.pageBase[o.piece] + o.page) * PAGE_PLACES + (i - slot * N);
  }

  /**
   * The traversal's cut (pool slot indices) to node slots: the nodes held
   * are reused, the others get a slot (the pool grown if the cut outgrew
   * it) and their records packed from their page's payload (a section, or
   * CMPC v4 runs) and written. The list keeps the cut's order. A node whose
   * run was dropped (fetched again) is drawn as its parent, else as its
   * children, when those are held; else left out until it lands.
   */
  private toNodeSlots(
    list: Uint32Array,
    path?: CompactRelitPath,
    near?: CutNeighbours,
  ) {
    // Typed arrays throughout (thread CU): no per-node objects, Sets or
    // comparator sorts; the misses are sorted as packed float keys.
    const alloc = this.nodes as NodeSlotAllocator;
    const pool = this.pool as CompactNodePool;
    const runs = this.runs;
    const store = this.payload;
    const t0 = performance.now();
    const d0 = runs ? runs.stats.inflates : (store?.stats.decodes ?? 0);
    const ms0 = runs ? runs.stats.inflateMs : (store?.stats.decodeMs ?? 0);
    if (list.length > alloc.capacity && pool.grow(list.length)) {
      alloc.grow(pool.capacity);
      path?.setPagedCloud(pool);
    }
    alloc.begin();
    const n = list.length;
    const N = this.slotNodes;
    // Pool index -> node key: i + kb[slot] (the slot's page's first key less its first index).
    const kb = this.slotKeyBases();
    const keys = new Int32Array(n);
    const out = new Int32Array(n).fill(-1);
    const failed = new Uint8Array(n);
    let failedN = 0;
    const inFlight = 0;
    // Misses as key * 2^23 + k (exact in a double; a cut under 2^23).
    const SH = 8388608;
    const missKeys = new Float64Array(n);
    let missN = 0;
    for (let k = 0; k < n; k++) {
      const i = list[k];
      const key = i + kb[(i / N) | 0];
      keys[k] = key;
      const s = alloc.use(key);
      if (s < 0) missKeys[missN++] = key * SH + k;
      else out[k] = s;
    }
    const tLoop = performance.now();
    const miss = missKeys.subarray(0, missN).sort();
    // The groups (a run's misses, v4, or a page's), in key order.
    const gStart: number[] = [];
    const gRun: number[] = [];
    {
      let q = 0;
      while (q < missN) {
        const key0 = keys[miss[q] % SH];
        const g = key0 >>> 12;
        const info = runs?.runInfo(g);
        const r = info ? ((key0 & 4095) / info.run) | 0 : -1;
        gStart.push(q);
        gRun.push(r);
        q += 1;
        while (q < missN) {
          const key = keys[miss[q] % SH];
          if (key >>> 12 !== g) break;
          if (info && (((key & 4095) / info.run) | 0) !== r) break;
          q++;
        }
      }
      gStart.push(missN);
    }
    const groups = gRun.length;
    // Decoded largest first (thread CU): a group's weight is its misses'
    // area on screen, so what the budget leaves out is the least visible.
    const screens = near?.screens;
    let order: Uint32Array;
    if (screens && groups > 1 && this.options.nodePool?.priority !== false) {
      const w = new Float64Array(groups);
      for (let x = 0; x < groups; x++) {
        let a = 0;
        for (let y = gStart[x]; y < gStart[x + 1]; y++) {
          const sc = screens[miss[y] % SH];
          a += sc * sc;
        }
        w[x] = a;
      }
      order = new Uint32Array(groups);
      for (let x = 0; x < groups; x++) order[x] = x;
      order.sort((a, b) => w[b] - w[a]);
    } else {
      order = new Uint32Array(groups);
      for (let x = 0; x < groups; x++) order[x] = x;
    }
    // Each miss: ready (payload at hand, under the budgets) or absent.
    const ready = new Int32Array(missN);
    let readyN = 0;
    this.deferred = false;

    const cap =
      this.options.nodePool?.uploadsPerUpdate ?? Number.POSITIVE_INFINITY;
    const inflateCap = this.options.nodePool?.inflatesPerUpdate ?? 256;
    let inflating = 0;
    let missDropped = 0;
    let missDeferred = 0;
    for (let o = 0; o < groups; o++) {
      const x = order[o];
      const q = gStart[x];
      const e = gStart[x + 1];
      const key0 = keys[miss[q] % SH];
      const g = key0 >>> 12;
      const r = gRun[x];
      const info = runs?.runInfo(g);
      let ok = true;
      if (runs) {
        if (!info) ok = false;
        else {
          // A run with a cut node still to upload stays (any protection mode).
          runs.mark(g, key0 & 4095, RUN_CUT);
          if (runs.missing(g, r)) {
            this.wantRun(g, r);
            ok = false;
            missDropped += e - q;
          } else if (!runs.isInflated(g, r)) {
            // The decode budget: runs not inflated yet, at most inflateCap.
            if (inflating >= inflateCap) {
              this.deferred = true;
              ok = false;
              missDeferred += e - q;
            } else inflating += 1;
          }
        }
      }
      for (let y = q; y < e; y++) {
        const k = miss[y] % SH;
        if (ok && readyN < cap) ready[readyN++] = k;
        else {
          failed[k] = 1;
          failedN += 1;
        }
      }
    }
    const tDecide = performance.now();
    // Slots for the ready (in key order: a run's nodes together).
    const got = new Int32Array(readyN);
    const gotSlot = new Float64Array(readyN);
    let gotN = 0;
    let noSlot = 0;
    for (let x = 0; x < readyN; x++) {
      const k = ready[x];
      const s = alloc.allocate(keys[k]);
      if (s < 0) {
        noSlot += 1;
        failed[k] = 1;
        failedN += 1;
        continue;
      }
      out[k] = s;
      gotSlot[gotN] = s * SH + gotN;
      got[gotN++] = k;
    }
    // Each record's place in slot order (one write a run of slots).
    const bySlot = gotSlot.subarray(0, gotN).sort();
    const pos = new Int32Array(gotN);
    const slots = new Uint32Array(gotN);
    for (let x = 0; x < gotN; x++) {
      const v = bySlot[x];
      const w = v % SH;
      pos[w] = x;
      slots[x] = (v - w) / SH;
    }
    // Packed group by group (a page, or a page's run: one inflate each),
    // straight into slot order.
    const R = NODE_RECORD_WORDS * 4;
    const records = new Uint8Array(gotN * R);
    const dropped: number[] = [];
    let j = 0;
    while (j < gotN) {
      const key0 = keys[got[j]];
      const g = key0 >>> 12;
      const info = runs?.runInfo(g);
      const r = info ? ((key0 & 4095) / info.run) | 0 : -1;
      let e = j + 1;
      while (e < gotN) {
        const key = keys[got[e]];
        if (key >>> 12 !== g) break;
        if (info && (((key & 4095) / info.run) | 0) !== r) break;
        e++;
      }
      const at = pos.subarray(j, e);
      if (runs && info) {
        const base = r * info.run;
        const m = Math.min(info.run, info.n - base);
        const planes = runs.planes(g, r);
        const locals = new Uint32Array(e - j);
        for (let x = j; x < e; x++)
          locals[x - j] = (keys[got[x]] & 4095) - base;
        if (planes) packRunRecords(planes, m, locals, g, base, records, at);
        else for (let x = j; x < e; x++) dropped.push(x);
      } else {
        const owner = this.pageOwner[g];
        const pn = this.pieces[owner.piece].rows[owner.page].count;
        const raw = store?.sectionOf(g) ?? null;
        const places = new Uint32Array(e - j);
        for (let x = j; x < e; x++) places[x - j] = keys[got[x]] & 4095;
        if (raw) {
          const tmp = new Uint8Array(places.length * R);
          packPageRecords(raw, pn, places, g, tmp, 0);
          for (let x = 0; x < places.length; x++) {
            records.set(tmp.subarray(x * R, x * R + R), at[x] * R);
          }
        } else for (let x = j; x < e; x++) dropped.push(x);
      }
      j = e;
    }
    // A dropped record's slot is written with nothing: left out of the writes.
    let wSlots = slots;
    let wRecords = records;
    if (dropped.length) {
      const bad = new Uint8Array(gotN);
      for (const x of dropped) {
        const k = got[x];
        bad[pos[x]] = 1;
        alloc.release(out[k]);
        failed[k] = 1;
        failedN += 1;
      }
      const keepN = gotN - dropped.length;
      wSlots = new Uint32Array(keepN);
      wRecords = new Uint8Array(keepN * R);
      let y = 0;
      for (let x = 0; x < gotN; x++) {
        if (bad[x]) continue;
        wSlots[y] = slots[x];
        wRecords.set(records.subarray(x * R, x * R + R), y * R);
        y++;
      }
    }
    this.stats.writes = wSlots.length ? pool.writeRecords(wSlots, wRecords) : 0;
    this.stats.uploaded = wSlots.length;
    this.stats.decodes =
      (runs ? runs.stats.inflates : (store?.stats.decodes ?? 0)) - d0;
    this.stats.decodeMs =
      (runs ? runs.stats.inflateMs : (store?.stats.decodeMs ?? 0)) - ms0;
    // The rest: drawn as their parent or children when held, else not yet.
    // The cut's drawn part (its own slots, in its order) once; a refresh
    // then only walks the misses (thread CU).
    const base = new Uint32Array(n - failedN);
    const failedList = new Int32Array(failedN);
    let allArea = 0;
    {
      let b0 = 0;
      let f0 = 0;
      const scr = near?.screens;
      for (let k = 0; k < n; k++) {
        if (failed[k]) failedList[f0++] = k;
        else base[b0++] = out[k];
        if (scr) allArea += scr[k] * scr[k];
      }
    }
    this.cutState = { keys, out, near, kb, n, base, failedList, allArea };
    const tB = performance.now();
    const result = this.buildDrawList();
    this.stats.sub.build = performance.now() - tB;
    this.stats.missDropped = missDropped;
    this.stats.missDeferred = missDeferred;
    this.stats.inFlight = inFlight;
    this.stats.noSlot = noSlot;
    const tC = performance.now();
    alloc.commit(result);
    this.stats.sub.commit = performance.now() - tC;
    this.stats.sub.loop = tLoop - t0;
    this.stats.sub.decide = tDecide - tLoop;
    this.stats.sub.alloc = tB - tDecide - (this.stats.sub.send ?? 0);
    this.stats.packMs = performance.now() - t0 - this.stats.decodeMs;
    return result;
  }

  /** The last cut (node keys, slots, misses, neighbours), for `refresh`. */
  private cutState: {
    keys: Int32Array;
    out: Int32Array;
    near?: CutNeighbours;
    kb: Int32Array;
    n: number;
    /** Its nodes drawn (slots, the cut's order), then those whose records came since. */
    base: Uint32Array;
    /** Its misses (cut indices) still waiting. */
    failedList: Int32Array;
    /** The cut's screen area (sum of its nodes' angle squared). */
    allArea: number;
  } | null = null;

  /**
   * The draw list of the last cut: its nodes held and written (`base`),
   * those of its misses whose records came since (they join `base`), the
   * other misses as their parent or children when those are held, else
   * left out; the stats of what is missing (count, screen area, holes).
   * Substitutes are never cut nodes (a cut is an antichain): only they are
   * deduplicated.
   */
  private buildDrawList() {
    const st = this.cutState as NonNullable<typeof this.cutState>;
    const { keys, out, near, kb } = st;
    const alloc = this.nodes as NodeSlotAllocator;
    const N = this.slotNodes;
    const screens = near?.screens;
    // Misses whose records are now written: drawn from now on.
    const fl = st.failedList;
    let resolved = 0;
    for (let x = 0; x < fl.length; x++) {
      const s = out[fl[x]];
      if (s >= 0 && alloc.keyIn(s) === keys[fl[x]]) resolved++;
    }
    if (resolved) {
      const base = new Uint32Array(st.base.length + resolved);
      base.set(st.base);
      let b0 = st.base.length;
      let f0 = 0;
      for (let x = 0; x < fl.length; x++) {
        const k = fl[x];
        const s = out[k];
        if (s >= 0 && alloc.keyIn(s) === keys[k]) {
          base[b0++] = s;
          alloc.use(keys[k]);
        } else fl[f0++] = k;
      }
      st.base = base;
      st.failedList = fl.subarray(0, f0);
    }
    const misses = st.failedList;
    let substituted = 0;
    let holes = 0;
    let holeArea = 0;
    let missedArea = 0;
    let result: Uint32Array;
    if (misses.length) {
      const seen = this.seenStamps(alloc.capacity);
      const epoch = ++this.seenEpoch;
      let subs = new Uint32Array(Math.max(64, misses.length * 2));
      let sn = 0;
      const push = (s: number) => {
        if (seen[s] === epoch) return;
        seen[s] = epoch;
        if (sn === subs.length) {
          const g2 = new Uint32Array(subs.length * 2);
          g2.set(subs);
          subs = g2;
        }
        subs[sn++] = s;
      };
      const held = (i: number) => {
        const s = alloc.use(i + kb[(i / N) | 0]);
        return s;
      };
      for (let x = 0; x < misses.length; x++) {
        const k = misses[x];
        const a = screens ? screens[k] * screens[k] : 0;
        missedArea += a;
        if (!near) {
          holes += 1;
          continue;
        }
        // Its own slot stays stamped (a record on its way stays held).
        if (out[k] >= 0) alloc.use(keys[k]);
        const parent = near.parents[k];
        const ps = parent !== 0xffffffff ? held(parent) : -1;
        if (ps >= 0) {
          push(ps);
          substituted += 1;
          continue;
        }
        const first = near.firstKids[k];
        const count = near.kidCounts[k];
        let c = 0;
        if (first !== 0xffffffff && count) {
          if (this.kidScratch.length < count)
            this.kidScratch = new Int32Array(count * 2);
          const ks = this.kidScratch;
          for (; c < count; c++) {
            const s = held(first + c);
            if (s < 0) break;
            ks[c] = s;
          }
          if (c === count) {
            for (let y = 0; y < count; y++) push(ks[y]);
            substituted += 1;
            continue;
          }
        }
        holes += 1;
        holeArea += a;
      }
      result = new Uint32Array(st.base.length + sn);
      result.set(st.base);
      result.set(subs.subarray(0, sn), st.base.length);
    } else {
      result = st.base;
    }
    if (screens) {
      this.stats.missingArea = st.allArea > 0 ? missedArea / st.allArea : 0;
      this.stats.holeArea = st.allArea > 0 ? holeArea / st.allArea : 0;
    }
    this.stats.holes = holes;
    this.stats.missing = misses.length;
    this.stats.substituted = substituted;
    return result;
  }

  private seenEpoch = 0;
  private seen = new Uint32Array(0);
  private kidScratch = new Int32Array(64);

  /** An epoch stamp a node slot (grown with the pool). */
  private seenStamps(capacity: number) {
    if (this.seen.length < capacity) {
      const s = new Uint32Array(capacity);
      s.set(this.seen);
      this.seen = s;
    }
    return this.seen;
  }

  /** Each pool slot's key base: a node key is its pool index plus it. */
  private slotKeyBases() {
    const N = this.slotNodes;
    const kb = new Int32Array(this.slotOwner.length);
    for (let s = 0; s < kb.length; s++) {
      const o = this.slotOwner[s];
      // A slot with no page: keys below 0 (never held).
      kb[s] = o
        ? (this.pageBase[o.piece] + o.page) * PAGE_PLACES - s * N
        : -(1 << 30);
    }
    return kb;
  }

  /** The views of the last updates (trees' space), for the view ahead. */
  private views: { t: number; m: THREE.Matrix4 }[] = [];

  /**
   * The view `nodePool.aheadMs` (default 2.5 updates, at most 600 ms)
   * ahead, extrapolated from the last two updates' (position linearly, the
   * rotation by the same turn); null when still or no history.
   */
  private predictView(view: THREE.Matrix4): THREE.Matrix4 | null {
    const now = performance.now();
    const h = this.views;
    h.push({ t: now, m: view.clone() });
    if (h.length > 2) h.shift();
    if (h.length < 2 || this.options.nodePool?.aheadMs === 0) return null;
    const [a, b] = h;
    const dt = b.t - a.t;
    if (!(dt > 0) || dt > 1000) return null;
    const pa = new THREE.Vector3();
    const pb = new THREE.Vector3();
    const qa = new THREE.Quaternion();
    const qb = new THREE.Quaternion();
    const sc = new THREE.Vector3();
    a.m.decompose(pa, qa, sc);
    b.m.decompose(pb, qb, sc);
    const turn = qb.clone().multiply(qa.clone().invert());
    const angle = 2 * Math.acos(Math.min(1, Math.abs(turn.w)));
    if (pa.distanceTo(pb) < 1.0e-4 && angle < 1.0e-4) return null;
    const lead = Math.min(this.options.nodePool?.aheadMs ?? 2.5 * dt, 600) / dt;
    const p = pb.clone().addScaledVector(pb.clone().sub(pa), lead);
    // The turn raised to `lead` (about the same axis).
    const axis = new THREE.Vector3(turn.x, turn.y, turn.z);
    const q = qb.clone();
    if (axis.lengthSq() > 1.0e-12 && angle >= 1.0e-4) {
      const sign = turn.w < 0 ? -1 : 1;
      axis.multiplyScalar(sign).normalize();
      q.premultiply(
        new THREE.Quaternion().setFromAxisAngle(axis, angle * lead),
      );
    }
    return new THREE.Matrix4().compose(p, q, sc.set(1, 1, 1));
  }

  /**
   * The cut of the view ahead (one more traversal, no neighbours): the runs
   * of its nodes not held are kept this update, and those dropped asked
   * for again (after the cut's own, which went first).
   */
  private async prefetchAhead(
    ahead: THREE.Matrix4,
    instances: Record<string, unknown>,
    pixel: number,
    errorLimit: number,
    resume?: () => void,
  ) {
    const runs = this.runs as RunPayloadStore;
    const alloc = this.nodes as NodeSlotAllocator;
    const inst: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(instances)) {
      inst[k] = {
        ...(v as Record<string, unknown>),
        viewToObjectCols: ahead.elements.slice(),
      };
    }
    const t0 = performance.now();
    const ahead_ = (await this.options.lod("traverseLodTrees", {
      maxSplats:
        this.pool instanceof CompactNodePool
          ? this.pool.maxCapacity
          : this.options.slots * this.slotNodes,
      pixelScaleLimit: pixel,
      instances: inst,
      traverseMode: "standard",
      errorLimit,
      backCullMin: this.options.backCull === false ? undefined : -1,
      occlusionCull: this.options.occlusion === true,
      neighbours: false,
      unordered: true,
      stream: !!this.remote,
    })) as StreamedCut & {
      keyIndices: Record<string, { numSplats: number; indices: Uint32Array }>;
    };
    this.stats.aheadMs = performance.now() - t0;
    resume?.();
    const keyIndices = ahead_.keyIndices;
    if (this.remote) {
      const parts = (ahead_.parts ?? []).map((k) => ({
        numSplats: k.numSplats,
        indices: k.indices,
      }));
      this.post(
        { t: "ahead", kb: this.slotKeyBases(), parts },
        parts.map((p) => p.indices.buffer),
      );
      return;
    }
    const N = this.slotNodes;
    const kb = this.slotKeyBases();
    let wanted = 0;
    let lastRun = -1;
    for (const k of Object.values(keyIndices)) {
      const idx = k.indices;
      for (let j = 0; j < k.numSplats; j++) {
        const i = idx[j];
        const key = i + kb[(i / N) | 0];
        if (key < 0 || alloc.slotOf.slot(key) >= 0) continue;
        const g = key >>> 12;
        const info = runs.runInfo(g);
        if (!info) continue;
        const r = ((key & 4095) / info.run) | 0;
        const rk = g * 64 + r;
        if (rk === lastRun) continue;
        lastRun = rk;
        runs.mark(g, key & 4095, RUN_CUT);
        if (runs.missing(g, r) && !this.wantedRuns.get(g)?.has(r)) {
          this.wantRun(g, r);
          wanted += 1;
        }
      }
    }
    this.stats.prefetchRuns = wanted;
  }

  // ------------------------------------- the node pool's worker (thread CU)

  /** The node pool's CPU side (nodeWorker.ts), for CMPC v4 pieces. */
  private remote: Worker | null = null;
  private lastPath: CompactRelitPath | undefined;
  /** Pages posted to the worker, not landed back yet. */
  private unpacking = 0;
  private remoteBusy = false;
  private remoteStats: Record<string, number | boolean> = {};
  private keysWaiting = new Map<number, (k: Int32Array) => void>();
  private remoteKeys: Int32Array | null = null;

  private post(m: ToCutter, transfer: Transferable[] = []) {
    this.remote?.postMessage(m, transfer);
  }

  /** The worker's messages: records and draw lists, landed pages, runs to fetch. */
  /** Main-thread ms in the worker's messages since the reader reset it (thread CU). */
  cutterMs = 0;
  cutterMsgs = 0;
  cutterTypes: Record<string, number> = {};

  private onCutter(m: FromCutter) {
    const t0 = performance.now();
    try {
      this.onCutterInner(m);
    } finally {
      this.cutterMs += performance.now() - t0;
      this.cutterMsgs += 1;
      this.cutterTypes[m.t] = (this.cutterTypes[m.t] ?? 0) + 1;
    }
  }

  private onCutterInner(m: FromCutter) {
    if (m.t === "frame") {
      const pool = this.pool as CompactNodePool;
      // The worker's size exactly (a grow, or a shrink with its moves).
      if (m.grow && (m.grow !== pool.capacity || m.moves?.length)) {
        if (pool.resize(m.grow, m.moves ?? undefined)) {
          this.lastPath?.setPagedCloud(pool);
        }
      }
      if (m.slots.length) pool.writeRecords(m.slots, m.records);
      if (m.drawList) {
        // Owned by the path (no copy); the list it replaces goes back to the
        // worker for its next one (no allocation churn on this thread: the
        // GC paused frames for 7-13 ms).
        const path = this.lastPath;
        const old = path?.drawListHeld;
        this.drawList = m.drawList;
        path?.setDrawList(m.drawList, true);
        if (old && old !== m.drawList && path?.drawListHeld === m.drawList) {
          this.post({ t: "recycle", buffer: old.buffer as ArrayBuffer }, [
            old.buffer,
          ]);
        }
      }
      const st = m.stats;
      this.remoteStats = { ...st };
      this.remoteBusy = st.busy;
      const s = this.stats;
      s.cut = m.drawList ? m.drawList.length : s.cut;
      s.missing = st.missing;
      s.missDropped = st.missDropped;
      s.missDeferred = st.missDeferred;
      s.substituted = st.substituted;
      s.holes = st.holes;
      s.missingArea = st.missingArea;
      s.holeArea = st.holeArea;
      s.uploaded = st.uploaded;
      s.decodes = st.decodes;
      s.decodeMs = st.decodeMs;
      s.noSlot = st.noSlot;
      s.runRanges = st.runRanges;
      s.runBytes = st.runBytesRead;
      s.prefetchRuns = st.prefetchRuns;
      s.packMs = st.sliceMs;
      s.workerCutMs = st.cutMs;
      s.workerSlices = st.slices;
    } else if (m.t === "landed") {
      this.unpacking -= 1;
      const o = this.pageOwner[m.g];
      const piece = this.pieces[o.piece];
      this.landed.push({
        piece,
        page: o.page,
        raw: null,
        stored: null,
        input: m.input,
      });
    } else if (m.t === "want") {
      for (const spans of m.ranges) this.runQueue.push(spans);
      this.fetchQueued();
    } else if (m.t === "keys") {
      this.remoteKeys = m.keys;
      this.keysWaiting.get(m.id)?.(m.keys);
      this.keysWaiting.delete(m.id);
    }
  }

  /** The worker's runs to fetch (a page's at a time), 12 Ranges in flight. */
  private runQueue: [number, number, number, number][][] = [];
  private remoteFetches = 0;

  private fetchQueued() {
    while (this.remoteFetches < 16 && this.runQueue.length) {
      const m = this.runQueue.shift() as [number, number, number, number][];
      this.remoteFetches += 1;
      this.fetchWanted(m).finally(() => {
        this.remoteFetches -= 1;
        this.fetchQueued();
      });
    }
  }

  private fetchWanted(spans: [number, number, number, number][]) {
    const first = spans[0];
    const last = spans[spans.length - 1];
    const piece = this.pieces[this.pageOwner[first[0]].piece].piece;
    const at = first[2];
    return piece
      .read(at, last[2] + last[3] - at)
      .then((data) => {
        // The Range whole (transferred): the worker cuts its runs out.
        const buf =
          data.byteOffset === 0 && data.byteLength === data.buffer.byteLength
            ? data
            : data.slice();
        this.post({ t: "runs", data: buf, at, spans }, [buf.buffer]);
      })
      .catch((e) => {
        console.warn(e);
        this.post({
          t: "runs",
          asked: spans.map(([g, r]) => g * 64 + r),
        });
      });
  }

  /** The worker's slot keys (for nodeOf), fetched now. */
  async fetchKeys() {
    if (!this.remote) return null;
    const id = Math.random();
    const p = new Promise<Int32Array>((ok) => this.keysWaiting.set(id, ok));
    this.post({ t: "keys", id });
    return p;
  }

  /** A cut's parts (the traversal's chunks, transferred) to the worker. */
  private postCut(cut: StreamedCut) {
    const parts: CutPart[] = [...(cut.parts ?? [])];
    for (const c of cut.culled ?? []) {
      parts.push({ numSplats: 0, indices: new Uint32Array(0), culled: c });
    }
    const transfer: Transferable[] = [];
    for (const p of parts) {
      for (const a of [
        p.indices,
        p.parents,
        p.firstKids,
        p.kidCounts,
        p.culled,
        p.screens,
      ]) {
        if (a && !transfer.includes(a.buffer)) transfer.push(a.buffer);
      }
    }
    const msg: CutMsg = { t: "cut", kb: this.slotKeyBases(), parts };
    this.post(msg, transfer);
  }

  /** Nodes left for the next update by the decode budget. */
  private deferred = false;

  /** Runs to fetch again (dropped runs a cut node needs): page -> runs. */
  private wantedRuns = new Map<number, Set<number>>();
  private runFetches = 0;

  private wantRun(page: number, r: number) {
    const set = this.wantedRuns.get(page) ?? new Set<number>();
    set.add(r);
    this.wantedRuns.set(page, set);
  }

  /**
   * The dropped runs the cut asked for, by Range: a page's wanted runs in
   * one request (the runs between them read too while that at most doubles
   * the bytes), `runFetchers` (8) requests in flight.
   */
  private fetchRuns() {
    const runs = this.runs;
    if (!runs) return;
    const max = this.options.nodePool?.runFetchers ?? 12;
    for (const [g, set] of this.wantedRuns) {
      if (this.runFetches >= max) break;
      this.wantedRuns.delete(g);
      const owner = this.pageOwner[g];
      const piece = this.pieces[owner.piece].piece;
      const list = [...set]
        .filter((r) => runs.missing(g, r))
        .sort((a, b) => a - b);
      // Groups: a run joins the group before it unless the gap would more
      // than double the group's bytes.
      const groups: number[][] = [];
      for (const r of list) {
        const last = groups[groups.length - 1];
        if (last) {
          const a = runs.fileSpan(g, last[0]) as [number, number];
          const b = runs.fileSpan(g, last[last.length - 1]) as [number, number];
          const c = runs.fileSpan(g, r) as [number, number];
          const need = last.reduce(
            (n, q) => n + (runs.fileSpan(g, q) as [number, number])[1],
            c[1],
          );
          if (c[0] + c[1] - a[0] <= 2 * need || c[0] === b[0] + b[1]) {
            last.push(r);
            continue;
          }
        }
        groups.push([r]);
      }
      for (const rs of groups) {
        const first = runs.fileSpan(g, rs[0]) as [number, number];
        const last = runs.fileSpan(g, rs[rs.length - 1]) as [number, number];
        this.runFetches += 1;
        this.stats.runRanges += 1;
        this.stats.runBytes += last[0] + last[1] - first[0];
        piece
          .read(first[0], last[0] + last[1] - first[0])
          .then((data) => {
            for (const r of rs) {
              const span = runs.fileSpan(g, r);
              if (!span) continue;
              runs.putRun(
                g,
                r,
                data.slice(span[0] - first[0], span[0] - first[0] + span[1]),
              );
            }
          })
          .catch((e) => console.warn(e))
          .finally(() => {
            this.runFetches -= 1;
            // The next ones at once, not at the next update (thread CU).
            if (this.wantedRuns.size) this.fetchRuns();
          });
      }
    }
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
      this.runs?.drop(this.pageBase[key.index] + page);
      if (this.remote) {
        this.post({ t: "drop", g: this.pageBase[key.index] + page });
      }
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
    const v3 = this.tablesV3;
    const occlusion =
      this.assetOcclusion || (this.options.asset?.occlusion.length ?? 0) > 0;
    return (
      (v3 && this.options.backCull !== false) ||
      (v3 && occlusion && this.options.occlusion === true)
    );
  }

  /** Some piece's tables are CMPT 3 (cones, occlusion indices). */
  private tablesV3 = false;

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
          if (p.runs && this.remote) {
            // The worker unpacks it (its runs kept there) and sends the
            // traversal's input back ("landed").
            const c = stored.slice();
            this.unpacking += 1;
            this.post(
              {
                t: "page",
                g: this.pageBase[p.index] + page,
                container: c,
                fileAt: span.offset,
              },
              [c.buffer],
            );
            continue;
          }
          if (p.runs) {
            // CMPC v4: the traversal's member gunzipped, the runs inflated
            // once (the positions the traversal reads are in them).
            const c = stored.slice();
            const h = parseRunsContainer(c);
            const trav = await this.options.gunzip(
              c.subarray(h.travAt, h.travAt + h.travBytes),
            );
            const nodePool = this.pool instanceof CompactNodePool;
            const u = unpackRunsContainer(c, h, trav, !nodePool);
            this.landed.push({
              piece: p,
              page,
              raw: u.raw,
              stored: null,
              input: u.input,
              container: nodePool ? c : undefined,
              fileAt: span.offset,
              planes: u.planes,
            });
            continue;
          }
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
        gpuRecordsUsed:
          ((this.remote
            ? (this.remoteStats.nodesHeld as number)
            : this.nodes?.used) ?? 0) *
          NODE_RECORD_WORDS *
          4,
        cpuTraversal,
        cpuCompressed:
          (this.remote
            ? (this.remoteStats.runBytesHeld as number)
            : (this.runs?.bytes ?? this.payload?.compressedBytes)) ?? 0,
        cpuDecoded:
          (this.remote
            ? (this.remoteStats.inflatedBytes as number)
            : (this.runs?.inflatedBytes ?? this.payload?.decodedBytes)) ?? 0,
        /** The node pool's index (slots by node) and per-slot arrays (in the worker with one). */
        cpuNodeIndex:
          (this.remote
            ? (this.remoteStats.indexBytes as number)
            : this.nodes?.bytes) ?? 0,
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
    if (this.remote) {
      const key = this.remoteKeys ? this.remoteKeys[i] : -1;
      if (key === undefined || key < 0) return null;
      const g = Math.floor(key / PAGE_PLACES);
      return { ...this.pageOwner[g], node: key - g * PAGE_PLACES };
    }
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
    this.remote?.terminate();
    this.remote = null;
  }
}

// --- in the browser ----------------------------------------------------------

/** A paged .athc with its compact tier, read over HTTP Range requests. */
export async function openCompactPiece(
  url: string,
  name = url,
): Promise<CompactPiece> {
  const { layout, prefix } = await readAthcLayout({ url });
  shareOcclusionCodebook(layout as { occlusionCodebook?: ArrayLike<number> });
  return {
    name,
    layout: layout as AthcV3Layout,
    prefix,
    read: async (offset, bytes) =>
      (await readRange({ url }, offset, bytes)).data,
    url:
      typeof location !== "undefined" ? new URL(url, location.href).href : url,
  };
}

/** The occlusion codebooks the pieces' layouts hold, one of each. */
const occlusionCodebooks: Uint32Array[] = [];

/**
 * A layout's occlusion codebook as one typed array the pieces with the same
 * share (thread CU: the asset's, as a JS array of numbers in each of a car's
 * nine layouts, was 4.3 MB of the page's heap). WgpuLod reads it with
 * `new Uint32Array(...)` as before.
 */
function shareOcclusionCodebook(layout: {
  occlusionCodebook?: ArrayLike<number>;
}) {
  const book = layout.occlusionCodebook;
  if (!book || book instanceof Uint32Array) return;
  const words = Uint32Array.from(book);
  const same = occlusionCodebooks.find(
    (b) => b.length === words.length && b.every((v, i) => v === words[i]),
  );
  if (!same && occlusionCodebooks.length >= 8) occlusionCodebooks.shift();
  if (!same) occlusionCodebooks.push(words);
  layout.occlusionCodebook = same ?? words;
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

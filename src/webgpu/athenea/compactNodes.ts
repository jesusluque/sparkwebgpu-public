// THE NODE POOL (thread CT): the compact path's GPU payload for the DRAWN cut
// only, a record a node, instead of whole band pages in pool slots.
//
// A band page's nodes are about a quarter drawn (the rest are the cut's
// ancestors, the nodes the culls remove, finer nodes the page's box reaches):
// the page pool (CompactPagePool) kept 40 GPU bytes for every one of them.
// Here:
//   - NodeSlotAllocator: node slots with a free list and an idle FIFO (nodes
//     that left the cut stay until their slot is wanted: the hysteresis);
//   - PagePayloadStore: each resident page's CMPC section as it was fetched
//     (gzip, ~29 B a node), and a small LRU of decompressed sections;
//   - CompactNodePool: one GPU buffer, the pieces' tables, every page's head
//     (16 words a page, all pages of all pieces: ~230 KB for the whole car)
//     then the records (NODE_RECORD_WORDS a node: the GPU's 40 bytes and the
//     node's origin), grown by copy when the cut outgrows it.
// compact.slang reads a record when nodeWords > 0.

import { gunzipSync } from "fflate";
import type { CompactCloud, CompactLayout } from "./compact";
import {
  COMPACT_PLANE_WIDTHS,
  type CompactPieceTables,
  compactPlaneOffset,
} from "./compactPages";

/** Words a node record: the GPU's 40 bytes, then its origin word. */
export const NODE_RECORD_WORDS = 11;
/** Words a page's head in the head table (compact.slang kCompactHeadWords). */
export const NODE_HEAD_WORDS = 16;
/** Places a page in a node key (CMPC's place: 12 bits). */
export const PAGE_PLACES = 4096;

/**
 * The GPU planes (compactPages GPU_PLANES' order, plane k of a CMPC section)
 * and their byte offset in a record: the 4-byte fields first, every 2-byte
 * field at an even byte, so no read crosses more than one word boundary.
 */
const RECORD_FIELDS: [name: string, plane: number, at: number][] = [
  ["rotation", 6, 0],
  ["weights", 11, 4],
  ["weights1", 12, 8],
  ["weights2", 13, 12],
  ["posX", 0, 16],
  ["posY", 1, 18],
  ["posZ", 2, 20],
  ["material", 9, 22],
  ["shadow", 14, 24],
  ["albedo", 3, 26],
  ["scale", 5, 29],
  ["normal", 7, 32],
  ["curvature", 8, 35],
  ["opacity", 4, 38],
  ["cluster", 10, 39],
];

/** Each field's byte offset in a record, by plane name. */
export function nodeRecordLayout(): Record<string, number> {
  const words: Record<string, number> = {};
  for (const [name, , at] of RECORD_FIELDS) words[name] = at;
  return words;
}

/**
 * Plane offsets of a section of n nodes, once per n (compactPlaneOffset
 * sums the widths each call).
 */
function planeOffsets(n: number) {
  return RECORD_FIELDS.map(([, plane, off]) => [
    compactPlaneOffset(plane, n),
    COMPACT_PLANE_WIDTHS[plane],
    off,
  ]);
}

// ------------------------------------------------------------- the slots

/**
 * Node slots: a key (a node's page and place, `page * PAGE_PLACES + place`)
 * to a slot. A frame's cut is `use`d (its slots stamped with the frame);
 * the nodes that were in the last cut and are not in this one go to an
 * idle FIFO, and a slot wanted with none free is taken from its head (the
 * node that left the cut longest ago), never from this frame's cut.
 */
export class NodeSlotAllocator {
  private keyOf: Int32Array;
  private stamp: Uint32Array;
  private free: number[] = [];
  private idle: number[] = [];
  private idleHead = 0;
  readonly slotOf = new Map<number, number>();
  /** The frame being built (`begin`); a slot stamped with it is in its cut. */
  frame = 1;
  private lastCut: Uint32Array = new Uint32Array(0);
  readonly stats = { allocated: 0, reused: 0, evicted: 0, failed: 0 };

  constructor(public capacity: number) {
    this.keyOf = new Int32Array(capacity).fill(-1);
    this.stamp = new Uint32Array(capacity);
    for (let s = capacity - 1; s >= 0; s--) this.free.push(s);
  }

  /** Slots in use (a node in them, in the cut or idle). */
  get used() {
    return this.slotOf.size;
  }

  /** More slots (the GPU buffer grown to hold them): free at the end. */
  grow(capacity: number) {
    if (capacity <= this.capacity) return;
    const keyOf = new Int32Array(capacity).fill(-1);
    keyOf.set(this.keyOf);
    const stamp = new Uint32Array(capacity);
    stamp.set(this.stamp);
    this.keyOf = keyOf;
    this.stamp = stamp;
    // The new slots taken lowest first (after any already free).
    const added: number[] = [];
    for (let s = capacity - 1; s >= this.capacity; s--) added.push(s);
    this.free = added.concat(this.free);
    this.capacity = capacity;
  }

  /** A new frame's cut begins. */
  begin() {
    this.frame += 1;
    this.lastCutAt = 0;
  }

  /** The slot of `key` if it is held (and in this frame's cut from now), else -1. */
  use(key: number) {
    const s = this.slotOf.get(key);
    if (s === undefined) return -1;
    this.stamp[s] = this.frame;
    this.stats.reused += 1;
    return s;
  }

  /** A slot for `key` (not held), in this frame's cut; -1: none left. */
  allocate(key: number) {
    let s = this.free.pop();
    while (s === undefined && this.idleHead < this.idle.length) {
      const c = this.idle[this.idleHead++];
      // Back in a cut since it went idle, or already taken: skip it.
      if (this.stamp[c] === this.frame || this.keyOf[c] < 0) continue;
      if (this.isInLastCut(c)) continue;
      this.slotOf.delete(this.keyOf[c]);
      this.stats.evicted += 1;
      s = c;
    }
    if (this.idleHead > 4096 && this.idleHead * 2 > this.idle.length) {
      this.idle = this.idle.slice(this.idleHead);
      this.idleHead = 0;
    }
    // Then a node of the last cut not in this one: the new cut and its
    // records go up together (setDrawList after the writes), so the frames
    // drawn before it never read the slot.
    while (s === undefined && this.lastCutAt < this.lastCut.length) {
      const c = this.lastCut[this.lastCutAt++];
      if (this.stamp[c] === this.frame || this.keyOf[c] < 0) continue;
      this.slotOf.delete(this.keyOf[c]);
      this.stats.evicted += 1;
      s = c;
    }
    if (s === undefined) {
      this.stats.failed += 1;
      return -1;
    }
    this.keyOf[s] = key;
    this.stamp[s] = this.frame;
    this.slotOf.set(key, s);
    this.stats.allocated += 1;
    return s;
  }

  private lastCutStamp = 0;
  private lastCutAt = 0;
  /** A slot of the last cut still drawn until this frame's goes up. */
  private isInLastCut(s: number) {
    return this.stamp[s] === this.lastCutStamp && this.lastCutStamp > 0;
  }

  /**
   * This frame's cut is complete (`cut` its slots): the last cut's slots
   * not in it go idle, oldest first to be taken.
   */
  commit(cut: Uint32Array) {
    for (const s of this.lastCut) {
      if (this.stamp[s] !== this.frame && this.keyOf[s] >= 0) this.idle.push(s);
    }
    this.lastCut = cut;
    this.lastCutStamp = this.frame;
  }

  /** Slot s back to the free list (a node that could not be written). */
  release(s: number) {
    const key = this.keyOf[s];
    if (key < 0) return;
    this.slotOf.delete(key);
    this.keyOf[s] = -1;
    this.free.push(s);
  }

  /** The key held in slot s, or -1. */
  keyIn(s: number) {
    return this.keyOf[s];
  }

  /** Forget every node (slots all free). */
  clear() {
    this.keyOf.fill(-1);
    this.stamp.fill(0);
    this.slotOf.clear();
    this.idle = [];
    this.idleHead = 0;
    this.lastCut = new Uint32Array(0);
    this.lastCutAt = 0;
    this.free = [];
    for (let s = this.capacity - 1; s >= 0; s--) this.free.push(s);
  }
}

// ----------------------------------------------------- the pages' payload

/**
 * The resident pages' CMPC sections on the CPU: compressed as fetched, and
 * the last few decompressed (an LRU), from which the cut's records are
 * packed. `sectionOf` gunzips synchronously (fflate) and times it.
 */
export class PagePayloadStore {
  private compressed = new Map<number, { bytes: Uint8Array; n: number }>();
  /** Insertion order is recency: the first entry is the least recent. */
  private decoded = new Map<number, Uint8Array>();
  compressedBytes = 0;
  decodedBytes = 0;
  readonly stats = { decodes: 0, decodeMs: 0, hits: 0 };

  constructor(
    /** Decompressed sections kept. */
    public decodedPages = 8,
    private gunzip: (b: Uint8Array) => Uint8Array = (b) => gunzipSync(b),
  ) {}

  /** A page landed: its stored section (null: stored uncompressed, `raw` is it) and its raw. */
  put(key: number, n: number, stored: Uint8Array | null, raw: Uint8Array) {
    this.drop(key);
    if (stored) {
      this.compressed.set(key, { bytes: stored, n });
      this.compressedBytes += stored.byteLength;
      this.remember(key, raw);
    } else {
      // Uncompressed in the file: the raw is the stored form.
      this.compressed.set(key, { bytes: raw, n });
      this.compressedBytes += raw.byteLength;
    }
  }

  has(key: number) {
    return this.compressed.has(key);
  }

  /** The page's raw section (decompressing it if it is not in the LRU), or null. */
  sectionOf(key: number): Uint8Array | null {
    const hit = this.decoded.get(key);
    if (hit) {
      this.decoded.delete(key);
      this.decoded.set(key, hit);
      this.stats.hits += 1;
      return hit;
    }
    const c = this.compressed.get(key);
    if (!c) return null;
    if (this.isRaw(c)) return c.bytes;
    const t0 = performance.now();
    const raw = this.gunzip(c.bytes);
    this.stats.decodeMs += performance.now() - t0;
    this.stats.decodes += 1;
    this.remember(key, raw);
    return raw;
  }

  private isRaw(c: { bytes: Uint8Array; n: number }) {
    // A gzip member starts 1f 8b; a raw section is n x node bytes.
    return !(c.bytes[0] === 0x1f && c.bytes[1] === 0x8b);
  }

  private remember(key: number, raw: Uint8Array) {
    if (this.decodedPages <= 0) return;
    const old = this.decoded.get(key);
    if (old) {
      this.decoded.delete(key);
      this.decodedBytes -= old.byteLength;
    }
    this.decoded.set(key, raw);
    this.decodedBytes += raw.byteLength;
    while (this.decoded.size > this.decodedPages) {
      const [k, v] = this.decoded.entries().next().value as [
        number,
        Uint8Array,
      ];
      this.decoded.delete(k);
      this.decodedBytes -= v.byteLength;
    }
  }

  /** The page left the pool. */
  drop(key: number) {
    const c = this.compressed.get(key);
    if (c) {
      this.compressed.delete(key);
      this.compressedBytes -= c.bytes.byteLength;
    }
    const d = this.decoded.get(key);
    if (d) {
      this.decoded.delete(key);
      this.decodedBytes -= d.byteLength;
    }
  }

  get pages() {
    return this.compressed.size;
  }
}

// ------------------------------------------------------------ the GPU pool

/**
 * The GPU side: the pieces' tables, every page's head, then `capacity`
 * records (grown by copy). Its `cloud` is the compact path's layout
 * (CompactRelitPath.setPagedCloud; again after a grow, as the buffer is new).
 */
export class CompactNodePool {
  buffer: GPUBuffer;
  readonly tableWords: number;
  readonly headTable: number;
  readonly recordsBase: number;
  cloud: CompactCloud;
  capacity: number;
  /** Each piece's codebook and materials (words in the buffer). */
  readonly pieceBase: { codebook: number; materials: number }[] = [];
  private tables: Uint32Array;
  private heads: Uint32Array;

  constructor(
    readonly device: GPUDevice,
    readonly pieces: CompactPieceTables[],
    /** Each piece's page heads (12 words a page: the page-table row's words 17..28). */
    pageHeads: ArrayLike<number>[][],
    capacity: number,
    readonly maxCapacity = 1 << 22,
  ) {
    if (!pieces.length) throw new Error("CompactNodePool: no piece");
    const first = pieces[0];
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
    this.headTable = Math.ceil(tables.length / 16) * 16;
    const pages = pageHeads.reduce((n, h) => n + h.length, 0);
    if (pages > 65536) throw new Error("CompactNodePool: over 65536 pages");
    const heads = new Uint32Array(pages * NODE_HEAD_WORDS);
    let g = 0;
    pageHeads.forEach((list, piece) => {
      list.forEach((head, page) => {
        const at = g * NODE_HEAD_WORDS;
        for (let k = 0; k < 12; k++) heads[at + k] = head[k] >>> 0;
        heads[at + 9] = this.pieceBase[piece].codebook;
        heads[at + 10] = this.pieceBase[piece].materials;
        // The faint dither's seed: as CompactPagePool's (the node's place in its piece).
        heads[at + 11] = ((piece << 26) + page * 4096) >>> 0;
        g += 1;
      });
    });
    this.tableWords = this.headTable;
    this.recordsBase = this.headTable + heads.length;
    this.tables = new Uint32Array(tables);
    this.heads = heads;
    this.capacity = Math.max(1024, Math.min(capacity, maxCapacity));
    this.buffer = this.createBuffer(this.capacity);
    device.queue.writeBuffer(this.buffer, 0, this.tables);
    device.queue.writeBuffer(this.buffer, this.headTable * 4, heads);
    const words = nodeRecordLayout();
    const layout: CompactLayout = {
      count: this.capacity,
      chunks: 0,
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
        weightsStride: 4,
        chunkTable: 0,
        pieceTable: 0,
        materials: 0,
        basis,
      },
      bytes: this.bytes(),
      paged: {
        slotNodes: 0,
        slotWords: 0,
        slotsBase: this.recordsBase,
        nodeWords: NODE_RECORD_WORDS,
        headTable: this.headTable,
      },
    };
    this.cloud = { data: new Uint32Array(0), layout };
  }

  private createBuffer(capacity: number) {
    return this.device.createBuffer({
      label: "compact nodes",
      size: (this.recordsBase + capacity * NODE_RECORD_WORDS) * 4,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });
  }

  bytes() {
    return {
      tables: this.headTable * 4,
      heads: this.heads.byteLength,
      records: this.capacity * NODE_RECORD_WORDS * 4,
      total: (this.recordsBase + this.capacity * NODE_RECORD_WORDS) * 4,
    };
  }

  /**
   * At least `need` records (x1.5 steps, at most maxCapacity): a new buffer
   * with the old one's words copied over. Returns whether it grew.
   */
  grow(need: number) {
    if (need <= this.capacity || this.capacity >= this.maxCapacity) {
      return false;
    }
    let next = this.capacity;
    while (next < need) next = Math.ceil(next * 1.5);
    next = Math.min(next, this.maxCapacity);
    const buffer = this.createBuffer(next);
    const enc = this.device.createCommandEncoder({ label: "node pool grow" });
    enc.copyBufferToBuffer(this.buffer, 0, buffer, 0, this.buffer.size);
    this.device.queue.submit([enc.finish()]);
    this.buffer.destroy();
    this.buffer = buffer;
    this.capacity = next;
    const layout = this.cloud.layout as CompactLayout;
    this.cloud = {
      data: this.cloud.data,
      layout: { ...layout, count: next, bytes: this.bytes() },
    };
    return true;
  }

  /**
   * Records into their slots: `slots` (ascending) and the packed records in
   * the same order; a run of consecutive slots is one writeBuffer.
   */
  writeRecords(slots: Uint32Array, records: Uint8Array) {
    const R = NODE_RECORD_WORDS * 4;
    let k = 0;
    let writes = 0;
    while (k < slots.length) {
      let j = k + 1;
      while (j < slots.length && slots[j] === slots[j - 1] + 1) j++;
      this.device.queue.writeBuffer(
        this.buffer,
        (this.recordsBase + slots[k] * NODE_RECORD_WORDS) * 4,
        records,
        k * R,
        (j - k) * R,
      );
      writes += 1;
      k = j;
    }
    return writes;
  }

  dispose() {
    this.buffer.destroy();
  }
}

/**
 * The records of nodes `places` of a page's raw section (n nodes), packed in
 * order into `out` from record `first`; `origin(place)` their origin words.
 */
export function packPageRecords(
  raw: Uint8Array,
  n: number,
  places: ArrayLike<number>,
  originBase: number,
  out: Uint8Array,
  first: number,
) {
  const offs = planeOffsets(n);
  const R = NODE_RECORD_WORDS * 4;
  for (let k = 0; k < places.length; k++) {
    const place = places[k];
    const at = (first + k) * R;
    for (const [src, w, off] of offs) {
      const from = src + place * w;
      for (let b = 0; b < w; b++) out[at + off + b] = raw[from + b];
    }
    const origin = (originBase + place * 65536) >>> 0;
    out[at + 40] = origin & 0xff;
    out[at + 41] = (origin >>> 8) & 0xff;
    out[at + 42] = (origin >>> 16) & 0xff;
    out[at + 43] = (origin >>> 24) & 0xff;
  }
}

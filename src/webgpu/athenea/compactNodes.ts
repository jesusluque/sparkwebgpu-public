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

import { gunzipSync, inflateSync } from "fflate";
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
 * Node keys (`page * PAGE_PLACES + place`) to slots: an Int32Array of a
 * page's places for each page with a node held (-1 none), freed when its
 * last node goes. About 16 KB a page, where a Map takes ~40 bytes a node.
 */
export class NodeIndex {
  private pages = new Map<number, { slots: Int32Array; count: number }>();
  size = 0;

  /** Places a page holds at most (a pool slot's nodes). */
  constructor(readonly places = PAGE_PLACES) {}

  get(key: number): number | undefined {
    const p = this.pages.get(Math.floor(key / PAGE_PLACES));
    if (!p) return undefined;
    const s = p.slots[key % PAGE_PLACES];
    return s < 0 ? undefined : s;
  }

  set(key: number, slot: number) {
    const g = Math.floor(key / PAGE_PLACES);
    let p = this.pages.get(g);
    if (!p) {
      p = { slots: new Int32Array(this.places).fill(-1), count: 0 };
      this.pages.set(g, p);
    }
    const k = key % PAGE_PLACES;
    if (p.slots[k] < 0) {
      p.count += 1;
      this.size += 1;
    }
    p.slots[k] = slot;
  }

  delete(key: number) {
    const g = Math.floor(key / PAGE_PLACES);
    const p = this.pages.get(g);
    if (!p) return;
    const k = key % PAGE_PLACES;
    if (p.slots[k] < 0) return;
    p.slots[k] = -1;
    p.count -= 1;
    this.size -= 1;
    if (p.count === 0) this.pages.delete(g);
  }

  clear() {
    this.pages.clear();
    this.size = 0;
  }

  /** Bytes held by the index. */
  get bytes() {
    return this.pages.size * this.places * 4;
  }
}

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
  readonly slotOf: NodeIndex;
  /** The frame being built (`begin`); a slot stamped with it is in its cut. */
  frame = 1;
  private lastCut: Uint32Array = new Uint32Array(0);
  readonly stats = { allocated: 0, reused: 0, evicted: 0, failed: 0 };

  constructor(
    public capacity: number,
    /** Places a page holds at most (NodeIndex). */
    places = PAGE_PLACES,
  ) {
    this.slotOf = new NodeIndex(places);
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

  /** JS bytes of the allocator: its index and its per-slot arrays. */
  get bytes() {
    return this.slotOf.bytes + this.capacity * 8;
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
    // A tenth over the cut, at least a quarter more (a copy each time).
    const next = Math.min(
      Math.ceil(Math.max(need * 1.1, this.capacity * 1.25)),
      this.maxCapacity,
    );
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

// ------------------------------------------------- CMPC v4 runs (thread CT)

/** "CMP4": a page's CMPC v4 container (rust athc_compact encode_runs). */
export const RUNS_MAGIC = 0x34504d43;
/** The compact section's encoding for CMPC v4 containers. */
export const ENCODING_COMPACT_RUNS = 16;
const RUNS_HEAD_WORDS = 5;
/** Bytes a node of the GPU's planes, and of the traversal's (v3). */
const GPU_BYTES = 40;
const TRAVERSAL_BYTES = 24;

/** A CMPC v4 container's head: its nodes, nodes a run, the traversal member, each run's end. */
export interface RunsContainer {
  n: number;
  run: number;
  runs: number;
  /** The traversal member (gzip), within the container. */
  travAt: number;
  travBytes: number;
  /** The first run's byte within the container; run r is [runAt + start(r), runAt + ends[r]). */
  runAt: number;
  ends: Uint32Array;
}

export function parseRunsContainer(c: Uint8Array): RunsContainer {
  const v = new DataView(c.buffer, c.byteOffset, c.byteLength);
  if (v.getUint32(0, true) !== RUNS_MAGIC) {
    throw new Error("CMPC v4: not a container");
  }
  const n = v.getUint32(4, true);
  const run = v.getUint32(8, true);
  const travBytes = v.getUint32(12, true);
  const runs = v.getUint32(16, true);
  const ends = new Uint32Array(runs);
  for (let r = 0; r < runs; r++) {
    ends[r] = v.getUint32(4 * (RUNS_HEAD_WORDS + r), true);
  }
  const travAt = 4 * (RUNS_HEAD_WORDS + runs);
  return { n, run, runs, travAt, travBytes, runAt: travAt + travBytes, ends };
}

/** Run r's bytes within the container: [start, end). */
export function runSpan(h: RunsContainer, r: number): [number, number] {
  return [h.runAt + (r > 0 ? h.ends[r - 1] : 0), h.runAt + h.ends[r]];
}

/** Nodes of run r. */
function runCount(n: number, run: number, r: number) {
  return Math.min(run, n - r * run);
}

/** A run's planes (inflated): plane k of a run of m nodes at prefix(k) x m. */
function runPlaneOffsets(m: number) {
  const offs: number[] = [];
  let at = 0;
  for (let k = 0; k < 15; k++) {
    offs.push(at);
    at += COMPACT_PLANE_WIDTHS[k] * m;
  }
  return offs;
}

/**
 * A container (as fetched) to the page's planes: the traversal's input
 * (positions then the traversal's planes, 30 bytes a node) and, with
 * `full`, the whole raw section (NODE_BYTES a node) a page pool writes.
 * `travRaw`: the traversal member gunzipped. Returns the inflated runs too.
 */
export function unpackRunsContainer(
  c: Uint8Array,
  h: RunsContainer,
  travRaw: Uint8Array,
  full: boolean,
) {
  const { n, run, runs } = h;
  const input = new Uint8Array(6 * n + TRAVERSAL_BYTES * n);
  input.set(travRaw.subarray(0, TRAVERSAL_BYTES * n), 6 * n);
  const raw = full ? new Uint8Array(64 * n) : null;
  if (raw) raw.set(travRaw.subarray(0, TRAVERSAL_BYTES * n), GPU_BYTES * n);
  const planes: Uint8Array[] = [];
  for (let r = 0; r < runs; r++) {
    const [a, b] = runSpan(h, r);
    const p = inflateSync(c.subarray(a, b));
    planes.push(p);
    const m = runCount(n, run, r);
    const offs = runPlaneOffsets(m);
    // The positions (planes 0-2, 2 bytes each) into the traversal's input.
    for (let k = 0; k < 3; k++) {
      input.set(p.subarray(offs[k], offs[k] + 2 * m), 2 * k * n + 2 * r * run);
    }
    if (raw) {
      for (let k = 0; k < 15; k++) {
        const w = COMPACT_PLANE_WIDTHS[k];
        raw.set(
          p.subarray(offs[k], offs[k] + w * m),
          compactPlaneOffset(k, n) + r * run * w,
        );
      }
    }
  }
  return { input, raw, planes };
}

/**
 * Records of nodes `locals` (places within run r's nodes) from the run's
 * inflated planes (m nodes), packed in order into `out` from record
 * `first`; `page` their head-table page, `base` the run's first place.
 */
export function packRunRecords(
  planes: Uint8Array,
  m: number,
  locals: ArrayLike<number>,
  page: number,
  base: number,
  out: Uint8Array,
  first: number,
) {
  const offs = runPlaneOffsets(m);
  const R = NODE_RECORD_WORDS * 4;
  for (let k = 0; k < locals.length; k++) {
    const j = locals[k];
    const at = (first + k) * R;
    for (const [, plane, off] of RECORD_FIELDS) {
      const w = COMPACT_PLANE_WIDTHS[plane];
      const from = offs[plane] + j * w;
      for (let b = 0; b < w; b++) out[at + off + b] = planes[from + b];
    }
    const origin = (page + (base + j) * 65536) >>> 0;
    out[at + 40] = origin & 0xff;
    out[at + 41] = (origin >>> 8) & 0xff;
    out[at + 42] = (origin >>> 16) & 0xff;
    out[at + 43] = (origin >>> 24) & 0xff;
  }
}

/** A run's priority: kept for the cut (its nodes, parents, children), culled, other. */
export const RUN_CUT = 0;
export const RUN_CULLED = 1;
export const RUN_OTHER = 2;

interface RunPage {
  n: number;
  run: number;
  /** The container's first byte in the file, its head. */
  fileAt: number;
  head: RunsContainer;
  /** Each run's deflate bytes, or null once dropped (fetched again by Range). */
  bytes: (Uint8Array | null)[];
  /** Each run's class and the update it was last classed at. */
  cls: Uint8Array;
  stamp: Uint32Array;
}

/**
 * The resident pages' payload as CMPC v4 runs: each run's deflate bytes,
 * kept or dropped by class under a byte budget (the cut's runs never), a
 * small LRU of inflated runs, and the runs to fetch again.
 */
export class RunPayloadStore {
  private pages = new Map<number, RunPage>();
  /** Insertion order is recency. */
  private inflated = new Map<number, Uint8Array>();
  bytes = 0;
  inflatedBytes = 0;
  frame = 0;
  readonly stats = {
    inflates: 0,
    inflateMs: 0,
    hits: 0,
    dropped: 0,
    fetched: 0,
  };

  constructor(
    /** Bytes of deflate runs kept beyond the cut's (Infinity: all). */
    public budget = Number.POSITIVE_INFINITY,
    /** Inflated runs kept. */
    public inflatedRuns = 256,
  ) {}

  private key(page: number, r: number) {
    return page * 64 + r;
  }

  /** A page landed: its container (its own copy) and runs already inflated. */
  put(page: number, c: Uint8Array, fileAt: number, planes?: Uint8Array[]) {
    this.drop(page);
    const head = parseRunsContainer(c);
    const bytes: Uint8Array[] = [];
    for (let r = 0; r < head.runs; r++) {
      const [a, b] = runSpan(head, r);
      bytes.push(c.slice(a, b));
      this.bytes += b - a;
    }
    this.pages.set(page, {
      n: head.n,
      run: head.run,
      fileAt,
      head,
      bytes,
      cls: new Uint8Array(head.runs).fill(RUN_OTHER),
      stamp: new Uint32Array(head.runs).fill(this.frame),
    });
    if (planes) {
      planes.forEach((p, r) => this.remember(this.key(page, r), p));
    }
  }

  has(page: number) {
    return this.pages.has(page);
  }

  /** The nodes of a page's run r, and its first place. */
  runOf(page: number, place: number) {
    const p = this.pages.get(page);
    if (!p) return null;
    const r = Math.floor(place / p.run);
    return { r, base: r * p.run, m: runCount(p.n, p.run, r) };
  }

  /** Classes run r of a page for this update (the lowest class wins). */
  mark(page: number, place: number, cls: number) {
    const p = this.pages.get(page);
    if (!p) return;
    const r = Math.floor(place / p.run);
    if (p.stamp[r] !== this.frame || cls < p.cls[r]) {
      p.cls[r] = cls;
      p.stamp[r] = this.frame;
    }
  }

  /** A new update's classes begin. */
  begin() {
    this.frame += 1;
  }

  /**
   * The runs over budget dropped: not this update's cut runs, classes
   * other first, then culled, the longest unmarked first.
   */
  trim() {
    if (this.bytes <= this.budget) return 0;
    const cands: [number, number, number, number][] = [];
    for (const [page, p] of this.pages) {
      for (let r = 0; r < p.bytes.length; r++) {
        if (!p.bytes[r]) continue;
        const cls = p.stamp[r] === this.frame ? p.cls[r] : RUN_OTHER;
        if (cls === RUN_CUT) continue;
        cands.push([cls, p.stamp[r], page, r]);
      }
    }
    cands.sort((a, b) => b[0] - a[0] || a[1] - b[1]);
    let dropped = 0;
    for (const [, , page, r] of cands) {
      if (this.bytes <= this.budget) break;
      const p = this.pages.get(page) as RunPage;
      this.bytes -= (p.bytes[r] as Uint8Array).byteLength;
      p.bytes[r] = null;
      dropped += 1;
    }
    this.stats.dropped += dropped;
    return dropped;
  }

  /** Whether run r is inflated already (no inflate to pack it). */
  isInflated(page: number, r: number) {
    return this.inflated.has(this.key(page, r));
  }

  /** Run r's inflated planes, or null when the run was dropped. */
  planes(page: number, r: number): Uint8Array | null {
    const k = this.key(page, r);
    const hit = this.inflated.get(k);
    if (hit) {
      this.inflated.delete(k);
      this.inflated.set(k, hit);
      this.stats.hits += 1;
      return hit;
    }
    const rp = this.pages.get(page);
    const b = rp?.bytes[r];
    if (!rp || !b) return null;
    // Used now: last to go by recency (classes left as marked).
    if (rp.stamp[r] !== this.frame) {
      rp.stamp[r] = this.frame;
      rp.cls[r] = RUN_OTHER;
    }
    const t0 = performance.now();
    const p = inflateSync(b);
    this.stats.inflateMs += performance.now() - t0;
    this.stats.inflates += 1;
    this.remember(k, p);
    return p;
  }

  /** Where run r is in the file, to fetch it again: [offset, bytes]. */
  fileSpan(page: number, r: number): [number, number] | null {
    const p = this.pages.get(page);
    if (!p) return null;
    const [a, b] = runSpan(p.head, r);
    return [p.fileAt + a, b - a];
  }

  /** A dropped run fetched again. */
  putRun(page: number, r: number, bytes: Uint8Array) {
    const p = this.pages.get(page);
    if (!p || p.bytes[r]) return;
    p.bytes[r] = bytes;
    this.bytes += bytes.byteLength;
    this.stats.fetched += 1;
  }

  missing(page: number, r: number) {
    const p = this.pages.get(page);
    return !!p && !p.bytes[r] && !this.inflated.has(this.key(page, r));
  }

  private remember(k: number, p: Uint8Array) {
    if (this.inflatedRuns <= 0) return;
    const old = this.inflated.get(k);
    if (old) {
      this.inflated.delete(k);
      this.inflatedBytes -= old.byteLength;
    }
    this.inflated.set(k, p);
    this.inflatedBytes += p.byteLength;
    while (this.inflated.size > this.inflatedRuns) {
      const [key, v] = this.inflated.entries().next().value as [
        number,
        Uint8Array,
      ];
      this.inflated.delete(key);
      this.inflatedBytes -= v.byteLength;
    }
  }

  /** The page left the traversal: its runs go. */
  drop(page: number) {
    const p = this.pages.get(page);
    if (!p) return;
    for (let r = 0; r < p.bytes.length; r++) {
      const b = p.bytes[r];
      if (b) this.bytes -= b.byteLength;
      const k = this.key(page, r);
      const i = this.inflated.get(k);
      if (i) {
        this.inflated.delete(k);
        this.inflatedBytes -= i.byteLength;
      }
    }
    this.pages.delete(page);
  }

  /** Runs held, and by class at this update. */
  census() {
    const out = {
      pages: this.pages.size,
      runs: 0,
      held: 0,
      cut: 0,
      culled: 0,
      other: 0,
    };
    for (const p of this.pages.values()) {
      for (let r = 0; r < p.bytes.length; r++) {
        out.runs += 1;
        if (!p.bytes[r]) continue;
        out.held += 1;
        const cls = p.stamp[r] === this.frame ? p.cls[r] : RUN_OTHER;
        if (cls === RUN_CUT) out.cut += 1;
        else if (cls === RUN_CULLED) out.culled += 1;
        else out.other += 1;
      }
    }
    return out;
  }
}

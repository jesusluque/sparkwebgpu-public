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
import type { CompactPieceTables } from "./compactPages";
import {
  COMPACT_PLANE_WIDTHS,
  NODE_RECORD_WORDS,
  RECORD_FIELDS,
  compactPlaneOffset,
  packRunRecords,
  runPlaneOffsets,
} from "./compactRecords";

export { NODE_RECORD_WORDS, packRunRecords };

/** Words a page's head in the head table (compact.slang kCompactHeadWords). */
export const NODE_HEAD_WORDS = 16;
/** Places a page in a node key (CMPC's place: 12 bits). */
export const PAGE_PLACES = 4096;

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
  /**
   * A page's held places as a bitmap with the bits before each word
   * counted, and the slots in place order (thread CU): a lookup is a bit
   * and a popcount, a page about 0.6 KB plus 4 bytes a node held (the
   * cut's pages hold ~20 % of their nodes; an Int32Array of every place was
   * 12 KB a page, 6-8 MB for the whole car).
   */
  private pages: (IndexPage | undefined)[] = [];
  private held = 0;
  private slotBytes = 0;
  size = 0;

  /** Places a page holds at most (a pool slot's nodes). */
  constructor(readonly places = PAGE_PLACES) {}

  get(key: number): number | undefined {
    const s = this.slot(key);
    return s < 0 ? undefined : s;
  }

  /** The slot of `key`, or -1 (get without the undefined). */
  slot(key: number): number {
    const p = this.pages[key >>> 12];
    if (p === undefined) return -1;
    const k = key & 4095;
    const w = k >>> 5;
    const word = p.bits[w];
    const bit = 1 << (k & 31);
    if ((word & bit) === 0) return -1;
    return p.slots[p.pre[w] + popcount(word & (bit - 1))];
  }

  /** The slots page g holds (a copy), or undefined. */
  slotsIn(g: number): Int32Array | undefined {
    const p = this.pages[g];
    return p ? p.slots.slice(0, p.n) : undefined;
  }

  set(key: number, slot: number) {
    const g = key >>> 12;
    let p = this.pages[g];
    if (p === undefined) {
      const words = Math.ceil(this.places / 32);
      p = {
        bits: new Int32Array(words),
        pre: new Uint16Array(words),
        slots: new Int32Array(16),
        n: 0,
      };
      this.pages[g] = p;
      this.held += 1;
      this.slotBytes += 16 * 4;
    }
    const k = key & 4095;
    const w = k >>> 5;
    const bit = 1 << (k & 31);
    const r = p.pre[w] + popcount(p.bits[w] & (bit - 1));
    if (p.bits[w] & bit) {
      p.slots[r] = slot;
      return;
    }
    if (p.n === p.slots.length) {
      const grown = new Int32Array(
        Math.min(this.places, Math.ceil(p.slots.length * 1.25) + 8),
      );
      grown.set(p.slots);
      this.slotBytes += (grown.length - p.slots.length) * 4;
      p.slots = grown;
    }
    p.slots.copyWithin(r + 1, r, p.n);
    p.slots[r] = slot;
    p.n += 1;
    p.bits[w] |= bit;
    for (let x = w + 1; x < p.pre.length; x++) p.pre[x] += 1;
    this.size += 1;
  }

  delete(key: number) {
    const g = key >>> 12;
    const p = this.pages[g];
    if (p === undefined) return;
    const k = key & 4095;
    const w = k >>> 5;
    const bit = 1 << (k & 31);
    if ((p.bits[w] & bit) === 0) return;
    const r = p.pre[w] + popcount(p.bits[w] & (bit - 1));
    p.slots.copyWithin(r, r + 1, p.n);
    p.n -= 1;
    p.bits[w] &= ~bit;
    for (let x = w + 1; x < p.pre.length; x++) p.pre[x] -= 1;
    this.size -= 1;
    if (p.n === 0) {
      this.pages[g] = undefined;
      this.held -= 1;
      this.slotBytes -= p.slots.length * 4;
    }
  }

  clear() {
    this.pages = [];
    this.held = 0;
    this.slotBytes = 0;
    this.size = 0;
  }

  /** Bytes held by the index. */
  get bytes() {
    return this.held * Math.ceil(this.places / 32) * 6 + this.slotBytes;
  }
}

interface IndexPage {
  bits: Int32Array;
  pre: Uint16Array;
  slots: Int32Array;
  n: number;
}

/** Set bits of a 32-bit word. */
function popcount(v: number) {
  let x = v - ((v >>> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  return (Math.imul((x + (x >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24) & 0xff;
}

/**
 * Node slots: a key (a node's page and place, `page * PAGE_PLACES + place`)
 * to a slot. A frame's cut is `use`d (its slots stamped with the frame);
 * the nodes that were in the last cut and are not in this one go to an
 * idle FIFO, and a slot wanted with none free is taken from its head (the
 * node that left the cut longest ago), never from this frame's cut.
 */
/**
 * The allocator's slots to take, as one ring of the capacity with a flag a
 * slot (each slot in it once): free slots at the front, taken first, and
 * idle ones (a node still in them) at the back, oldest first (thread CU
 * round 4: the free list was a stack of its own, 4 bytes a slot more).
 */
class SlotDeque {
  q: Int32Array;
  /** One bit a slot: in the ring (thread CU round 10: was a byte). */
  inQ: Uint32Array;
  head = 0;
  len = 0;
  constructor(capacity: number) {
    this.q = new Int32Array(Math.max(capacity, 1));
    this.inQ = new Uint32Array((Math.max(capacity, 1) + 31) >>> 5);
  }
  /** An idle slot, behind the others. */
  push(s: number) {
    if (this.inQ[s >>> 5] & (1 << (s & 31))) return;
    this.inQ[s >>> 5] |= 1 << (s & 31);
    this.q[(this.head + this.len) % this.q.length] = s;
    this.len += 1;
  }
  /** A free slot, in front (taken next). */
  unshift(s: number) {
    if (this.inQ[s >>> 5] & (1 << (s & 31))) return;
    this.inQ[s >>> 5] |= 1 << (s & 31);
    this.head = (this.head + this.q.length - 1) % this.q.length;
    this.q[this.head] = s;
    this.len += 1;
  }
  shift(): number | undefined {
    if (!this.len) return undefined;
    const s = this.q[this.head];
    this.head = (this.head + 1) % this.q.length;
    this.len -= 1;
    this.inQ[s >>> 5] &= ~(1 << (s & 31));
    return s;
  }
  /** Its slots, front first (a copy). */
  toArray() {
    const out = new Int32Array(this.len);
    for (let k = 0; k < this.len; k++)
      out[k] = this.q[(this.head + k) % this.q.length];
    return out;
  }
  /**
   * Rebuilt for `capacity` slots: `front` (in order) then its own slots
   * that `keep` keeps, in order.
   */
  rebuild(
    capacity: number,
    keep: (s: number) => boolean,
    front: ArrayLike<number> = [],
  ) {
    const old = this.toArray();
    this.q = new Int32Array(Math.max(capacity, 1));
    this.inQ = new Uint32Array((Math.max(capacity, 1) + 31) >>> 5);
    this.head = 0;
    this.len = 0;
    for (let k = 0; k < front.length; k++) this.push(front[k]);
    for (const s of old) if (s < capacity && keep(s)) this.push(s);
  }
}

/** Stamps are u16 frames; renumbered before they wrap (thread CU round 4). */
const STAMP_WRAP = 0xfff0;

export class NodeSlotAllocator {
  private keyOf: Int32Array;
  private stamp: Uint16Array;
  /** Free slots in front, idle ones behind. */
  private idle: SlotDeque;
  readonly slotOf: NodeIndex;
  /** The frame being built (`begin`); a slot stamped with it is in its cut. */
  frame = 1;
  private lastCut: Uint32Array = new Uint32Array(0);
  /**
   * Pinned slots (thread CU round 9): a cut the page keeps to draw at once
   * (the fast cut while a still frame draws a finer one): never taken,
   * idled nor moved until unpinned.
   */
  private pinned = new Uint32Array(0);
  pinnedN = 0;
  readonly stats = { allocated: 0, reused: 0, evicted: 0, failed: 0 };

  constructor(
    public capacity: number,
    /** Places a page holds at most (NodeIndex). */
    places = PAGE_PLACES,
  ) {
    this.slotOf = new NodeIndex(places);
    this.keyOf = new Int32Array(capacity).fill(-1);
    this.stamp = new Uint16Array(capacity);
    this.idle = new SlotDeque(capacity);
    for (let s = 0; s < capacity; s++) this.idle.push(s);
  }

  /** Pins `slots` (those holding a node); replaces any earlier pin. */
  pin(slots: ArrayLike<number>) {
    this.unpin();
    this.pinned = new Uint32Array((this.capacity + 31) >>> 5);
    let n = 0;
    for (let k = 0; k < slots.length; k++) {
      const s = slots[k];
      if (s < this.capacity && this.keyOf[s] >= 0 && !this.isPinnedBit(s)) {
        this.pinned[s >>> 5] |= 1 << (s & 31);
        n += 1;
      }
    }
    this.pinnedN = n;
  }

  /** Lets the pinned slots go: those not in use go idle. */
  unpin() {
    if (!this.pinnedN) {
      this.pinned = new Uint32Array(0);
      return;
    }
    const p = this.pinned;
    this.pinned = new Uint32Array(0);
    this.pinnedN = 0;
    for (let s = 0; s < this.capacity; s++) {
      if (
        p[s >>> 5] & (1 << (s & 31)) &&
        this.keyOf[s] >= 0 &&
        this.stamp[s] !== this.frame &&
        !this.isInLastCut(s)
      ) {
        this.idle.push(s);
      }
    }
  }

  private isPinned(s: number) {
    return this.pinnedN > 0 && this.isPinnedBit(s);
  }

  private isPinnedBit(s: number) {
    return (this.pinned[s >>> 5] & (1 << (s & 31))) !== 0;
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
    const stamp = new Uint16Array(capacity);
    stamp.set(this.stamp);
    this.keyOf = keyOf;
    this.stamp = stamp;
    // The new slots taken first, lowest first.
    const added = new Int32Array(capacity - this.capacity);
    for (let k = 0; k < added.length; k++) added[k] = this.capacity + k;
    this.idle.rebuild(capacity, () => true, added);
    if (this.pinned.length) {
      const pinned = new Uint32Array((capacity + 31) >>> 5);
      pinned.set(this.pinned);
      this.pinned = pinned;
    }
    this.capacity = capacity;
  }

  /** A new frame's cut begins. */
  begin() {
    if (this.frame >= STAMP_WRAP) this.renumber();
    this.frame += 1;
    this.lastCutAt = 0;
  }

  /**
   * Stamps back to small numbers before the u16 wraps: this frame's 3,
   * the last cut's 2, any other 1 (they only ever compare equal).
   */
  private renumber() {
    const st = this.stamp;
    const f = this.frame;
    const l = this.lastCutStamp;
    for (let s = 0; s < st.length; s++) {
      const v = st[s];
      st[s] = v === 0 ? 0 : v === f ? 3 : l > 0 && v === l ? 2 : 1;
    }
    if (l > 0) this.lastCutStamp = 2;
    this.frame = 3;
  }

  /** The slot of `key` if it is held (and in this frame's cut from now), else -1. */
  use(key: number) {
    const s = this.slotOf.slot(key);
    if (s < 0) return -1;
    if (this.stamp[s] !== this.frame) {
      this.stamp[s] = this.frame;
      this.touch(s);
    }
    this.stats.reused += 1;
    return s;
  }

  /**
   * The slots this frame stamped (its cut, the substitutes, the records on
   * their way: whatever it used, thread CU), each once: at the next commit
   * those the next frame does not use go idle.
   */
  private cur = new Uint32Array(1024);
  private curN = 0;

  private touch(s: number) {
    if (this.curN === this.cur.length) {
      const c = new Uint32Array(this.cur.length * 2);
      c.set(this.cur);
      this.cur = c;
    }
    this.cur[this.curN++] = s;
  }

  /** A slot for `key` (not held), in this frame's cut; -1: none left. */
  allocate(key: number) {
    let s: number | undefined;
    while (s === undefined && this.idle.len > 0) {
      const c = this.idle.shift() as number;
      // Free (in front, or let go while idle): take it.
      if (this.keyOf[c] < 0) {
        s = c;
        break;
      }
      // Back in a cut since it went idle, or pinned: skip it.
      if (this.stamp[c] === this.frame || this.isPinned(c)) continue;
      if (this.isInLastCut(c)) continue;
      this.slotOf.delete(this.keyOf[c]);
      this.stats.evicted += 1;
      s = c;
    }
    // Then a node of the last cut not in this one: the new cut and its
    // records go up together (setDrawList after the writes), so the frames
    // drawn before it never read the slot.
    while (s === undefined && this.lastCutAt < this.lastCut.length) {
      const c = this.lastCut[this.lastCutAt++];
      if (this.stamp[c] === this.frame || this.keyOf[c] < 0) continue;
      if (this.isPinned(c)) continue;
      this.slotOf.delete(this.keyOf[c]);
      this.stats.evicted += 1;
      s = c;
    }
    if (s === undefined) {
      this.stats.failed += 1;
      return -1;
    }
    this.keyOf[s] = key;
    if (this.stamp[s] !== this.frame) {
      this.stamp[s] = this.frame;
      this.touch(s);
    }
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
   * This frame's cut is complete: the last frame's slots not used in this
   * one go idle, oldest first to be taken. (Every slot a frame stamps
   * counts, not only `cut`: a substitute or a record on its way that a
   * later draw list leaves out still goes idle, thread CU.)
   */
  commit(_cut?: Uint32Array) {
    const last = this.lastCut;
    const stamp = this.stamp;
    const keyOf = this.keyOf;
    const frame = this.frame;
    for (let k = 0; k < last.length; k++) {
      const s = last[k];
      if (stamp[s] !== frame && keyOf[s] >= 0 && !this.isPinned(s)) {
        this.idle.push(s);
      }
    }
    this.lastCut = this.cur.slice(0, this.curN);
    this.curN = 0;
    this.lastCutStamp = this.frame;
  }

  /**
   * Page g left the traversal (thread CU): its nodes not used this frame
   * nor the last go back to the free list (their page's index array with
   * them), so the index holds about the resident pages only.
   */
  releasePage(g: number) {
    const p = this.slotOf.slotsIn(g);
    if (!p) return 0;
    let n = 0;
    for (let k = 0; k < p.length; k++) {
      const s = p[k];
      if (
        s < 0 ||
        this.stamp[s] === this.frame ||
        this.isInLastCut(s) ||
        this.isPinned(s)
      ) {
        continue;
      }
      this.release(s);
      n += 1;
    }
    return n;
  }

  /** Slot s back to the free list (a node that could not be written). */
  release(s: number) {
    const key = this.keyOf[s];
    if (key < 0 || this.isPinned(s)) return;
    this.slotOf.delete(key);
    this.keyOf[s] = -1;
    this.idle.unshift(s);
  }

  /**
   * Fewer slots (thread CU: the pool after a large cut): the nodes past
   * `capacity` used this frame or the last (drawn) move to free or idle
   * slots below it, the others are let go. Returns the moves (from, to
   * pairs), or null when there is not room below (nothing changed).
   */
  shrink(capacity: number): Uint32Array | null {
    if (capacity >= this.capacity) return new Uint32Array(0);
    // Pinned slots do not move (the page draws them by slot).
    if (this.pinnedN > 0) return null;
    const keyOf = this.keyOf;
    const stamp = this.stamp;
    const live = (s: number) =>
      keyOf[s] >= 0 &&
      (stamp[s] === this.frame ||
        (this.lastCutStamp > 0 && stamp[s] === this.lastCutStamp));
    // Room below: free slots, then idle ones (not live).
    const roomSet = new Set<number>();
    const idle = this.idle.toArray();
    for (let k = 0; k < idle.length; k++) {
      const s = idle[k];
      if (s < capacity && keyOf[s] < 0) roomSet.add(s);
    }
    let need = 0;
    for (let s = capacity; s < this.capacity; s++) if (live(s)) need += 1;
    for (let k = 0; k < idle.length && roomSet.size < need; k++) {
      const s = idle[k];
      if (s < capacity && keyOf[s] >= 0 && !live(s)) roomSet.add(s);
    }
    if (roomSet.size < need) return null;
    const room = [...roomSet];
    const moves: number[] = [];
    const used = new Set<number>();
    for (let s = capacity; s < this.capacity; s++) {
      const key = keyOf[s];
      if (key < 0) continue;
      if (!live(s)) {
        this.slotOf.delete(key);
        keyOf[s] = -1;
        continue;
      }
      const t = room.pop() as number;
      used.add(t);
      if (keyOf[t] >= 0) {
        this.slotOf.delete(keyOf[t]);
        this.stats.evicted += 1;
      }
      keyOf[t] = key;
      stamp[t] = stamp[s];
      this.slotOf.set(key, t);
      keyOf[s] = -1;
      moves.push(s, t);
    }
    const map = new Map<number, number>();
    for (let k = 0; k < moves.length; k += 2) map.set(moves[k], moves[k + 1]);
    const remap = (a: ArrayLike<number>, n: number) => {
      const out: number[] = [];
      for (let k = 0; k < n; k++) {
        const s = a[k];
        const t = s < capacity ? s : map.get(s);
        if (t !== undefined) out.push(t);
      }
      return out;
    };
    this.lastCut = Uint32Array.from(remap(this.lastCut, this.lastCut.length));
    const cur = remap(this.cur, this.curN);
    this.cur.set(cur);
    this.curN = cur.length;
    this.idle.rebuild(capacity, (s) => !used.has(s));
    this.lastCutAt = 0;
    this.keyOf = keyOf.slice(0, capacity);
    this.stamp = stamp.slice(0, capacity);
    this.capacity = capacity;
    return Uint32Array.from(moves);
  }

  /** The key held in slot s, or -1. */
  keyIn(s: number) {
    return this.keyOf[s];
  }

  /** JS bytes of the allocator: its index and its per-slot arrays. */
  get bytes() {
    return (
      this.slotOf.bytes +
      this.capacity * 10.125 +
      this.pinned.byteLength +
      this.cur.byteLength +
      this.lastCut.byteLength
    );
  }

  /** Forget every node (slots all free). */
  clear() {
    this.keyOf.fill(-1);
    this.stamp.fill(0);
    this.slotOf.clear();
    this.idle = new SlotDeque(this.capacity);
    for (let s = 0; s < this.capacity; s++) this.idle.push(s);
    this.lastCut = new Uint32Array(0);
    this.lastCutAt = 0;
    this.curN = 0;
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
  private headBytes: number;

  constructor(
    readonly device: GPUDevice,
    /** Not kept (thread CU): they hold the pieces' prefixes, on the GPU from here. */
    pieces: CompactPieceTables[],
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
    const tableWords = new Uint32Array(tables);
    this.headBytes = heads.byteLength;
    this.capacity = Math.max(1024, Math.min(capacity, maxCapacity));
    this.buffer = this.createBuffer(this.capacity);
    device.queue.writeBuffer(this.buffer, 0, tableWords);
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
      heads: this.headBytes,
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
   * Exactly `capacity` records (thread CU: the node pool's worker decides):
   * a new buffer with the tables, the heads and the records below both
   * capacities copied, then `moves` (from, to slot pairs: records past a
   * smaller capacity kept in a slot below it).
   */
  private movePipeline: GPUComputePipeline | null = null;

  /** Records `moves` (from, to slot pairs) from this buffer into `dst`. */
  private moveRecords(
    enc: GPUCommandEncoder,
    dst: GPUBuffer,
    moves: Uint32Array,
  ) {
    const d = this.device;
    this.movePipeline ??= d.createComputePipeline({
      label: "node pool moves",
      layout: "auto",
      compute: {
        module: d.createShaderModule({
          label: "node pool moves",
          code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<storage, read> moves: array<u32>;
@group(0) @binding(3) var<uniform> p: vec4<u32>; // moves, records base (words), words a record, threads a row
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.y * p.w + g.x;
  let m = t / p.z;
  if (m >= p.x) { return; }
  let w = t % p.z;
  dst[p.y + moves[2u * m + 1u] * p.z + w] = src[p.y + moves[2u * m] * p.z + w];
}`,
        }),
        entryPoint: "main",
      },
    });
    const n = moves.length >> 1;
    const threads = n * NODE_RECORD_WORDS;
    const groups = Math.ceil(threads / 64);
    const gx = Math.min(groups, 65535);
    const gy = Math.ceil(groups / gx);
    const mb = d.createBuffer({
      label: "node pool moves",
      size: moves.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    d.queue.writeBuffer(
      mb,
      0,
      moves.buffer,
      moves.byteOffset,
      moves.byteLength,
    );
    const ub = d.createBuffer({
      label: "node pool moves params",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    d.queue.writeBuffer(
      ub,
      0,
      new Uint32Array([n, this.recordsBase, NODE_RECORD_WORDS, gx * 64]),
    );
    const pass = enc.beginComputePass({ label: "node pool moves" });
    pass.setPipeline(this.movePipeline);
    pass.setBindGroup(
      0,
      d.createBindGroup({
        layout: this.movePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.buffer } },
          { binding: 1, resource: { buffer: dst } },
          { binding: 2, resource: { buffer: mb } },
          { binding: 3, resource: { buffer: ub } },
        ],
      }),
    );
    pass.dispatchWorkgroups(gx, gy);
    pass.end();
    // The params buffer is small: let it go with the moves.
    const both = mb;
    queueMicrotask(() => ub.destroy());
    return both;
  }

  resize(capacity: number, moves?: Uint32Array) {
    const next = Math.min(Math.max(capacity, 1024), this.maxCapacity);
    if (next === this.capacity && !moves?.length) return false;
    const buffer = this.createBuffer(next);
    const enc = this.device.createCommandEncoder({ label: "node pool resize" });
    const keep =
      (this.recordsBase + Math.min(next, this.capacity) * NODE_RECORD_WORDS) *
      4;
    enc.copyBufferToBuffer(this.buffer, 0, buffer, 0, keep);
    const R = NODE_RECORD_WORDS * 4;
    let movesBuffer: GPUBuffer | null = null;
    if (moves && moves.length > 2 * 64) {
      // Many moves (a still frame's pool given back): one dispatch, a
      // thread a word, instead of a copy command a record (thread CU
      // round 10: ~100k copies stalled the first moving frame ~90 ms).
      movesBuffer = this.moveRecords(enc, buffer, moves);
    } else {
      for (let k = 0; moves && k < moves.length; k += 2) {
        enc.copyBufferToBuffer(
          this.buffer,
          this.recordsBase * 4 + moves[k] * R,
          buffer,
          this.recordsBase * 4 + moves[k + 1] * R,
          R,
        );
      }
    }
    this.device.queue.submit([enc.finish()]);
    movesBuffer?.destroy();
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
  /**
   * Not trimmed before this update (thread CU): a page that just landed
   * waits for the cuts that can use it (its traversal input lands an update
   * later when the store is in a worker).
   */
  grace: number;
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
  /** Pages held (thread CU: the memory census). */
  get pageCount() {
    return this.pages.size;
  }
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
    /** Updates a landed page's runs are not trimmed (0: CT's; the worker's store 2). */
    public landGrace = 0,
    /** Updates a run marked for a cut stays (1: this update's, CT's). */
    public keepCut = 1,
    /**
     * Runs kept inflated (thread CU): a page's as it lands (inflated then
     * anyway for the positions), a run fetched again as it comes; the
     * budget counts their inflated bytes (deflate saved only ~25 %, at
     * ~31 us of fflate a run to pack it).
     */
    public raw = false,
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
      if (this.raw && planes) {
        // Kept inflated (thread CU): no inflate to pack its records.
        bytes.push(planes[r]);
        this.bytes += planes[r].byteLength;
        continue;
      }
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
      grace: this.frame + this.landGrace,
      stamp: new Uint32Array(head.runs).fill(this.frame),
    });
    if (planes && !this.raw) {
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
    // Candidates as sort keys (one typed sort, no tuples): class other
    // first, then culled, the longest unmarked first; the low 26 bits a
    // candidate's index into `who`.
    let count = 0;
    for (const p of this.pages.values()) count += p.bytes.length;
    const keys = new Float64Array(count);
    const who: [number, RunPage, number][] = [];
    let k = 0;
    for (const [page, p] of this.pages) {
      if (this.frame < p.grace) continue;
      for (let r = 0; r < p.bytes.length; r++) {
        if (!p.bytes[r]) continue;
        const cls = p.stamp[r] === this.frame ? p.cls[r] : RUN_OTHER;
        if (cls === RUN_CUT) continue;
        // A cut's (or the view ahead's) runs of the last updates stay too.
        if (p.cls[r] === RUN_CUT && this.frame - p.stamp[r] < this.keepCut) {
          continue;
        }
        const age = Math.min(p.stamp[r], 0xffffff);
        keys[k] = ((2 - cls) * 0x1000000 + age) * 0x4000000 + who.length;
        who.push([page, p, r]);
        k += 1;
      }
    }
    const order = keys.subarray(0, k).sort();
    let dropped = 0;
    for (let q = 0; q < k && this.bytes > this.budget; q++) {
      const [, p, r] = who[order[q] % 0x4000000];
      this.bytes -= (p.bytes[r] as Uint8Array).byteLength;
      p.bytes[r] = null;
      dropped += 1;
    }
    this.stats.dropped += dropped;
    return dropped;
  }

  /** A page's runs: its nodes and nodes a run (undefined: not held). */
  runInfo(page: number): { n: number; run: number } | undefined {
    return this.pages.get(page);
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
    if (this.raw) return b;
    const t0 = performance.now();
    const p = inflateSync(b);
    this.stats.inflateMs += performance.now() - t0;
    this.stats.inflates += 1;
    this.remember(k, p);
    return p;
  }

  /** Run r's deflate bytes (null: dropped), used now (the last to go). */
  runBytes(page: number, r: number): Uint8Array | null {
    const p = this.pages.get(page);
    const b = p?.bytes[r];
    if (!p || !b) return null;
    if (p.stamp[r] !== this.frame) {
      p.stamp[r] = this.frame;
      p.cls[r] = RUN_OTHER;
    }
    return b;
  }

  /** Run r's inflated planes if in the LRU (no inflate, no recency change). */
  inflatedPlanes(page: number, r: number): Uint8Array | null {
    return this.inflated.get(this.key(page, r)) ?? null;
  }

  /** Where run r is in the file, to fetch it again: [offset, bytes]. */
  fileSpan(page: number, r: number): [number, number] | null {
    const p = this.pages.get(page);
    if (!p) return null;
    const [a, b] = runSpan(p.head, r);
    return [p.fileAt + a, b - a];
  }

  /** A dropped run fetched again. */
  putRun(page: number, r: number, stored: Uint8Array) {
    const p = this.pages.get(page);
    if (!p || p.bytes[r]) return;
    let bytes = stored;
    if (this.raw) {
      const t0 = performance.now();
      bytes = inflateSync(stored);
      this.stats.inflateMs += performance.now() - t0;
      this.stats.inflates += 1;
    }
    p.bytes[r] = bytes;
    this.bytes += bytes.byteLength;
    this.stats.fetched += 1;
    // Fetched for a use: the most recent (the last to go over budget).
    p.stamp[r] = this.frame;
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

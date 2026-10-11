// The node pool's CPU side in a worker (thread CU, round 2): everything an
// update did on the main thread for CMPC v4 pieces -- the cut's nodes to
// slots, the runs' payload (compressed, by class, under a budget), their
// inflate and the records' packing, the draw list with its substitutes,
// the dropped runs fetched again by Range, the landed pages' traversal
// input -- so the main thread only writes records and sets draw lists.
//
// `NodeCutter` is the logic (no worker API: the unit tests drive it); the
// module's tail is the worker's glue.

import { gunzipSync } from "fflate";
import {
  NodeSlotAllocator,
  PAGE_PLACES,
  RUN_CULLED,
  RUN_CUT,
  RunPayloadStore,
  parseRunsContainer,
  unpackRunsContainer,
} from "./compactNodes";
import { NODE_RECORD_WORDS, packRunRecords } from "./compactRecords";

export interface CutterInit {
  t: "init";
  capacity: number;
  maxCapacity: number;
  /** Nodes a pool slot (the traversal's page slot). */
  slotNodes: number;
  runBudget: number;
  inflatedRuns: number;
  /** Keep the cut's parents' and children's runs whatever the budget. */
  protect: boolean;
  priority: boolean;
  /** Each head-table page's piece, and each piece's URL (absolute; "" none: the main thread fetches). */
  pagePiece: Uint16Array;
  urls: string[];
  runFetchers: number;
}

export interface CutPart {
  numSplats: number;
  indices: Uint32Array;
  parents?: Uint32Array;
  firstKids?: Uint32Array;
  kidCounts?: Uint32Array;
  culled?: Uint32Array;
  screens?: Uint32Array;
}

export interface CutMsg {
  t: "cut";
  /** Each pool slot's key base (a node key is its pool index plus it). */
  kb: Int32Array;
  parts: CutPart[];
}

export interface AheadMsg {
  t: "ahead";
  kb: Int32Array;
  parts: { numSplats: number; indices: Uint32Array }[];
}

export type ToCutter =
  | CutterInit
  | CutMsg
  | AheadMsg
  | { t: "page"; g: number; container: Uint8Array; fileAt: number }
  | { t: "drop"; g: number }
  | {
      t: "runs";
      /** A Range's bytes (from file offset `at`) and its runs; none: it failed. */
      data?: Uint8Array;
      at?: number;
      spans?: [g: number, r: number, offset: number, bytes: number][];
      /** A failed Range's runs (to ask again). */
      asked?: number[];
    }
  | { t: "recycle"; buffer: ArrayBuffer }
  | { t: "keys"; id: number };

export interface CutterStats {
  cut: number;
  missing: number;
  missDropped: number;
  missDeferred: number;
  substituted: number;
  holes: number;
  missingArea: number;
  holeArea: number;
  uploaded: number;
  decodes: number;
  decodeMs: number;
  sliceMs: number;
  /** The last cut's own work (before its first slice), ms; slices since it. */
  cutMs: number;
  slices: number;
  /** All ms in cuts and slices, and the slices, since the start. */
  busyMs: number;
  sliceCount: number;
  noSlot: number;
  capacity: number;
  nodesHeld: number;
  indexBytes: number;
  runBytesHeld: number;
  inflatedBytes: number;
  runRanges: number;
  runBytesRead: number;
  prefetchRuns: number;
  /** Mean ms of a run Range, and Ranges in flight now. */
  fetchMs: number;
  fetches: number;
  waitingRuns: number;
  busy: boolean;
}

export type FromCutter =
  | {
      t: "frame";
      /** Resize the GPU pool to exactly this many records first (0: no). */
      grow: number;
      /** With a shrink: records to move (from, to slot pairs). */
      moves?: Uint32Array | null;
      /** Records to write: slots ascending, records in the same order. */
      slots: Uint32Array;
      records: Uint8Array;
      /** The draw list now (null: unchanged). */
      drawList: Uint32Array | null;
      stats: CutterStats;
    }
  | { t: "landed"; g: number; input: Uint8Array }
  | {
      t: "want";
      /**
       * Ranges of a piece's file, each its runs (page, run, offset, bytes)
       * by offset; one message for all a fetchRuns asks (thread CU).
       */
      ranges: [g: number, r: number, offset: number, bytes: number][][];
    }
  | { t: "keys"; id: number; keys: Int32Array };

const SH = 8388608;

/**
 * `f` in a new task without setTimeout's 4 ms clamp on nested timers (a
 * MessageChannel's message; setTimeout where there is none).
 */
const laterTask: (f: () => void) => void = (() => {
  if (typeof MessageChannel === "undefined")
    return (f: () => void) => setTimeout(f, 0);
  const ch = new MessageChannel();
  const q: (() => void)[] = [];
  ch.port1.onmessage = () => (q.shift() as () => void)();
  return (f: () => void) => {
    q.push(f);
    ch.port2.postMessage(0);
  };
})();

/** The node pool's CPU side (the worker's logic, driven by messages). */
export class NodeCutter {
  readonly alloc: NodeSlotAllocator;
  readonly runs: RunPayloadStore;
  private o: CutterInit;
  private gen = 0;
  /** The current cut. */
  private keys = new Int32Array(0);
  private out = new Int32Array(0);
  private failed = new Uint8Array(0);
  private failedList: Int32Array<ArrayBufferLike> = new Int32Array(0);
  private base: Uint32Array<ArrayBufferLike> = new Uint32Array(0);
  private near: {
    parents: Uint32Array;
    firstKids: Uint32Array;
    kidCounts: Uint32Array;
    screens?: Float32Array;
  } | null = null;
  private kb: Int32Array<ArrayBufferLike> = new Int32Array(0);
  private allArea = 0;
  /** Groups to pack, largest first: [start, end) in `miss`, run, page. */
  /**
   * The cut's groups as typed arrays (thread CU: no object a group): the
   * misses [gA, gB) of `miss`, their page and run, their screen weight.
   */
  private gA = new Int32Array(0);
  private gB = new Int32Array(0);
  private gG = new Int32Array(0);
  private gR = new Int32Array(0);
  private gW = new Float64Array(0);
  /** Groups to pack, by id, in order (appended to as runs come). */
  private queue = new Int32Array(64);
  private queueLen = 0;
  private queueAt = 0;
  /** Groups waiting for a dropped run: run key (page x 64 + r) -> groups. */
  private waiting = new Map<number, number[]>();
  private miss = new Float64Array(0);
  private grow = 0;
  /** Moves of a shrink, sent with the next frame's resize. */
  private moves: Uint32Array | null = null;
  /** The last cuts' sizes. */
  private recent: number[] = [];
  /** A view that stays: the pool down to its cut (thread CU). */
  private idleTimer: ReturnType<typeof setTimeout> | null = null;
  private asked = new Set<number>();
  private fetches = 0;
  private stats: CutterStats = {
    cut: 0,
    missing: 0,
    missDropped: 0,
    missDeferred: 0,
    substituted: 0,
    holes: 0,
    missingArea: 0,
    holeArea: 0,
    uploaded: 0,
    decodes: 0,
    decodeMs: 0,
    sliceMs: 0,
    cutMs: 0,
    slices: 0,
    busyMs: 0,
    sliceCount: 0,
    noSlot: 0,
    capacity: 0,
    nodesHeld: 0,
    indexBytes: 0,
    runBytesHeld: 0,
    inflatedBytes: 0,
    runRanges: 0,
    runBytesRead: 0,
    prefetchRuns: 0,
    fetchMs: 0,
    fetches: 0,
    waitingRuns: 0,
    busy: false,
  };
  /** A frame's mark a slot (an epoch byte, cleared as it wraps). */
  private seen = new Uint8Array(0);
  private seenEpoch = 0;
  private kidScratch = new Int32Array(64);

  constructor(
    init: CutterInit,
    /** Posts a message to the main thread (transfers its buffers). */
    readonly post: (m: FromCutter) => void,
    /** Runs `f` soon (a new task: messages may come first). */
    readonly later: (f: () => void) => void = laterTask,
    /** A Range of piece p's file (default fetch of its URL). */
    readonly read?: (
      piece: number,
      offset: number,
      bytes: number,
    ) => Promise<Uint8Array>,
  ) {
    this.o = init;
    this.alloc = new NodeSlotAllocator(init.capacity, init.slotNodes);
    this.runs = new RunPayloadStore(init.runBudget, 0, 3, 3, true);
  }

  handle(m: ToCutter) {
    switch (m.t) {
      case "page":
        this.land(m.g, m.container, m.fileAt);
        break;
      case "drop":
        this.runs.drop(m.g);
        this.alloc.releasePage(m.g);
        break;
      case "recycle":
        if (this.spare.length < 1) this.spare.push(m.buffer);
        break;
      case "runs":
        if (m.data && m.spans) {
          const at = m.at ?? 0;
          for (const [g, r, off, len] of m.spans) {
            this.gotRun(g, r, m.data.subarray(off - at, off - at + len));
          }
        }
        // A want's reply (its Range done, or failed: its runs asked again).
        if (m.asked) for (const rk of m.asked) this.asked.delete(rk);
        this.fetches = Math.max(0, this.fetches - 1);
        this.fetchRuns();
        this.wake();
        break;
      case "cut":
        this.cut(m);
        break;
      case "ahead":
        this.ahead(m);
        break;
      case "keys": {
        const keys = new Int32Array(this.alloc.capacity);
        for (let s = 0; s < keys.length; s++) keys[s] = this.alloc.keyIn(s);
        this.post({ t: "keys", id: m.id, keys });
        break;
      }
    }
  }

  /** A page landed: its runs kept, the traversal's input back. */
  private land(g: number, c: Uint8Array, fileAt: number) {
    const h = parseRunsContainer(c);
    const trav = gunzipSync(c.subarray(h.travAt, h.travAt + h.travBytes));
    const u = unpackRunsContainer(c, h, trav, false);
    this.runs.put(g, c, fileAt, u.planes);
    this.post({ t: "landed", g, input: u.input });
  }

  // ------------------------------------------------------------- the cut

  private cut(m: CutMsg) {
    const t0 = performance.now();
    this.gen += 1;
    const alloc = this.alloc;
    const runs = this.runs;
    const N = this.o.slotNodes;
    const kb = m.kb;
    let n = 0;
    for (const p of m.parts) n += p.numSplats;
    // The pool after a large cut (thread CU): its size the largest of the
    // last cuts and 6 %, once that frees over a megabyte; the nodes in
    // use past it move down (the GPU copies them with the resize).
    this.recent.push(n);
    if (this.recent.length > 8) this.recent.shift();
    const most = Math.max(...this.recent);
    const target = Math.max(1024, Math.ceil(most * 1.06));
    if (
      this.recent.length >= 4 &&
      alloc.capacity > most * 1.15 &&
      (alloc.capacity - target) * NODE_RECORD_WORDS * 4 > 1 << 20
    ) {
      const moves = alloc.shrink(target);
      if (moves) {
        this.grow = target;
        this.moves = moves;
      }
    }
    if (this.idleTimer) clearTimeout(this.idleTimer);
    this.idleTimer = setTimeout(() => this.idleShrink(n), 1500);
    if (n > alloc.capacity && alloc.capacity < this.o.maxCapacity) {
      const next = Math.min(
        Math.ceil(Math.max(n * 1.06, alloc.capacity * 1.1)),
        this.o.maxCapacity,
      );
      alloc.grow(next);
      this.grow = next;
    }
    alloc.begin();
    // Thread CU: only the misses' data is kept (indexed by miss), the held
    // nodes' slots go straight into the drawn base; the parts (transferred)
    // are let go with the message.
    const base = new Uint32Array(n);
    let b0 = 0;
    const missAt = new Int32Array(n);
    const missKeyAll = new Int32Array(n);
    let missN = 0;
    let allArea = 0;
    runs.begin();
    const protect = this.o.protect;
    const markI = (i: number) => {
      if (i === 0xffffffff) return;
      const key = i + kb[(i / N) | 0];
      if (key >= 0) runs.mark(key >>> 12, key & 4095, RUN_CUT);
    };
    m.parts.forEach((p, pi) => {
      const c = p.numSplats;
      const idx = p.indices;
      const sc = p.screens
        ? new Float32Array(p.screens.buffer, p.screens.byteOffset, c)
        : null;
      for (let j = 0; j < c; j++) {
        const i = idx[j];
        const key = i + kb[(i / N) | 0];
        if (sc) allArea += sc[j] * sc[j];
        const s = alloc.use(key);
        if (s < 0) {
          missAt[missN] = pi * SH + j;
          missKeyAll[missN++] = key;
        } else base[b0++] = s;
        if (protect) {
          runs.mark(key >>> 12, key & 4095, RUN_CUT);
          if (p.parents) markI(p.parents[j]);
          const f = p.firstKids?.[j] ?? 0xffffffff;
          if (f !== 0xffffffff) {
            for (let k = 0; k < (p.kidCounts?.[j] ?? 0); k++) markI(f + k);
          }
        }
      }
      // The culled nodes' runs next to keep (the rest go first).
      for (const i of p.culled ?? []) {
        const key = i + kb[(i / N) | 0];
        if (key >= 0) runs.mark(key >>> 12, key & 4095, RUN_CULLED);
      }
    });
    const hasScreens = m.parts.every((p) => p.screens || !p.numSplats);
    const keys = missKeyAll.slice(0, missN);
    const out = new Int32Array(missN).fill(-1);
    const failed = new Uint8Array(missN).fill(1);
    const parents = new Uint32Array(missN);
    const firstKids = new Uint32Array(missN);
    const kidCounts = new Uint32Array(missN);
    const screens = hasScreens ? new Float32Array(missN) : undefined;
    const missKeys = new Float64Array(missN);
    const scr = m.parts.map((p) =>
      p.screens
        ? new Float32Array(p.screens.buffer, p.screens.byteOffset, p.numSplats)
        : null,
    );
    for (let x = 0; x < missN; x++) {
      const pi = (missAt[x] / SH) | 0;
      const j = missAt[x] - pi * SH;
      const p = m.parts[pi];
      parents[x] = p.parents ? p.parents[j] : 0xffffffff;
      firstKids[x] = p.firstKids ? p.firstKids[j] : 0xffffffff;
      kidCounts[x] = p.kidCounts ? p.kidCounts[j] : 0;
      if (screens) screens[x] = (scr[pi] as Float32Array)[j];
      missKeys[x] = keys[x] * SH + x;
    }
    const near = { parents, firstKids, kidCounts, screens };
    const miss = missKeys.subarray(0, missN).sort();
    // The groups (a run's misses), largest on screen first.
    const gA = new Int32Array(missN);
    const gB = new Int32Array(missN);
    const gG = new Int32Array(missN);
    const gR = new Int32Array(missN);
    const gW = new Float64Array(missN);
    let groups = 0;
    let q = 0;
    while (q < missN) {
      const key0 = keys[miss[q] % SH];
      const g = key0 >>> 12;
      const info = runs.runInfo(g);
      const r = info ? ((key0 & 4095) / info.run) | 0 : -1;
      let e = q + 1;
      while (e < missN) {
        const key = keys[miss[e] % SH];
        if (key >>> 12 !== g) break;
        if (info && (((key & 4095) / info.run) | 0) !== r) break;
        e++;
      }
      let w = 0;
      if (screens) {
        for (let y = q; y < e; y++) {
          const sc = screens[miss[y] % SH];
          w += sc * sc;
        }
      }
      gA[groups] = q;
      gB[groups] = e;
      gG[groups] = g;
      gR[groups] = r;
      gW[groups] = w;
      groups += 1;
      q = e;
    }
    const order = new Int32Array(groups);
    for (let x = 0; x < groups; x++) order[x] = x;
    if (screens && this.o.priority) order.sort((x, y) => gW[y] - gW[x]);
    this.gA = gA;
    this.gB = gB;
    this.gG = gG;
    this.gR = gR;
    this.gW = gW;
    this.queue = new Int32Array(Math.max(64, groups));
    this.queueLen = 0;
    this.queueAt = 0;
    this.waiting = new Map();
    let dropped = 0;
    for (let o = 0; o < groups; o++) {
      const x = order[o];
      const g = gG[x];
      const r = gR[x];
      if (r < 0) continue; // its page left: it comes back with it
      runs.mark(g, r * (runs.runInfo(g) as { run: number }).run, RUN_CUT);
      if (runs.missing(g, r)) {
        const rk = g * 64 + r;
        const list = this.waiting.get(rk) ?? [];
        list.push(x);
        this.waiting.set(rk, list);
        this.want(g, r, false, gW[x]);
        dropped += gB[x] - gA[x];
      } else this.enqueue(x);
    }
    this.keys = keys;
    this.out = out;
    this.failed = failed;
    this.miss = miss;
    this.near = near;
    this.kb = kb;
    this.allArea = allArea;
    const fl = new Int32Array(missN);
    for (let x = 0; x < missN; x++) fl[x] = x;
    this.failedList = fl;
    this.base = base.slice(0, b0);
    this.stats.missDropped = dropped;
    this.stats.cut = n;
    this.fetchRuns();
    this.stats.cutMs = performance.now() - t0;
    this.stats.busyMs += this.stats.cutMs;
    this.stats.slices = 0;
    // The first slice at once (a short one: the list goes up soon), the
    // rest in later tasks.
    this.slice(performance.now() - t0 < 4 ? 4 : 1, true);
  }

  /** The current cut's queued groups packed for about `ms`, then a frame posted. */
  private slice(ms: number, first = false) {
    const t0 = performance.now();
    const gen = this.gen;
    const alloc = this.alloc;
    const runs = this.runs;
    const R = NODE_RECORD_WORDS * 4;
    const slots: number[] = [];
    const chunks: { rec: Uint8Array; slots: number[] }[] = [];
    const i0 = runs.stats.inflates;
    const ms0 = runs.stats.inflateMs;
    let noSlot = 0;
    while (this.queueAt < this.queueLen) {
      if (performance.now() - t0 > ms && slots.length) break;
      const x0 = this.queue[this.queueAt++];
      const grG = this.gG[x0];
      const grR = this.gR[x0];
      const info = runs.runInfo(grG);
      if (!info) continue;
      const planes = runs.planes(grG, grR);
      if (!planes) {
        // Dropped meanwhile: wait for it again.
        const rk = grG * 64 + grR;
        const list = this.waiting.get(rk) ?? [];
        list.push(x0);
        this.waiting.set(rk, list);
        this.want(grG, grR, false, this.gW[x0]);
        continue;
      }
      const base = grR * info.run;
      const m = Math.min(info.run, info.n - base);
      const locals: number[] = [];
      const gs: number[] = [];
      for (let y = this.gA[x0]; y < this.gB[x0]; y++) {
        const k = this.miss[y] % SH;
        if (!this.failed[k]) continue;
        const key = this.keys[k];
        let s = alloc.use(key);
        if (s < 0) s = alloc.allocate(key);
        if (s < 0) {
          noSlot += 1;
          continue;
        }
        this.out[k] = s;
        this.failed[k] = 0;
        locals.push((key & 4095) - base);
        gs.push(s);
      }
      if (!locals.length) continue;
      const rec = new Uint8Array(locals.length * R);
      packRunRecords(planes, m, Uint32Array.from(locals), grG, base, rec, 0);
      chunks.push({ rec, slots: gs });
      for (const s of gs) slots.push(s);
    }
    // Records in slot order.
    const order = new Float64Array(slots.length);
    let x = 0;
    const where: [number, number][] = [];
    chunks.forEach((c, ci) => {
      for (let j = 0; j < c.slots.length; j++) {
        order[x] = c.slots[j] * SH + x;
        where.push([ci, j]);
        x++;
      }
    });
    order.sort();
    const outSlots = new Uint32Array(x);
    const records = new Uint8Array(x * R);
    for (let y = 0; y < x; y++) {
      const v = order[y];
      const w = v % SH;
      outSlots[y] = (v - w) / SH;
      const [ci, j] = where[w];
      records.set(chunks[ci].rec.subarray(j * R, j * R + R), y * R);
    }
    // The list at most every ~30 ms (records go up at once: a later slice
    // never takes a slot the shown list draws).
    const emitList = first || performance.now() - this.lastListAt > 30;
    const drawList = emitList ? this.buildDrawList() : null;
    if (emitList) this.lastListAt = performance.now();
    else if (x) this.listLater();
    if (first) alloc.commit();
    const s = this.stats;
    s.uploaded = x;
    s.noSlot = noSlot;
    s.decodes = runs.stats.inflates - i0;
    s.decodeMs = runs.stats.inflateMs - ms0;
    s.missDeferred = 0;
    for (let y = this.queueAt; y < this.queueLen; y++) {
      s.missDeferred += this.gB[this.queue[y]] - this.gA[this.queue[y]];
    }
    if (first) runs.trim();
    this.fillStats();
    s.sliceMs = performance.now() - t0;
    s.slices += 1;
    s.sliceCount += 1;
    s.busyMs += s.sliceMs;
    const grow = this.grow;
    this.grow = 0;
    const moves = this.moves;
    this.moves = null;
    // Nothing to write nor show: no message (each is a task on the main thread).
    if (x || drawList || grow) {
      this.post({
        t: "frame",
        grow,
        moves,
        slots: outSlots,
        records,
        drawList,
        stats: { ...s },
      });
    }
    void gen;
    this.schedule();
  }

  /**
   * No new cut for a while: the pool shrinks to this cut and 6 % (the
   * largest of the last cuts held it up while the view moved).
   */
  private idleShrink(n: number) {
    this.idleTimer = null;
    const alloc = this.alloc;
    const target = Math.max(1024, Math.ceil(n * 1.06));
    if (this.queueAt < this.queueLen || this.scheduled) {
      this.idleTimer = setTimeout(() => this.idleShrink(n), 500);
      return;
    }
    if (
      this.grow ||
      this.moves ||
      alloc.capacity <= n * 1.1 ||
      (alloc.capacity - target) * NODE_RECORD_WORDS * 4 <= 1 << 20
    )
      return;
    const moves = alloc.shrink(target);
    if (!moves) return;
    this.recent = [n];
    // The held cut's slots follow their moves.
    if (moves.length) {
      const to = new Map<number, number>();
      for (let k = 0; k < moves.length; k += 2) to.set(moves[k], moves[k + 1]);
      for (const a of [this.base, this.out]) {
        for (let i = 0; i < a.length; i++) {
          const t = to.get(a[i]);
          if (t !== undefined) a[i] = t;
        }
      }
    }
    // The list names the moved slots: a new one with the resize.
    const drawList = this.buildDrawList();
    this.lastListAt = performance.now();
    this.fillStats();
    this.post({
      t: "frame",
      grow: target,
      moves,
      slots: new Uint32Array(0),
      records: new Uint8Array(0),
      drawList,
      stats: { ...this.stats },
    });
  }

  /** Group x to pack (the queue grows as runs come). */
  private enqueue(x: number) {
    if (this.queueLen === this.queue.length) {
      const q = new Int32Array(this.queue.length * 2);
      q.set(this.queue);
      this.queue = q;
    }
    this.queue[this.queueLen++] = x;
  }

  private scheduled = false;
  private lastListAt = 0;
  private listTimer = false;

  /** The list (with the records written since) in at most ~30 ms. */
  private listLater() {
    if (this.listTimer) return;
    this.listTimer = true;
    setTimeout(
      () => {
        this.listTimer = false;
        if (performance.now() - this.lastListAt < 25) return;
        const drawList = this.buildDrawList();
        this.lastListAt = performance.now();
        this.fillStats();
        this.post({
          t: "frame",
          grow: 0,
          moves: null,
          slots: new Uint32Array(0),
          records: new Uint8Array(0),
          drawList,
          stats: { ...this.stats },
        });
      },
      Math.max(0, 30 - (performance.now() - this.lastListAt)),
    );
  }

  /** One slice later (one at a time: fetches' replies get their turns). */
  private schedule() {
    if (this.scheduled || this.queueAt >= this.queueLen) return;
    this.scheduled = true;
    this.later(() => {
      this.scheduled = false;
      if (this.queueAt < this.queueLen) this.slice(8);
    });
  }

  private fillStats() {
    const s = this.stats;
    s.capacity = this.alloc.capacity;
    s.nodesHeld = this.alloc.used;
    s.indexBytes = this.alloc.bytes;
    s.runBytesHeld = this.runs.bytes;
    s.inflatedBytes = this.runs.inflatedBytes;
    s.fetches = this.fetches;
    s.waitingRuns = this.waiting.size;
    s.busy =
      this.queueAt < this.queueLen ||
      this.waiting.size > 0 ||
      this.fetches > 0 ||
      this.wantNow.size > 0 ||
      this.wantAhead.size > 0;
  }

  /** Packing resumed (a dropped run came): in at most ~8 ms, with the others. */
  private wake() {
    if (this.wakeTimer) return;
    this.wakeTimer = true;
    setTimeout(() => {
      this.wakeTimer = false;
      this.schedule();
    }, 8);
  }

  private wakeTimer = false;

  /**
   * The draw list of the current cut: its nodes held, the misses as their
   * parent or children when held, else left out (with the missing stats).
   */
  private buildDrawList() {
    const alloc = this.alloc;
    const N = this.o.slotNodes;
    const near = this.near;
    const kb = this.kb;
    const screens = near?.screens;
    const fl = this.failedList;
    let resolved = 0;
    for (let x = 0; x < fl.length; x++) if (!this.failed[fl[x]]) resolved++;
    if (resolved) {
      const base = new Uint32Array(this.base.length + resolved);
      base.set(this.base);
      let b0 = this.base.length;
      let f0 = 0;
      for (let x = 0; x < fl.length; x++) {
        const k = fl[x];
        if (!this.failed[k]) base[b0++] = this.out[k];
        else fl[f0++] = k;
      }
      this.base = base;
      this.failedList = fl.subarray(0, f0);
    }
    const misses = this.failedList;
    let substituted = 0;
    let holes = 0;
    let holeArea = 0;
    let missedArea = 0;
    let result: Uint32Array;
    if (misses.length && near) {
      if (this.seen.length < alloc.capacity)
        this.seen = new Uint8Array(alloc.capacity);
      const seen = this.seen;
      if (++this.seenEpoch > 255) {
        seen.fill(0);
        this.seenEpoch = 1;
      }
      const epoch = this.seenEpoch;
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
      const held = (i: number) => alloc.use(i + kb[(i / N) | 0]);
      for (let x = 0; x < misses.length; x++) {
        const k = misses[x];
        const a = screens ? screens[k] * screens[k] : 0;
        missedArea += a;
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
      result = this.listBuffer(this.base.length + sn);
      result.set(this.base);
      result.set(subs.subarray(0, sn), this.base.length);
    } else {
      result = this.listBuffer(this.base.length);
      result.set(this.base);
      holes = misses.length;
    }
    const s = this.stats;
    s.missingArea = this.allArea > 0 ? missedArea / this.allArea : 0;
    s.holeArea = this.allArea > 0 ? holeArea / this.allArea : 0;
    s.holes = holes;
    s.missing = misses.length;
    s.substituted = substituted;
    return result;
  }

  /** Draw lists the main thread gave back, for the next ones. */
  private spare: ArrayBuffer[] = [];

  /** A list of n words, in a recycled buffer when one is big enough. */
  private listBuffer(n: number) {
    const i = this.spare.findIndex((b) => b.byteLength >= 4 * n);
    if (i >= 0) {
      const b = this.spare.splice(i, 1)[0];
      return new Uint32Array(b, 0, n);
    }
    // Room to grow: the next lists of the cut fit it.
    return new Uint32Array(
      new ArrayBuffer(4 * Math.ceil(n * 1.1 + 1024)),
      0,
      n,
    );
  }

  // ---------------------------------------------------- the view ahead

  private ahead(m: AheadMsg) {
    const runs = this.runs;
    const N = this.o.slotNodes;
    let wanted = 0;
    let lastRun = -1;
    for (const p of m.parts) {
      for (let j = 0; j < p.numSplats; j++) {
        const i = p.indices[j];
        const key = i + m.kb[(i / N) | 0];
        if (key < 0 || this.alloc.slotOf.slot(key) >= 0) continue;
        const g = key >>> 12;
        const info = runs.runInfo(g);
        if (!info) continue;
        const r = ((key & 4095) / info.run) | 0;
        const rk = g * 64 + r;
        if (rk === lastRun) continue;
        lastRun = rk;
        runs.mark(g, key & 4095, RUN_CUT);
        if (runs.missing(g, r) && !this.asked.has(rk)) {
          this.want(g, r, true);
          wanted += 1;
        }
      }
    }
    this.stats.prefetchRuns = wanted;
    // Older predictions go first when too many wait.
    for (const rk of this.wantAhead) {
      if (this.wantAhead.size <= 1024) break;
      this.wantAhead.delete(rk);
    }
    this.fetchRuns();
  }

  // ------------------------------------------------ dropped runs again

  /**
   * Run (page g, run r) to fetch: the cut's first (`now`), the view
   * ahead's after it.
   */
  private want(g: number, r: number, ahead = false, w = 0) {
    const rk = g * 64 + r;
    if (this.asked.has(rk)) return;
    if (ahead) {
      if (!this.wantNow.has(rk)) this.wantAhead.add(rk);
    } else {
      this.wantAhead.delete(rk);
      this.wantNow.set(rk, (this.wantNow.get(rk) ?? 0) + w + 1e-12);
    }
  }

  /** The cut's runs to fetch, by their misses' area on screen; the view ahead's. */
  private wantNow = new Map<number, number>();
  private wantAhead = new Set<number>();
  /** Bytes a Range reads over to join the next wanted run (fewer requests). */
  gap = 4096;

  private gotRun(g: number, r: number, bytes: Uint8Array) {
    const rk = g * 64 + r;
    this.asked.delete(rk);
    this.runs.putRun(g, r, bytes);
    const list = this.waiting.get(rk);
    if (list) {
      this.waiting.delete(rk);
      for (const x of list) this.enqueue(x);
    }
  }

  /**
   * The wanted runs in Ranges, `runFetchers` in flight: up to 512 runs at a
   * time (the cut's first), sorted by file offset, joined across pages
   * while the gap is under 4 KB and a Range under 1 MB.
   */
  private outWants: [number, number, number, number][][] = [];

  private fetchRuns() {
    this.fetchRunsInner();
    // One message for the Ranges of the next few ms (each is a task on the
    // main thread; a reply a Range freed one at a time).
    if (this.outWants.length && !this.wantTimer) {
      this.wantTimer = true;
      setTimeout(() => {
        this.wantTimer = false;
        this.fetchRunsInner();
        if (this.outWants.length) {
          this.post({ t: "want", ranges: this.outWants });
          this.outWants = [];
        }
      }, 4);
    }
  }

  private wantTimer = false;

  private fetchRunsInner() {
    const runs = this.runs;
    for (const set of [this.wantNow, this.wantAhead] as (
      | Map<number, number>
      | Set<number>
    )[]) {
      if (this.fetches >= this.o.runFetchers) return;
      if (!set.size) continue;
      const weight = set instanceof Map ? set : null;
      // [piece, offset, bytes, run key], by piece and offset.
      const spans: [number, number, number, number][] = [];
      for (const rk of set.keys()) {
        const g = Math.floor(rk / 64);
        const r = rk - g * 64;
        if (!runs.missing(g, r) || this.asked.has(rk)) {
          set.delete(rk);
          continue;
        }
        const sp = runs.fileSpan(g, r) as [number, number];
        spans.push([this.o.pagePiece[g], sp[0], sp[1], rk]);
      }
      spans.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
      // A run joins the Range before it while that at most doubles its
      // bytes, or the gap is under `gap`, under 1 MB.
      const groups: (typeof spans)[] = [];
      let need = 0;
      for (const sp of spans) {
        const last = groups[groups.length - 1];
        if (last) {
          const a = last[0];
          const b = last[last.length - 1];
          const total = sp[1] + sp[2] - a[1];
          if (
            a[0] === sp[0] &&
            total <= 1 << 20 &&
            (sp[1] - (b[1] + b[2]) <= this.gap || total <= 2 * (need + sp[2]))
          ) {
            last.push(sp);
            need += sp[2];
            continue;
          }
        }
        groups.push([sp]);
        need = sp[2];
      }
      // The cut's Ranges largest on screen first, the view ahead's with the
      // most runs first, within the ones in flight; the rest wait (and
      // grow) for the next free one.
      const score = (grp: typeof spans) =>
        weight
          ? grp.reduce((n, sp) => n + (weight.get(sp[3]) ?? 0), 0)
          : grp.length;
      const scored = groups.map((grp) => [score(grp), grp] as const);
      scored.sort((x, y) => y[0] - x[0]);
      groups.length = 0;
      for (const [, grp] of scored) groups.push(grp);
      for (const grp of groups) {
        if (this.fetches >= this.o.runFetchers) return;
        for (const sp of grp) set.delete(sp[3]);
        this.fetchGroup(grp);
      }
    }
  }

  private fetchGroup(grp: (readonly [number, number, number, number])[]) {
    const runs = this.runs;
    const piece = grp[0][0];
    const offset = grp[0][1];
    const lastSp = grp[grp.length - 1];
    const bytes = lastSp[1] + lastSp[2] - offset;
    for (const sp of grp) this.asked.add(sp[3]);
    this.stats.runRanges += 1;
    this.stats.runBytesRead += bytes;
    const url = this.o.urls[piece];
    if (!url && !this.read) {
      // The main thread fetches the group's one Range, posted back as "runs"
      // (one reply a want: in flight until then).
      this.fetches += 1;
      this.outWants.push(
        grp.map((sp) => {
          const g = Math.floor(sp[3] / 64);
          return [g, sp[3] - g * 64, sp[1], sp[2]] as [
            number,
            number,
            number,
            number,
          ];
        }),
      );
      return;
    }
    this.fetches += 1;
    const tf = performance.now();
    const read = this.read
      ? this.read(piece, offset, bytes)
      : fetch(url, {
          headers: { Range: `bytes=${offset}-${offset + bytes - 1}` },
        }).then(async (res) => {
          if (!res.ok) throw new Error(`${url}: ${res.status}`);
          return new Uint8Array(await res.arrayBuffer());
        });
    read
      .then((data) => {
        this.stats.fetchMs +=
          0.1 * (performance.now() - tf - this.stats.fetchMs);
        for (const sp of grp) {
          const g = Math.floor(sp[3] / 64);
          const r = sp[3] - g * 64;
          const now = runs.fileSpan(g, r);
          if (!now || now[0] !== sp[1]) {
            this.asked.delete(sp[3]);
            continue;
          }
          this.gotRun(g, r, data.slice(sp[1] - offset, sp[1] - offset + sp[2]));
        }
      })
      .catch((e) => {
        for (const sp of grp) this.asked.delete(sp[3]);
        console.warn(e);
      })
      .finally(() => {
        this.fetches -= 1;
        this.fetchRuns();
        this.wake();
      });
  }
}

/** The run key's page, for the tests. */
export const runPage = (key: number) => Math.floor(key / PAGE_PLACES);

// A Web Worker only (the module also loads on its own, for the checks).
if (typeof self !== "undefined" && typeof window === "undefined") {
  let cutter: NodeCutter | null = null;
  const post = (m: FromCutter) => {
    const t: Transferable[] = [];
    if (m.t === "frame") {
      t.push(m.slots.buffer, m.records.buffer);
      if (m.moves) t.push(m.moves.buffer);
      if (m.drawList) t.push(m.drawList.buffer);
    } else if (m.t === "landed") t.push(m.input.buffer);
    else if (m.t === "keys") t.push(m.keys.buffer);
    (self as unknown as Worker).postMessage(m, t);
  };
  self.onmessage = (e: MessageEvent<ToCutter>) => {
    const m = e.data;
    if (m.t === "init") {
      cutter = new NodeCutter(m, post);
      return;
    }
    cutter?.handle(m);
  };
}

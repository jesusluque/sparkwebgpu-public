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
      items: { g: number; r: number; bytes: Uint8Array }[];
      /** A failed Range's runs (to ask again). */
      asked?: number[];
    }
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
      /** Grow the GPU pool to this many records first (0: no). */
      grow: number;
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
      /** One Range of a piece's file: its runs (page, run, offset, bytes), by offset. */
      spans: [g: number, r: number, offset: number, bytes: number][];
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
  private queue: { a: number; b: number; g: number; r: number; w: number }[] =
    [];
  private queueAt = 0;
  /** Groups waiting for a dropped run: run key (page x 64 + r) -> groups. */
  private waiting = new Map<
    number,
    { a: number; b: number; g: number; r: number; w: number }[]
  >();
  private miss = new Float64Array(0);
  private grow = 0;
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
  private seen = new Uint32Array(0);
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
        break;
      case "runs":
        for (const { g, r, bytes } of m.items) this.gotRun(g, r, bytes);
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
    if (n > alloc.capacity && alloc.capacity < this.o.maxCapacity) {
      const next = Math.min(
        Math.ceil(Math.max(n * 1.1, alloc.capacity * 1.25)),
        this.o.maxCapacity,
      );
      alloc.grow(next);
      this.grow = next;
    }
    alloc.begin();
    const keys = new Int32Array(n);
    const out = new Int32Array(n).fill(-1);
    const failed = new Uint8Array(n);
    const parents = new Uint32Array(n);
    const firstKids = new Uint32Array(n);
    const kidCounts = new Uint32Array(n);
    const hasScreens = m.parts.every((p) => p.screens);
    const screens = hasScreens ? new Float32Array(n) : undefined;
    const missKeys = new Float64Array(n);
    let missN = 0;
    let at = 0;
    let allArea = 0;
    runs.begin();
    for (const p of m.parts) {
      const c = p.numSplats;
      const idx = p.indices;
      if (p.parents) parents.set(p.parents.subarray(0, c), at);
      if (p.firstKids) firstKids.set(p.firstKids.subarray(0, c), at);
      if (p.kidCounts) kidCounts.set(p.kidCounts.subarray(0, c), at);
      if (screens && p.screens) {
        screens.set(
          new Float32Array(p.screens.buffer, p.screens.byteOffset, c),
          at,
        );
      }
      for (let j = 0; j < c; j++) {
        const i = idx[j];
        const k = at + j;
        const key = i + kb[(i / N) | 0];
        keys[k] = key;
        const s = alloc.use(key);
        if (s < 0) missKeys[missN++] = key * SH + k;
        else out[k] = s;
      }
      // The culled nodes' runs next to keep (the rest go first).
      for (const i of p.culled ?? []) {
        const key = i + kb[(i / N) | 0];
        if (key >= 0) runs.mark(key >>> 12, key & 4095, RUN_CULLED);
      }
      at += c;
    }
    if (screens) for (let k = 0; k < n; k++) allArea += screens[k] * screens[k];
    const near = { parents, firstKids, kidCounts, screens };
    if (this.o.protect) {
      const mark = (i: number) => {
        if (i === 0xffffffff) return;
        const key = i + kb[(i / N) | 0];
        if (key >= 0) runs.mark(key >>> 12, key & 4095, RUN_CUT);
      };
      for (let k = 0; k < n; k++) {
        runs.mark(keys[k] >>> 12, keys[k] & 4095, RUN_CUT);
        mark(parents[k]);
        const f = firstKids[k];
        if (f !== 0xffffffff)
          for (let c = 0; c < kidCounts[k]; c++) mark(f + c);
      }
    }
    const miss = missKeys.subarray(0, missN).sort();
    // The groups (a run's misses), largest on screen first.
    const groups: { a: number; b: number; g: number; r: number; w: number }[] =
      [];
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
      groups.push({ a: q, b: e, g, r, w });
      q = e;
    }
    if (screens && this.o.priority) groups.sort((x, y) => y.w - x.w);
    for (let y = 0; y < missN; y++) failed[miss[y] % SH] = 1;
    this.queue = [];
    this.queueAt = 0;
    this.waiting = new Map();
    let dropped = 0;
    for (const gr of groups) {
      if (gr.r < 0) continue; // its page left: it comes back with it
      runs.mark(
        gr.g,
        gr.r * (runs.runInfo(gr.g) as { run: number }).run,
        RUN_CUT,
      );
      if (runs.missing(gr.g, gr.r)) {
        const rk = gr.g * 64 + gr.r;
        const list = this.waiting.get(rk) ?? [];
        list.push(gr);
        this.waiting.set(rk, list);
        this.want(gr.g, gr.r, false, gr.w);
        dropped += gr.b - gr.a;
      } else this.queue.push(gr);
    }
    this.keys = keys;
    this.out = out;
    this.failed = failed;
    this.miss = miss;
    this.near = near;
    this.kb = kb;
    this.allArea = allArea;
    const fl = new Int32Array(missN);
    const base = new Uint32Array(n - missN);
    let f0 = 0;
    let b0 = 0;
    for (let k = 0; k < n; k++) {
      if (failed[k]) fl[f0++] = k;
      else base[b0++] = out[k];
    }
    this.failedList = fl;
    this.base = base;
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
    while (this.queueAt < this.queue.length) {
      if (performance.now() - t0 > ms && slots.length) break;
      const gr = this.queue[this.queueAt++];
      const info = runs.runInfo(gr.g);
      if (!info) continue;
      const planes = runs.planes(gr.g, gr.r);
      if (!planes) {
        // Dropped meanwhile: wait for it again.
        const rk = gr.g * 64 + gr.r;
        const list = this.waiting.get(rk) ?? [];
        list.push(gr);
        this.waiting.set(rk, list);
        this.want(gr.g, gr.r, false, gr.w);
        continue;
      }
      const base = gr.r * info.run;
      const m = Math.min(info.run, info.n - base);
      const locals: number[] = [];
      const gs: number[] = [];
      for (let y = gr.a; y < gr.b; y++) {
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
      packRunRecords(planes, m, Uint32Array.from(locals), gr.g, base, rec, 0);
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
    for (let y = this.queueAt; y < this.queue.length; y++) {
      s.missDeferred += this.queue[y].b - this.queue[y].a;
    }
    if (first) runs.trim();
    this.fillStats();
    s.sliceMs = performance.now() - t0;
    s.slices += 1;
    s.sliceCount += 1;
    s.busyMs += s.sliceMs;
    const grow = this.grow;
    this.grow = 0;
    this.post({
      t: "frame",
      grow,
      slots: outSlots,
      records,
      drawList,
      stats: { ...s },
    });
    void gen;
    this.schedule();
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
    if (this.scheduled || this.queueAt >= this.queue.length) return;
    this.scheduled = true;
    this.later(() => {
      this.scheduled = false;
      if (this.queueAt < this.queue.length) this.slice(8);
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
      this.queueAt < this.queue.length ||
      this.waiting.size > 0 ||
      this.fetches > 0 ||
      this.wantNow.size > 0 ||
      this.wantAhead.size > 0;
  }

  /** Packing resumed (a dropped run came). */
  private wake() {
    this.schedule();
  }

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
        this.seen = new Uint32Array(alloc.capacity);
      const seen = this.seen;
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
      result = new Uint32Array(this.base.length + sn);
      result.set(this.base);
      result.set(subs.subarray(0, sn), this.base.length);
    } else {
      result = this.base.slice();
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
      for (const gr of list) this.queue.push(gr);
    }
  }

  /**
   * The wanted runs in Ranges, `runFetchers` in flight: up to 512 runs at a
   * time (the cut's first), sorted by file offset, joined across pages
   * while the gap is under 4 KB and a Range under 1 MB.
   */
  private fetchRuns() {
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
      this.post({
        t: "want",
        spans: grp.map((sp) => {
          const g = Math.floor(sp[3] / 64);
          return [g, sp[3] - g * 64, sp[1], sp[2]] as [
            number,
            number,
            number,
            number,
          ];
        }),
      });
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

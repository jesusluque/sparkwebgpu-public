// The node pool's CPU parts (thread CT, compactNodes.ts): the slot
// allocator (free list, idle FIFO, the last cut's slots, growth) and the
// pages' payload store (compressed as fetched, an LRU of decompressed).

import { deflateRawSync, gzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import {
  NODE_RECORD_WORDS,
  NodeIndex,
  NodeSlotAllocator,
  PagePayloadStore,
  RUN_CULLED,
  RUN_CUT,
  RunPayloadStore,
  nodeRecordLayout,
  packPageRecords,
  parseRunsContainer,
  unpackRunsContainer,
} from "../../src/webgpu/athenea/compactNodes";
import {
  COMPACT_NODE_BYTES,
  COMPACT_PLANE_WIDTHS,
  compactPlaneOffset,
} from "../../src/webgpu/athenea/compactPages";

/** One frame: keys used or allocated, then committed. */
function frame(a: NodeSlotAllocator, keys: number[]) {
  a.begin();
  const out = keys.map((k) => a.use(k));
  keys.forEach((k, j) => {
    if (out[j] < 0) out[j] = a.allocate(k);
  });
  const cut = Uint32Array.from(out.filter((s) => s >= 0));
  a.commit(cut);
  return out;
}

describe("NodeSlotAllocator", () => {
  it("gives each node one slot and keeps it while it stays in the cut", () => {
    const a = new NodeSlotAllocator(8);
    const f1 = frame(a, [10, 11, 12]);
    expect(new Set(f1).size).toBe(3);
    expect(f1).toEqual([0, 1, 2]); // lowest first: runs of slots
    const f2 = frame(a, [12, 11, 13]);
    expect(f2[0]).toBe(f1[2]);
    expect(f2[1]).toBe(f1[1]);
    expect(a.used).toBe(4); // 10 idle, still held
    expect(a.stats.reused).toBe(2);
  });

  it("reuses an idle node's slot when it comes back (the hysteresis)", () => {
    const a = new NodeSlotAllocator(8);
    const f1 = frame(a, [1, 2]);
    frame(a, [3]); // 1, 2 idle
    frame(a, [4]);
    const f4 = frame(a, [1]);
    expect(f4[0]).toBe(f1[0]);
    expect(a.stats.allocated).toBe(4);
  });

  it("takes idle slots oldest first, never this cut's", () => {
    const a = new NodeSlotAllocator(4);
    frame(a, [1, 2]); // slots 0, 1
    frame(a, [3, 4]); // slots 2, 3; 1, 2 idle
    frame(a, [3, 4]); // nothing changes
    const f = frame(a, [3, 4, 5]); // full: an idle one goes
    expect(f.slice(0, 2)).toEqual([2, 3]);
    expect([0, 1]).toContain(f[2]);
    expect(a.stats.evicted).toBe(1);
    expect(a.used).toBe(4);
    // The evicted key is gone, the other idle one held.
    a.begin();
    const held = [1, 2].map((k) => a.use(k)).filter((s) => s >= 0);
    expect(held.length).toBe(1);
  });

  it("takes the last cut's slots when no idle one is left", () => {
    const a = new NodeSlotAllocator(4);
    frame(a, [1, 2, 3, 4]);
    const f = frame(a, [5, 6, 3]);
    expect(f.every((s) => s >= 0)).toBe(true);
    expect(new Set(f).size).toBe(3);
    expect(a.stats.failed).toBe(0);
    // A cut over the capacity fails the rest.
    const g = frame(a, [7, 8, 9, 10, 11]);
    expect(g.filter((s) => s < 0).length).toBe(1);
  });

  it("grows with the new slots free", () => {
    const a = new NodeSlotAllocator(2);
    frame(a, [1, 2]);
    a.grow(4);
    const f = frame(a, [1, 2, 3, 4]);
    expect(f).toEqual([0, 1, 2, 3]);
    expect(a.stats.failed).toBe(0);
  });

  it("idles every slot a frame used, not only its cut (thread CU)", () => {
    // A substitute (a parent drawn for a missing child) or a record on its
    // way is used by a frame but not in the list it commits: it must go
    // idle once a frame no longer uses it, never stay held for good.
    const a = new NodeSlotAllocator(4);
    a.begin();
    const p = a.allocate(1); // the parent, drawn as a substitute
    const c = a.allocate(2); // the child, its record on its way
    // The list committed is the one drawn last: the child had come by then.
    a.commit(Uint32Array.of(c));
    a.begin();
    a.use(2);
    a.commit(Uint32Array.of(c));
    a.begin();
    a.use(2);
    a.commit(Uint32Array.of(c));
    // The full pool takes the parent's slot (idle), not a failure.
    a.begin();
    a.use(2);
    const got = [3, 4, 5].map((k) => a.allocate(k));
    expect(got.every((s) => s >= 0)).toBe(true);
    expect(got).toContain(p);
    expect(a.use(1)).toBe(-1);
  });

  it("shrinks: the drawn nodes past the new size move down, the idle go", () => {
    const a = new NodeSlotAllocator(8);
    frame(a, [1, 2, 3, 4, 5, 6, 7, 8]); // slots 0..7
    frame(a, [2, 7, 8]); // 1, 3..6 idle; 2, 7, 8 drawn (slots 1, 6, 7)
    const moves = a.shrink(4) as Uint32Array;
    expect(moves).not.toBeNull();
    expect(a.capacity).toBe(4);
    // Every drawn node still held, below 4, the moves from where it was.
    for (const k of [2, 7, 8]) {
      a.begin();
      const s = a.use(k);
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThan(4);
    }
    const from = new Set<number>();
    for (let i = 0; i < moves.length; i += 2) {
      from.add(moves[i]);
      expect(moves[i + 1]).toBeLessThan(4);
    }
    expect([...from].sort()).toEqual([6, 7]);
    // Too small for the drawn ones: refused.
    const b = new NodeSlotAllocator(4);
    frame(b, [1, 2, 3, 4]);
    expect(b.shrink(2)).toBeNull();
    expect(b.capacity).toBe(4);
  });

  it("releases a slot", () => {
    const a = new NodeSlotAllocator(2);
    a.begin();
    const s = a.allocate(9);
    a.release(s);
    expect(a.keyIn(s)).toBe(-1);
    expect(a.use(9)).toBe(-1);
    expect(a.allocate(10)).toBe(s);
  });

  it("takes a released slot before evicting an idle node (one deque)", () => {
    const a = new NodeSlotAllocator(4);
    frame(a, [1, 2, 3, 4]);
    frame(a, [1, 2]); // 3, 4 idle
    frame(a, [1, 2]);
    a.begin();
    a.release(a.use(2));
    a.commit();
    const f = frame(a, [1, 5]);
    expect(a.use(3)).toBeGreaterThanOrEqual(0); // 3 still held
    expect(f[1]).toBeGreaterThanOrEqual(0);
    expect(a.stats.evicted).toBe(0);
  });

  it("keeps pinned slots out of reach until unpinned (thread CU round 9)", () => {
    const a = new NodeSlotAllocator(4);
    const f1 = frame(a, [1, 2]);
    a.pin(Uint32Array.from(f1));
    // A finer cut that wants every slot: the pinned ones are not taken.
    const f2 = frame(a, [3, 4, 5, 6]);
    expect(f2.filter((s) => s >= 0).some((s) => f1.includes(s))).toBe(false);
    expect(f2.filter((s) => s < 0).length).toBe(2);
    expect(a.use(1)).toBe(f1[0]);
    expect(a.shrink(2)).toBeNull();
    a.unpin();
    const f3 = frame(a, [7]);
    frame(a, [8]);
    // Unpinned and out of use: idle, taken again.
    const f5 = frame(a, [9, 10, 11]);
    expect(new Set([...f5, ...f3]).size).toBeGreaterThan(0);
    expect(f5.every((s) => s >= 0)).toBe(true);
  });

  it("keeps its hysteresis across the u16 stamps' renumbering", () => {
    const a = new NodeSlotAllocator(4);
    const f1 = frame(a, [1, 2]);
    for (let k = 0; k < 0x10010; k++) frame(a, [1]);
    // 2 idle all along, 1 in every cut: both keep their slots.
    const f = frame(a, [1, 2, 3]);
    expect(f[0]).toBe(f1[0]);
    expect(f[1]).toBe(f1[1]);
    const g = frame(a, [4]); // 1, 2, 3 in the last cut: 4 takes the free one
    expect(f).not.toContain(g[0]);
    expect(g[0]).toBeGreaterThanOrEqual(0);
  });
});

describe("NodeIndex", () => {
  it("maps keys by page and place, freeing a page with its last node", () => {
    const x = new NodeIndex(3072);
    x.set(5 * 4096 + 7, 11);
    x.set(5 * 4096 + 3000, 12);
    x.set(9000 * 4096 + 1, 13);
    expect(x.get(5 * 4096 + 7)).toBe(11);
    expect(x.slot(5 * 4096 + 8)).toBe(-1);
    expect(x.get(9000 * 4096 + 1)).toBe(13);
    expect(x.bytes).toBeLessThan(2 * 1024);
    x.delete(5 * 4096 + 7);
    x.delete(5 * 4096 + 3000);
    expect(x.size).toBe(1);
    expect(x.slot(-5)).toBe(-1);
  });

  it("matches a Map over random sets and deletes (rank order kept)", () => {
    const x = new NodeIndex(3072);
    const ref = new Map<number, number>();
    let seed = 3;
    const rnd = () => {
      seed = (Math.imul(seed, 1103515245) + 12345) >>> 0;
      return seed;
    };
    for (let step = 0; step < 40000; step++) {
      const key = (rnd() % 7) * 4096 + (rnd() % 3072);
      if (ref.has(key) && rnd() % 3 === 0) {
        x.delete(key);
        ref.delete(key);
      } else {
        const s = rnd() % 100000;
        x.set(key, s);
        ref.set(key, s);
      }
      if (step % 4999 === 0) {
        for (let g = 0; g < 7; g++) {
          for (let k = 0; k < 3072; k++) {
            expect(x.slot(g * 4096 + k)).toBe(ref.get(g * 4096 + k) ?? -1);
          }
        }
      }
    }
    expect(x.size).toBe(ref.size);
  });
});

describe("PagePayloadStore", () => {
  const section = (n: number, seed: number) =>
    Uint8Array.from(
      { length: n * COMPACT_NODE_BYTES },
      (_, i) => (i * 7 + seed) & 0xff,
    );

  it("keeps the stored form, decompresses on a miss, LRU of decoded pages", () => {
    const s = new PagePayloadStore(2);
    const raws = [0, 1, 2].map((p) => section(10, p));
    raws.forEach((raw, p) => {
      s.put(p, 10, new Uint8Array(gzipSync(raw)), raw);
    });
    expect(s.pages).toBe(3);
    expect(s.decodedBytes).toBe(2 * raws[0].length); // the last two landed
    expect(s.sectionOf(2)).toBe(raws[2]); // a hit, the same bytes
    expect(s.stats.hits).toBe(1);
    const again = s.sectionOf(0) as Uint8Array; // a miss: gunzipped
    expect(s.stats.decodes).toBe(1);
    expect(Array.from(again)).toEqual(Array.from(raws[0]));
    expect(s.decodedBytes).toBe(2 * raws[0].length);
    // 1 was the least recent: gone from the LRU, 2 and 0 kept.
    s.sectionOf(2);
    s.sectionOf(0);
    expect(s.stats.decodes).toBe(1);
    s.drop(0);
    expect(s.sectionOf(0)).toBeNull();
    expect(s.pages).toBe(2);
    const compressed = [1, 2].reduce((n, p) => n + gzipSync(raws[p]).length, 0);
    expect(s.compressedBytes).toBe(compressed);
  });

  it("keeps an uncompressed section as it is", () => {
    const s = new PagePayloadStore(0);
    const raw = section(4, 3);
    s.put(5, 4, null, raw);
    expect(s.sectionOf(5)).toBe(raw);
    expect(s.stats.decodes).toBe(0);
  });
});

describe("node records", () => {
  it("hold each GPU plane's bytes at the record's offsets, and the origin", () => {
    const n = 5;
    const raw = Uint8Array.from(
      { length: n * COMPACT_NODE_BYTES },
      (_, i) => (i * 13 + 1) & 0xff,
    );
    const out = new Uint8Array(2 * NODE_RECORD_WORDS * 4);
    packPageRecords(raw, n, [3, 1], 77, out, 0);
    const at = nodeRecordLayout();
    const planes: Record<string, number> = {
      posX: 0,
      posY: 1,
      posZ: 2,
      albedo: 3,
      opacity: 4,
      scale: 5,
      rotation: 6,
      normal: 7,
      curvature: 8,
      material: 9,
      cluster: 10,
      weights: 11,
      weights1: 12,
      weights2: 13,
      shadow: 14,
    };
    const used = new Set<number>();
    [3, 1].forEach((place, r) => {
      for (const [name, k] of Object.entries(planes)) {
        const w = COMPACT_PLANE_WIDTHS[k];
        for (let b = 0; b < w; b++) {
          const byte = r * 44 + at[name] + b;
          if (r === 0) used.add(byte);
          expect(out[byte]).toBe(raw[compactPlaneOffset(k, n) + place * w + b]);
        }
      }
      const o = new DataView(out.buffer).getUint32(r * 44 + 40, true);
      expect(o & 0xffff).toBe(77);
      expect(o >>> 16).toBe(place);
    });
    // The 40 payload bytes, each once.
    expect(used.size).toBe(40);
    expect(Math.max(...used)).toBe(39);
  });
});

describe("RunPayloadStore", () => {
  /** A container like rust encode_runs: head, ends, a traversal member, runs. */
  function container(n: number, run: number) {
    const runs = Math.ceil(n / run);
    const trav = new Uint8Array(gzipSync(new Uint8Array(24 * n)));
    const parts: Uint8Array[] = [];
    for (let r = 0; r < runs; r++) {
      const m = Math.min(run, n - r * run);
      const planes = Uint8Array.from(
        { length: 40 * m },
        (_, i) => (i + r) & 0xff,
      );
      parts.push(new Uint8Array(deflateRawSync(planes)));
    }
    const head = new Uint32Array(5 + runs);
    head.set([0x34504d43, n, run, trav.length, runs]);
    let end = 0;
    parts.forEach((p, r) => {
      end += p.length;
      head[5 + r] = end;
    });
    const out = new Uint8Array(head.byteLength + trav.length + end);
    out.set(new Uint8Array(head.buffer), 0);
    out.set(trav, head.byteLength);
    let at = head.byteLength + trav.length;
    for (const p of parts) {
      out.set(p, at);
      at += p.length;
    }
    return { out, parts };
  }

  it("keeps the cut's runs, drops the others over budget, fetches them again", () => {
    const { out, parts } = container(200, 64);
    const s = new RunPayloadStore(0, 2);
    s.put(7, out, 1000);
    expect(s.bytes).toBe(parts.reduce((n, p) => n + p.length, 0));
    s.begin();
    s.mark(7, 70, RUN_CUT); // run 1
    s.mark(7, 130, RUN_CULLED); // run 2
    s.trim();
    const c = s.census();
    expect(c).toMatchObject({ runs: 4, held: 1, cut: 1 });
    expect(s.missing(7, 0)).toBe(true);
    expect(s.planes(7, 0)).toBeNull();
    const p1 = s.planes(7, 1) as Uint8Array;
    expect(p1.length).toBe(40 * 64);
    expect(p1[0]).toBe(1);
    // Run 3 (8 nodes) fetched again from its file span.
    const span = s.fileSpan(7, 3) as [number, number];
    const local = span[0] - 1000;
    s.putRun(7, 3, out.slice(local, local + span[1]));
    expect((s.planes(7, 3) as Uint8Array).length).toBe(40 * 8);
    expect(s.stats.fetched).toBe(1);
    s.drop(7);
    expect(s.bytes).toBe(0);
    expect(s.inflatedBytes).toBe(0);
  });

  it("unpacks a container to the traversal's input with the run positions", () => {
    const { out } = container(100, 64);
    const h = parseRunsContainer(out);
    const u = unpackRunsContainer(out, h, new Uint8Array(24 * 100), true);
    expect(u.input.length).toBe(30 * 100);
    // posX of node 70 (run 1, local 6): run 1's planes start with posX.
    expect(u.input[2 * 70]).toBe((2 * 6 + 1) & 0xff);
    expect(u.raw?.length).toBe(64 * 100);
  });
});

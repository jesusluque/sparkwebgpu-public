import { describe, expect, it } from "vitest";
import { loadTick } from "../../src/webgpu/athenea/compactRecords";
import { LoadMarks } from "../../src/webgpu/athenea/nodeWorker";
import { atheneaAdapterCompact as K } from "../../src/webgpu/generated/constants";

/** A list of slots `slots` for nodes `keys` (parents, pending) marked at `now`. */
function marked(
  g: LoadMarks,
  slots: number[],
  keys: number[],
  parents: number[],
  pending: boolean[],
  now: number,
) {
  const list = Uint32Array.from(slots);
  const until = g.mark(
    list,
    (x) => keys[x],
    (x) => parents[x],
    (x) => pending[x],
    now,
  );
  return { list, until };
}

const fading = (s: number, t: number) =>
  (s | K.kCompactLoadFading | (loadTick(t) << 24)) >>> 0;

describe("LoadMarks", () => {
  it("marks the waiting nodes, fades their children in, then lets them be", () => {
    const g = new LoadMarks();
    const t0 = 1_000_000;
    // Node 7 (slot 3) waits for its children; node 8 (slot 4) does not.
    let r = marked(g, [3, 4], [7, 8], [-1, -1], [true, false], t0);
    expect(Array.from(r.list)).toEqual([(3 | K.kCompactLoadPending) >>> 0, 4]);
    expect(r.until).toBe(0);
    // Its children 70 and 71 arrive (slots 5, 6) at t1: they fade from t1.
    const t1 = t0 + 2000;
    r = marked(
      g,
      [5, 6, 4],
      [70, 71, 8],
      [7, 7, -1],
      [false, false, false],
      t1,
    );
    expect(Array.from(r.list)).toEqual([fading(5, t1), fading(6, t1), 4]);
    expect(r.until).toBe(t1 + 1000);
    // A grandchild 700 in 0.2 s later inherits its parent's arrival.
    r = marked(
      g,
      [9, 6, 4],
      [700, 71, 8],
      [70, 7, -1],
      [false, false, false],
      t1 + 200,
    );
    expect(Array.from(r.list)).toEqual([fading(9, t1), fading(6, t1), 4]);
    // Past the half second: no marks, nothing to wait for.
    r = marked(
      g,
      [9, 6, 4],
      [700, 71, 8],
      [70, 7, -1],
      [false, false, false],
      t1 + 1000,
    );
    expect(Array.from(r.list)).toEqual([9, 6, 4]);
    expect(r.until).toBe(0);
  });

  it("fades a node that stops waiting, and marks nothing in a loaded cut", () => {
    const g = new LoadMarks();
    marked(g, [1], [10], [-1], [true], 0);
    const r = marked(g, [1], [10], [-1], [false], 100);
    expect(r.list[0]).toBe(fading(1, 100));
    const h = new LoadMarks();
    const q = marked(h, [1, 2], [10, 11], [-1, 10], [false, false], 0);
    expect(Array.from(q.list)).toEqual([1, 2]);
  });
});

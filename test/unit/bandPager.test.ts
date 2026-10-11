import { describe, expect, it } from "vitest";
import type { AthcPageRow } from "../../src/athc";
import {
  type BandMesh,
  pageNeeded,
  planBand,
} from "../../src/webgpu/WgpuBandPager";

// A row of a page table: a box on the x axis from x0 to x1, parent errors.
function row(
  x0: number,
  x1: number,
  epsMax: number,
  epsMin = epsMax / 2,
  count = 100,
): AthcPageRow {
  return {
    band: 0,
    count,
    first: 0,
    lo: [x0, -0.1, -0.1],
    hi: [x1, 0.1, 0.1],
    epsMax,
    epsMin,
    offset: 0,
    tierBytes: [1000, 1000, 1000],
  };
}

// Page 0 the root; coarse pages over everything, then finer ones along x.
const rows: AthcPageRow[] = [
  row(0, 4, 3e38, 3e38),
  row(0, 4, 0.08),
  row(0, 2, 0.02),
  row(2, 4, 0.02),
  row(0, 1, 0.005),
  row(1, 2, 0.005),
  row(2, 3, 0.005),
  row(3, 4, 0.005),
];

function mesh(
  eye: number[],
  limit: number,
  resident: Set<number> = new Set(),
  children: Record<number, number[]> = {},
  inFlight: Set<number> = new Set(),
): BandMesh<string> {
  return {
    key: "m",
    rows,
    pageBytes: rows.map((_, p) => (p === 1 ? 4000 : 1000)),
    eye,
    limit,
    resident: (p) => resident.has(p),
    inFlight: (p) => inFlight.has(p),
    children: (p) => children[p],
  };
}

describe("band pages (athenea's error-band paging)", () => {
  it("tests a page by its largest parent error from its box's nearest point", () => {
    // 0.02 / 1 = 0.02 > 0.01: needed; from 3 away, 0.0067: not.
    expect(pageNeeded(rows[2], [3, 0, 0], 0.01)).toBe(true);
    expect(pageNeeded(rows[2], [5, 0, 0], 0.01)).toBe(false);
    // Inside the box: the distance is the floor (1e-3).
    expect(pageNeeded(rows[7], [3.5, 0, 0], 4)).toBe(true);
  });

  it("needs what the eye is near, always page 0, and asks by error per byte", () => {
    // From x = -0.5 with limit 0.01: page 1 (0.08 / 0.5), 2 (0.02 / 0.5),
    // 4 (0.005 / 0.5 = 0.01, not past), 3 (0.02 / 2.5) not.
    const plan = planBand([mesh([-0.5, 0, 0], 0.01)], 100);
    expect([...(plan.needed.get("m") as Set<number>)].sort()).toEqual([
      0, 1, 2,
    ]);
    expect(plan.neededPages).toBe(3);
    expect(plan.neededNodes).toBe(300);
    // Pages 0, 1, 2 are consecutive: one Range request.
    expect(plan.requests).toHaveLength(1);
    expect(plan.requests[0].pages).toEqual([0, 1, 2]);
  });

  it("coalesces only consecutive pages, the best first", () => {
    // Near x = 2.5: pages 0..3 and the fine pages 5 (x 1-2), 6 (2-3), 7 (3-4)
    // within 0.5 m; 4 (x 0-1) is 1.5 m away.
    const plan = planBand(
      [mesh([2.5, 0, 0], 0.009, new Set([0, 1, 2, 3]))],
      100,
    );
    const needed = plan.needed.get("m") as Set<number>;
    expect(needed.has(4)).toBe(false);
    expect([...needed].sort()).toEqual([0, 1, 2, 3, 5, 6, 7]);
    expect(plan.requests).toHaveLength(1);
    expect(plan.requests[0].pages).toEqual([5, 6, 7]);
  });

  it("stays within the budget: the best missing pages fill what is free", () => {
    const resident = new Set([0, 1]);
    // Everything near x = 2 is needed; capacity 4 leaves room for two more.
    const plan = planBand([mesh([2, 0.15, 0], 0.001, resident)], 4);
    expect(plan.neededPages).toBe(8);
    const asked = plan.requests.flatMap((r) => r.pages);
    expect(asked).toHaveLength(2);
    // The coarser pages (error 0.02 a page) before the finer (0.005).
    expect(asked.sort()).toEqual([2, 3]);
  });

  it("evicts from the highest page down, never a page a resident page depends on, never page 0", () => {
    // Resident: everything. Needed from far away: only 0 and 1.
    const all = new Set(rows.map((_, p) => p));
    // Page 2's nodes have children in 4 and 5; 3's in 6 and 7; 1's in 2, 3.
    const children = { 1: [2, 3], 2: [4, 5], 3: [6, 7] };
    const plan = planBand([mesh([40, 0, 0], 0.001, all, children)], 100);
    expect([...(plan.needed.get("m") as Set<number>)].sort()).toEqual([0, 1]);
    // All of 2..7 may go (their dependents go too), finest first.
    expect(plan.evictable.map((e) => e.page)).toEqual([7, 6, 5, 4, 3, 2]);
    // With page 6 needed (say a second view), page 3 must stay.
    const near6 = planBand([mesh([2.5, 0, 0], 0.004, all, children)], 100);
    const needed = near6.needed.get("m") as Set<number>;
    expect(needed.has(6)).toBe(true);
    expect(needed.has(3)).toBe(true);
    // A page whose child page is still coming is not evicted either.
    const coming = planBand(
      [mesh([40, 0, 0], 0.001, new Set([0, 1, 2, 3]), children, new Set([4]))],
      100,
    );
    expect(coming.evictable.map((e) => e.page)).toEqual([3]);
  });
});

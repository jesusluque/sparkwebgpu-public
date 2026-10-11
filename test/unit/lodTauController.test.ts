// lodErrorTauAuto's controller (WgpuLod LodTauController): the detail scale
// that divides tau steps down under pressure, up with room, holds between,
// at most once an interval, within [min, 1].

import { describe, expect, it } from "vitest";
import { LodTauController } from "../../src/webgpu/WgpuLod";

function run(
  c: LodTauController,
  frames: number,
  dt: number,
  fill: number | null,
  t0: number,
) {
  let t = t0;
  for (let i = 0; i < frames; ++i) {
    t += dt;
    c.frame(t, fill);
  }
  return t;
}

describe("LodTauController", () => {
  it("coarsens over the frame budget, down to its floor, never past 1", () => {
    const c = new LodTauController();
    let t = run(c, 10, 16, 0.5, 0);
    expect(c.scale).toBe(1); // under budget with room: already at 1
    t = run(c, 600, 40, 0.5, t); // 24 s at 25 fps
    expect(c.scale).toBeCloseTo(0.3, 5);
    // Rate-limited: about one step every 250 ms, not one a frame.
    const d = new LodTauController();
    run(d, 50, 40, 0.5, 0); // 2 s
    expect(d.scale).toBeGreaterThan(0.97 ** 9);
    expect(d.scale).toBeLessThan(0.97 ** 5);
  });

  it("coarsens on a full pool, holds in the band, refines with room", () => {
    const c = new LodTauController();
    let t = run(c, 100, 16, 0.95, 0);
    const low = c.scale;
    expect(low).toBeLessThan(1);
    t = run(c, 200, 16, 0.78, t); // between low and high: hold
    expect(c.scale).toBe(low);
    t = run(c, 2000, 16, 0.5, t);
    expect(c.scale).toBe(1);
    // Fast frames near the budget's edge (0.85..1 x frameMs) also hold.
    const d = new LodTauController();
    run(d, 100, 16, 0.95, 0);
    const held = d.scale;
    run(d, 200, 18, 0.5, 1600);
    expect(d.scale).toBe(held);
  });
});

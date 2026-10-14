import { describe, expect, it } from "vitest";
import {
  type FitTauState,
  fitTauStep,
} from "../../src/webgpu/athenea/compactPages";

/** A pool that holds the cut at tau >= minFit (a still view); the traversal's wanted set at
 * tau T reports (as page bounds do) the fit at T: 0 above minFit, else a
 * value far above it (the bound's overshoot, iPad: 11.8 -> 23). */
function run(hold: boolean, steps: number, minFit: number, start: number) {
  const st: FitTauState = {
    tau: start,
    tooLow: 0,
    fell: false,
    hold,
    held: 0,
  };
  const taus: number[] = [];
  for (let i = 0; i < steps; i++) {
    const fit = st.tau >= minFit ? 0.5 : 2 * minFit;
    st.tau = fitTauStep(st, fit, true);
    taus.push(st.tau);
  }
  return { st, taus };
}

describe("fitTauStep", () => {
  it("a still view settles on one tau and stays", () => {
    const { st, taus } = run(true, 60, 12, 23);
    const tail = taus.slice(-30);
    expect(new Set(tail).size).toBe(1);
    // At most one step under the lowest tau that fits (the pool keeps what
    // it holds there).
    expect(tail[0]).toBeGreaterThanOrEqual(12 / 1.25);
    expect(st.held).toBe(tail[0]);
    // Never coarser than the start, and monotone.
    for (let i = 1; i < taus.length; i++) {
      expect(taus[i]).toBeLessThanOrEqual(taus[i - 1]);
    }
  });
  it("a still view never goes coarser", () => {
    const st: FitTauState = {
      tau: 10,
      tooLow: 0,
      fell: false,
      hold: true,
      held: 0,
    };
    // The pool's fit (or the frame-time floor) above tau: tau stays.
    expect(fitTauStep(st, 11, true)).toBe(10);
    expect(fitTauStep(st, 40, true)).toBe(10);
    // It still refines.
    expect(fitTauStep(st, 1, true)).toBe(8);
    // A step down that did not fit: held there, for good.
    st.tau = 8;
    expect(fitTauStep(st, 30, true)).toBe(8);
    expect(st.held).toBe(8);
    expect(fitTauStep(st, 1, true)).toBe(8);
  });
  it("random fits: a still tau never rises", () => {
    let seed = 7;
    const rnd = () => {
      seed = (seed * 16807) % 2147483647;
      return seed / 2147483647;
    };
    const st: FitTauState = {
      tau: 20,
      tooLow: 0,
      fell: false,
      hold: true,
      held: 0,
    };
    for (let i = 0; i < 500; i++) {
      const next = fitTauStep(st, 40 * rnd(), true);
      expect(next).toBeLessThanOrEqual(st.tau);
      st.tau = next;
    }
  });
  it("moving: up at once, down by at most 1/1.25 a step", () => {
    const st: FitTauState = {
      tau: 10,
      tooLow: 0,
      fell: false,
      hold: false,
      held: 0,
    };
    expect(fitTauStep(st, 1, true)).toBe(8);
    st.tau = 8;
    expect(fitTauStep(st, 20, true)).toBe(20);
  });
});

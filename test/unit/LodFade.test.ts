// LodFade (src/webgpu/LodFade.ts): LoD transitions that fade. The splats
// entering a cut fade in over the fade time while those leaving it stay
// drawn, then fade out; at rest the drawn indices are the cut itself.

import { describe, expect, it } from "vitest";
import {
  LOD_FADE_MAX_INDEX,
  LOD_FADE_MAX_STEP_MS,
  LodFade,
} from "../../src/webgpu/LodFade";

const fadeOf = (word: number) => 1 - (word >>> 24) / 255;
const indexOf = (word: number) => word & LOD_FADE_MAX_INDEX;

/** index -> how drawn it is, from a frame. */
function weights(frame: { indices: Uint32Array; faded?: Uint32Array }) {
  const out = new Map<number, number>();
  frame.indices.forEach((i, k) => {
    const w = frame.faded ? fadeOf(frame.faded[k]) : 1;
    if (frame.faded) expect(indexOf(frame.faded[k])).toBe(i);
    out.set(i, w);
  });
  return out;
}

describe("LodFade", () => {
  it("draws the first cut at once, as it is", () => {
    const fade = new LodFade(100);
    const cut = new Uint32Array([5, 3, 9]);
    const frame = fade.setCut(cut, 0);
    expect(frame.indices).toBe(cut);
    expect(frame.faded).toBeUndefined();
    expect(fade.active).toBe(false);
    expect(fade.step(16)).toBeNull();
  });

  it("fades the new splats in, holds the old, then fades them out", () => {
    const fade = new LodFade(100);
    fade.setCut(new Uint32Array([0, 1, 2]), 0);
    // A node (2) splits into its children (3, 4).
    const cut = new Uint32Array([0, 1, 3, 4]);
    let w = weights(fade.setCut(cut, 0));
    expect(w.get(3)).toBe(0);
    expect(w.get(4)).toBe(0);
    expect(w.get(2)).toBe(1);
    expect(fade.active).toBe(true);
    // Halfway: the children half in, the parent still whole.
    let t = 0;
    for (; t < 50; t += 10) fade.step(t + 10);
    w = weights(fade.step(50) ?? { indices: new Uint32Array() });
    expect(w.get(3)).toBeCloseTo(0.5, 1);
    expect(w.get(2)).toBe(1);
    expect(w.get(0)).toBe(1);
    // Children in: the parent starts to go.
    for (t = 50; t < 150; t += 10) fade.step(t + 10);
    w = weights(fade.step(150) ?? { indices: new Uint32Array() });
    expect(w.get(3)).toBe(1);
    expect(w.get(2)).toBeLessThan(0.6);
    expect(w.get(2)).toBeGreaterThan(0.3);
    // Done: the cut itself, in its order, once; then nothing.
    let last = null;
    for (t = 150; t < 260; t += 10) last = fade.step(t + 10) ?? last;
    expect(fade.active).toBe(false);
    expect(last?.indices).toBe(cut);
    expect(last?.faded).toBeUndefined();
    expect(fade.step(400)).toBeNull();
  });

  it("covers the surface throughout: the old stay until the new are in", () => {
    const fade = new LodFade(80);
    fade.setCut(new Uint32Array([10]), 0);
    fade.setCut(new Uint32Array([11, 12]), 0);
    for (let t = 10; t <= 200; t += 10) {
      const frame = fade.step(t);
      if (!frame) break;
      const w = weights(frame);
      const parent = w.get(10) ?? 0;
      const child = Math.min(w.get(11) ?? 0, w.get(12) ?? 0);
      // Either the parent is whole or the children are.
      expect(Math.max(parent, child)).toBeGreaterThan(0.99);
    }
  });

  it("brings back a splat fading out from where it is", () => {
    const fade = new LodFade(100);
    fade.setCut(new Uint32Array([1]), 0);
    fade.setCut(new Uint32Array([2]), 0);
    for (let t = 10; t <= 150; t += 10) fade.step(t);
    const out = weights(fade.step(150) ?? { indices: new Uint32Array() });
    const before = out.get(1) ?? 0;
    expect(before).toBeLessThan(1);
    expect(before).toBeGreaterThan(0);
    // The camera turns back: 1 is in the cut again, 2 leaves.
    const back = weights(fade.setCut(new Uint32Array([1]), 150));
    expect(back.get(1)).toBeCloseTo(before, 2);
    expect(back.get(2)).toBe(1);
  });

  it("holds a frame's step to a thirtieth of a second", () => {
    const fade = new LodFade(200);
    fade.setCut(new Uint32Array([1]), 0);
    fade.setCut(new Uint32Array([2]), 0);
    // A stall of a second moves the fade one step only.
    const w = weights(fade.step(1000) ?? { indices: new Uint32Array() });
    expect(w.get(2)).toBeCloseTo(LOD_FADE_MAX_STEP_MS / 200, 2);
  });

  it("draws without a fade when it is off or an index is too large", () => {
    const off = new LodFade(0);
    off.setCut(new Uint32Array([1]), 0);
    const plain = off.setCut(new Uint32Array([2]), 0);
    expect(plain.faded).toBeUndefined();
    expect(off.active).toBe(false);
    const big = new LodFade(100);
    big.setCut(new Uint32Array([1]), 0);
    const cut = new Uint32Array([LOD_FADE_MAX_INDEX + 1]);
    const frame = big.setCut(cut, 0);
    expect(frame.indices).toBe(cut);
    expect(frame.faded).toBeUndefined();
  });

  it("starts again at once after a reset (a mesh hidden and shown)", () => {
    const fade = new LodFade(100);
    fade.setCut(new Uint32Array([1]), 0);
    fade.reset();
    const frame = fade.setCut(new Uint32Array([2]), 10);
    expect(frame.faded).toBeUndefined();
    expect(fade.active).toBe(false);
  });
});

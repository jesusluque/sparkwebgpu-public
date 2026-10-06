// The relight pass's footprint prefilter (slang/athenea_adapter/relight.slang
// footprintRoughness): a splat shaded once and drawn as one colour has its
// lobes widened by how far the normal turns over its gaussian, so a coarse
// cloud on a curved mirror blurs a sharp reflection instead of showing its
// splats' layout as a moire (the light Corvette's doors). Checked on Dawn
// against the formula: GGX alpha grows in quadrature by gain x the normal's
// spread, sqrt(su^2 |S e_u|^2 + sv^2 |S e_v|^2), at most 1.

import { describe, expect, it } from "vitest";
import { KernelRegistry } from "../../src/webgpu/KernelRegistry";
import { FOOTPRINT_GAIN } from "../../src/webgpu/athenea/relightPlugin";
import footprintTest from "../../src/webgpu/generated/tests/relight_footprint";
import { device, readBack, storage } from "./device";

type Case = {
  shape: [number, number, number];
  roughness: number;
  scales: [number, number, number];
  gain: number;
};

function expected({ shape, roughness, scales, gain }: Case): number {
  const [xx, xy, yy] = shape;
  const variance =
    scales[0] ** 2 * (xx * xx + xy * xy) + scales[1] ** 2 * (xy * xy + yy * yy);
  const alpha = roughness * roughness;
  const add = gain * Math.sqrt(variance);
  const widened = Math.min(Math.sqrt(alpha * alpha + add * add), 1);
  return Math.max(Math.sqrt(widened), roughness);
}

describe.skipIf(!device)("relight footprint prefilter", () => {
  const registry = new KernelRegistry(device as GPUDevice);

  async function run(cases: Case[]): Promise<Float32Array> {
    const data = new Float32Array(cases.length * 8);
    cases.forEach((c, i) => {
      data.set([...c.shape, c.roughness, ...c.scales, c.gain], i * 8);
    });
    const out = storage(cases.length * 4);
    registry.get(footprintTest, "footprintTest").run({
      grid: [cases.length],
      buffers: { footprintCases: storage(data), footprintOut: out },
    });
    return new Float32Array(await readBack(out));
  }

  it("widens a coarse splat's lobes on a curved panel, as the formula says", async () => {
    // The light Corvette's paint: a 9 mm cell, widened (sigma about 5 mm),
    // on a door bent at 1-3 per metre, under a 0.05 coat; and the detailed
    // cloud's 1.5 mm splats on the same door.
    const cases: Case[] = [
      {
        shape: [2, 0, 0.5],
        roughness: 0.05,
        scales: [0.005, 0.004, 1e-4],
        gain: FOOTPRINT_GAIN,
      },
      {
        shape: [3, 0.5, 1],
        roughness: 0.05,
        scales: [0.005, 0.005, 1e-4],
        gain: 1,
      },
      {
        shape: [2, 0, 0.5],
        roughness: 0.05,
        scales: [0.0015, 0.0012, 1e-4],
        gain: FOOTPRINT_GAIN,
      },
      {
        shape: [-20, 4, 12],
        roughness: 0.3,
        scales: [0.02, 0.01, 1e-3],
        gain: 0.5,
      },
      // A crease: far past what a lobe can widen, held at alpha 1.
      {
        shape: [400, 0, 0],
        roughness: 0.05,
        scales: [0.01, 0.01, 1e-4],
        gain: 1,
      },
    ];
    const got = await run(cases);
    cases.forEach((c, i) => {
      expect(got[i]).toBeCloseTo(expected(c), 5);
      expect(got[i]).toBeGreaterThan(c.roughness);
    });
    expect(got[4]).toBeCloseTo(1, 6);
    // The coarse splat's coat half as rough again; the fine one's hardly.
    expect(got[0]).toBeGreaterThan(0.07);
    expect(got[2]).toBeLessThan(0.055);
  });

  it("leaves a flat or a fine splat's roughness as it is", async () => {
    const got = await run([
      {
        shape: [0, 0, 0],
        roughness: 0.05,
        scales: [0.01, 0.01, 1e-4],
        gain: 1,
      },
      { shape: [2, 0, 2], roughness: 0.4, scales: [0, 0, 0], gain: 1 },
      {
        shape: [2, 0, 2],
        roughness: 0.05,
        scales: [0.01, 0.01, 1e-4],
        gain: 0,
      },
      {
        shape: [Number.NaN, 0, 0],
        roughness: 0.2,
        scales: [0.01, 0.01, 1e-4],
        gain: 1,
      },
    ]);
    expect(got[0]).toBeCloseTo(0.05, 6);
    expect(got[1]).toBeCloseTo(0.4, 6);
    expect(got[2]).toBeCloseTo(0.05, 6);
    expect(got[3]).toBeCloseTo(0.2, 6);
  });
});

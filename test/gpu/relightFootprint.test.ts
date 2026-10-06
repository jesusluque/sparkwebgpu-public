// The relight pass's footprint prefilter (slang/athenea_adapter/relight.slang
// footprintRoughness): a splat shaded once and drawn as one colour has its
// lobes widened by how far the normal turns over its gaussian, so a coarse
// cloud on a curved mirror blurs a sharp reflection instead of showing its
// splats' layout as a moire (the light Corvette's doors). Checked on Dawn
// against the filtered-GGX form of geometric specular anti-aliasing
// (Kaplanyan et al. 2016, Tokuyoshi & Kaplanyan 2019): alpha^2 grows by
// min(gain^2 x variance, kappa), the variance the larger of the splat's
// (su^2 |S e_u|^2 + sv^2 |S e_v|^2) and the pixel's, alpha^2 at most 1.

import { describe, expect, it } from "vitest";
import { KernelRegistry } from "../../src/webgpu/KernelRegistry";
import {
  FOOTPRINT_CLAMP,
  FOOTPRINT_CLAMP_GGX,
  FOOTPRINT_GAIN,
  FOOTPRINT_GAIN_GGX,
} from "../../src/webgpu/athenea/relightPlugin";
import footprintTest from "../../src/webgpu/generated/tests/relight_footprint";
import { device, readBack, storage } from "./device";

type Case = {
  shape: [number, number, number];
  roughness: number;
  scales: [number, number, number];
  gain: number;
  kappa?: number;
  pixel?: number;
  /** The merged normals' variance (CURV's fourth half). */
  spread?: number;
};

function expected({
  shape,
  roughness,
  scales,
  gain,
  kappa = FOOTPRINT_CLAMP,
  pixel = 0,
  spread = 0,
}: Case): number {
  const [xx, xy, yy] = shape;
  const splat =
    scales[0] ** 2 * (xx * xx + xy * xy) + scales[1] ** 2 * (xy * xy + yy * yy);
  const alpha = roughness * roughness;
  const add = Math.min(gain * gain * Math.max(splat, pixel, spread), kappa);
  const widened = Math.min(alpha * alpha + add, 1);
  return Math.max(Math.sqrt(Math.sqrt(widened)), roughness);
}

describe.skipIf(!device)("relight footprint prefilter", () => {
  const registry = new KernelRegistry(device as GPUDevice);

  async function run(cases: Case[]): Promise<Float32Array> {
    const data = new Float32Array(cases.length * 12);
    cases.forEach((c, i) => {
      data.set(
        [
          ...c.shape,
          c.roughness,
          ...c.scales,
          c.gain,
          c.kappa ?? FOOTPRINT_CLAMP,
          c.pixel ?? 0,
          c.spread ?? 0,
          0,
        ],
        i * 12,
      );
    });
    const out = storage(cases.length * 4);
    registry.get(footprintTest, "footprintTest").run({
      grid: [cases.length],
      buffers: { footprintCases: storage(data), footprintOut: out },
    });
    return new Float32Array(await readBack(out));
  }

  it("widens a coarse splat's lobes on a curved panel, as filtered GGX says", async () => {
    // The light Corvette's paint: a 9 mm cell, widened (sigma about 5 mm),
    // on a door bent at 1-3 per metre, under a 0.05 coat; and the detailed
    // cloud's 1.5 mm splats on the same door.
    const cases: Case[] = [
      {
        shape: [2, 0, 0.5],
        roughness: 0.05,
        scales: [0.005, 0.004, 1e-4],
        gain: FOOTPRINT_GAIN_GGX,
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
        gain: FOOTPRINT_GAIN_GGX,
      },
      {
        shape: [-20, 4, 12],
        roughness: 0.3,
        scales: [0.02, 0.01, 1e-3],
        gain: 0.5,
      },
      // A crease: far past what a lobe can widen, held at kappa (a rough
      // lobe, not a diffuse one).
      {
        shape: [400, 0, 0],
        roughness: 0.05,
        scales: [0.01, 0.01, 1e-4],
        gain: FOOTPRINT_GAIN_GGX,
        kappa: FOOTPRINT_CLAMP_GGX,
      },
      // The same crease under the default gain and clamp: a diffuse lobe.
      {
        shape: [400, 0, 0],
        roughness: 0.05,
        scales: [0.01, 0.01, 1e-4],
        gain: FOOTPRINT_GAIN,
      },
    ];
    const got = await run(cases);
    cases.forEach((c, i) => {
      expect(got[i]).toBeCloseTo(expected(c), 5);
      expect(got[i]).toBeGreaterThan(c.roughness);
    });
    // The crease: alpha^2 = kappa, roughness kappa^(1/4) (about 0.65).
    expect(got[4]).toBeCloseTo(FOOTPRINT_CLAMP_GGX ** 0.25, 4);
    expect(got[5]).toBeCloseTo(FOOTPRINT_CLAMP ** 0.25, 4);
    expect(got[4]).toBeLessThan(0.7);
    // The coarse splat's coat over twice as rough; the fine one's less.
    expect(got[0]).toBeGreaterThan(0.11);
    expect(got[2]).toBeLessThan(0.075);
  });

  it("takes the merged normals' variance where it is the larger", async () => {
    // A merged cell over a crease: flat to the curvature, its members'
    // normals 0.3 rad apart.
    const cell = {
      shape: [0.5, 0, 0.5] as [number, number, number],
      roughness: 0.05,
      scales: [0.005, 0.005, 1e-4] as [number, number, number],
      gain: FOOTPRINT_GAIN,
    };
    const cases: Case[] = [cell, { ...cell, spread: 0.09 }];
    const got = await run(cases);
    cases.forEach((c, i) => expect(got[i]).toBeCloseTo(expected(c), 5));
    expect(got[1]).toBeGreaterThan(3 * got[0]);
  });

  it("takes the pixel's footprint where it is the larger", async () => {
    // A 1.5 mm splat on a door bent at 2 per metre, seen from far: a
    // pixel spans 2 cm of the door, so the normal turns by 0.04 a pixel.
    const fine = {
      shape: [2, 0, 2] as [number, number, number],
      roughness: 0.05,
      scales: [0.0015, 0.0015, 1e-4] as [number, number, number],
      gain: FOOTPRINT_GAIN_GGX,
    };
    const pixel = 0.25 * 2 * 0.04 ** 2;
    const cases: Case[] = [
      fine,
      { ...fine, pixel },
      // The pixel smaller than the splat: the splat's term alone.
      { ...fine, scales: [0.02, 0.02, 1e-3], pixel },
      { ...fine, scales: [0.02, 0.02, 1e-3] },
    ];
    const got = await run(cases);
    cases.forEach((c, i) => expect(got[i]).toBeCloseTo(expected(c), 5));
    expect(got[1]).toBeGreaterThan(1.5 * got[0]);
    expect(got[2]).toBeCloseTo(got[3], 6);
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

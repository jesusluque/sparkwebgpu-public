// The dome prepared on the GPU (AtheneaSky, slang/athenea_adapter/env.slang)
// against athenea's own environment tests (tests/technique/
// test_environment.cpp at txf 89a04d9): the same skies, the same checks --
// athenea's environment_check kernels (slang/tests/athenea_env.slang), which
// read the map through its envIrradiance, envRadiance and envTexelAt -- and
// the same tolerances.

import { describe, expect, it } from "vitest";
import { KernelRegistry } from "../../src/webgpu/KernelRegistry";
import { AtheneaSky, type SkyImage } from "../../src/webgpu/athenea/AtheneaSky";
import envTest from "../../src/webgpu/generated/tests/athenea_env";
import { UniformWriter } from "../../src/webgpu/uniforms";
import { device, readBack, storage } from "./device";

const PI = Math.PI;

/** An 8-bit image, as athenea's tests write their PNGs ("raw": code / 255). */
function skyImage(
  w: number,
  h: number,
  fill: (x: number, y: number) => number,
): SkyImage {
  const data = new Float32Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = fill(x, y);
      data.set([v, v, v, 1], (y * w + x) * 4);
    }
  }
  return { width: w, height: h, data };
}

const grey = (v: number) => Math.round(Math.min(Math.max(v, 0), 1) * 255) / 255;

/** The source's integral over the sphere: each texel its band. */
function sourceIntegral(image: SkyImage) {
  const { width: w, height: h, data } = image;
  let sum = 0;
  for (let y = 0; y < h; y++) {
    const band =
      ((2 * PI) / w) * (Math.cos((PI * y) / h) - Math.cos((PI * (y + 1)) / h));
    for (let x = 0; x < w; x++) sum += data[(y * w + x) * 4] * band;
  }
  return sum;
}

describe.skipIf(!device)("athenea sky (env prepare)", () => {
  const d = device as GPUDevice;
  const registry = new KernelRegistry(d);

  async function prepare(image: SkyImage | null, intensity = 1) {
    const sky = new AtheneaSky(d, registry);
    sky.set({ image, intensity });
    sky.prepareNow();
    const { sun } = await sky.read();
    return {
      sky,
      axis: Array.from(sun.subarray(0, 4)),
      power: Array.from(sun.subarray(4, 8)),
    };
  }

  async function run(
    sky: AtheneaSky,
    entry: string,
    values: Record<string, number>,
  ) {
    const stats = storage(32);
    const worst = storage(32);
    registry.get(envTest, entry).run({
      grid: [1],
      buffers: {
        envTexels: sky.envTexels as GPUBuffer,
        envSh: sky.envSh,
        stats,
        worst,
      },
      uniforms: UniformWriter.for(envTest).setAll({
        dome: 0,
        baseSide: sky.baseSide,
        ...values,
      }).data,
    });
    const counts = Array.from(new Uint32Array(await readBack(stats)));
    const readings = Array.from(new Float32Array(await readBack(worst)));
    stats.destroy();
    worst.destroy();
    return { counts, values: readings };
  }

  const check = (
    sky: AtheneaSky,
    directions: number,
    expected: number,
    tolerance: number,
    up: number,
    down: number,
    rough: number,
  ) =>
    run(sky, "environmentCheck", {
      directions,
      expected,
      tolerance,
      upExpected: up,
      downExpected: down,
      roughExpected: rough,
    });

  it("a sky of one colour prepares to the closed form a dome always had", async () => {
    const { sky } = await prepare(
      skyImage(64, 32, () => 1),
      2,
    );
    const r = await check(sky, 64, PI * 2, 0.02, 2, 2, 2);
    console.log(
      `  flat sky: irradiance ${r.values[1].toFixed(4)} (want ${(2 * PI).toFixed(4)}), worst ${(r.values[0] * 100).toFixed(3)}%; up ${r.values[4].toFixed(3)} down ${r.values[5].toFixed(3)} rough ${r.values[6].toFixed(3)}`,
    );
    expect(r.counts[1]).toBe(64);
    expect(r.counts.slice(0, 1).concat(r.counts.slice(2, 5))).toEqual([
      0, 0, 0, 0,
    ]);
    sky.destroy();
  });

  it("a sky with no image is its colour, as the closed form", async () => {
    const { sky, axis } = await prepare(null, 3);
    const r = await check(sky, 64, PI * 3, 1e-4, 3, 3, 3);
    expect(r.counts[0]).toBe(0);
    expect(r.counts.slice(2, 5)).toEqual([0, 0, 0]);
    expect(axis[3]).toBe(0);
    sky.destroy();
  });

  it("a sky that varies arrives with its variation", async () => {
    const { sky } = await prepare(skyImage(64, 32, (_, y) => (y < 16 ? 1 : 0)));
    const r = await check(sky, 0, 0, 0.07, 1, 0, 1);
    console.log(
      `  halves: map up ${r.values[4].toFixed(3)}, down ${r.values[5].toFixed(3)}, rough ${r.values[6].toFixed(3)}`,
    );
    expect(r.counts.slice(2, 5)).toEqual([0, 0, 0]);
    sky.destroy();
  });

  it("a sun is found, taken out of the harmonics and handed over as a light", async () => {
    const W = 256;
    const H = 128;
    const kSkyGrey = 0.02;
    const kSunValue = 20;
    const { sky, axis, power } = await prepare(
      skyImage(W, H, (x, y) =>
        x >= 127 && x <= 129 && y >= 63 && y <= 65 ? grey(1) : grey(kSkyGrey),
      ),
      kSunValue,
    );
    const cone = axis[3];
    const solidAngle = 2 * PI * (1 - Math.cos(cone));
    const irradiance = power[1];
    const texel = (2 * PI * PI) / (W * H);
    const disc = 9 * texel * kSunValue;
    const rest = Math.max(solidAngle - 9 * texel, 0) * kSkyGrey * kSunValue;
    console.log(
      `  sun: direction ${axis.slice(0, 3).map((v) => v.toFixed(3))}, cone ${cone.toFixed(4)}, irradiance ${irradiance.toFixed(4)}, closed form ${(disc + rest).toFixed(4)}`,
    );
    expect(cone).toBeGreaterThan(0);
    expect(cone).toBeLessThan(0.08);
    expect(
      Math.abs(Math.hypot(axis[0], axis[1], axis[2]) ** 2 - 1),
    ).toBeLessThan(1e-3);
    expect(Math.abs(irradiance - (disc + rest))).toBeLessThan(
      0.1 * (disc + rest),
    );
    const residual = kSkyGrey * kSunValue;
    const r = await check(sky, 64, PI * residual, 0.12, -1, -1, -1);
    console.log(
      `  residual sky: irradiance ${r.values[1].toFixed(4)} (want ${(PI * residual).toFixed(4)}), worst ${(r.values[0] * 100).toFixed(2)}%`,
    );
    expect(r.counts[0]).toBe(0);
    sky.destroy();
  });

  it("an even sky has no sun taken out of it", async () => {
    const { sky, axis } = await prepare(
      skyImage(128, 64, (_, y) => (200 - y * 2) / 255),
    );
    expect(axis[3]).toBe(0);
    sky.destroy();
  });

  it("a compact bright source prefilters to one lobe and keeps the sky's light", async () => {
    const image = skyImage(512, 256, (x, y) =>
      x >= 150 && x < 152 && y >= 160 && y < 162 ? 20 : 1,
    );
    const { sky } = await prepare(image);
    for (const level of [4, 5]) {
      const r = await run(sky, "environmentPeaks", { level });
      console.log(
        `  level ${level}: ${r.counts[0]} peaks (brightest ${r.values[0].toFixed(2)})`,
      );
      expect(r.counts[0]).toBeGreaterThanOrEqual(1);
      expect(r.counts[0]).toBeLessThanOrEqual(2);
    }
    const source = sourceIntegral(image);
    for (const level of [2, 3, 4, 5, 6]) {
      const r = await run(sky, "environmentEnergy", { level });
      console.log(
        `  level ${level} holds ${(r.values[0] / source).toFixed(4)} of the source`,
      );
      expect(Math.abs(r.values[0] - source)).toBeLessThan(0.02 * source);
    }
    for (const readRoughness of [
      0.0816, 0.1837, 0.3265, 0.3421, 0.5102, 0.7347,
    ]) {
      const r = await run(sky, "environmentReadEnergy", {
        count: 262144,
        readRoughness,
      });
      console.log(
        `  read at ${readRoughness}: ${(r.values[0] / source).toFixed(4)} of the source`,
      );
      expect(Math.abs(r.values[0] - source)).toBeLessThan(0.02 * source);
    }
    sky.destroy();
  });

  it("a sun is taken out of the prefiltered sky", async () => {
    const kSky = 0.02;
    const image = skyImage(512, 256, (x, y) =>
      x >= 150 && x < 152 && y >= 160 && y < 162 ? 30000 : kSky,
    );
    const { sky, axis, power } = await prepare(image);
    expect(axis[3]).toBeGreaterThan(0);
    const source = sourceIntegral(image);
    const residual = source - power[1];
    for (const level of [2, 3, 4, 5, 6]) {
      const held = (await run(sky, "environmentEnergy", { level })).values[0];
      const brightest = (await run(sky, "environmentPeaks", { level }))
        .values[0];
      console.log(
        `  level ${level} holds ${held.toFixed(4)} (sky without its sun ${residual.toFixed(4)}), brightest ${brightest.toFixed(3)}`,
      );
      expect(Math.abs(held - residual)).toBeLessThanOrEqual(0.02 * source);
      expect(brightest).toBeLessThan(50 * kSky);
    }
    sky.destroy();
  });

  it("a scatter of lamps is not taken out as a sun", async () => {
    const { sky, axis } = await prepare(
      skyImage(512, 256, (x, y) =>
        y === 128 && (x === 250 || x === 257 || x === 264) ? 3000 : 0.05,
      ),
    );
    expect(axis[3]).toBe(0);
    sky.destroy();
  });
});

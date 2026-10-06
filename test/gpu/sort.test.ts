// GpuSorter against the CPU sort (sortBackToFront) on the same metrics.

import { describe, expect, it } from "vitest";
import { GpuSorter } from "../../src/webgpu/GpuSorter";
import { KernelRegistry } from "../../src/webgpu/KernelRegistry";
import { sortBackToFront } from "../../src/webgpu/WgpuSplatRenderer";
import { device, readBack, storage } from "./device";

// sparse: most splats inactive, as after generate's culling.
function metrics(n: number, seed: number, sparse = false) {
  const m = new Float32Array(n);
  let s = seed;
  for (let i = 0; i < n; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    const r = s / 2 ** 32;
    // Some inactive splats, some exact ties, a wide range of magnitudes.
    m[i] = (sparse ? i % 9 !== 0 : i % 53 === 0)
      ? Number.POSITIVE_INFINITY
      : i % 7 === 0
        ? 2.5
        : 0.001 + r * r * 5000;
  }
  return m;
}

// The plain scatter, and the subgroup one where the device has subgroups.
const scatters = device?.features.has("subgroups") ? [false, true] : [false];

describe.skipIf(!device).each(scatters)("GpuSorter (subgroups: %s)", (sg) => {
  const registry = new KernelRegistry(device as GPUDevice);
  const sorter = new GpuSorter(registry, "sort", sg);

  it.each([
    [1, false],
    [1000, false],
    [100003, false],
    [1_200_000, false],
    [300_001, true],
    [5, true],
  ])("matches the CPU order for %i splats (sparse: %s)", async (n, sparse) => {
    const m = metrics(n, n, sparse);
    const d = device as GPUDevice;
    const enc = d.createCommandEncoder();
    const pass = enc.beginComputePass();
    sorter.encode(pass, storage(m), n);
    pass.end();
    registry.submit(enc.finish());

    const expected = sortBackToFront(m, n);
    const args = new Uint32Array(await readBack(sorter.drawArgs));
    expect(args[0]).toBe(4);
    expect(args[1]).toBe(expected.length);
    const got = new Uint32Array(await readBack(sorter.ordering)).subarray(
      0,
      expected.length,
    );
    let firstBad = -1;
    for (let i = 0; i < expected.length; i++) {
      if (got[i] !== expected[i]) {
        firstBad = i;
        break;
      }
    }
    expect(firstBad).toBe(-1);
  });
});

describe.skipIf(!device)("GpuSorter with fewer key bits", () => {
  const registry = new KernelRegistry(device as GPUDevice);
  const sorter = new GpuSorter(registry);

  it.each([16, 24] as const)(
    "sorts stably on the top %i bits",
    async (bits) => {
      const n = 300_000;
      const m = metrics(n, 77);
      const d = device as GPUDevice;
      const enc = d.createCommandEncoder();
      const pass = enc.beginComputePass();
      sorter.encode(pass, storage(m), n, bits);
      pass.end();
      registry.submit(enc.finish());

      // CPU reference: finite metrics, stable by the truncated inverted key.
      const u = new Uint32Array(m.buffer);
      const idx = Array.from({ length: n }, (_, i) => i).filter((i) =>
        Number.isFinite(m[i]),
      );
      const key = (i: number) =>
        Math.min(~u[i] >>> 0, 0xfffffffe) >>> (32 - bits);
      idx.sort((a, b) => key(a) - key(b) || a - b);

      const got = new Uint32Array(await readBack(sorter.ordering)).subarray(
        0,
        idx.length,
      );
      expect(new Uint32Array(await readBack(sorter.drawArgs))[1]).toBe(
        idx.length,
      );
      let firstBad = -1;
      for (let i = 0; i < idx.length; i++) {
        if (got[i] !== idx[i]) {
          firstBad = i;
          break;
        }
      }
      expect(firstBad).toBe(-1);
    },
  );
});

describe.skipIf(!device)("GpuSorter with depth-range keys", () => {
  const registry = new KernelRegistry(device as GPUDevice);
  const sorter = new GpuSorter(registry);

  it("sorts by the place in the range on 24 bits, stably, clamping outside", async () => {
    const n = 300_001;
    const lo = 100.05;
    const hi = 300;
    const m = new Float32Array(n);
    let s = 5;
    for (let i = 0; i < n; i++) {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      const r = s / 2 ** 32;
      m[i] =
        i % 61 === 0
          ? Number.POSITIVE_INFINITY
          : i % 97 === 0
            ? lo - 1 // below the range: the nearest
            : i % 89 === 0
              ? hi + 3 // above: the farthest
              : i % 7 === 0
                ? 150 // exact ties
                : lo + r * (hi - lo);
    }
    const d = device as GPUDevice;
    const enc = d.createCommandEncoder();
    const pass = enc.beginComputePass();
    sorter.encode(pass, storage(m), n, 32, { lo, hi });
    pass.end();
    registry.submit(enc.finish());

    // CPU reference with the kernel's f32 maths.
    const keys = 16777215;
    const scale = Math.fround(keys / (hi - lo));
    const flo = Math.fround(lo);
    const key = (i: number) => {
      const t = Math.min(
        Math.max(Math.fround(Math.fround(m[i] - flo) * scale), 0),
        keys,
      );
      return keys - Math.floor(t);
    };
    const idx = Array.from({ length: n }, (_, i) => i).filter((i) =>
      Number.isFinite(m[i]),
    );
    idx.sort((a, b) => key(a) - key(b) || a - b);
    expect(new Uint32Array(await readBack(sorter.drawArgs))[1]).toBe(
      idx.length,
    );
    const got = new Uint32Array(await readBack(sorter.ordering)).subarray(
      0,
      idx.length,
    );
    let firstBad = -1;
    for (let i = 0; i < idx.length; i++) {
      if (got[i] !== idx[i]) {
        firstBad = i;
        break;
      }
    }
    expect(firstBad).toBe(-1);
    // Back to front: never a nearer metric before a farther one by more
    // than a key step.
    const step = (hi - lo) / keys;
    for (let i = 1; i < idx.length; i++) {
      const a = Math.min(Math.max(m[got[i - 1]], lo), hi);
      const b = Math.min(Math.max(m[got[i]], lo), hi);
      if (b > a + 2 * step) throw new Error(`order at ${i}: ${a} then ${b}`);
    }
  });
});

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

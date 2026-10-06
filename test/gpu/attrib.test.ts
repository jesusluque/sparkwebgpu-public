// slang/core/attrib.slang against the TS packer in
// src/webgpu/attributes/schema.ts: every format, odd component counts.

import { describe, expect, it } from "vitest";
import { KernelRegistry } from "../../src/webgpu/KernelRegistry";
import { PagedAttribPool } from "../../src/webgpu/attributes/PagedAttribPool";
import { planAttribPaging } from "../../src/webgpu/attributes/attribPaging";
import {
  type AttribFormat,
  AttribPool,
  columnBits,
  decodeComponent,
} from "../../src/webgpu/attributes/schema";
import attribTest from "../../src/webgpu/generated/tests/attrib";
import { UniformWriter } from "../../src/webgpu/uniforms";
import { device, readBack, storage } from "./device";

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

const N = 333;
const r = rng(99);

// Values each format represents (or rounds to) exactly.
function sample(format: AttribFormat): number {
  switch (format) {
    case "f32":
      return (r() - 0.5) * 1e4;
    case "f16":
      return (r() - 0.5) * 100;
    case "unorm8":
      return r();
    case "snorm8":
      return r() * 2 - 1;
    case "u8":
      return Math.floor(r() * 256);
    case "u16":
      return Math.floor(r() * 65536);
    case "u32":
      return Math.floor(r() * 2 ** 24); // exact through float re-encoding
  }
}

const cases: [AttribFormat, number][] = [
  ["f32", 3],
  ["f16", 8],
  ["f16", 3],
  ["unorm8", 1],
  ["snorm8", 3],
  ["u8", 1],
  ["u8", 6],
  ["u16", 5],
  ["u32", 2],
  ["f32", 1],
];

const pool = new AttribPool(N);
cases.forEach(([format, components], k) => {
  const data = Array.from({ length: N * components }, () => sample(format));
  pool.setAttribute(`a${k}`, data, format, components);
});
const { words, layout } = pool.pack();
const maxComps = Math.max(...cases.map((c) => c[1]));
const maxGroups = Math.ceil(maxComps / 4);
const A = cases.length;

describe.skipIf(!device)("attrib.slang", () => {
  const registry = new KernelRegistry(device as GPUDevice);
  const params = UniformWriter.for(attribTest).setAll({
    count: N,
    numAttribs: A,
    maxGroups,
    maxComps,
  });

  it("decodes every format as the CPU does, and reads raw bits", async () => {
    const decoded = storage(N * A * maxGroups * 16);
    const raw = storage(N * A * maxComps * 4);
    registry.get(attribTest, "attribDecode").run({
      grid: [N],
      buffers: { pool: storage(words), decoded, raw },
      uniforms: params.data,
    });
    const dec = new Float32Array(await readBack(decoded));
    const bits = new Uint32Array(await readBack(raw));
    pool.columns.forEach((col, a) => {
      const { format, components } = col.spec;
      for (let i = 0; i < N; i++) {
        for (let c = 0; c < maxComps; c++) {
          const want = c < components ? columnBits(col, i, c) : 0;
          expect(bits[(i * A + a) * maxComps + c]).toBe(want);
        }
        for (let c = 0; c < 4 * maxGroups; c++) {
          const got = dec[((i * A + a) * maxGroups + (c >> 2)) * 4 + (c & 3)];
          const want =
            c < components ? decodeComponent(format, columnBits(col, i, c)) : 0;
          if (format === "f32") expect(got).toBe(Math.fround(want));
          else expect(got).toBeCloseTo(want, 5);
        }
      }
    });
  });

  it("re-encodes to the same words, padding and neighbours intact", async () => {
    const rewritten = new Uint32Array(words.length);
    rewritten.set(words.subarray(0, layout.headerWords));
    const out = storage(rewritten);
    registry.get(attribTest, "attribEncode").run({
      grid: [N],
      buffers: { pool: storage(words), rewritten: out },
      uniforms: params.data,
    });
    const got = new Uint32Array(await readBack(out)).subarray(0, words.length);
    expect(Array.from(got)).toEqual(Array.from(words));
  });

  it("reads a paged attribute where its page is resident, zeros elsewhere", async () => {
    const d = device as GPUDevice;
    const PAGE = 65536;
    // Every attribute in one group of one page, over a pool of two.
    const plan = planAttribPaging(pool.schema, 2, {
      budget: 1e9,
      tier: 3,
      pages: { core: 1 },
    });
    expect(plan.groups.map((g) => [g.slots, g.paged])).toEqual([[1, true]]);
    const paged = new PagedAttribPool(d, plan.specs, 2 * PAGE, plan);
    const group = paged.groups[0];
    paged.assign(group, 1, 0, N, pool);
    const count = 2 * PAGE;
    const decoded = storage(count * A * maxGroups * 16);
    const raw = storage(count * A * maxComps * 4);
    registry.get(attribTest, "attribDecode").run({
      grid: [count],
      buffers: { pool: paged.buffer, decoded, raw },
      uniforms: UniformWriter.for(attribTest).setAll({
        count,
        numAttribs: A,
        maxGroups,
        maxComps,
      }).data,
    });
    const dec = new Float32Array(await readBack(decoded));
    const bits = new Uint32Array(await readBack(raw));
    pool.columns.forEach((col, a) => {
      const { format, components } = col.spec;
      for (let i = 0; i < N; i += 5) {
        for (let c = 0; c < components; c++) {
          // Page 1 holds the splats; page 0 is not resident.
          const at = ((PAGE + i) * A + a) * maxComps + c;
          expect(bits[at]).toBe(columnBits(col, i, c));
          expect(bits[(i * A + a) * maxComps + c]).toBe(0);
          const got =
            dec[(((PAGE + i) * A + a) * maxGroups + (c >> 2)) * 4 + (c & 3)];
          const want = decodeComponent(format, columnBits(col, i, c));
          if (format === "f32") expect(got).toBe(Math.fround(want));
          else expect(got).toBeCloseTo(want, 5);
        }
      }
    });
    // Evicted, the page reads as zeros too.
    paged.evict(group, 1);
    registry.get(attribTest, "attribDecode").run({
      grid: [count],
      buffers: { pool: paged.buffer, decoded, raw },
      uniforms: UniformWriter.for(attribTest).setAll({
        count,
        numAttribs: A,
        maxGroups,
        maxComps,
      }).data,
    });
    const after = new Uint32Array(await readBack(raw));
    expect(
      after
        .subarray(PAGE * A * maxComps, (PAGE + N) * A * maxComps)
        .every((w) => w === 0),
    ).toBe(true);
    decoded.destroy();
    raw.destroy();
    paged.destroy();
  });
});

// The synthetic light sidecar (rust/build-lod athenea-lights, in
// examples/webgpu/athenea-lights/) through the real WASM reader: the three
// files agree, the layers are sparse, one page's layers read alone from its
// byte range, and an AthlPager holds the chunks it is asked for.

import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";

// test/unit/setup.ts stubs the wasm package; these tests need the real one,
// initialised here (src/wasm.ts would compile the stub's empty module).
vi.mock("spark-rs", async () => await vi.importActual("spark-rs"));
vi.mock("../../src/wasm", () => ({
  initialization: Promise.resolve(),
  isInitialized: () => true,
}));

const wasm = await vi.importActual<typeof import("spark-rs")>("spark-rs");
wasm.initSync({
  module: readFileSync(
    new URL("../../rust/spark-rs/pkg/spark_rs_bg.wasm", import.meta.url),
  ),
});

const {
  AthlStore,
  athcCloudHash,
  athlChunkRange,
  decodeAthl,
  openAthl,
  readAthlHeader,
  decodeAthlChunk,
} = await import("../../src/webgpu/athenea/athl");
const { parseLightSidecar } = await import(
  "../../src/webgpu/athenea/lightSidecar"
);

const file = (n: string) =>
  new Uint8Array(
    readFileSync(
      new URL(`../../examples/webgpu/athenea-lights/${n}`, import.meta.url),
    ),
  );
const athc = file("car.athc");
const bytes = file("car.lights.athl");
const sidecar = parseLightSidecar(
  new TextDecoder().decode(file("car.lights.usda")),
);
const fetchRange = async (start: number, end: number) => {
  ranges.push([start, end]);
  return bytes.slice(start, Math.min(end, bytes.length));
};
let ranges: [number, number][] = [];

describe("the synthetic light sidecar", () => {
  it("decodes, matches its cloud and its sidecar, and is sparse", async () => {
    const data = await decodeAthl(bytes);
    expect(data.groups.map((g) => g.name)).toEqual(
      sidecar.groups.map((g) => g.name),
    );
    expect(sidecar.warnings).toEqual([]);
    expect(data.header.cloudHash).toBe(athcCloudHash(athc));
    expect(sidecar.cloudHash).toBe(data.header.cloudHash);
    const { merged, splatBase, splatCount, elementCount } = data.header;
    expect(splatBase % 65536).toBe(0);
    expect(elementCount).toBe(splatBase + splatCount);
    expect(merged).toBeGreaterThan(0);
    // Indirect layers reach some blocks of each chunk, not all.
    const blocks = Math.ceil(merged / 256) + Math.ceil(splatCount / 256);
    for (let g = 0; g < 3; g++) {
      const n = data.layers
        .filter((l) => l.group === g && l.kind === 0)
        .reduce((a, l) => a + l.blocks.length, 0);
      expect(n).toBeGreaterThan(0);
      expect(n).toBeLessThan(blocks);
    }
    const packed = new AthlStore(data).pack();
    expect(packed[0]).toBe(3);
    expect(packed[1]).toBe(Math.ceil(elementCount / 65536));
  });

  it("reads one page's layers from its byte range", async () => {
    const data = await decodeAthl(bytes);
    const header = await readAthlHeader(fetchRange);
    const chunk = Math.floor(header.splatBase / 65536);
    const range = athlChunkRange(header, chunk) as [number, number];
    const layers = await decodeAthlChunk(
      header,
      chunk,
      bytes.slice(range[0], range[1]),
      range[0],
    );
    expect(layers.length).toBeGreaterThan(0);
    for (const l of layers) {
      const whole = data.layers.find(
        (x) => x.chunk === chunk && x.group === l.group && x.kind === l.kind,
      );
      expect(whole && [...whole.blocks]).toEqual([...l.blocks]);
      expect(whole && [...whole.data]).toEqual([...l.data]);
    }
  });

  it("pages: holds the chunks asked for, one range each, and drops the rest", async () => {
    ranges = [];
    const pager = await openAthl(fetchRange);
    expect(pager.store.layers()).toEqual([]);
    expect(pager.store.groups.length).toBe(3);
    const metaEnd = Math.max(...ranges.map((r) => r[1]));
    const splatChunk = pager.header.splatBase / 65536;
    ranges = [];
    pager.update([splatChunk]);
    await pager.settled();
    expect(pager.store.heldChunks).toEqual([splatChunk]);
    expect(ranges).toEqual([athlChunkRange(pager.header, splatChunk)]);
    expect(ranges[0][0]).toBeGreaterThanOrEqual(metaEnd);
    const one = pager.store.pack().length;
    pager.update([0, splatChunk]);
    await pager.settled();
    expect(pager.store.heldChunks).toEqual([0, splatChunk]);
    expect(pager.store.pack().length).toBeGreaterThan(one);
    // The whole file's packing, once every chunk is held.
    const whole = new AthlStore(await decodeAthl(bytes)).pack();
    expect([...pager.store.pack()]).toEqual([...whole]);
    pager.update([0]);
    expect(pager.store.heldChunks).toEqual([0]);
  });
});

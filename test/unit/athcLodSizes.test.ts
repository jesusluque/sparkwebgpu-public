// LoD sizes by error (rust athc_lod_error.rs, the v3 section LODS) through
// the browser's paths, on the real WASM: a whole file, and a paged one
// (src/athc.ts openAthc: the levels' sections, the loader worker's merged
// pages, a chunk page). The decoder writes each merged node's size in place
// of its geometric one in the LoD tree (Spark's traversal reads it there),
// and a splat keeps its own.

import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";

vi.mock("spark-rs", async () => {
  const { readFileSync } = await import("node:fs");
  const actual = await vi.importActual<Record<string, unknown>>("spark-rs");
  (actual.initSync as (o: { module: Buffer }) => void)({
    module: readFileSync(
      new URL("../../rust/spark-rs/pkg/spark_rs_bg.wasm", import.meta.url),
    ),
  });
  return { ...actual, default: async () => ({}) };
});

vi.mock("../../src/SplatWorker", async () => {
  const wasm = (await import("spark-rs")) as unknown as Record<
    string,
    // biome-ignore lint/suspicious/noExplicitAny: the WASM bindings
    (...args: any[]) => any
  >;
  const handlers: Record<string, (args: Record<string, unknown>) => unknown> = {
    athc3MergedPages: ({ tables, levels }) => {
      const { pages, headers, ...tree } = wasm.athc3_merged_pages(
        tables,
        levels,
      );
      return { tree, pages, headers };
    },
  };
  const worker = {
    call: async (name: string, args: Record<string, unknown>) =>
      handlers[name](args),
  };
  return {
    SplatWorker: class {},
    workerPool: {
      withWorker: async <T>(fn: (w: typeof worker) => Promise<T>) => fn(worker),
    },
  };
});

const { fetchAthcPage, openAthc } = await import("../../src/athc");
const wasm = (await import("spark-rs")) as unknown as {
  decode_to_packedsplats: (...a: unknown[]) => {
    push(b: Uint8Array): void;
    finish(): { numSplats: number; lodTree: Uint32Array };
  };
};

const fixture = (name: string) =>
  new Uint8Array(
    readFileSync(new URL(`../fixtures/athc/${name}`, import.meta.url)),
  );
// two_cards.athc as v3 with its levels' coverage and LoD sizes:
// athc-convert two_cards.athc two_cards.lods.athc --coverage --lod-sizes --gzip --planes
const SIZED = fixture("two_cards.lods.athc");
const PLAIN = fixture("two_cards.athc");

function decode(bytes: Uint8Array) {
  const d = wasm.decode_to_packedsplats(
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
  );
  d.push(bytes);
  return d.finish();
}

// A LoD node's size: the high f16 of its second word.
const half = new DataView(new ArrayBuffer(2));
function size(tree: Uint32Array, k: number) {
  half.setUint16(0, tree[k * 4 + 1] >>> 16, true);
  const bits = half.getUint16(0, true);
  const e = (bits >> 10) & 31;
  const m = bits & 1023;
  return e === 0 ? m * 2 ** -24 : (1 + m / 1024) * 2 ** (e - 15);
}

describe(".athc LoD sizes by error (LODS)", () => {
  const plain = decode(PLAIN);
  const sized = decode(SIZED);
  const merged = 715 + 1; // two_cards' levels, and the root merged from level 1

  it("replace the merged nodes' geometric sizes in a whole file", () => {
    expect(sized.numSplats).toBe(plain.numSplats);
    let larger = 0;
    for (let k = 0; k < merged; ++k) {
      // Children as before; the size at least the geometric one.
      expect(sized.lodTree[k * 4 + 2]).toBe(plain.lodTree[k * 4 + 2]);
      expect(sized.lodTree[k * 4 + 3]).toBe(plain.lodTree[k * 4 + 3]);
      if (k > 0) {
        expect(size(sized.lodTree, k)).toBeGreaterThanOrEqual(
          size(plain.lodTree, k) * 0.99,
        );
      }
      if (size(sized.lodTree, k) > size(plain.lodTree, k) * 1.01) ++larger;
    }
    expect(larger).toBeGreaterThan(0);
    // The splats keep their own sizes.
    const from = merged * 4;
    expect(
      Array.from(sized.lodTree.subarray(from, from + 4000)).map(
        (_, i) => sized.lodTree[from + i] >>> 16,
      ),
    ).toEqual(
      Array.from(plain.lodTree.subarray(from, from + 4000)).map(
        (_, i) => plain.lodTree[from + i] >>> 16,
      ),
    );
  });

  it("travel in the paged merged pages, and not in a chunk's", async () => {
    const paging = await openAthc({ fileBytes: SIZED, keepLinear: true });
    expect(paging.tree.merged).toBe(merged);
    const page = await fetchAthcPage(paging, 0, { fileBytes: SIZED });
    const view = new DataView(page.buffer, page.byteOffset);
    // Keep-linear and LoD sizes, both.
    expect(view.getUint32(152, true)).toBe(3);
    const m = decode(page);
    expect(m.numSplats).toBe(merged);
    for (let k = 0; k < merged; ++k) {
      expect(size(m.lodTree, k)).toBe(size(sized.lodTree, k));
    }
    const first = paging.tree.splatBase / 65536;
    const chunk = decode(
      await fetchAthcPage(paging, first, { fileBytes: SIZED }),
    );
    for (let i = 0; i < 100; ++i) {
      expect(chunk.lodTree[i * 4 + 1] >>> 16).toBe(
        sized.lodTree[(merged + i) * 4 + 1] >>> 16,
      );
    }
  });
});

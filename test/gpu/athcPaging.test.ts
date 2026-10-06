// Paging a .athc's streams by group (attributes/attribPaging.ts): the real
// PagedSplats and WgpuSplatPager, with the loader worker run in-process on
// the real WASM. every_stream (SH3, TX transfer of 112 values, 16x16
// shadow bits, material) is paged in two pages (its merged nodes, its
// splats) with one page of relight streams: the transfer and shadow bits
// are resident where the traversal ranks first and fall back elsewhere,
// read through slang/athenea_adapter as a relight kernel reads them.

import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";
import { PagedSplats } from "../../src/PagedSplats";
import { KernelRegistry } from "../../src/webgpu/KernelRegistry";
import { WgpuSplatPager } from "../../src/webgpu/WgpuSplatPager";
import { specsFromRadMeta } from "../../src/webgpu/attributes/PagedAttribPool";
import { ATTRIB_NONE, AttribPool } from "../../src/webgpu/attributes/schema";
import athcTest from "../../src/webgpu/generated/tests/athc_adapter";
import { UniformWriter } from "../../src/webgpu/uniforms";
import { device, readBack, storage } from "./device";

// The real WASM (test/unit/setup.ts stubs it), initialized up front.
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

// The loader worker's handlers, in-process (src/worker.ts does the same).
vi.mock("../../src/SplatWorker", async () => {
  const wasm = (await import("spark-rs")) as unknown as Record<
    string,
    // biome-ignore lint/suspicious/noExplicitAny: the WASM bindings
    (...args: any[]) => any
  >;
  const handlers: Record<string, (args: Record<string, unknown>) => unknown> = {
    athcMergedPages: ({ prefix, fileBytes }) => {
      const { pages, ...tree } = wasm.athc_merged_pages(prefix, fileBytes);
      return { tree, pages };
    },
    athc3MergedPages: ({ tables, levels }) => {
      const { pages, headers, ...tree } = wasm.athc3_merged_pages(
        tables,
        levels,
      );
      return { tree, pages, headers };
    },
    loadPackedSplats: ({
      fileBytes,
      pathName,
      sh1Codes,
      sh2Codes,
      sh3Codes,
    }) => {
      const decoder = wasm.decode_to_packedsplats(
        undefined,
        pathName || undefined,
        undefined,
        sh1Codes,
        sh2Codes,
        sh3Codes,
      );
      decoder.push(fileBytes);
      const r = decoder.finish();
      return {
        lodSplats: {
          numSplats: r.numSplats,
          packedArray: r.packed,
          extra: {
            sh1: r.sh1,
            sh2: r.sh2,
            sh3: r.sh3,
            sh1Codes: r.sh1Codes,
            sh2Codes: r.sh2Codes,
            sh3Codes: r.sh3Codes,
            lodTree: r.lodTree,
            attribs: r.attribSpecs?.length
              ? {
                  count: r.numSplats,
                  specs: r.attribSpecs,
                  values: r.attribColumns,
                }
              : undefined,
          },
          splatEncoding: r.splatEncoding,
        },
      };
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

const wasm = (await import("spark-rs")) as unknown as {
  decode_to_packedsplats: (...a: unknown[]) => {
    push(b: Uint8Array): void;
    finish(): {
      numSplats: number;
      attribSpecs: { name: string; format: string; components: number }[];
      attribColumns: Float64Array[];
    };
  };
};

const PAGE = 65536;
const fixture = (name: string) =>
  new Uint8Array(
    readFileSync(new URL(`../fixtures/athc/${name}`, import.meta.url)),
  );
const V2 = fixture("every_stream.athc");
const V3 = fixture("every_stream.v3.athc");
const MERGED = 9;
const COUNT = 300;

// The whole file decoded at once: merged nodes 0..8, then the splats.
const whole = (() => {
  const decoder = wasm.decode_to_packedsplats(
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
    undefined,
  );
  decoder.push(V2);
  const r = decoder.finish();
  return AttribPool.fromValues({
    count: r.numSplats,
    specs: r.attribSpecs,
    values: r.attribColumns,
  });
})();

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

describe.skipIf(!device)("a .athc paged by stream group", () => {
  const d = device as GPUDevice;
  const registry = new KernelRegistry(d);

  async function open(
    bytes: Uint8Array,
    attributes: ConstructorParameters<typeof WgpuSplatPager>[1]["attributes"],
  ) {
    const pager = new WgpuSplatPager(d, {
      maxSplats: 2 * PAGE,
      autoDrive: false,
      onUpdate: () => {},
      attributes,
    });
    const splats = new PagedSplats({ fileBytes: bytes, pager });
    const { meta } = await splats.getRadMeta();
    pager.setAttribSchema(
      specsFromRadMeta(
        meta.attributes as Parameters<typeof specsFromRadMeta>[0],
      ),
    );
    return { pager, splats };
  }

  // Fetches what the priority says, uploads it, and lets upgrades land.
  async function settle(pager: WgpuSplatPager) {
    for (let round = 0; round < 3; round++) {
      pager.driveFetchers();
      while (pager.isFetching() || (pager.residency?.pending ?? 0) > 0) {
        await sleep(2);
      }
      pager.consumeLodTreeUpdates();
      pager.processUploads();
    }
    await d.queue.onSubmittedWorkDone();
  }

  // The adapter's view of every element of both pages: 28 floats each.
  async function adapterView(pager: WgpuSplatPager) {
    const pool = pager.attribs?.pool as AttribPool;
    const id = (name: string) => {
      const k = pool.id(name);
      return k < 0 ? ATTRIB_NONE : k;
    };
    const count = 2 * PAGE;
    const params = UniformWriter.for(athcTest).setAll({
      count,
      normalOct: id("normalOct"),
      emission: id("emission"),
      pbr: id("pbr"),
      lobes: id("lobes"),
      transfer: id("transfer"),
      shadowBits: id("shadowBits"),
      transferCount: pool.column("transfer")?.spec.components ?? 0,
      shadowWords: pool.column("shadowBits")?.spec.components ?? 0,
    });
    const out = storage(count * 28 * 4);
    registry.get(athcTest, "athcDecode").run({
      grid: [count],
      buffers: { pool: pool.gpuBuffer as GPUBuffer, decoded: out },
      uniforms: params.data,
    });
    const got = new Float32Array(await readBack(out));
    out.destroy();
    return { got, bits: new Uint32Array(got.buffer) };
  }

  // Checks page `page` (holding whole-file elements first .. first + n)
  // as resident or not for the relight streams.
  function expectPage(
    view: { got: Float32Array; bits: Uint32Array },
    page: number,
    first: number,
    n: number,
    resident: boolean,
    transferValues = 112,
  ) {
    for (let i = 0; i < n; i += 7) {
      const o = (page * PAGE + i) * 28;
      const ref = first + i;
      expect(view.got[o + 26]).toBe(resident ? 1 : 0);
      // The material is interleaved: always there.
      const pbr = whole.getAttribute("pbr", ref)[0];
      expect(view.got[o + 6]).toBeCloseTo((pbr & 0xff) / 255, 6);
      if (resident) {
        const t = whole.getAttribute("transfer", ref);
        expect(view.got[o + 23]).toBe(t[0]);
        expect(view.got[o + 24]).toBe(t[transferValues - 1]);
        expect(view.bits[o + 25]).toBe(
          whole.getAttribute("shadowBits", ref)[0],
        );
      } else {
        // Not resident: no transfer, every direction open.
        expect(view.got[o + 23]).toBe(0);
        expect(view.bits[o + 25]).toBe(0xffffffff);
      }
    }
  }

  it.each([
    ["v3 (sections, gzip)", V3],
    ["v2", V2],
  ])(
    "keeps the relight streams of the first-ranked page, %s",
    async (_, bytes) => {
      const { pager, splats } = await open(bytes, {
        attributes: "all",
        pages: { relight: 1 },
      });
      const plan = pager.attributePlan();
      expect(plan?.groups.map((g) => [g.name, g.slots, g.paged])).toEqual([
        ["core", 2, false],
        ["material", 2, false],
        ["relight", 1, true],
      ]);
      pager.fetchPriority = [
        { splats, chunk: 0 },
        { splats, chunk: 1 },
      ];
      await settle(pager);
      const p0 = pager.getSplatsChunk(splats, 0)?.page as number;
      const p1 = pager.getSplatsChunk(splats, 1)?.page as number;
      expect(new Set([p0, p1])).toEqual(new Set([0, 1]));
      const relight = pager.attribs?.group("relight");
      expect(relight?.resident(p0)).toBe(true);
      expect(relight?.resident(p1)).toBe(false);
      let view = await adapterView(pager);
      expectPage(view, p0, 0, MERGED, true);
      expectPage(view, p1, MERGED, COUNT, false);

      // The splats move first: their page takes the slot (fetched on its own).
      const before = pager.residency?.upgrades ?? 0;
      pager.fetchPriority = [
        { splats, chunk: 1 },
        { splats, chunk: 0 },
      ];
      await settle(pager);
      expect(pager.residency?.upgrades).toBe(before + 1);
      expect(relight?.resident(p1)).toBe(true);
      expect(relight?.resident(p0)).toBe(false);
      view = await adapterView(pager);
      expectPage(view, p1, MERGED, COUNT, true);
      expectPage(view, p0, 0, MERGED, false);
      pager.dispose();
    },
  );

  it("keeps the transfer's direct and indirect halves only when asked", async () => {
    const { pager, splats } = await open(V3, {
      attributes: "all",
      pages: { relight: 2 },
      transferForm: "indirect",
    });
    pager.fetchPriority = [
      { splats, chunk: 0 },
      { splats, chunk: 1 },
    ];
    await settle(pager);
    const pool = pager.attribs?.pool as AttribPool;
    expect(pool.column("transfer")?.spec.components).toBe(64);
    const p0 = pager.getSplatsChunk(splats, 0)?.page as number;
    const p1 = pager.getSplatsChunk(splats, 1)?.page as number;
    const view = await adapterView(pager);
    expectPage(view, p0, 0, MERGED, true, 64);
    expectPage(view, p1, MERGED, COUNT, true, 64);
    pager.dispose();
  });

  it("loads no relight stream no plugin asks for", async () => {
    const { pager, splats } = await open(V3, { attributes: ["emission"] });
    pager.fetchPriority = [
      { splats, chunk: 0 },
      { splats, chunk: 1 },
    ];
    await settle(pager);
    const pool = pager.attribs?.pool as AttribPool;
    expect(pool.schema.map((s) => s.name)).toEqual(["athcGroup", "emission"]);
    expect(pager.residency).toBe(null);
    pager.dispose();
  });

  it("accounts for every byte of the pool", async () => {
    const { pager, splats } = await open(V3, {
      attributes: "all",
      pages: { relight: 1 },
    });
    pager.fetchPriority = [{ splats, chunk: 0 }];
    await settle(pager);
    const attribs = pager.attribs;
    if (!attribs) throw new Error("no attribute pool");
    const mem = attribs.memory();
    expect(mem.total).toBe(attribs.buffer.size);
    expect(mem.total).toBe(pager.attributePlan()?.bytes);
    // Header: 4 + 4 x 7 words. Interleaved: athcGroup 2, normal, emission,
    // pbr 1 each, lobes 3 words, over 2 pages. Relight: per attribute a
    // table of 2 pages and 1 slot of 65 536 records (8 and 56 words).
    const header = (4 + 4 * 7) * 4;
    const interleaved = 2 * PAGE * (2 + 1 + 1 + 1 + 3) * 4;
    const relight = (2 + PAGE * 8) * 4 + (2 + PAGE * 56) * 4;
    expect(mem.interleaved).toBe(interleaved);
    expect(mem.groups).toEqual([
      { name: "relight", slots: 1, resident: 1, bytes: relight },
    ]);
    expect(mem.total).toBe(header + interleaved + relight);
    pager.dispose();
  });
});

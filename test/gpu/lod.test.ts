// LoD on the WebGPU path with real trees: the WASM tiny-LoD build and
// traversal (as the LoD worker runs them), generate's index remap over the
// result, and WgpuSplatPager's page pool addressed by the paged traversal.

import { readFileSync } from "node:fs";
import * as THREE from "three";
import { describe, expect, it, vi } from "vitest";
import type { PagedSplats } from "../../src/PagedSplats";
import { decodeExtSplat, setPackedSplat, unpackSplat } from "../../src/utils";
import { GpuSplatSource } from "../../src/webgpu/GpuSplatSource";
import { KernelRegistry } from "../../src/webgpu/KernelRegistry";
import { WgpuSplatPager } from "../../src/webgpu/WgpuSplatPager";
import { kernelsGenerate } from "../../src/webgpu/generated/constants";
import generate from "../../src/webgpu/generated/kernels/generate";
import { UniformWriter } from "../../src/webgpu/uniforms";
import { device, readBack, storage } from "./device";

const {
  GEN_USE_LOD: USE_LOD,
  GEN_LOD_OPACITY: LOD_OPACITY,
  GEN_OUT_EXT: OUT_EXT,
  GEN_LOD_FADE: LOD_FADE,
} = kernelsGenerate;

// test/unit/setup.ts stubs the wasm package; these tests need the real one.
const wasm = await vi.importActual<typeof import("spark-rs")>("spark-rs");
wasm.initSync({
  module: readFileSync(
    new URL("../../rust/spark-rs/pkg/spark_rs_bg.wasm", import.meta.url),
  ),
});

const PAGE = 65536;

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

// A cloud dense enough for a multi-level tree, small enough for one page.
const N = 20000;
const r = rng(3);
const packed = new Uint32Array(N * 4);
for (let i = 0; i < N; i++) {
  const s = 0.01 + 0.02 * r();
  setPackedSplat(
    packed,
    i,
    r() * 4 - 2,
    r() * 4 - 2,
    r() * 4 - 2,
    s,
    s,
    s,
    0,
    0,
    0,
    1,
    0.3 + 0.6 * r(),
    r(),
    r(),
    r(),
  );
}
const encoding = {
  rgbMin: 0,
  rgbMax: 1,
  lnScaleMin: -12,
  lnScaleMax: 9,
  sh1Max: 1,
  sh2Max: 1,
  sh3Max: 1,
  lodOpacity: false,
};
const lod = wasm.tiny_lod_packedsplats(
  N,
  packed.slice(),
  {},
  1.5,
  false,
  undefined,
  encoding,
) as {
  numSplats: number;
  packed: Uint32Array;
  lodTree: Uint32Array;
  splatEncoding: { lodOpacity: boolean };
};

// Camera 3 units back looking down -z; each tree traversed with a budget
// well below its size so the selection is a real cut through the tree.
function traverse(lodId: number, rootPage: number) {
  const cameraToWorld = new THREE.Matrix4().makeTranslation(0, 0, 3);
  const result = wasm.traverse_lod_trees(
    5000,
    (2 * Math.tan(Math.PI / 6)) / 600,
    undefined,
    new Uint32Array([lodId]),
    new Uint32Array([rootPage]),
    new Float32Array(cameraToWorld.elements),
    new Float32Array([1]),
    new Float32Array([0.2]),
    new Float32Array([0.4]),
    new Float32Array([90]),
    new Float32Array([120]),
  ) as {
    instanceIndices: { numSplats: number; indices: Uint32Array }[];
  };
  const { numSplats, indices } = result.instanceIndices[0];
  return indices.slice(0, numSplats);
}

describe.skipIf(!device)("LoD", () => {
  const d = device as GPUDevice;
  const registry = new KernelRegistry(d);

  async function runWords(
    source: GpuSplatSource,
    indices: Uint32Array,
    fade = false,
  ) {
    const out = storage(indices.length * 32);
    const params = UniformWriter.for(generate).setAll({
      numSplats: indices.length,
      outBase: 0,
      flags:
        USE_LOD |
        OUT_EXT |
        (source.lodOpacity ? LOD_OPACITY : 0) |
        (fade ? LOD_FADE : 0),
      numSh: 0,
      srcCount: source.count,
      rotate: [0, 0, 0, 1],
      translateScale: [0, 0, 0, 1],
      recolor: [1, 1, 1, 1],
      encoding: source.encoding,
      viewDir: [0, 0, -1, 0],
    });
    registry.get(generate, "generate").run({
      grid: [indices.length],
      buffers: {
        src: source.src,
        sh: storage(16),
        lodIndices: storage(indices),
        outSplats: out,
        sortMetric: storage(indices.length * 4),
      },
      uniforms: params.data,
    });
    return new Uint32Array(await readBack(out));
  }

  async function run(source: GpuSplatSource, indices: Uint32Array) {
    const words = await runWords(source, indices);
    return Array.from(
      indices,
      (_, i) => new Float32Array(words.buffer, 32 * i, 3),
    );
  }

  function expectCenters(
    centers: Float32Array[],
    indices: Uint32Array,
    base = 0,
  ) {
    indices.forEach((index, i) => {
      const e = unpackSplat(lod.packed, index - base);
      expect(centers[i][0]).toBeCloseTo(e.center.x, 3);
      expect(centers[i][1]).toBeCloseTo(e.center.y, 3);
      expect(centers[i][2]).toBeCloseTo(e.center.z, 3);
    });
  }

  it("builds a tree whose traversal picks a cut of the LoD splats", () => {
    expect(lod.numSplats).toBeGreaterThan(N);
    expect(lod.numSplats).toBeLessThan(PAGE);
    const { lodId } = wasm.init_lod_tree(
      lod.numSplats,
      lod.lodTree.slice(),
    ) as {
      lodId: number;
    };
    const indices = traverse(lodId, 0);
    wasm.dispose_lod_tree(lodId);
    expect(indices.length).toBeGreaterThan(1000);
    expect(indices.length).toBeLessThanOrEqual(5000);
    expect(Math.max(...indices)).toBeLessThan(lod.numSplats);
    expect(new Set(indices).size).toBe(indices.length);
  });

  it("generates the traversal's splats from the LoD source", async () => {
    const { lodId } = wasm.init_lod_tree(
      lod.numSplats,
      lod.lodTree.slice(),
    ) as {
      lodId: number;
    };
    const indices = traverse(lodId, 0);
    wasm.dispose_lod_tree(lodId);
    const source = GpuSplatSource.fromPacked(d, lod.packed, lod.numSplats, {
      lodOpacity: lod.splatEncoding.lodOpacity,
    });
    expectCenters(await run(source, indices), indices);
    source.destroy();
  });

  it("fades a LoD transition: the index's top byte scales the alpha", async () => {
    const { lodId } = wasm.init_lod_tree(
      lod.numSplats,
      lod.lodTree.slice(),
    ) as {
      lodId: number;
    };
    const indices = traverse(lodId, 0);
    wasm.dispose_lod_tree(lodId);
    const source = GpuSplatSource.fromPacked(d, lod.packed, lod.numSplats, {
      lodOpacity: lod.splatEncoding.lodOpacity,
    });
    // LodFade's encoding: 0 drawn, 255 gone.
    const gone = Array.from(indices, (_, i) => (i * 37) % 256);
    const faded = Uint32Array.from(
      indices,
      (index, i) => (index | (gone[i] << 24)) >>> 0,
    );
    const plain = await runWords(source, indices);
    const got = await runWords(source, faded, true);
    // Without the flag the plain indices give the same splats.
    const a = [
      plain.filter((_, k) => k % 8 < 4),
      plain.filter((_, k) => k % 8 >= 4),
    ];
    const b = [
      got.filter((_, k) => k % 8 < 4),
      got.filter((_, k) => k % 8 >= 4),
    ];
    let checked = 0;
    indices.forEach((_, i) => {
      // decodeExtSplat hands back one object it reuses: copy.
      const pd = decodeExtSplat(a as [Uint32Array, Uint32Array], i);
      const p = { center: pd.center.clone(), opacity: pd.opacity };
      const fd = decodeExtSplat(b as [Uint32Array, Uint32Array], i);
      const f = { center: fd.center.clone(), opacity: fd.opacity };
      expect(f.center.distanceTo(p.center)).toBeLessThan(1e-6);
      if (gone[i] === 255) {
        // Gone: nothing drawn (an inactive splat or alpha 0).
        expect(f.opacity).toBeLessThan(1e-3);
        return;
      }
      expect(f.opacity).toBeCloseTo(p.opacity * (1 - gone[i] / 255), 2);
      checked++;
    });
    expect(checked).toBeGreaterThan(1000);
    // Gone == 0 is the plain splat, bit for bit.
    indices.forEach((_, i) => {
      if (gone[i] !== 0) return;
      for (let k = 0; k < 8; k++) expect(got[8 * i + k]).toBe(plain[8 * i + k]);
    });
    source.destroy();
  });

  it("addresses the page pool through the paged traversal", async () => {
    const pager = new WgpuSplatPager(d, {
      maxSplats: 4 * PAGE,
      onUpdate: () => {},
    });
    // The chunk lands in page 2, as SplatPager.processFetched would place it.
    const page = 2;
    (
      pager as unknown as {
        uploadPage(p: number, a: Uint32Array, sh: Uint32Array[]): void;
      }
    ).uploadPage(page, lod.packed, []);

    const { lodId: poolId } = wasm.new_lod_tree(pager.maxSplats) as {
      lodId: number;
    };
    const { lodId } = wasm.new_shared_lod_tree(poolId) as { lodId: number };
    wasm.update_lod_trees(
      new Uint32Array([lodId]),
      new Uint32Array([page * PAGE]),
      new Uint32Array([0]),
      new Uint32Array([lod.numSplats]),
      [lod.lodTree.slice()],
    );
    const indices = traverse(lodId, page);
    wasm.dispose_lod_tree(lodId);
    wasm.dispose_lod_tree(poolId);

    // Pool indices: page base plus the offset the unpaged tree would give.
    const { lodId: plainId } = wasm.init_lod_tree(
      lod.numSplats,
      lod.lodTree.slice(),
    ) as { lodId: number };
    const plain = traverse(plainId, 0);
    wasm.dispose_lod_tree(plainId);
    expect(Array.from(indices, (i) => i - page * PAGE).sort()).toEqual(
      Array.from(plain).sort(),
    );

    const source = pager.source({
      numSh: 0,
      maxSh: 3,
      splatEncoding: { ...encoding },
    } as unknown as PagedSplats);
    expect(source.count).toBe(4 * PAGE);
    expectCenters(await run(source, indices), indices, page * PAGE);
    pager.dispose();
  });

  it("interleaves ext pages and SH in the pool layout", async () => {
    const pager = new WgpuSplatPager(d, {
      extSplats: true,
      maxSplats: 2 * PAGE,
      onUpdate: () => {},
    });
    const count = 100;
    const fill = (k: number, words: number) =>
      Uint32Array.from({ length: count * words }, (_, i) => k * 1000000 + i);
    const shArrays = [fill(3, 4), fill(4, 4), fill(5, 4), fill(6, 4)];
    (
      pager as unknown as {
        uploadPage(
          p: number,
          a: Uint32Array,
          sh: Uint32Array[],
          b: Uint32Array,
        ): void;
      }
    ).uploadPage(1, fill(1, 4), shArrays, fill(2, 4));
    expect(pager.curSh).toBe(3);
    // The CPU copy of the core pool that SplatMesh.raycast reads.
    const ext1 = pager.packedTexture.value.image.data as Uint32Array;
    const ext2 = pager.extTexture.value.image.data as Uint32Array;
    expect(ext1.length).toBe(pager.maxSplats * 4);
    expect(ext1[(PAGE + 57) * 4 + 1]).toBe(1000000 + 4 * 57 + 1);
    expect(ext2[(PAGE + 57) * 4 + 2]).toBe(2000000 + 4 * 57 + 2);
    const core = new Uint32Array(
      await readBack(pager.pools.core.buffer as GPUBuffer),
    );
    const sh = new Uint32Array(
      await readBack(pager.pools.sh.buffer as GPUBuffer),
    );
    for (const i of [0, 57, count - 1]) {
      const c = (PAGE + i) * 8;
      expect(core[c + 1]).toBe(1000000 + 4 * i + 1);
      expect(core[c + 6]).toBe(2000000 + 4 * i + 2);
      const s = (PAGE + i) * 16;
      for (let k = 0; k < 4; k++) {
        expect(sh[s + 4 * k + 3]).toBe((3 + k) * 1000000 + 4 * i + 3);
      }
    }
    pager.dispose();
  });
});

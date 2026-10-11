// src/worker.ts's LoD calls in-process (the compact streamer's LodCall), for
// the tests that run the WASM traversal without a worker.

import type { LodCall } from "../../src/webgpu/athenea/compactPages";

// biome-ignore lint/suspicious/noExplicitAny: the WASM bindings
type Wasm = Record<string, any>;

export interface LodInstanceArgs {
  lodId: number;
  rootPage: number;
  viewToObjectCols: number[];
  lodScale: number;
  behindFoveate: number;
  coneFov0: number;
  coneFov: number;
  coneFoveate: number;
}

export function makeLodCall(wasm: Wasm): LodCall {
  return async (name, a) => {
    if (name === "newLodTree") {
      return wasm.new_lod_tree(
        a.capacity as number,
        a.slot as number,
        a.compact as boolean | undefined,
      );
    }
    if (name === "newSharedLodTree") {
      return wasm.new_shared_lod_tree(a.lodId as number);
    }
    if (name === "updateLodTrees") {
      const r = a.ranges as {
        lodId: number;
        pageBase: number;
        chunkBase: number;
        count: number;
        lodTreeData?: Uint32Array;
        lodErrorData?: Uint32Array;
      }[];
      wasm.update_lod_trees(
        new Uint32Array(r.map((x) => x.lodId)),
        new Uint32Array(r.map((x) => x.pageBase)),
        new Uint32Array(r.map((x) => x.chunkBase)),
        new Uint32Array(r.map((x) => x.count)),
        r.map((x) => x.lodTreeData),
        r.some((x) => x.lodErrorData)
          ? r.map((x) => x.lodErrorData)
          : undefined,
        undefined,
      );
      return {};
    }
    if (name === "updateCompactLodTrees") {
      const r = a.ranges as {
        lodId: number;
        pageBase: number;
        chunkBase: number;
        count: number;
        input?: Uint8Array;
        box?: ArrayLike<number>;
      }[];
      const boxes = new Float32Array(6 * r.length);
      r.forEach((x, k) => {
        if (x.box) boxes.set(Array.from(x.box).slice(0, 6), 6 * k);
      });
      wasm.update_compact_lod_trees(
        new Uint32Array(r.map((x) => x.lodId)),
        new Uint32Array(r.map((x) => x.pageBase)),
        new Uint32Array(r.map((x) => x.chunkBase)),
        new Uint32Array(r.map((x) => x.count)),
        r.map((x) => x.input),
        boxes,
      );
      return {};
    }
    const inst = Object.entries(a.instances as Record<string, LodInstanceArgs>);
    const f = (k: (i: LodInstanceArgs) => number) =>
      new Float32Array(inst.map(([, i]) => k(i)));
    const result = wasm.traverse_lod_trees(
      a.maxSplats as number,
      a.pixelScaleLimit as number,
      undefined,
      new Uint32Array(inst.map(([, i]) => i.lodId)),
      new Uint32Array(inst.map(([, i]) => i.rootPage)),
      new Float32Array(inst.flatMap(([, i]) => i.viewToObjectCols)),
      f((i) => i.lodScale),
      f((i) => i.behindFoveate),
      f((i) => i.coneFoveate),
      f((i) => i.coneFov0),
      f((i) => i.coneFov),
      a.errorLimit as number | undefined,
      undefined,
      undefined,
    ) as {
      instanceIndices: { numSplats: number; indices: Uint32Array }[];
    };
    const keyIndices: Record<
      string,
      { numSplats: number; indices: Uint32Array }
    > = {};
    inst.forEach(([key], k) => {
      keyIndices[key] = result.instanceIndices[k];
    });
    return { keyIndices };
  };
}

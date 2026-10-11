// Splat sources on the GPU, in the interleaved layout
// slang/kernels/generate.slang reads.

import type { ExtSplats } from "../ExtSplats";
import type { PackedSplats } from "../PackedSplats";
import {
  type AttribFormat,
  AttribPool,
  type AttributeSpec,
} from "./attributes/schema";
import { upload } from "./gpuBuffers";

/** Splats on the GPU in the layout kernels/generate.slang reads. */
export class GpuSplatSource {
  /** Extra per-Gaussian attributes (src/webgpu/attributes), if any. */
  attribs: AttribPool | null = null;
  /**
   * Bumped by setAttribute, so renderers regenerate. Changes made straight
   * on the AttribPool need renderer.markDirty() instead.
   */
  version = 0;
  /**
   * uint4s per splat in `sh`; 0 for all degrees (3 packed, 4 ext). Pools
   * sized for fewer degrees (WgpuSplatPager) set it.
   */
  shStride = 0;

  constructor(
    readonly format: "packed" | "ext",
    readonly count: number,
    readonly src: GPUBuffer,
    readonly sh: GPUBuffer | null = null,
    readonly numSh = 0,
    readonly shMax: [number, number, number] = [1, 1, 1],
    readonly encoding: [number, number, number, number] = [0, 1, -12, 9],
    readonly lodOpacity = false,
  ) {}

  /** From a PackedSplats array (4 words per splat). */
  static fromPacked(
    device: GPUDevice,
    packed: Uint32Array,
    count: number,
    options: {
      sh1?: Uint32Array;
      sh2?: Uint32Array;
      sh3?: Uint32Array;
      shMax?: [number, number, number];
      encoding?: [number, number, number, number];
      lodOpacity?: boolean;
    } = {},
  ) {
    const src = upload(device, packed.subarray(0, count * 4), "packed splats");
    const numSh = options.sh3 ? 3 : options.sh2 ? 2 : options.sh1 ? 1 : 0;
    let sh: GPUBuffer | null = null;
    if (numSh > 0) {
      // 3 x uint4 per splat: sh1 (2 words + pad), sh2, sh3.
      const words = new Uint32Array(count * 12);
      for (let i = 0; i < count; i++) {
        if (options.sh1)
          words.set(options.sh1.subarray(2 * i, 2 * i + 2), 12 * i);
        if (options.sh2)
          words.set(options.sh2.subarray(4 * i, 4 * i + 4), 12 * i + 4);
        if (options.sh3)
          words.set(options.sh3.subarray(4 * i, 4 * i + 4), 12 * i + 8);
      }
      sh = upload(device, words, "packed SH");
    }
    return new GpuSplatSource(
      "packed",
      count,
      src,
      sh,
      numSh,
      options.shMax,
      options.encoding,
      options.lodOpacity,
    );
  }

  /** From the two ExtSplats arrays (4 words per splat each). */
  static fromExt(
    device: GPUDevice,
    a: Uint32Array,
    b: Uint32Array,
    count: number,
    options: {
      sh1?: Uint32Array;
      sh2?: Uint32Array;
      sh3a?: Uint32Array;
      sh3b?: Uint32Array;
      /** Opacities past 1 are Spark's LoD opacity (merged nodes). */
      lodOpacity?: boolean;
    } = {},
  ) {
    const words = new Uint32Array(count * 8);
    for (let i = 0; i < count; i++) {
      words.set(a.subarray(4 * i, 4 * i + 4), 8 * i);
      words.set(b.subarray(4 * i, 4 * i + 4), 8 * i + 4);
    }
    const src = upload(device, words, "ext splats");
    const numSh = options.sh3a ? 3 : options.sh2 ? 2 : options.sh1 ? 1 : 0;
    let sh: GPUBuffer | null = null;
    if (numSh > 0) {
      const shWords = new Uint32Array(count * 16);
      const parts = [options.sh1, options.sh2, options.sh3a, options.sh3b];
      for (let i = 0; i < count; i++) {
        parts.forEach((p, k) => {
          if (p) shWords.set(p.subarray(4 * i, 4 * i + 4), 16 * i + 4 * k);
        });
      }
      sh = upload(device, shWords, "ext SH");
    }
    return new GpuSplatSource(
      "ext",
      count,
      src,
      sh,
      numSh,
      undefined,
      undefined,
      options.lodOpacity ?? false,
    );
  }

  /** From a loaded PackedSplats (await packedSplats.initialized first). */
  static fromPackedSplats(device: GPUDevice, splats: PackedSplats) {
    if (!splats.packedArray) throw new Error("PackedSplats has no data");
    const e = splats.splatEncoding;
    const extra = splats.extra as Record<string, Uint32Array | undefined>;
    const source = GpuSplatSource.fromPacked(
      device,
      splats.packedArray,
      splats.numSplats,
      {
        sh1: extra.sh1,
        sh2: extra.sh1 ? extra.sh2 : undefined,
        sh3: extra.sh1 && extra.sh2 ? extra.sh3 : undefined,
        shMax: [e.sh1Max, e.sh2Max, e.sh3Max],
        encoding: [e.rgbMin, e.rgbMax, e.lnScaleMin, e.lnScaleMax],
        lodOpacity: e.lodOpacity,
      },
    );
    source.attribs = attribsOf(splats.extra);
    return source;
  }

  /** From a loaded ExtSplats (await extSplats.initialized first). */
  static fromExtSplats(device: GPUDevice, splats: ExtSplats) {
    const [a, b] = splats.extArrays;
    const extra = splats.extra as Record<string, Uint32Array | undefined>;
    const sh3 = extra.sh3a && extra.sh3b;
    const source = GpuSplatSource.fromExt(device, a, b, splats.numSplats, {
      sh1: extra.sh1,
      sh2: extra.sh1 ? extra.sh2 : undefined,
      sh3a: sh3 ? extra.sh3a : undefined,
      sh3b: sh3 ? extra.sh3b : undefined,
    });
    source.attribs = attribsOf(splats.extra);
    return source;
  }

  /**
   * Adds or replaces a per-Gaussian attribute from `count * components`
   * values (see AttribPool.setAttribute); toDraw makes it readable when drawn.
   */
  setAttribute(
    name: string,
    data: ArrayLike<number>,
    format?: AttribFormat,
    components = 1,
    options: Partial<
      Omit<AttributeSpec, "name" | "format" | "components">
    > = {},
  ): AttributeSpec {
    this.attribs ??= new AttribPool(this.count);
    this.version += 1;
    return this.attribs.setAttribute(name, data, format, components, options);
  }

  destroy() {
    this.src.destroy();
    this.sh?.destroy();
  }
}

// extra.attribs as loaded (the worker's AttribValues), or a pool.
function attribsOf(extra: Record<string, unknown>): AttribPool | null {
  const a = extra.attribs as Parameters<typeof AttribPool.from>[0] | undefined;
  if (!a) return null;
  const pool = AttribPool.from(a);
  extra.attribs = pool;
  return pool.columns.length ? pool : null;
}

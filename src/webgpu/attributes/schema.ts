// Generic per-Gaussian attributes: a schema of named, typed vectors, stored
// per attribute on the CPU and packed into the one-buffer pool that
// slang/core/attrib.slang decodes at run time (layout documented there).
//
// AttribPool holds only plain data, so it survives postMessage; the loader
// worker sends the WASM decoders' values instead (AttribValues), and
// AttribPool.from() makes a pool of either on this side.

import type { AttribValues } from "../../defines";
import { coreAttrib } from "../generated/constants";

export type AttribFormat =
  | "f32"
  | "f16"
  | "unorm8"
  | "snorm8"
  | "u8"
  | "u16"
  | "u32";

/** How LOD merges children into a parent (rust/spark-lib LOD builders). */
export type LodMerge =
  | "weightedMean"
  | "normalizeMean"
  | "max"
  | "mode"
  | "first";

export interface AttributeSpec {
  name: string;
  format: AttribFormat;
  /** Components per splat, 1 or more. */
  components: number;
  lodMerge?: LodMerge;
  /** Copied by generate into the draw pool, readable in the draw shaders. */
  toDraw?: boolean;
  /** A 3-vector that turns with its object (normals), rotated by generate. */
  direction?: boolean;
}

/** Format codes, ATTRIB_* in slang/core/attrib.slang. */
export const ATTRIB_FORMATS: Record<AttribFormat, number> = {
  f32: coreAttrib.ATTRIB_F32,
  f16: coreAttrib.ATTRIB_F16,
  unorm8: coreAttrib.ATTRIB_UNORM8,
  snorm8: coreAttrib.ATTRIB_SNORM8,
  u8: coreAttrib.ATTRIB_U8,
  u16: coreAttrib.ATTRIB_U16,
  u32: coreAttrib.ATTRIB_U32,
};
export const { ATTRIB_NONE, ATTRIB_DIRECTION, ATTRIB_PAGED } = coreAttrib;

export function formatBytes(format: AttribFormat): number {
  return format === "f32" || format === "u32"
    ? 4
    : format === "f16" || format === "u16"
      ? 2
      : 1;
}

export function isIntegerFormat(format: AttribFormat) {
  return format === "u8" || format === "u16" || format === "u32";
}

export function attribWords(
  spec: Pick<AttributeSpec, "format" | "components">,
) {
  return Math.ceil((spec.components * formatBytes(spec.format)) / 4);
}

/**
 * Word span of components 4 * comp4 .. + 3 within a splat's record of the
 * attribute: first word and count (0 past the end), as attribGroupWords in
 * slang/core/attrib.slang.
 */
export function attribGroupWords(
  spec: Pick<AttributeSpec, "format" | "components">,
  comp4: number,
): [number, number] {
  const per = formatBytes(spec.format); // words per four components
  const first = comp4 * per;
  const total = attribWords(spec);
  return [first, first < total ? Math.min(per, total - first) : 0];
}

export function defaultLodMerge(spec: AttributeSpec): LodMerge {
  if (spec.direction) return "normalizeMean";
  return isIntegerFormat(spec.format) ? "mode" : "weightedMean";
}

/** One component to its stored bits (not yet shifted into its word). */
export function encodeComponent(format: AttribFormat, v: number): number {
  switch (format) {
    case "f32":
      F32[0] = v;
      return U32[0];
    case "f16":
      return toHalf(v);
    case "unorm8":
      return Math.round(Math.min(Math.max(v, 0), 1) * 255);
    case "snorm8":
      return Math.round(Math.min(Math.max(v, -1), 1) * 127) & 0xff;
    case "u8":
      return Math.min(Math.max(Math.round(v), 0), 255);
    case "u16":
      return Math.min(Math.max(Math.round(v), 0), 65535);
    case "u32":
      return Math.max(Math.round(v), 0) >>> 0;
  }
}

/** Stored bits to the value readAttrib returns. */
export function decodeComponent(format: AttribFormat, bits: number): number {
  switch (format) {
    case "f32":
      U32[0] = bits;
      return F32[0];
    case "f16":
      return fromHalf(bits);
    case "unorm8":
      return bits / 255;
    case "snorm8":
      return Math.max(((bits << 24) >> 24) / 127, -1);
    default:
      return bits >>> 0;
  }
}

const F32 = new Float32Array(1);
const U32 = new Uint32Array(F32.buffer);

// Half floats without src/utils.ts, which would pull three.js into the
// loader worker. Float16Array where the browser has it.
const F16Array = (
  globalThis as unknown as { Float16Array?: Float32ArrayConstructor }
).Float16Array;
const F16 = F16Array ? new F16Array(1) : null;
const U16 = F16 ? new Uint16Array(F16.buffer) : null;

export function toHalf(v: number): number {
  if (F16 && U16) {
    F16[0] = v;
    return U16[0];
  }
  return toHalfJs(v);
}

/**
 * toHalf without Float16Array: round to nearest even, from the double itself
 * (rounding through float32 first can round twice).
 */
export function toHalfJs(v: number): number {
  if (Number.isNaN(v)) return 0x7e00;
  const sign = v < 0 || Object.is(v, -0) ? 0x8000 : 0;
  const a = Math.abs(v);
  // 65520 is halfway between 65504, the largest half, and 2^16.
  if (a >= 65520) return sign | 0x7c00;
  // Subnormals count 2^-24 units; 1024 of them carry into the exponent.
  if (a < 2 ** -14) return sign | roundEven(a * 2 ** 24);
  let e = Math.floor(Math.log2(a));
  if (2 ** e > a) e -= 1;
  else if (2 ** (e + 1) <= a) e += 1;
  // Exact: a / 2^e is in [1, 2). A carry into the exponent is still correct.
  const m = roundEven((a / 2 ** e - 1) * 1024);
  return sign | (((e + 15) << 10) + m);
}

function roundEven(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  return d > 0.5 || (d === 0.5 && f % 2 === 1) ? f + 1 : f;
}

export function fromHalf(h: number): number {
  if (F16 && U16) {
    U16[0] = h;
    return F16[0];
  }
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >>> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24;
  if (exp === 31) return mant ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

/** Where each attribute sits in a pool built from `specs`. */
export interface PoolLayout {
  specs: AttributeSpec[];
  offsets: number[];
  strideWords: number;
  headerWords: number;
}

export function poolLayout(specs: AttributeSpec[]): PoolLayout {
  const offsets: number[] = [];
  let stride = 0;
  for (const s of specs) {
    offsets.push(stride);
    stride += attribWords(s);
  }
  // Four words of sizes, then a uint4 descriptor per attribute.
  const headerWords = 4 + 4 * specs.length;
  return { specs, offsets, strideWords: stride, headerWords };
}

/** A pool's header words: sizes, then one descriptor per attribute. */
export function poolHeader(layout: PoolLayout, count: number): Uint32Array {
  const h = new Uint32Array(layout.headerWords);
  h.set([layout.strideWords, layout.specs.length, layout.headerWords, count]);
  layout.specs.forEach((s, k) => {
    h.set(
      [
        layout.offsets[k],
        ATTRIB_FORMATS[s.format],
        s.components,
        s.direction ? ATTRIB_DIRECTION : 0,
      ],
      4 + 4 * k,
    );
  });
  return h;
}

export function poolWords(layout: PoolLayout, count: number) {
  return layout.headerWords + Math.max(count, 1) * layout.strideWords;
}

/** An attribute's packed words, `attribWords(spec)` per splat. */
export interface AttributeColumn {
  spec: AttributeSpec;
  words: Uint32Array;
}

function inferFormat(data: ArrayLike<number>): AttribFormat {
  if (data instanceof Uint8Array) return "u8";
  if (data instanceof Uint16Array) return "u16";
  if (data instanceof Uint32Array) return "u32";
  if (data instanceof Int8Array) return "snorm8";
  return "f32";
}

/** Packs `count * components` values into a column of `spec`'s format. */
export function packColumn(
  spec: AttributeSpec,
  data: ArrayLike<number>,
  count: number,
): Uint32Array {
  const n = spec.components;
  if (data.length < count * n) {
    throw new Error(
      `attribute '${spec.name}': ${data.length} values for ${count} x ${n}`,
    );
  }
  const per = attribWords(spec);
  const bytes = formatBytes(spec.format);
  const words = new Uint32Array(count * per);
  // 32-bit formats: a typed-array copy (u32 only from integers, exact).
  if (
    spec.format === "f32" ||
    (spec.format === "u32" && data instanceof Uint32Array)
  ) {
    const values = ArrayBuffer.isView(data)
      ? (data as Float64Array).subarray(0, count * n)
      : Array.prototype.slice.call(data, 0, count * n);
    (spec.format === "f32" ? new Float32Array(words.buffer) : words).set(
      values,
    );
    return words;
  }
  for (let i = 0; i < count; i++) {
    for (let c = 0; c < n; c++) {
      const bits = encodeComponent(spec.format, data[i * n + c]);
      const byte = c * bytes;
      words[i * per + (byte >> 2)] |= bits << ((byte & 3) * 8);
    }
  }
  return words;
}

/** Raw bits of component `c` of splat `i` in a column. */
export function columnBits(col: AttributeColumn, i: number, c: number) {
  const bytes = formatBytes(col.spec.format);
  const byte = c * bytes;
  const w = col.words[i * attribWords(col.spec) + (byte >> 2)];
  if (bytes === 4) return w >>> 0;
  return (w >>> ((byte & 3) * 8)) & ((1 << (8 * bytes)) - 1);
}

/**
 * The extra attributes of a set of splats: columns by name, packed for the
 * GPU on demand. Index i is splat i of the PackedSplats/ExtSplats/source.
 */
export class AttribPool {
  columns: AttributeColumn[] = [];
  /** Bumped on every change, so GPU copies know to re-upload. */
  version = 0;
  /**
   * The packed pool already on the GPU (paged .rad attributes, see
   * PagedAttribPool): renderers bind it instead of uploading pack().
   */
  gpuBuffer?: GPUBuffer;

  constructor(public count: number) {}

  /**
   * Restores the methods of a pool that went through postMessage, or packs
   * the values the loader worker sends.
   */
  static from(
    data:
      | AttribPool
      | { count: number; columns: AttributeColumn[] }
      | AttribValues,
  ) {
    if (data instanceof AttribPool) return data;
    if ("values" in data) return AttribPool.fromValues(data);
    const pool = new AttribPool(data.count);
    pool.columns = data.columns;
    return pool;
  }

  /**
   * A pool of the WASM decoders' attributes (decoded from a PLY or .rad, or
   * merged by a LOD build), each drawable.
   */
  static fromValues({ count, specs, values }: AttribValues): AttribPool {
    const pool = new AttribPool(count);
    specs.forEach(({ name, format, components, lodMerge }, k) => {
      // The file keeps no flags: a renormalized 3-vector is a direction. An
      // all-zero one is a 3DGS trainer's placeholder normal.
      const direction = lodMerge === "normalizeMean" && components === 3;
      if (direction && values[k].every((v) => v === 0)) return;
      pool.setAttribute(name, values[k], format as AttribFormat, components, {
        lodMerge: lodMerge as LodMerge,
        direction,
        toDraw: true,
      });
    });
    return pool;
  }

  get schema(): AttributeSpec[] {
    return this.columns.map((c) => c.spec);
  }

  id(name: string): number {
    return this.columns.findIndex((c) => c.spec.name === name);
  }

  column(name: string): AttributeColumn | undefined {
    return this.columns[this.id(name)];
  }

  /**
   * Adds or replaces attribute `name` from `count * components` values:
   * floats for f32/f16, [0, 1] for unorm8, [-1, 1] for snorm8, integers for
   * the u formats. The format defaults from the array type.
   */
  setAttribute(
    name: string,
    data: ArrayLike<number>,
    format: AttribFormat = inferFormat(data),
    components = 1,
    options: Partial<
      Omit<AttributeSpec, "name" | "format" | "components">
    > = {},
  ): AttributeSpec {
    const spec: AttributeSpec = { name, format, components, ...options };
    spec.lodMerge ??= defaultLodMerge(spec);
    this.setColumn({ spec, words: packColumn(spec, data, this.count) });
    return spec;
  }

  /** Adds or replaces an already packed column. */
  setColumn(col: AttributeColumn) {
    if (col.words.length < this.count * attribWords(col.spec)) {
      throw new Error(`attribute '${col.spec.name}': column too short`);
    }
    const k = this.id(col.spec.name);
    if (k >= 0) this.columns[k] = col;
    else this.columns.push(col);
    this.version += 1;
  }

  removeAttribute(name: string) {
    const k = this.id(name);
    if (k >= 0) {
      this.columns.splice(k, 1);
      this.version += 1;
    }
  }

  /** The decoded components of splat `index`. */
  getAttribute(name: string, index: number): number[] {
    const col = this.column(name);
    if (!col) throw new Error(`no attribute '${name}'`);
    return Array.from({ length: col.spec.components }, (_, c) =>
      decodeComponent(col.spec.format, columnBits(col, index, c)),
    );
  }

  /** The interleaved pool of the columns `filter` keeps, with its header. */
  pack(filter: (spec: AttributeSpec) => boolean = () => true): {
    layout: PoolLayout;
    words: Uint32Array;
  } {
    const cols = this.columns.filter((c) => filter(c.spec));
    const layout = poolLayout(cols.map((c) => c.spec));
    const words = new Uint32Array(poolWords(layout, this.count));
    words.set(poolHeader(layout, this.count));
    const stride = layout.strideWords;
    cols.forEach((col, k) => {
      const per = attribWords(col.spec);
      let at = layout.headerWords + layout.offsets[k];
      for (let i = 0; i < this.count; i++, at += stride) {
        words.set(col.words.subarray(i * per, (i + 1) * per), at);
      }
    });
    return { layout, words };
  }
}

/** A column's decoded values, components per splat. */
export function columnValues(
  col: AttributeColumn,
  count: number,
): Float64Array {
  const n = col.spec.components;
  const out = new Float64Array(count * n);
  for (let i = 0; i < count; i++) {
    for (let c = 0; c < n; c++) {
      out[i * n + c] = decodeComponent(col.spec.format, columnBits(col, i, c));
    }
  }
  return out;
}

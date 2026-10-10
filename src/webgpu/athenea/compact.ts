// THE COMPACT RELIT PATH (thread CB, research/iphone-compact.md): a relit TX
// cloud in one compact buffer (athenea's proposal 097 layout, with joint CPCA
// weights and a shadowBits dictionary), relit per drawn splat in the frame's
// shading pass (slang/athenea_adapter/compact.slang). Opt-in and
// self-contained: it draws one cloud under one dome on its own, beside or
// instead of WgpuSplatRenderer + atheneaRelightPlugin, and keeps:
//
//   per gaussian   the compact planes (encodeCompact; ~52 B)
//   per drawn      the sort's four u32 buffers (16 B), reused as the draw's
//                  records once sorted, and its scan levels
//   fixed          the cluster projections (147 KB at C = 256), the dome's
//                  prefiltered map and harmonics (its source images freed
//                  once prepared), a uniform
//
// Usage (see test/gpu/iphoneCompact.test.ts):
//   const cloud = encodeCompact({ count, ext0, ext1, normalOct, pbr, lobes,
//                                 curvature, shadowBits, q8, basis, clusters });
//   const path = new CompactRelitPath(device, { hdri });
//   path.setCloud(cloud);
//   path.render(encoder, camera, view, width, height);

import * as THREE from "three";
import { toHalf } from "../../utils";
import type { GpuProfiler } from "../GpuProfiler";
import { GpuSorter } from "../GpuSorter";
import { KernelRegistry } from "../KernelRegistry";
import compactModule from "../generated/athenea_adapter/compact";
import { atheneaAdapterCompact as K } from "../generated/constants";
import { createReadback, createStorage, readAndDestroy } from "../gpuBuffers";
import {
  type ReflectedRenderPipeline,
  createBindGroups,
  createReflectedRenderPipeline,
} from "../renderPipeline";
import { UniformWriter } from "../uniforms";
import { AtheneaSky, type SkyImage } from "./AtheneaSky";

/** Gaussians a chunk (positions, scales and curvature are ranged per chunk). */
export const COMPACT_CHUNK = K.kCompactChunk;
/** Gaussians a piece of the shadowBits dictionary (athc-convert --shadow-vq). */
export const COMPACT_PIECE = K.kCompactPiece;
const CHUNK_WORDS = K.kCompactChunkWords;
const CLUSTER_WORDS = K.kCompactClusterWords;
const CURVATURE_FLOOR = 0.01;

export interface CompactInput {
  count: number;
  /** ExtSplats words (4 + 4 a splat). */
  ext0: Uint32Array;
  ext1: Uint32Array;
  /** athenea's packNormal (16 + 16), a word a splat. */
  normalOct: Uint32Array;
  pbr: Uint32Array;
  /** 3 words a splat. */
  lobes: Uint32Array;
  /** f16 x 4 (xx, xy, yy, spread), 2 words a splat; absent: no curvature. */
  curvature?: Uint32Array | null;
  /** 8 words a splat (a TX cloud's 256 cells). */
  shadowBits: Uint32Array;
  /** athenea's joint CPCA records (cpcaJoint.ts, 7 words a splat). */
  q8: Uint32Array;
  /** cpcaJoint.ts packJointBasis (1008 floats a cluster). */
  basis: Float32Array;
  clusters: number;
  /** The stored colour is linear light (else sRGB). */
  linear?: boolean;
}

/** Where each plane and table starts (words) and how wide (bytes a gaussian). */
export interface CompactLayout {
  count: number;
  chunks: number;
  pieces: number;
  materials: number;
  codebookEntries: number;
  clusters: number;
  linear: boolean;
  curvature: boolean;
  words: Record<string, number>;
  /** Bytes by category: per-gaussian planes and the tables. */
  bytes: Record<string, number>;
}

export interface CompactCloud {
  data: Uint32Array;
  layout: CompactLayout;
}

/** The opacity code (compact.slang opacityOf): sqrt up to 1, then 1..2 by 1/63. */
export function encodeOpacity(a: number): number {
  if (!(a > 0)) return 0;
  if (a <= 1) return Math.round(Math.sqrt(a) * 191);
  return Math.min(192 + Math.round((a - 1) * 63), 255);
}
export function decodeOpacity(code: number): number {
  if (code <= 191) return (code / 191) ** 2;
  return 1 + (code - 192) / 63;
}

const halfToFloat = (h: number) => {
  const s = h >> 15;
  const e = (h >> 10) & 31;
  const m = h & 1023;
  const v =
    e === 0
      ? m * 2 ** -24
      : e === 31
        ? m
          ? Number.NaN
          : Number.POSITIVE_INFINITY
        : (1 + m / 1024) * 2 ** (e - 15);
  return s ? -v : v;
};

/** athenea's unpackNormal. */
function unpackNormal(word: number): [number, number, number] {
  const fx = ((word & 0xffff) / 65535) * 2 - 1;
  const fy = ((word >>> 16) / 65535) * 2 - 1;
  return octVector(fx, fy);
}

function octVector(fx: number, fy: number): [number, number, number] {
  let x = fx;
  let y = fy;
  const z = 1 - Math.abs(fx) - Math.abs(fy);
  const t = Math.min(Math.max(-z, 0), 1);
  x += x >= 0 ? -t : t;
  y += y >= 0 ? -t : t;
  const l = Math.hypot(x, y, z) || 1;
  return [x / l, y / l, z / l];
}

/** An 8 + 8 octahedral code of n, the best of the four around it. */
export function encodeNormal8(n: [number, number, number]): number {
  const l1 = Math.abs(n[0]) + Math.abs(n[1]) + Math.abs(n[2]) || 1;
  let px = n[0] / l1;
  let py = n[1] / l1;
  if (n[2] < 0) {
    const ox = (1 - Math.abs(py)) * (px >= 0 ? 1 : -1);
    const oy = (1 - Math.abs(px)) * (py >= 0 ? 1 : -1);
    px = ox;
    py = oy;
  }
  const ux = (px * 0.5 + 0.5) * 255;
  const uy = (py * 0.5 + 0.5) * 255;
  let best = 0;
  let bestDot = -2;
  for (const qx of [Math.floor(ux), Math.ceil(ux)]) {
    for (const qy of [Math.floor(uy), Math.ceil(uy)]) {
      const cx = Math.min(Math.max(qx, 0), 255);
      const cy = Math.min(Math.max(qy, 0), 255);
      const v = decodeNormal8(cx | (cy << 8));
      const d = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
      if (d > bestDot) {
        bestDot = d;
        best = cx | (cy << 8);
      }
    }
  }
  return best;
}

export function decodeNormal8(q: number): [number, number, number] {
  return octVector(((q & 0xff) / 255) * 2 - 1, ((q >> 8) / 255) * 2 - 1);
}

/** athenea's packNormal (16 + 16). */
export function packNormal16(n: [number, number, number]): number {
  const l1 = Math.abs(n[0]) + Math.abs(n[1]) + Math.abs(n[2]) || 1;
  let px = n[0] / l1;
  let py = n[1] / l1;
  if (n[2] < 0) {
    const ox = (1 - Math.abs(py)) * (px >= 0 ? 1 : -1);
    const oy = (1 - Math.abs(px)) * (py >= 0 ? 1 : -1);
    px = ox;
    py = oy;
  }
  const q = (v: number) =>
    Math.round(Math.min(Math.max(v * 0.5 + 0.5, 0), 1) * 65535);
  return (q(px) | (q(py) << 16)) >>> 0;
}

/** Byte planes in one word buffer, each 64-byte aligned. */
class PlaneWriter {
  words = 0;
  private planes: { at: number; bytes: Uint8Array }[] = [];
  add(bytes: Uint8Array): number {
    const at = Math.ceil(this.words / 16) * 16;
    this.planes.push({ at, bytes });
    this.words = at + Math.ceil(bytes.length / 4);
    return at;
  }
  addWords(words: Uint32Array): number {
    return this.add(
      new Uint8Array(words.buffer, words.byteOffset, words.byteLength),
    );
  }
  finish(): Uint32Array {
    const out = new Uint32Array(Math.ceil(this.words / 16) * 16);
    const bytes = new Uint8Array(out.buffer);
    for (const p of this.planes) bytes.set(p.bytes, p.at * 4);
    return out;
  }
}

/**
 * The compact buffer of a relit TX cloud (compact.slang's head): planes per
 * gaussian, then the chunk ranges, the pieces' codebooks, the materials and
 * athenea's basis in f16.
 */
export function encodeCompact(input: CompactInput): CompactCloud {
  const n = input.count;
  const chunks = Math.ceil(n / COMPACT_CHUNK);
  const pieces = Math.ceil(n / COMPACT_PIECE);
  const posX = new Uint16Array(n);
  const posY = new Uint16Array(n);
  const posZ = new Uint16Array(n);
  const albedo = new Uint8Array(3 * n);
  const opacity = new Uint8Array(n);
  const scale = new Uint8Array(3 * n);
  const rotation = new Uint32Array(n);
  const normal = new Uint16Array(n);
  const curvature = new Uint8Array(3 * n);
  const material = new Uint16Array(n);
  const cluster = new Uint8Array(n);
  const weights = [0, 1, 2].map(() => new Uint32Array(2 * n));
  const shadow = new Uint16Array(n);
  const chunkTable = new Float32Array(chunks * CHUNK_WORDS);
  const f32 = new Float32Array(
    input.ext0.buffer,
    input.ext0.byteOffset,
    input.ext0.length,
  );
  const hasCurvature = Boolean(input.curvature);
  const curv = (i: number, k: number) => {
    const w = (input.curvature as Uint32Array)[2 * i + (k >> 1)];
    return halfToFloat(k & 1 ? w >>> 16 : w & 0xffff);
  };
  for (let c = 0; c < chunks; c++) {
    const first = c * COMPACT_CHUNK;
    const last = Math.min(first + COMPACT_CHUNK, n);
    const lo = [
      Number.POSITIVE_INFINITY,
      Number.POSITIVE_INFINITY,
      Number.POSITIVE_INFINITY,
    ];
    const hi = [
      Number.NEGATIVE_INFINITY,
      Number.NEGATIVE_INFINITY,
      Number.NEGATIVE_INFINITY,
    ];
    let lnLo = Number.POSITIVE_INFINITY;
    let lnHi = Number.NEGATIVE_INFINITY;
    let cMax = 0;
    for (let i = first; i < last; i++) {
      for (let k = 0; k < 3; k++) {
        const p = f32[4 * i + k];
        lo[k] = Math.min(lo[k], p);
        hi[k] = Math.max(hi[k], p);
      }
      const b = input.ext1;
      for (const h of [
        b[4 * i + 1] >>> 16,
        b[4 * i + 2] & 0xffff,
        b[4 * i + 2] >>> 16,
      ]) {
        const ln = halfToFloat(h);
        if (ln <= -60000) continue;
        lnLo = Math.min(lnLo, ln);
        lnHi = Math.max(lnHi, ln);
      }
      if (hasCurvature) {
        for (let k = 0; k < 3; k++) {
          const v = Math.abs(curv(i, k));
          if (Number.isFinite(v)) cMax = Math.max(cMax, v);
        }
      }
    }
    if (!(lnLo <= lnHi)) {
      lnLo = 0;
      lnHi = 0;
    }
    const range = Math.log2(1 + cMax / CURVATURE_FLOOR);
    const t = c * CHUNK_WORDS;
    chunkTable.set([...lo, ...hi, lnLo, lnHi, range], t);
    for (let i = first; i < last; i++) {
      const q = (k: number) =>
        hi[k] > lo[k]
          ? Math.round(((f32[4 * i + k] - lo[k]) / (hi[k] - lo[k])) * 65535)
          : 0;
      posX[i] = q(0);
      posY[i] = q(1);
      posZ[i] = q(2);
      const b = input.ext1;
      const rgb = [
        halfToFloat(b[4 * i] & 0xffff),
        halfToFloat(b[4 * i] >>> 16),
        halfToFloat(b[4 * i + 1] & 0xffff),
      ];
      for (let k = 0; k < 3; k++) {
        albedo[3 * i + k] = Math.round(Math.min(Math.max(rgb[k], 0), 1) * 255);
      }
      opacity[i] = encodeOpacity(halfToFloat(input.ext0[4 * i + 3] & 0xffff));
      const lns = [
        b[4 * i + 1] >>> 16,
        b[4 * i + 2] & 0xffff,
        b[4 * i + 2] >>> 16,
      ];
      for (let k = 0; k < 3; k++) {
        const ln = halfToFloat(lns[k]);
        scale[3 * i + k] =
          ln <= -60000
            ? 0
            : 1 +
              (lnHi > lnLo
                ? Math.round(((ln - lnLo) / (lnHi - lnLo)) * 254)
                : 0);
      }
      rotation[i] = b[4 * i + 3];
      normal[i] = encodeNormal8(unpackNormal(input.normalOct[i]));
      if (hasCurvature && range > 0) {
        for (let k = 0; k < 3; k++) {
          const v = curv(i, k);
          const m = Number.isFinite(v)
            ? Math.round(
                (127 * Math.log2(1 + Math.abs(v) / CURVATURE_FLOOR)) / range,
              )
            : 0;
          curvature[3 * i + k] = (v < 0 ? -m : m) & 0xff;
        }
      }
    }
  }
  // Materials: every distinct (pbr, lobes).
  const materialOf = new Map<string, number>();
  const materialWords: number[] = [];
  for (let i = 0; i < n; i++) {
    const w = [
      input.pbr[i],
      input.lobes[3 * i],
      input.lobes[3 * i + 1],
      input.lobes[3 * i + 2],
    ];
    const key = w.join(",");
    let m = materialOf.get(key);
    if (m === undefined) {
      m = materialOf.size;
      if (m > 0xffff) throw new Error("compact: over 65536 materials");
      materialOf.set(key, m);
      materialWords.push(...w);
    }
    material[i] = m;
  }
  // The shadowBits dictionary: each piece's distinct masks.
  const codebooks: number[] = [];
  const pieceStart = new Uint32Array(pieces);
  let entries = 0;
  for (let p = 0; p < pieces; p++) {
    const seen = new Map<string, number>();
    const start = codebooks.length;
    pieceStart[p] = start;
    const last = Math.min((p + 1) * COMPACT_PIECE, n);
    for (let i = p * COMPACT_PIECE; i < last; i++) {
      const w = input.shadowBits.subarray(8 * i, 8 * i + 8);
      const key = w.join(",");
      let k = seen.get(key);
      if (k === undefined) {
        k = seen.size;
        if (k > 0xffff) throw new Error("compact: over 65536 masks a piece");
        seen.set(key, k);
        for (const x of w) codebooks.push(x);
      }
      shadow[i] = k;
    }
    entries += seen.size;
  }
  for (let i = 0; i < n; i++) {
    for (let b = 0; b < 3; b++) {
      weights[b][2 * i] = input.q8[7 * i + 2 * b];
      weights[b][2 * i + 1] = input.q8[7 * i + 2 * b + 1];
    }
    cluster[i] = input.q8[7 * i + 6];
    if (input.q8[7 * i + 6] > 255)
      throw new Error("compact: over 256 clusters");
  }
  // athenea's basis in f16.
  const basisHalves = new Uint16Array(input.clusters * CLUSTER_WORDS * 2);
  for (let k = 0; k < input.basis.length; k++) {
    basisHalves[k] = toHalf(input.basis[k]);
  }

  const w = new PlaneWriter();
  const bytes = (a: ArrayBufferView) =>
    new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  const words: Record<string, number> = {
    posX: w.add(bytes(posX)),
    posY: w.add(bytes(posY)),
    posZ: w.add(bytes(posZ)),
    albedo: w.add(albedo),
    opacity: w.add(opacity),
    scale: w.add(scale),
    rotation: w.addWords(rotation),
    normal: w.add(bytes(normal)),
    curvature: w.add(curvature),
    material: w.add(bytes(material)),
    cluster: w.add(cluster),
    weights: w.addWords(weights[0]),
  };
  const w1 = w.addWords(weights[1]);
  w.addWords(weights[2]);
  words.weightsStride = w1 - words.weights;
  words.shadow = w.add(bytes(shadow));
  const perGaussianWords = w.words;
  words.chunkTable = w.addWords(new Uint32Array(chunkTable.buffer));
  const codebookStart = Math.ceil(w.words / 16) * 16;
  const pieceTable = new Uint32Array(pieces);
  // The piece table holds absolute word offsets of each codebook.
  words.pieceTable = w.addWords(pieceTable);
  words.codebooks = w.addWords(new Uint32Array(codebooks));
  words.materials = w.addWords(new Uint32Array(materialWords));
  words.basis = w.add(bytes(basisHalves));
  const data = w.finish();
  for (let p = 0; p < pieces; p++) {
    data[words.pieceTable + p] = words.codebooks + pieceStart[p];
  }
  void codebookStart;
  const layout: CompactLayout = {
    count: n,
    chunks,
    pieces,
    materials: materialOf.size,
    codebookEntries: entries,
    clusters: input.clusters,
    linear: Boolean(input.linear),
    curvature: hasCurvature,
    words,
    bytes: {
      position: 6 * n,
      albedo: 3 * n,
      opacity: n,
      scale: 3 * n,
      rotation: 4 * n,
      normal: 2 * n,
      curvature: 3 * n,
      material: 2 * n,
      cluster: n,
      weights: 24 * n,
      shadowIndex: 2 * n,
      planesPadded: perGaussianWords * 4,
      chunkTable: chunkTable.byteLength,
      pieceTable: pieceTable.byteLength,
      codebooks: codebooks.length * 4,
      materialTable: materialWords.length * 4,
      basis: basisHalves.byteLength,
      total: data.byteLength,
    },
  };
  return { data, layout };
}

/** Splat i of a compact cloud, decoded on the CPU as compact.slang does. */
export function decodeCompactSplat(cloud: CompactCloud, i: number) {
  const { data, layout } = cloud;
  const W = layout.words;
  const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const field = (plane: number, width: number) => {
    let v = 0;
    for (let k = 0; k < width; k++)
      v |= bytes[plane * 4 + i * width + k] << (8 * k);
    return v >>> 0;
  };
  const f = new Float32Array(data.buffer, data.byteOffset, data.length);
  const t = W.chunkTable + Math.floor(i / COMPACT_CHUNK) * CHUNK_WORDS;
  const lo = [f[t], f[t + 1], f[t + 2]];
  const hi = [f[t + 3], f[t + 4], f[t + 5]];
  const center = [field(W.posX, 2), field(W.posY, 2), field(W.posZ, 2)].map(
    (q, k) => Math.fround(lo[k] + q * Math.fround((hi[k] - lo[k]) / 65535)),
  );
  const s = field(W.scale, 3);
  const step = (f[t + 7] - f[t + 6]) / 254;
  const scales = [0, 1, 2].map((k) => {
    const c = (s >> (8 * k)) & 0xff;
    return c === 0 ? 0 : Math.exp(f[t + 6] + (c - 1) * step);
  });
  const a = field(W.albedo, 3);
  const rgb = [a & 0xff, (a >> 8) & 0xff, (a >> 16) & 0xff].map((v) => v / 255);
  const opacity = decodeOpacity(field(W.opacity, 1));
  const normal = decodeNormal8(field(W.normal, 2));
  const cw = field(W.curvature, 3);
  const curvature = [0, 1, 2].map((k) => {
    const c = ((cw >> (8 * k)) << 24) >> 24;
    const m = CURVATURE_FLOOR * (2 ** ((Math.abs(c) / 127) * f[t + 8]) - 1);
    return c < 0 ? -m : m;
  });
  const m = W.materials + field(W.material, 2) * 4;
  const piece = data[W.pieceTable + Math.floor(i / COMPACT_PIECE)];
  const cells = data.subarray(
    piece + field(W.shadow, 2) * 8,
    piece + field(W.shadow, 2) * 8 + 8,
  );
  return {
    center,
    scales,
    rotation: data[W.rotation + i],
    rgb,
    opacity,
    normal,
    curvature,
    pbr: data[m],
    lobes: [data[m + 1], data[m + 2], data[m + 3]],
    cells,
  };
}

export interface CompactPathOptions {
  hdri?: SkyImage | null;
  /** Turns the dome about +Y (radians), as the relight plugin's. */
  rotation?: number;
  intensity?: number;
  /** The cloud's stage (the relight plugin's `frame`); default the world. */
  frame?: THREE.Matrix4 | null;
  indirect?: boolean;
  ior?: number;
  footprint?: number;
  footprintClamp?: number;
  glassCover?: number;
  /** Spark's draw defaults. */
  blurAmount?: number;
  minAlpha?: number;
  maxStdDev?: number;
  maxPixelRadius?: number;
  minPixelRadius?: number;
  clipXY?: number;
  faint?: boolean;
  /** Free the dome's source images once prepared (default true). */
  releaseSky?: boolean;
  /** Debug: the albedo, no relight. */
  plain?: boolean;
}

/**
 * The compact path: one cloud, one dome, its own sort and draw.
 */
export class CompactRelitPath {
  readonly registry: KernelRegistry;
  readonly sky: AtheneaSky;
  readonly sorter: GpuSorter;
  options: Required<Omit<CompactPathOptions, "hdri" | "frame">> & {
    frame: THREE.Matrix4 | null;
  };
  cloud: CompactCloud | null = null;
  /** The cloud's place in the world. */
  matrixWorld = new THREE.Matrix4();
  /** Sort slots (drawn splats); default the cloud's count. */
  capacity = 0;
  stats = { projected: 0, frames: 0 };
  private data: GPUBuffer | null = null;
  private frame: GPUBuffer;
  private drawUniform: GPUBuffer;
  private pipeline: ReflectedRenderPipeline | null = null;
  private pipelineFormat = "";
  private projectedKey = "";
  private frameKey = "";

  constructor(
    readonly device: GPUDevice,
    options: CompactPathOptions = {},
  ) {
    this.registry = new KernelRegistry(device);
    this.sky = new AtheneaSky(device, this.registry);
    this.sorter = new GpuSorter(this.registry, "compact sort");
    this.options = {
      rotation: 0,
      intensity: 1,
      indirect: true,
      ior: 0,
      footprint: 0.5,
      footprintClamp: 1,
      glassCover: 1,
      blurAmount: 0.3,
      minAlpha: 0.5 / 255,
      maxStdDev: Math.sqrt(8),
      maxPixelRadius: 512,
      minPixelRadius: 0,
      clipXY: 1.4,
      faint: true,
      releaseSky: true,
      plain: false,
      ...options,
      frame: options.frame ?? null,
    };
    this.sky.set({
      image: options.hdri ?? null,
      rotation: this.options.rotation,
      intensity: this.options.intensity,
      frame: this.options.frame,
    });
    this.frame = createStorage(device, 16, "compact frame");
    const bytes = UniformWriter.for(compactModule, "compactParams").data
      .byteLength;
    this.drawUniform = device.createBuffer({
      label: "compact draw params",
      size: bytes,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  /** The dome (as AtheneaSky.set); the projections follow. */
  setSky(o: Parameters<AtheneaSky["set"]>[0]) {
    this.sky.set(o);
  }

  setCloud(cloud: CompactCloud) {
    this.data?.destroy();
    this.cloud = cloud;
    this.data = createStorage(
      this.device,
      cloud.data.byteLength,
      "compact data",
    );
    this.device.queue.writeBuffer(
      this.data,
      0,
      cloud.data.buffer,
      cloud.data.byteOffset,
      cloud.data.byteLength,
    );
    const words = cloud.layout.clusters * K.kCompactProjWords;
    if (this.frame.size < words * 4) {
      this.frame.destroy();
      this.frame = createStorage(this.device, words * 4, "compact projections");
    }
    this.capacity ||= cloud.layout.count;
    this.projectedKey = "";
  }

  private flags() {
    const o = this.options;
    const l = this.cloud?.layout;
    return (
      (l?.linear ? K.kCompactLinear : 0) |
      (o.indirect ? K.kCompactIndirect : 0) |
      (l?.curvature ? K.kCompactCurvature : 0) |
      (o.faint ? K.kCompactFaint : 0) |
      (o.plain ? K.kCompactPlain : 0)
    );
  }

  /** The uniform for this frame's camera and target size. */
  params(camera: THREE.Camera, width: number, height: number) {
    const cloud = this.cloud as CompactCloud;
    const W = cloud.layout.words;
    const o = this.options;
    camera.updateMatrixWorld();
    const toStage = o.frame ? o.frame.clone().invert() : new THREE.Matrix4();
    const world = toStage.clone().multiply(this.matrixWorld);
    const e = world.elements;
    const row = (k: number) => [e[k], e[k + 4], e[k + 8], e[k + 12]];
    const eyeInWorld = new THREE.Vector3().setFromMatrixPosition(
      camera.matrixWorld,
    );
    const eye = eyeInWorld.clone().applyMatrix4(toStage);
    const eyeObject = eyeInWorld
      .clone()
      .applyMatrix4(this.matrixWorld.clone().invert());
    const toEye = new THREE.Matrix4().multiplyMatrices(
      camera.matrixWorldInverse,
      this.matrixWorld,
    ).elements;
    const eyeRow = (k: number, sign: number) => [
      sign * toEye[k],
      sign * toEye[k + 4],
      sign * toEye[k + 8],
      sign * toEye[k + 12],
    ];
    const p = (camera as THREE.PerspectiveCamera).projectionMatrix.elements;
    const near = (camera as THREE.PerspectiveCamera).near ?? 0.01;
    const far = (camera as THREE.PerspectiveCamera).far ?? 1000;
    return UniformWriter.for(compactModule, "compactParams").setAll({
      count: cloud.layout.count,
      flags: this.flags(),
      chunkTable: W.chunkTable,
      pieceTable: W.pieceTable,
      posX: W.posX,
      posY: W.posY,
      posZ: W.posZ,
      albedo: W.albedo,
      opacity: W.opacity,
      scale: W.scale,
      rotation: W.rotation,
      normal: W.normal,
      curvature: W.curvature,
      material: W.material,
      cluster: W.cluster,
      weights: W.weights,
      weightsStride: W.weightsStride,
      shadow: W.shadow,
      materials: W.materials,
      basis: W.basis,
      clusters: cloud.layout.clusters,
      tableBase: 0,
      envBaseSide: this.sky.baseSide,
      capacity: this.capacity,
      ior: o.ior,
      footprint: o.footprint,
      footprintClamp: o.footprintClamp,
      glassCover: o.glassCover,
      minAlpha: o.minAlpha,
      blurAmount: o.blurAmount,
      maxStdDev: o.maxStdDev,
      maxPixelRadius: o.maxPixelRadius,
      minPixelRadius: o.minPixelRadius,
      clipXY: o.clipXY,
      near,
      far,
      falloff: 1,
      keyScale: ((1 << K.kCompactKeyBits) - 1) / (far * 1.8),
      row0: row(0),
      row1: row(1),
      row2: row(2),
      eyeObject: [eyeObject.x, eyeObject.y, eyeObject.z, 1],
      eyeWorld: [eye.x, eye.y, eye.z, 1],
      view0: eyeRow(0, 1),
      view1: eyeRow(1, 1),
      view2: eyeRow(2, -1),
      focal: [
        (width / 2) * p[0],
        -(height / 2) * p[5],
        (width / 2) * (1 - p[8]),
        (height / 2) * (1 + p[9]),
      ],
      viewport: [width, height, 0, 0],
    });
  }

  private buffers() {
    const sky = this.sky;
    return {
      compactData: this.data as GPUBuffer,
      compactFrame: this.frame,
      compactFrameOut: this.frame,
      envSh: sky.envSh,
      envSun: sky.envSun,
      envTexels: sky.envTexels as GPUBuffer,
    };
  }

  /** The dome prepared (when changed) and the clusters projected on it. */
  private prepareLight(
    encoder: GPUCommandEncoder,
    params: UniformWriter,
    profiler: GpuProfiler | null,
  ) {
    if (this.sky.dirty) {
      // On a submit of its own, so its source images can go right after.
      this.sky.prepareNow();
      if (this.options.releaseSky) this.sky.releaseSources();
    }
    const key = `${this.sky.version}`;
    if (key === this.projectedKey) return;
    this.projectedKey = key;
    const pass = encoder.beginComputePass({
      label: "compact.project",
      timestampWrites: profiler?.timestampWrites("compact.project"),
    });
    this.registry.get(compactModule, "atheneaCompactProject").dispatch(pass, {
      grid: [(this.cloud as CompactCloud).layout.clusters],
      buffers: this.buffers(),
      uniforms: params.data,
    });
    pass.end();
    this.stats.projected += 1;
  }

  /** Forces the projections again (as a change of the lights would). */
  touchLight() {
    this.projectedKey = "";
    this.frameKey = "";
  }

  /**
   * Records a frame into `encoder`: light (when changed), cull, sort, shade,
   * draw into `view` (`format`, width x height), cleared to `clear` or loaded.
   */
  render(
    encoder: GPUCommandEncoder,
    camera: THREE.Camera,
    view: GPUTextureView,
    format: GPUTextureFormat,
    width: number,
    height: number,
    o: { clear?: GPUColor | null; profiler?: GpuProfiler | null } = {},
  ) {
    const cloud = this.cloud;
    if (!cloud) return;
    const profiler = o.profiler ?? null;
    const params = this.params(camera, width, height);
    // Nothing moved and the light is the same: the records in the sort's
    // buffers are this frame's, and only the draw runs.
    const key = `${new Uint32Array(params.data).join()}|${this.sky.version}|${this.projectedKey}|${this.capacity}`;
    const still =
      key === this.frameKey && !this.sky.dirty && this.projectedKey !== "";
    if (still && this.pipeline && this.pipelineFormat === format) {
      this.draw(encoder, view, format, params, o.clear ?? null, profiler);
      return;
    }
    this.prepareLight(encoder, params, profiler);
    this.frameKey = `${new Uint32Array(params.data).join()}|${this.sky.version}|${this.projectedKey}|${this.capacity}`;
    const { keys, vals } = this.sorter.keyInput(this.capacity);
    this.device.queue.writeBuffer(
      this.sorter.drawArgs,
      0,
      new Uint32Array([4, 0, 0, 0]),
    );
    const buffers = this.buffers();
    {
      const pass = encoder.beginComputePass({
        label: "compact.cull",
        timestampWrites: profiler?.timestampWrites("compact.cull"),
      });
      const n = cloud.layout.count;
      const row = K.kCompactDispatchRow;
      this.registry.get(compactModule, "atheneaCompactCull").dispatch(pass, {
        grid: [Math.min(n, row), Math.ceil(n / row)],
        buffers: {
          ...buffers,
          sortKeys: keys,
          sortVals: vals,
          sortArgs: this.sorter.drawArgs,
        },
        uniforms: params.data,
      });
      this.registry.get(compactModule, "atheneaCompactClamp").dispatch(pass, {
        grid: [1],
        buffers: { sortArgs: this.sorter.drawArgs },
        uniforms: params.data,
      });
      pass.end();
    }
    const sorted = this.sorter.encodeKeys(
      encoder,
      profiler,
      this.capacity,
      K.kCompactKeyBits,
    );
    // The sorted values (the splats, back to front) are rec0; the three
    // other sort buffers take the rest of the record.
    const all = this.sorter as unknown as {
      keys: GPUBuffer[];
      vals: GPUBuffer[];
    };
    const others = [...all.keys, ...all.vals].filter((b) => b !== sorted.vals);
    const rec = [sorted.vals, ...others];
    this.records = rec;
    {
      const pass = encoder.beginComputePass({
        label: "compact.shade",
        timestampWrites: profiler?.timestampWrites("compact.shade"),
      });
      this.registry.get(compactModule, "atheneaCompactShade").dispatchIndirect(
        pass,
        {
          buffers: {
            ...buffers,
            sortCount: this.sorter.drawArgs,
            rec0: rec[0],
            rec1: rec[1],
            rec2: rec[2],
            rec3: rec[3],
          },
          uniforms: params.data,
        },
        this.sorter.dispatchArgs,
      );
      pass.end();
    }
    this.draw(encoder, view, format, params, o.clear ?? null, profiler);
    this.stats.frames += 1;
  }

  private records: GPUBuffer[] = [];

  private draw(
    encoder: GPUCommandEncoder,
    view: GPUTextureView,
    format: GPUTextureFormat,
    params: UniformWriter,
    clear: GPUColor | null,
    profiler: GpuProfiler | null,
  ) {
    const rec = this.records;
    if (!this.pipeline || this.pipelineFormat !== format) {
      this.pipeline = createReflectedRenderPipeline(
        this.device,
        compactModule,
        {
          vertex: "atheneaCompactVertex",
          fragment: "atheneaCompactFragment",
          targets: [
            {
              format,
              blend: {
                color: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
                alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
              },
            },
          ],
          primitive: { topology: "triangle-strip" },
          label: "compact draw",
        },
      );
      this.pipelineFormat = format;
    }
    this.device.queue.writeBuffer(this.drawUniform, 0, params.data);
    const groups = createBindGroups(this.device, this.pipeline, {
      draw0: rec[0],
      draw1: rec[1],
      draw2: rec[2],
      draw3: rec[3],
      compactParams: this.drawUniform,
    });
    const pass = encoder.beginRenderPass({
      label: "compact.draw",
      colorAttachments: [
        {
          view,
          loadOp: clear ? "clear" : "load",
          clearValue: clear ?? undefined,
          storeOp: "store",
        },
      ],
      timestampWrites: profiler?.timestampWrites("compact.draw"),
    });
    pass.setPipeline(this.pipeline.pipeline);
    groups.forEach((g, i) => pass.setBindGroup(i, g));
    pass.drawIndirect(this.sorter.drawArgs, 0);
    pass.end();
  }

  /** Every gaussian's relit colour and alpha (float4), for checks. */
  async colours(camera: THREE.Camera, width: number, height: number) {
    const cloud = this.cloud as CompactCloud;
    const n = cloud.layout.count;
    const out = createStorage(this.device, n * 16, "compact colours");
    const encoder = this.device.createCommandEncoder();
    const params = this.params(camera, width, height);
    this.prepareLight(encoder, params, null);
    const pass = encoder.beginComputePass({ label: "compact.colours" });
    const row = K.kCompactDispatchRow;
    this.registry.get(compactModule, "atheneaCompactColours").dispatch(pass, {
      grid: [Math.min(n, row), Math.ceil(n / row)],
      buffers: { ...this.buffers(), debugOut: out },
      uniforms: params.data,
    });
    pass.end();
    const staging = createReadback(this.device, n * 16, "compact colours");
    encoder.copyBufferToBuffer(out, 0, staging, 0, n * 16);
    this.registry.submit(encoder.finish());
    const f = new Float32Array(await readAndDestroy(staging));
    out.destroy();
    return f;
  }

  /** The splats drawn last frame (a readback of the sort's count). */
  async drawnCount(): Promise<number> {
    const staging = createReadback(this.device, 16, "compact drawn");
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.sorter.drawArgs, 0, staging, 0, 16);
    this.device.queue.submit([encoder.finish()]);
    return new Uint32Array(await readAndDestroy(staging))[1];
  }

  /** Sizes the sort to `drawn` splats (plus `margin`), as a view needs. */
  fitCapacity(drawn: number, margin = 0.02) {
    const n = this.cloud?.layout.count ?? drawn;
    this.capacity = Math.min(n, Math.ceil(drawn * (1 + margin)));
    (this.sorter as unknown as { capacity: number }).capacity = 0;
    (this.sorter as unknown as { destroyBuffers(): void }).destroyBuffers();
    this.frameKey = "";
  }

  dispose() {
    this.data?.destroy();
    this.frame.destroy();
    this.drawUniform.destroy();
    this.sorter.destroy();
    this.sky.destroy();
    this.registry.destroy();
  }
}

/**
 * The compact cloud dequantized back into ExtSplats words and the .athc
 * streams (what WgpuSplatRenderer + atheneaRelightPlugin read), so today's
 * relight can be run on exactly the values the compact path draws: the
 * difference between the two is then the path's, not the quantization's.
 * (Re-encoding rounds the opacity, colour and ln scales to halves again, and
 * the normal to 16 bits an axis.)
 */
export function compactToSource(cloud: CompactCloud, q8: Uint32Array) {
  const n = cloud.layout.count;
  const ext0 = new Uint32Array(4 * n);
  const ext1 = new Uint32Array(4 * n);
  const f = new Float32Array(ext0.buffer);
  const normalOct = new Uint32Array(n);
  const pbr = new Uint32Array(n);
  const lobes = new Uint32Array(3 * n);
  const curvature = new Uint32Array(2 * n);
  const shadowBits = new Uint32Array(8 * n);
  const lnHalf = (s: number) =>
    toHalf(s > 0 ? Math.max(Math.log(s), -65504) : -65504);
  for (let i = 0; i < n; i++) {
    const s = decodeCompactSplat(cloud, i);
    f[4 * i] = s.center[0];
    f[4 * i + 1] = s.center[1];
    f[4 * i + 2] = s.center[2];
    ext0[4 * i + 3] = toHalf(s.opacity);
    ext1[4 * i] = toHalf(s.rgb[0]) | (toHalf(s.rgb[1]) << 16);
    ext1[4 * i + 1] = toHalf(s.rgb[2]) | (lnHalf(s.scales[0]) << 16);
    ext1[4 * i + 2] = lnHalf(s.scales[1]) | (lnHalf(s.scales[2]) << 16);
    ext1[4 * i + 3] = s.rotation;
    normalOct[i] = packNormal16(s.normal);
    pbr[i] = s.pbr;
    lobes.set(s.lobes, 3 * i);
    curvature[2 * i] = toHalf(s.curvature[0]) | (toHalf(s.curvature[1]) << 16);
    curvature[2 * i + 1] = toHalf(s.curvature[2]);
    shadowBits.set(s.cells, 8 * i);
  }
  return { ext0, ext1, normalOct, pbr, lobes, curvature, shadowBits, q8 };
}

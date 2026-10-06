// athenea's skinning on WebGPU (slang/athenea_adapter/skin.slang, the
// atheneaSkinPlugin's kernel) against athenea itself:
//
//   - athenea's own kernel, scene/splat_skin.slang as vendored (txf 89a04d9,
//     slang/tests/athenea_splat_skin.slang compiles it unchanged), on the
//     same splats in athenea's buffer layout: the posed centres must be the
//     same floats, the frames the same up to the two rest encodings (Spark's
//     ExtSplats quaternion, athenea's smallest three);
//   - the math athenea's kernel states, in float64: the centre is the
//     linear blend of the joints, and the posed in-plane covariance is
//     J E S^2 E^T J^T with J the whole Jacobian of that blend (the weights'
//     gradients, 7dff879), E the rest axes and S the sizes; the normal turns
//     by cof(J).
//
// The splats are athenea's own test case (7dff879): a strip bent a quarter
// turn across a ramp of weights, plus a sheared and a non-uniformly scaled
// joint. With the Jacobian off the adapter is the blend of the joints'
// linear parts (athenea before 7dff879), which the strip shows wrong.

import * as THREE from "three";
import { describe, expect, it } from "vitest";
import {
  decodeQuatOctXy1010R12,
  encodeExtSplat,
  fromHalf,
  toHalf,
} from "../../src/utils";
import type { KernelModule } from "../../src/webgpu/KernelModule";
import { KernelRegistry } from "../../src/webgpu/KernelRegistry";
import { transposeRows } from "../../src/webgpu/athenea/skinPlugin";
import { AttribPool } from "../../src/webgpu/attributes/schema";
import skinModule from "../../src/webgpu/generated/athenea_adapter/skin";
import { atheneaAdapterSkin as C } from "../../src/webgpu/generated/constants";
import vendoredSkin from "../../src/webgpu/generated/tests/athenea_splat_skin";
import {
  createReadback,
  createStorage,
  readAndDestroy,
  upload,
} from "../../src/webgpu/gpuBuffers";
import { UniformWriter } from "../../src/webgpu/uniforms";
import { wideDevice } from "./device";

// biome-ignore lint/suspicious/noApproximativeNumericConstant: athenea's kSqrtHalf as packing.slang writes it
const SQRT_HALF = 0.70710678;

/**
 * athenea's kernel binds eleven storage buffers and Dawn on Metal gives a
 * stage ten. Its host binds `skinningXforms` for `skinningXformsEnd` and
 * another output for `motion` whenever no shutter is open (SplatSkinner.cpp:
 * "Bound whether or not it is read, as every declared name must be"); this
 * makes the same aliasing in the compiled WGSL -- two declarations dropped,
 * their names pointed at those -- and leaves the code as compiled. With
 * motionOut 0 nothing reads the end pose or writes the motion.
 */
function aliasedVendoredSkin(): KernelModule {
  let wgsl = vendoredSkin.wgsl
    .replace(
      /@binding\(\d+\) @group\(0\) var<storage, read> skinningXformsEnd_0 : [^;]+;\n/,
      "",
    )
    .replace(
      /@binding\(\d+\) @group\(0\) var<storage, read_write> motion_0 : [^;]+;\n/,
      "",
    );
  wgsl = wgsl
    .replace(/\bskinningXformsEnd_0\b/g, "skinningXforms_0")
    .replace(/\bmotion_0\b/g, "shape_0");
  const drop = ["skinningXformsEnd", "motion"];
  const r = vendoredSkin.reflection;
  return {
    name: "tests/athenea_splat_skin (aliased)",
    wgsl,
    reflection: {
      ...r,
      entries: r.entries.map((e) => ({
        ...e,
        uses: e.uses.filter((u: string) => !drop.includes(u)),
      })),
      bindings: r.bindings.filter((b) => !drop.includes(b.name)),
    },
  } as unknown as KernelModule;
}
const refModule = aliasedVendoredSkin();

// --- athenea's packing.slang and influences.slang, in JS -------------------

function encodeQuaternion(q: number[]): number {
  const l = Math.hypot(...q);
  let v = q.map((c) => c / l);
  let largest = 0;
  for (let k = 1; k < 4; k++) {
    if (Math.abs(v[k]) > Math.abs(v[largest])) largest = k;
  }
  if (v[largest] < 0) v = v.map((c) => -c);
  let word = largest;
  let shift = 22;
  for (let k = 0; k < 4; k++) {
    if (k === largest) continue;
    const unit = Math.min(Math.max((v[k] / SQRT_HALF) * 0.5 + 0.5, 0), 1);
    word |= Math.round(unit * 1023) << shift;
    shift -= 10;
  }
  return word >>> 0;
}

function decodeQuaternion(word: number): number[] {
  const largest = word & 3;
  const small = [
    (word >>> 22) & 1023,
    (word >>> 12) & 1023,
    (word >>> 2) & 1023,
  ].map((v) => ((v / 1023) * 2 - 1) * SQRT_HALF);
  const rest = Math.sqrt(Math.max(0, 1 - small.reduce((s, v) => s + v * v, 0)));
  const q = [0, 0, 0, 0];
  let next = 0;
  for (let k = 0; k < 4; k++) q[k] = k === largest ? rest : small[next++];
  return q;
}

const packHalves = (a: number, b: number) =>
  (toHalf(a) | (toHalf(b) << 16)) >>> 0;

function packNormal(n: number[]): number {
  const l1 = Math.abs(n[0]) + Math.abs(n[1]) + Math.abs(n[2]);
  const v = n.map((c) => c / l1);
  let p = [v[0], v[1]];
  if (v[2] < 0) {
    p = [
      (1 - Math.abs(v[1])) * (v[0] >= 0 ? 1 : -1),
      (1 - Math.abs(v[0])) * (v[1] >= 0 ? 1 : -1),
    ];
  }
  const q = p.map((c) =>
    Math.round(Math.min(Math.max(c * 0.5 + 0.5, 0), 1) * 65535),
  );
  return (q[0] | (q[1] << 16)) >>> 0;
}

function unpackNormal(word: number): number[] {
  const f = [(word & 0xffff) / 65535, (word >>> 16) / 65535].map(
    (c) => c * 2 - 1,
  );
  const n = [f[0], f[1], 1 - Math.abs(f[0]) - Math.abs(f[1])];
  const t = Math.min(Math.max(-n[2], 0), 1);
  n[0] += n[0] >= 0 ? -t : t;
  n[1] += n[1] >= 0 ? -t : t;
  const l = Math.hypot(...n);
  return n.map((c) => c / l);
}

/** splatInfluencesPack (390670e), the packed layout: the running sum rounded. */
function packInfluences(joints: number[], weights: number[]): number[] {
  let sum = 0;
  let soFar = 0;
  return joints.map((j, k) => {
    sum += Math.max(weights[k], 0);
    const upTo = Math.floor(Math.min(Math.max(sum, 0), 1) * 65535 + 0.5);
    const steps = Math.max(upTo - soFar, 0);
    soFar = Math.max(upTo, soFar);
    return ((j & 0xffff) | (Math.min(steps, 0xffff) << 16)) >>> 0;
  });
}

// --- small float64 linear algebra (column vectors, row-major 3x3/4x4) ------

type M3 = number[]; // 9, row-major
const m3 = (m4: ArrayLike<number>): M3 =>
  [0, 1, 2, 4, 5, 6, 8, 9, 10].map((k) => m4[k]);
const mulM3 = (a: M3, b: M3): M3 =>
  [0, 1, 2].flatMap((r) =>
    [0, 1, 2].map(
      (c) =>
        a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c],
    ),
  );
const mulV3 = (a: M3, v: number[]) =>
  [0, 1, 2].map(
    (r) => a[r * 3] * v[0] + a[r * 3 + 1] * v[1] + a[r * 3 + 2] * v[2],
  );
const mulP4 = (m: ArrayLike<number>, p: number[]) =>
  [0, 1, 2, 3].map(
    (r) =>
      m[r * 4] * p[0] +
      m[r * 4 + 1] * p[1] +
      m[r * 4 + 2] * p[2] +
      m[r * 4 + 3] * p[3],
  );
const add = (a: number[], b: number[]) => a.map((v, k) => v + b[k]);
const scale = (a: number[], s: number) => a.map((v) => v * s);
const dot = (a: number[], b: number[]) =>
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: number[], b: number[]) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const outer = (a: number[], s: number): M3 =>
  [0, 1, 2].flatMap((r) => [0, 1, 2].map((c) => a[r] * a[c] * s));
const frob = (a: M3) => Math.sqrt(a.reduce((s, v) => s + v * v, 0));

/** axesOfQuaternion: the rotated basis, as columns. */
function axes(q: number[]): number[][] {
  const [x, y, z, w] = q;
  const m = [
    1 - 2 * (y * y + z * z),
    2 * (x * y - z * w),
    2 * (x * z + y * w),
    2 * (x * y + z * w),
    1 - 2 * (x * x + z * z),
    2 * (y * z - x * w),
    2 * (x * z - y * w),
    2 * (y * z + x * w),
    1 - 2 * (x * x + y * y),
  ];
  return [0, 1, 2].map((c) => [m[c], m[3 + c], m[6 + c]]);
}

// --- the case --------------------------------------------------------------

const SIDE = 64;
const N = SIDE * SIDE;
const JOINTS = 4;
const PER = 4;

/** USD rows (vectors on the left) of a three.js matrix. */
const usdRows = (m: THREE.Matrix4) => Array.from(m.elements); // column-major = USD's rows

function joints(): Float32Array {
  const out = new Float32Array(JOINTS * 16);
  const pivot = new THREE.Vector3(0.5, 0, 0);
  // 0: rest. 1: a quarter turn about y about the pivot, and a lift.
  const turn = new THREE.Matrix4()
    .makeTranslation(pivot.x, 0.2, 0)
    .multiply(new THREE.Matrix4().makeRotationY(Math.PI / 2))
    .multiply(new THREE.Matrix4().makeTranslation(-pivot.x, 0, 0));
  // 2: a shear. 3: a non-uniform scale and a twist.
  const shear = new THREE.Matrix4().set(
    1,
    0.4,
    0,
    0.05,
    0,
    1,
    0.2,
    0,
    0.1,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
  );
  const stretch = new THREE.Matrix4()
    .makeRotationZ(0.3)
    .multiply(new THREE.Matrix4().makeScale(1.6, 0.7, 1.1));
  [new THREE.Matrix4(), turn, shear, stretch].forEach((m, j) =>
    out.set(usdRows(m), j * 16),
  );
  return out;
}

/** The cloud's space into the joints' (a turn and a shift, as a rig's is). */
const GEOM_BIND = new THREE.Matrix4()
  .makeTranslation(0.1, -0.2, 0.3)
  .multiply(new THREE.Matrix4().makeRotationX(0.5));

interface Cloud {
  centers: number[][];
  quats: number[][]; // as made
  scales: number[][];
  influences: number[]; // PER words a splat
  gradients: number[]; // PER - 1 words a splat
  normals: number[]; // a word a splat
}

function cloud(): Cloud {
  const c: Cloud = {
    centers: [],
    quats: [],
    scales: [],
    influences: [],
    gradients: [],
    normals: [],
  };
  // The geom bind turns the strip: its rest axes and the ramp's gradient
  // are in the cloud's own space, where the conversion measured them.
  for (let i = 0; i < N; i++) {
    const x = (i % SIDE) / (SIDE - 1);
    const y = Math.floor(i / SIDE) / (SIDE - 1);
    c.centers.push([x, y, 0.01 * Math.sin(7 * x)]);
    const angle = 2.4 * i;
    const q = new THREE.Quaternion().setFromAxisAngle(
      new THREE.Vector3(0, 0, 1),
      angle,
    );
    c.quats.push([q.x, q.y, q.z, q.w]);
    c.scales.push([0.012 + 0.004 * Math.sin(i), 0.008, 0.0008]);
    // The ramp: joint 1 from x 0.25 to 0.75; a few on the sheared and
    // stretched joints.
    const w1 = Math.min(Math.max((x - 0.25) / 0.5, 0), 1);
    const dw1 = x > 0.25 && x < 0.75 ? 2 : 0;
    const extra = y > 0.5 ? 0.3 : 0;
    const w = [
      (1 - w1) * (1 - extra),
      w1 * (1 - extra),
      extra * 0.6,
      extra * 0.4,
    ];
    const ax = axes(c.quats[i]);
    const du = dw1 * ax[0][0] * (1 - extra);
    const dv = dw1 * ax[1][0] * (1 - extra);
    c.influences.push(...packInfluences([0, 1, 2, 3], w));
    // d w_k / d u, d w_k / d v for the first three joints (the fourth: minus their sum).
    c.gradients.push(
      packHalves(-du, -dv),
      packHalves(du, dv),
      packHalves(0, 0),
    );
    const tilt = [0.2 * Math.sin(i), 0.1, 1];
    c.normals.push(packNormal(tilt));
  }
  return c;
}

/** The float64 reference: athenea's statement of what its kernel does. */
function reference(
  center: number[],
  q: number[],
  sizes: number[],
  restNormal: number[],
  infl: number[],
  grads: number[],
  rows: Float32Array,
  geom: Float32Array,
  jacobian: boolean,
) {
  const bound = mulP4(geom, [...center, 1]);
  let moved = [0, 0, 0, 0];
  let blended: M3 = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  let alongU = [0, 0, 0];
  let alongV = [0, 0, 0];
  let sum = [0, 0];
  let last = [0, 0];
  let total = 0;
  for (let k = 0; k < PER; k++) {
    const joint = infl[k] & 0xffff;
    const w = (infl[k] >>> 16) / 65535;
    let g = [0, 0];
    if (jacobian) {
      if (k < PER - 1) {
        g = [fromHalf(grads[k] & 0xffff), fromHalf(grads[k] >>> 16)];
        last = [last[0] - g[0], last[1] - g[1]];
      } else g = last;
    }
    const x = rows.subarray(joint * 16, joint * 16 + 16);
    const carried = mulP4(x, bound);
    if (g[0] !== 0 || g[1] !== 0) {
      alongU = add(alongU, scale(carried.slice(0, 3), g[0]));
      alongV = add(alongV, scale(carried.slice(0, 3), g[1]));
      sum = [sum[0] + g[0], sum[1] + g[1]];
    }
    if (!(w > 1e-5)) continue;
    moved = add(moved, scale(carried, w));
    blended = add(blended, scale(m3(x), w));
    total += w;
  }
  expect(total).toBeGreaterThan(0.99);
  alongU = add(alongU, scale(moved.slice(0, 3), -sum[0]));
  alongV = add(alongV, scale(moved.slice(0, 3), -sum[1]));
  const chain = mulM3(blended, m3(geom));
  const [u, v] = axes(q);
  const ju = add(mulV3(chain, u), alongU);
  const jv = add(mulV3(chain, v), alongV);
  const cov = add(
    outer(ju, sizes[0] * sizes[0]),
    outer(jv, sizes[1] * sizes[1]),
  );
  const turn = (d: number[]) =>
    add(
      mulV3(chain, d),
      add(scale(alongU, dot(d, u)), scale(alongV, dot(d, v))),
    );
  const helper = Math.abs(restNormal[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const a0 = cross(helper, restNormal);
  const a = scale(a0, 1 / Math.hypot(...a0));
  const b = cross(restNormal, a);
  const n = cross(turn(a), turn(b));
  return {
    center: moved.slice(0, 3),
    cov,
    normal: scale(n, 1 / Math.hypot(...n)),
  };
}

/** A posed splat's in-plane covariance from its frame and two sizes. */
function covOf(q: number[], sizes: number[]): M3 {
  const [u, v] = axes(q);
  return add(outer(u, sizes[0] * sizes[0]), outer(v, sizes[1] * sizes[1]));
}

async function read(d: GPUDevice, buffer: GPUBuffer): Promise<ArrayBuffer> {
  const staging = createReadback(d, buffer.size, "skin readback");
  const encoder = d.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
  d.queue.submit([encoder.finish()]);
  return readAndDestroy(staging);
}

const rowsOf = (m: THREE.Matrix4) => Array.from(transposeRows(usdRows(m), 1));

describe.skipIf(!wideDevice)("athenea skinning", () => {
  const d = wideDevice as GPUDevice;
  const registry = new KernelRegistry(d);
  const c = cloud();
  const xformRows = transposeRows(joints(), JOINTS);
  const geomRows = new Float32Array(rowsOf(GEOM_BIND));

  // Spark's splats (ExtSplats, interleaved as GpuSplatSource.fromExt).
  const ext = new Uint32Array(N * 8);
  const a = new Uint32Array(N * 4);
  const b = new Uint32Array(N * 4);
  for (let i = 0; i < N; i++) {
    const [x, y, z] = c.centers[i];
    const [sx, sy, sz] = c.scales[i];
    const [qx, qy, qz, qw] = c.quats[i];
    encodeExtSplat(
      [a, b],
      i,
      x,
      y,
      z,
      sx,
      sy,
      sz,
      qx,
      qy,
      qz,
      qw,
      0.9,
      0.5,
      0.4,
      0.3,
    );
    ext.set(a.subarray(4 * i, 4 * i + 4), 8 * i);
    ext.set(b.subarray(4 * i, 4 * i + 4), 8 * i + 4);
  }
  const sparkRest = (i: number) => {
    const q = decodeQuatOctXy1010R12(b[4 * i + 3], new THREE.Quaternion());
    return {
      center: c.centers[i].map((v) => Math.fround(v)),
      q: [q.x, q.y, q.z, q.w],
      sizes: [
        Math.exp(fromHalf(b[4 * i + 1] >>> 16)),
        Math.exp(fromHalf(b[4 * i + 2] & 0xffff)),
      ],
    };
  };

  async function runAdapter(flags: number) {
    const pool = new AttribPool(N);
    pool.setAttribute("skinInfluences", c.influences, "u32", PER);
    pool.setAttribute("skinGradients", c.gradients, "u32", PER - 1);
    pool.setAttribute("normalOct", c.normals, "u32", 1);
    const poolBuffer = upload(d, pool.pack().words, "skin pool");
    const out = createStorage(d, ext.byteLength, "skin out");
    const params = UniformWriter.for(skinModule, "skinParams")
      .set("count", N)
      .set("flags", C.kSkinExt | C.kSkinNormals | flags)
      .set("perSplat", PER)
      .set("gradientWords", PER - 1)
      .set("influences", 0)
      .set("gradients", 1)
      .set("normals", 2)
      .set("joints", JOINTS)
      .set("encoding", [0, 1, -12, 9]);
    for (let k = 0; k < 4; k++)
      params.set(
        `geomBind${k}`,
        Array.from(geomRows.subarray(4 * k, 4 * k + 4)),
      );
    registry.get(skinModule, "atheneaSkin").run({
      grid: [N, 1],
      buffers: {
        skinRest: upload(d, ext, "skin rest"),
        skinSrc: out,
        skinPool: poolBuffer,
        skinRestNormals: upload(d, new Uint32Array(c.normals), "rest normals"),
        skinXforms: upload(d, new Uint32Array(xformRows.buffer), "xforms"),
      },
      uniforms: params.data,
    });
    const words = new Uint32Array(await read(d, out));
    const posed = new Uint32Array(await read(d, poolBuffer));
    const layout = pool.pack().layout;
    return Array.from({ length: N }, (_, i) => {
      const pa = words.subarray(8 * i, 8 * i + 4);
      const pb = words.subarray(8 * i + 4, 8 * i + 8);
      const f = new Float32Array(pa.buffer, pa.byteOffset, 3);
      const q = decodeQuatOctXy1010R12(pb[3], new THREE.Quaternion());
      const normalWord =
        posed[layout.headerWords + i * layout.strideWords + layout.offsets[2]];
      return {
        center: Array.from(f),
        q: [q.x, q.y, q.z, q.w],
        sizes: [
          Math.exp(fromHalf(pb[1] >>> 16)),
          Math.exp(fromHalf(pb[2] & 0xffff)),
        ],
        normal: unpackNormal(normalWord),
      };
    });
  }

  // athenea's kernel, as vendored, on athenea's layout of the same splats.
  async function runAthenea(gradientsIn: boolean) {
    const positions = new Float32Array(N * 4);
    const shape = new Uint32Array(N * 4);
    const pairs = new Float32Array(N * PER * 2);
    for (let i = 0; i < N; i++) {
      const r = sparkRest(i);
      positions.set([...r.center, 0.9], 4 * i);
      const ln = [
        Math.log(c.scales[i][0]),
        Math.log(c.scales[i][1]),
        Math.log(c.scales[i][2]),
      ];
      shape.set(
        [
          encodeQuaternion(r.q),
          packHalves(ln[0], ln[1]),
          packHalves(ln[2], 0.5),
          packHalves(0.4, 0.3),
        ],
        4 * i,
      );
      for (let k = 0; k < PER; k++) {
        const w = c.influences[i * PER + k];
        pairs[(i * PER + k) * 2] = w & 0xffff;
        pairs[(i * PER + k) * 2 + 1] = Math.fround((w >>> 16) / 65535);
      }
    }
    const outPositions = createStorage(d, N * 16, "ath positions");
    const outShape = createStorage(d, N * 16, "ath shape");
    const outNormals = createStorage(d, N * 4, "ath normals");
    const shapeBuffer = upload(d, shape, "ath rest shape");
    const xforms = upload(d, new Uint32Array(xformRows.buffer), "ath xforms");
    const identity = rowsOf(new THREE.Matrix4());
    const params = UniformWriter.for(refModule)
      .set("count", N)
      .set("perSplat", PER)
      .set("motionOut", 0)
      .set("normalsOut", 1)
      .set("gradientsIn", gradientsIn ? 1 : 0);
    for (let k = 0; k < 4; k++) {
      params.set(
        `geomBind${k}`,
        Array.from(geomRows.subarray(4 * k, 4 * k + 4)),
      );
      params.set(`skelToWorld${k}`, identity.slice(4 * k, 4 * k + 4));
      params.set(`worldToPrim${k}`, identity.slice(4 * k, 4 * k + 4));
    }
    registry.get(refModule, "splatSkin").run({
      grid: [N],
      buffers: {
        restPositions: upload(d, new Uint32Array(positions.buffer), "ath rest"),
        restShape: shapeBuffer,
        influences: upload(d, new Uint32Array(pairs.buffer), "ath influences"),
        skinningXforms: xforms,
        positions: outPositions,
        shape: outShape,
        restNormals: upload(d, new Uint32Array(c.normals), "ath rest normals"),
        normals: outNormals,
        weightGradients: upload(
          d,
          new Uint32Array(c.gradients),
          "ath gradients",
        ),
      },
      uniforms: params.data,
    });
    const p = new Float32Array(await read(d, outPositions));
    const s = new Uint32Array(await read(d, outShape));
    const n = new Uint32Array(await read(d, outNormals));
    return Array.from({ length: N }, (_, i) => ({
      center: Array.from(p.subarray(4 * i, 4 * i + 3)),
      q: decodeQuaternion(s[4 * i]),
      restQ: decodeQuaternion(shape[4 * i]),
      sizes: [
        Math.exp(fromHalf(s[4 * i + 1] & 0xffff)),
        Math.exp(fromHalf(s[4 * i + 1] >>> 16)),
      ],
      normal: unpackNormal(n[i]),
    }));
  }

  const ref = (i: number, q: number[], jacobian: boolean) =>
    reference(
      sparkRest(i).center,
      q,
      sparkRest(i).sizes,
      unpackNormal(c.normals[i]),
      c.influences.slice(i * PER, (i + 1) * PER),
      c.gradients.slice(i * (PER - 1), (i + 1) * (PER - 1)),
      xformRows,
      geomRows,
      jacobian,
    );

  const stats = (v: number[]) => {
    const s = [...v].sort((x, y) => x - y);
    return {
      median: s[s.length >> 1],
      p99: s[Math.floor(s.length * 0.99)],
      max: s[s.length - 1],
    };
  };

  it("poses the splats as athenea's kernel does, and as its math says", async () => {
    const ours = await runAdapter(C.kSkinJacobian);
    const theirs = await runAthenea(true);
    const centre: number[] = [];
    const centreRef: number[] = [];
    const covOurs: number[] = [];
    const covTheirs: number[] = [];
    const covBetween: number[] = [];
    const normalOurs: number[] = [];
    const normalBetween: number[] = [];
    for (let i = 0; i < N; i++) {
      const o = ours[i];
      const t = theirs[i];
      const r = ref(i, sparkRest(i).q, true);
      const rt = ref(i, t.restQ, true);
      centre.push(Math.hypot(...o.center.map((v, k) => v - t.center[k])));
      centreRef.push(Math.hypot(...o.center.map((v, k) => v - r.center[k])));
      const co = covOf(o.q, o.sizes);
      const ct = covOf(t.q, t.sizes);
      covOurs.push(frob(add(co, scale(r.cov, -1))) / frob(r.cov));
      covTheirs.push(frob(add(ct, scale(rt.cov, -1))) / frob(rt.cov));
      covBetween.push(frob(add(co, scale(ct, -1))) / frob(ct));
      normalOurs.push(Math.acos(Math.min(1, dot(o.normal, r.normal))));
      normalBetween.push(Math.acos(Math.min(1, dot(o.normal, t.normal))));
    }
    const report = {
      centreVsAthenea: stats(centre),
      centreVsFloat64: stats(centreRef),
      covVsFloat64: stats(covOurs),
      atheneaCovVsFloat64: stats(covTheirs),
      covVsAthenea: stats(covBetween),
      normalVsFloat64: stats(normalOurs),
      normalVsAthenea: stats(normalBetween),
    };
    console.log("athenea skin:", JSON.stringify(report));
    // The same floats where the rest encodings do not enter.
    expect(report.centreVsAthenea.max).toBeLessThan(1e-6);
    expect(report.centreVsFloat64.max).toBeLessThan(2e-6);
    // The frame: what f16 sizes (a step of 2^-9 to 2^-8 in ln s here) and a
    // 10-bit rotation leave of the posed covariance, each way.
    expect(report.covVsFloat64.p99).toBeLessThan(1e-2);
    expect(report.covVsFloat64.max).toBeLessThan(1.5e-2);
    expect(report.atheneaCovVsFloat64.p99).toBeLessThan(1e-2);
    expect(report.covVsAthenea.p99).toBeLessThan(1.2e-2);
    // athenea's metric (7dff879): none off by over 3% with the Jacobian.
    expect(covOurs.filter((e) => e > 0.03).length).toBe(0);
    expect(report.normalVsFloat64.max).toBeLessThan(1e-3);
    expect(report.normalVsAthenea.p99).toBeLessThan(3e-3);
  });

  it("is the blend of the joints' linear parts without the Jacobian, as athenea was", async () => {
    const blend = await runAdapter(0);
    const whole = await runAdapter(C.kSkinJacobian);
    const theirs = await runAthenea(false);
    let off = 0;
    const vsBlend: number[] = [];
    const vsAthenea: number[] = [];
    for (let i = 0; i < N; i++) {
      const r = ref(i, sparkRest(i).q, true);
      const rb = ref(i, sparkRest(i).q, false);
      const cb = covOf(blend[i].q, blend[i].sizes);
      vsBlend.push(frob(add(cb, scale(rb.cov, -1))) / frob(rb.cov));
      vsAthenea.push(
        frob(add(cb, scale(covOf(theirs[i].q, theirs[i].sizes), -1))) /
          frob(rb.cov),
      );
      // athenea's metric (7dff879): off by over 3% against J E S^2 E^T J^T.
      if (frob(add(cb, scale(r.cov, -1))) / frob(r.cov) > 0.03) off += 1;
      expect(blend[i].center).toEqual(whole[i].center);
    }
    console.log(
      "athenea skin, blend only:",
      JSON.stringify({
        off,
        vsBlend: stats(vsBlend),
        vsAthenea: stats(vsAthenea),
      }),
    );
    expect(stats(vsBlend).p99).toBeLessThan(1e-2);
    expect(stats(vsAthenea).p99).toBeLessThan(1.2e-2);
    // The ramp's splats are stretched across the bend: the blend misses it.
    expect(off).toBeGreaterThan(N / 8);
  });
});

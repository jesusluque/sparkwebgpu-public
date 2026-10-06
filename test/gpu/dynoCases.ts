// A graph per dyno op with its expected value under GLSL semantics, computed
// in JS. Inputs are uniforms so a compiler can't fold the ops away. Used by
// the WGSL GPU tests and the GLSL snapshot test.

import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { PackedSplats } from "../../src/PackedSplats";
import * as d from "../../src/dyno";

type Val = d.DynoVal<d.DynoType>;
export type Expected = number[] | ((v: number[]) => void);

export const graph = (type: d.DynoType, f: (index: Val) => Val) =>
  d.dynoBlock({ index: "int" }, { value: type }, ({ index }) => ({
    value: f(index as Val) as never,
  }));

// biome-ignore lint/suspicious/noExplicitAny: ops are generic over many types
const D = d as any;
const f1 = (v: number) => d.dynoFloat(v);
const v2 = (x: number, y: number) => d.dynoVec2([x, y]);
const v3 = (x: number, y: number, z: number) => d.dynoVec3([x, y, z]);
const v4 = (x: number, y: number, z: number, w: number) =>
  d.dynoVec4([x, y, z, w]);
const i1 = (v: number) => d.dynoInt(v);
const u1 = (v: number) => d.dynoUint(v);
const b1 = (v: boolean) => d.dynoBool(v);
const iv3 = (x: number, y: number, z: number) => d.dynoIvec3([x, y, z]);
const uv3 = (x: number, y: number, z: number) => d.dynoUvec3([x, y, z]);
const bv3 = (x: boolean, y: boolean, z: boolean) => d.dynoBvec3([x, y, z]);

const fract = (x: number) => x - Math.floor(x);
const glslMod = (x: number, y: number) => x - y * Math.floor(x / y);
const smooth = (e0: number, e1: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
};

// The PCG hash of src/dyno/util.ts.
const pcgNext = (s: number) => (Math.imul(s, 747796405) + 2891336453) >>> 0;
const pcgHash = (s: number) => {
  const h = Math.imul(((s >>> ((s >>> 28) + 4)) ^ s) >>> 0, 277803737) >>> 0;
  return ((h >>> 22) ^ h) >>> 0;
};
const hashWords = (seed: number, n: number) => {
  let s = seed >>> 0;
  const out = [];
  for (let i = 0; i < n; i++) {
    s = pcgNext(s);
    out.push(pcgHash(s));
  }
  return out;
};
const f32bits = (x: number) => new Uint32Array(new Float32Array([x]).buffer)[0];

const m3 = new THREE.Matrix3().set(2, 0, 1, 1, 3, 0, 0, 1, 4);
const m4 = new THREE.Matrix4().compose(
  new THREE.Vector3(1, 2, 3),
  new THREE.Quaternion().setFromEuler(new THREE.Euler(0.3, -0.5, 0.9)),
  new THREE.Vector3(1.5, 0.5, 2),
);
const q = new THREE.Quaternion()
  .setFromEuler(new THREE.Euler(0.4, 1.1, -0.3))
  .normalize();

export const cases: [string, d.DynoType, () => Val, Expected][] = [
  // math
  [
    "add vec3+float",
    "vec3",
    () => d.add(v3(1, 2, 3), f1(0.5)),
    [1.5, 2.5, 3.5],
  ],
  ["sub", "vec2", () => d.sub(v2(1, 2), v2(3, 5)), [-2, -3]],
  [
    "mul mat3*vec3",
    "vec3",
    () => d.mul(d.dynoMat3(m3), v3(1, 2, 3)),
    new THREE.Vector3(1, 2, 3).applyMatrix3(m3).toArray(),
  ],
  [
    "mul mat4*mat4",
    "mat4",
    () => d.mul(d.dynoMat4(m4), d.dynoMat4(m4)),
    new THREE.Matrix4().multiplyMatrices(m4, m4).toArray(),
  ],
  ["div ivec3/int", "ivec3", () => d.div(iv3(7, -7, 9), i1(2)), [3, -3, 4]],
  ["imod", "int", () => d.imod(i1(17), i1(5)), [2]],
  ["mod negative", "vec3", () => d.mod(v3(-1.5, 2.5, 7), f1(2)), [0.5, 0.5, 1]],
  [
    "modf",
    "vec2",
    () => {
      const m = d.modf(v2(2.75, -1.25));
      return d.add(m.fract, d.mul(m.integer, f1(10)));
    },
    [20.75, -10.25],
  ],
  ["neg", "vec3", () => d.neg(v3(1, -2, 0)), [-1, 2, -0]],
  ["abs int", "int", () => d.abs(i1(-4)), [4]],
  ["sign", "vec3", () => d.sign(v3(-3, 0, 2)), [-1, 0, 1]],
  [
    "floor/ceil/trunc/round/fract",
    "vec4",
    () =>
      d.combine({
        vectorType: "vec4",
        x: d.floor(f1(-1.5)),
        y: d.ceil(f1(-1.5)),
        z: d.trunc(f1(-1.5)),
        w: d.fract(f1(-1.25)),
      } as never),
    [-2, -1, -1, 0.75],
  ],
  ["round", "float", () => d.round(f1(2.4)), [2]],
  [
    "pow/exp/exp2",
    "vec3",
    () =>
      d.combine({
        vectorType: "vec3",
        x: d.pow(f1(2), f1(3)),
        y: d.exp(f1(1)),
        z: d.exp2(f1(5)),
      } as never),
    [8, Math.E, 32],
  ],
  [
    "log/log2",
    "vec2",
    () =>
      d.combine({
        vectorType: "vec2",
        x: d.log(f1(Math.E)),
        y: d.log2(f1(64)),
      } as never),
    [1, 6],
  ],
  [
    "sqr/sqrt/inversesqrt",
    "vec3",
    () =>
      d.combine({
        vectorType: "vec3",
        x: d.sqr(f1(3)),
        y: d.sqrt(f1(16)),
        z: d.inversesqrt(f1(4)),
      } as never),
    [9, 4, 0.5],
  ],
  ["min vec3,float", "vec3", () => d.min(v3(1, 5, 3), f1(2)), [1, 2, 2]],
  ["max ivec3", "ivec3", () => d.max(iv3(1, 5, -3), iv3(2, 2, 2)), [2, 5, 2]],
  [
    "clamp vec3,float,float",
    "vec3",
    () => d.clamp(v3(-1, 0.5, 3), f1(0), f1(1)),
    [0, 0.5, 1],
  ],
  [
    "mix float t",
    "vec3",
    () => d.mix(v3(0, 0, 0), v3(2, 4, 6), f1(0.25)),
    [0.5, 1, 1.5],
  ],
  [
    "mix bvec t",
    "vec3",
    () => d.mix(v3(0, 0, 0), v3(2, 4, 6), bv3(true, false, true) as never),
    [2, 0, 6],
  ],
  ["step", "vec3", () => d.step(v3(1, 1, 1), v3(0, 1, 2)), [0, 1, 1]],
  [
    "smoothstep",
    "vec3",
    () => d.smoothstep(f1(0), f1(2), v3(-1, 0.5, 3)),
    [0, smooth(0, 2, 0.5), 1],
  ],
  [
    "isNan/isInf",
    "ivec4",
    () =>
      d.combine({
        vectorType: "ivec4",
        x: d.int(d.isNan(d.div(f1(0), f1(0)))),
        y: d.int(d.isNan(f1(1))),
        z: d.int(d.isInf(d.div(f1(1), f1(0)))),
        w: d.int(d.isInf(f1(1))),
      } as never),
    [1, 0, 1, 0],
  ],
  // trig
  [
    "radians/degrees",
    "vec2",
    () =>
      d.combine({
        vectorType: "vec2",
        x: D.radians(f1(180)),
        y: D.degrees(f1(Math.PI / 2)),
      } as never),
    [Math.PI, 90],
  ],
  [
    "sin/cos/tan",
    "vec3",
    () =>
      d.combine({
        vectorType: "vec3",
        x: D.sin(f1(0.5)),
        y: D.cos(f1(0.5)),
        z: D.tan(f1(0.5)),
      } as never),
    [Math.sin(0.5), Math.cos(0.5), Math.tan(0.5)],
  ],
  [
    "asin/acos/atan",
    "vec3",
    () =>
      d.combine({
        vectorType: "vec3",
        x: D.asin(f1(0.5)),
        y: D.acos(f1(0.5)),
        z: D.atan(f1(2)),
      } as never),
    [Math.asin(0.5), Math.acos(0.5), Math.atan(2)],
  ],
  ["atan2", "float", () => D.atan2(f1(-1), f1(-2)), [Math.atan2(-1, -2)]],
  [
    "sinh/cosh/tanh",
    "vec3",
    () =>
      d.combine({
        vectorType: "vec3",
        x: D.sinh(f1(0.5)),
        y: D.cosh(f1(0.5)),
        z: D.tanh(f1(0.5)),
      } as never),
    [Math.sinh(0.5), Math.cosh(0.5), Math.tanh(0.5)],
  ],
  [
    "asinh/acosh/atanh",
    "vec3",
    () =>
      d.combine({
        vectorType: "vec3",
        x: D.asinh(f1(0.5)),
        y: D.acosh(f1(1.5)),
        z: D.atanh(f1(0.5)),
      } as never),
    [Math.asinh(0.5), Math.acosh(1.5), Math.atanh(0.5)],
  ],
  // vecmat
  [
    "length/distance/dot",
    "vec3",
    () =>
      d.combine({
        vectorType: "vec3",
        x: d.length(v3(3, 4, 12)),
        y: d.distance(v2(1, 1), v2(4, 5)),
        z: d.dot(v3(1, 2, 3), v3(4, 5, 6)),
      } as never),
    [13, 5, 32],
  ],
  ["cross", "vec3", () => d.cross(v3(1, 0, 0), v3(0, 1, 0)), [0, 0, 1]],
  ["normalize", "vec3", () => d.normalize(v3(3, 0, 4)), [0.6, 0, 0.8]],
  ["projectH", "vec3", () => d.projectH(v4(2, 4, 6, 2)), [1, 2, 3]],
  ["extendVec", "vec4", () => d.extendVec(v3(1, 2, 3), f1(4)), [1, 2, 3, 4]],
  [
    "faceforward",
    "vec3",
    () => d.faceforward(v3(0, 0, 1), v3(0, 0, 1), v3(0, 0, 1)),
    [0, 0, -1],
  ],
  ["reflect", "vec3", () => d.reflectVec(v3(1, -1, 0), v3(0, 1, 0)), [1, 1, 0]],
  [
    "refract",
    "vec3",
    () => d.refractVec(v3(0, -1, 0), v3(0, 1, 0), f1(0.5)),
    [0, -1, 0],
  ],
  [
    "compMult",
    "mat2",
    () =>
      d.compMult(
        d.dynoMat2(new Float32Array([1, 2, 3, 4]) as never),
        d.dynoMat2(new Float32Array([5, 6, 7, 8]) as never),
      ),
    [5, 12, 21, 32],
  ],
  [
    "outer vec3,vec2",
    "mat2x3",
    () => d.outer(v3(1, 2, 3), v2(10, 100)),
    [10, 20, 30, 100, 200, 300],
  ],
  [
    "transpose",
    "mat3",
    () => d.transpose(d.dynoMat3(m3)),
    m3.clone().transpose().toArray(),
  ],
  [
    "determinant",
    "float",
    () => d.determinant(d.dynoMat3(m3)),
    [m3.determinant()],
  ],
  [
    "inverse mat2",
    "mat2",
    () => d.inverse(d.dynoMat2(new Float32Array([4, 2, 7, 6]) as never)),
    [0.6, -0.2, -0.7, 0.4],
  ],
  [
    "inverse mat3",
    "mat3",
    () => d.inverse(d.dynoMat3(m3)),
    m3.clone().invert().toArray(),
  ],
  [
    "inverse mat4",
    "mat4",
    () => d.inverse(d.dynoMat4(m4)),
    m4.clone().invert().toArray(),
  ],
  [
    "split/swizzle",
    "vec4",
    () => {
      const s = d.split(v4(1, 2, 3, 4)).outputs;
      return d.combine({
        vectorType: "vec4",
        x: s.w,
        y: s.b,
        z: d.swizzle(v4(5, 6, 7, 8), "y" as never),
        w: s.r,
      } as never);
    },
    [4, 3, 6, 1],
  ],
  [
    "swizzle vec3",
    "vec3",
    () => d.swizzle(v4(5, 6, 7, 8), "wzx" as never),
    [8, 7, 5],
  ],
  [
    "combine with vector",
    "ivec3",
    () => d.combine({ vector: iv3(1, 2, 3), y: i1(9) } as never),
    [1, 9, 3],
  ],
  [
    "combine defaults",
    "uvec3",
    () => d.combine({ vectorType: "uvec3", z: u1(9) } as never),
    [0, 0, 9],
  ],
  // logic
  [
    "and/or/xor bool",
    "ivec3",
    () =>
      d.combine({
        vectorType: "ivec3",
        x: d.int(d.and(b1(true), b1(false))),
        y: d.int(d.or(b1(true), b1(false))),
        z: d.int(d.xor(b1(true), b1(true))),
      } as never),
    [0, 1, 0],
  ],
  [
    "and/or/xor uint",
    "uvec3",
    () =>
      d.combine({
        vectorType: "uvec3",
        x: d.and(u1(12), u1(10)),
        y: d.or(u1(12), u1(10)),
        z: d.xor(u1(12), u1(10)),
      } as never),
    [8, 14, 6],
  ],
  [
    "not bool/bvec/int",
    "ivec3",
    () =>
      d.combine({
        vectorType: "ivec3",
        x: d.int(d.not(b1(false))),
        y: d.int(d.any(d.not(bv3(true, true, true)))),
        z: d.not(i1(5)),
      } as never),
    [1, 0, -6],
  ],
  [
    "compare scalar",
    "ivec4",
    () =>
      d.combine({
        vectorType: "ivec4",
        x: d.int(d.lessThan(f1(1), f1(2))),
        y: d.int(d.lessThanEqual(i1(2), i1(2))),
        z: d.int(d.greaterThan(u1(1), u1(2))),
        w: d.int(d.greaterThanEqual(f1(3), f1(2))),
      } as never),
    [1, 1, 0, 1],
  ],
  [
    "compare vector",
    "bvec3",
    () => d.lessThan(v3(1, 2, 3), v3(2, 2, 2)),
    [1, 0, 0],
  ],
  [
    "equal/notEqual",
    "bvec3",
    () => d.notEqual(iv3(1, 2, 3), iv3(1, 0, 3)),
    [0, 1, 0],
  ],
  ["equal scalar", "bool", () => d.equal(f1(2), f1(2)), [1]],
  [
    "any/all",
    "ivec2",
    () =>
      d.combine({
        vectorType: "ivec2",
        x: d.int(d.any(bv3(false, true, false))),
        y: d.int(d.all(bv3(true, true, false))),
      } as never),
    [1, 0],
  ],
  ["select float", "float", () => d.select(b1(false), f1(1), f1(2)), [2]],
  [
    "select vec3",
    "vec3",
    () => d.select(b1(true), v3(1, 2, 3), v3(4, 5, 6)),
    [1, 2, 3],
  ],
  [
    "select mat2",
    "mat2",
    () =>
      d.select(
        b1(true),
        d.dynoMat2(new Float32Array([1, 2, 3, 4]) as never),
        d.dynoMat2(new Float32Array([5, 6, 7, 8]) as never),
      ),
    [1, 2, 3, 4],
  ],
  ["compXor bvec3", "bool", () => d.compXor(bv3(true, true, true)), [1]],
  ["compXor ivec3", "int", () => d.compXor(iv3(1, 2, 4)), [7]],
  [
    "shr/shl",
    "ivec2",
    () =>
      d.combine({
        vectorType: "ivec2",
        x: d.shr(i1(-16), i1(2)),
        y: d.shl(i1(3), i1(4)),
      } as never),
    [-4, 48],
  ],
  ["shr ivec3", "ivec3", () => d.shr(iv3(16, 32, 64), iv3(1, 2, 3)), [8, 8, 8]],
  // convert
  [
    "bool/int/uint/float",
    "vec4",
    () =>
      d.combine({
        vectorType: "vec4",
        x: d.float(d.bool(f1(0.5))),
        y: d.float(d.int(f1(-2.7))),
        z: d.float(d.uint(f1(3.9))),
        w: d.float(i1(-5)),
      } as never),
    [1, -2, 3, -5],
  ],
  ["bvec3(vec3)", "bvec3", () => d.bvec3(v3(0, 1, -1)), [0, 1, 1]],
  ["ivec3(vec3)", "ivec3", () => d.ivec3(v3(1.5, -2.5, 3)), [1, -2, 3]],
  ["uvec3(ivec3)", "uvec3", () => d.uvec3(iv3(1, 2, 3)), [1, 2, 3]],
  ["vec2(vec4)", "vec2", () => d.vec2(v4(1, 2, 3, 4)), [1, 2]],
  ["vec3(float)", "vec3", () => d.vec3(f1(2)), [2, 2, 2]],
  [
    "vec4(uvec4)",
    "vec4",
    () => d.vec4(d.dynoUvec4([1, 2, 3, 4])),
    [1, 2, 3, 4],
  ],
  ["mat3(float)", "mat3", () => d.mat3(f1(2)), [2, 0, 0, 0, 2, 0, 0, 0, 2]],
  [
    "mat3(mat4)",
    "mat3",
    () => d.mat3(d.dynoMat4(m4)),
    new THREE.Matrix3().setFromMatrix4(m4).toArray(),
  ],
  [
    "mat4(mat2)",
    "mat4",
    () => d.mat4(d.dynoMat2(new Float32Array([1, 2, 3, 4]) as never)),
    [1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
  ],
  [
    "float bits",
    "uvec4",
    () =>
      d.combine({
        vectorType: "uvec4",
        x: d.floatBitsToUint(f1(1)),
        y: d.uint(d.floatBitsToInt(f1(-2))),
        z: d.floatBitsToUint(d.intBitsToFloat(i1(7))),
        w: d.floatBitsToUint(d.uintBitsToFloat(u1(0x3f000000))),
      } as never),
    [f32bits(1), f32bits(-2), 7, 0x3f000000],
  ],
  [
    "pack/unpack",
    "vec4",
    () =>
      d.combine({
        vectorType: "vec4",
        x: d.swizzle(
          d.unpackHalf2x16(d.packHalf2x16(v2(1.5, -2))),
          "y" as never,
        ),
        y: d.swizzle(
          d.unpackSnorm2x16(d.packSnorm2x16(v2(-1, 0.5))),
          "x" as never,
        ),
        z: d.swizzle(
          d.unpackUnorm2x16(d.packUnorm2x16(v2(0, 1))),
          "y" as never,
        ),
        w: d.float(d.packHalf2x16(v2(1, 0))),
      } as never),
    [-2, -1, 1, 0x3c00],
  ],
  [
    "uintToRgba8",
    "vec4",
    () => d.uintToRgba8(u1(0x80ff0033)),
    [0x33 / 255, 0, 1, 0x80 / 255],
  ],
  // util
  ["remapIndex", "int", () => d.remapIndex(i1(10), i1(4), i1(100)), [106]],
  [
    "pcg",
    "uint",
    () => d.pcgHash(d.pcgNext(d.pcgMix(u1(42)))),
    [pcgHash(pcgNext(42))],
  ],
  ["hash int", "uint", () => d.hash(i1(7)), hashWords(7, 1)],
  [
    "hash2 uvec2",
    "uvec2",
    () => d.hash2(d.dynoUvec2([3, 5])),
    hashWords((3 + Math.imul(0x9e3779b9, 5)) >>> 0, 2),
  ],
  [
    "hash3 vec3",
    "uvec3",
    () => d.hash3(v3(1, 2, 3)),
    hashWords(
      (f32bits(1) +
        Math.imul(0x9e3779b9, f32bits(2)) +
        Math.imul(0x85ebca6b, f32bits(3))) >>>
        0,
      3,
    ),
  ],
  [
    "hash4 ivec4",
    "uvec4",
    () => d.hash4(d.dynoIvec4([1, 2, 3, 4])),
    hashWords(
      (1 +
        Math.imul(0x9e3779b9, 2) +
        Math.imul(0x85ebca6b, 3) +
        Math.imul(0xc2b2ae35, 4)) >>>
        0,
      4,
    ),
  ],
  [
    "hashFloat",
    "float",
    () => d.hashFloat(i1(9)),
    [Math.fround(hashWords(9, 1)[0]) / 2 ** 32],
  ],
  [
    "hashVec4",
    "vec4",
    () => d.hashVec4(i1(9)),
    hashWords(9, 4).map((w) => Math.fround(w) / 2 ** 32),
  ],
  [
    "normalizedDepth",
    "float",
    () => d.normalizedDepth(f1(5), f1(1), f1(10)),
    [(Math.log2(6) - 1) / (Math.log2(11) - 1)],
  ],
  [
    "debugColorHue",
    "vec3",
    () => d.debugColorHue(i1(3)),
    (v) => expect(v.every((x) => x >= 0.15 - 1e-6 && x <= 1)).toBe(true),
  ],
  // transform
  [
    "transformPos",
    "vec3",
    () =>
      d.transformPos(v3(1, 2, 3), {
        scale: f1(2),
        rotate: d.dynoVec4(q),
        translate: v3(1, 0, 0),
      }),
    new THREE.Vector3(2, 4, 6)
      .applyQuaternion(q)
      .add(new THREE.Vector3(1, 0, 0))
      .toArray(),
  ],
  [
    "transformDir",
    "vec3",
    () =>
      d.transformDir(v3(1, 2, 3), {
        scales: v3(1, 2, 3),
        rotate: d.dynoVec4(q),
      }),
    new THREE.Vector3(1, 4, 9).applyQuaternion(q).toArray(),
  ],
  [
    "transformQuat",
    "vec4",
    () => d.transformQuat(v4(0, 0, 0, 1), { rotate: d.dynoVec4(q) }),
    q.toArray(),
  ],
  // values
  [
    "consts",
    "vec4",
    () =>
      d.combine({
        vectorType: "vec4",
        x: d.dynoConst("float", 1.5),
        y: d.float(d.dynoConst("int", -3)),
        z: d.float(d.dynoConst("uint", 4)),
        w: d.float(d.dynoConst("bool", true)),
      } as never),
    [1.5, -3, 4, 1],
  ],
  [
    "const vectors",
    "ivec3",
    () =>
      d.add(
        d.dynoConst("ivec3", [1, 2, 3]),
        d.ivec3(d.dynoConst("uvec3", [1, 1, 1])),
      ),
    [2, 3, 4],
  ],
  [
    "const bvec/mat",
    "vec3",
    () =>
      d.mul(
        d.dynoConst("mat3", m3),
        d.vec3(d.dynoConst("bvec3", [true, false, true])),
      ),
    new THREE.Vector3(1, 0, 1).applyMatrix3(m3).toArray(),
  ],
  [
    "const infinity",
    "bool",
    () => d.isInf(d.dynoConst("float", Number.NEGATIVE_INFINITY)),
    [1],
  ],
  ["literal", "float", () => d.dynoLiteral("float", "PI"), [Math.PI]],
  [
    "uniform bool/bvec",
    "ivec3",
    () =>
      d.combine({
        vectorType: "ivec3",
        x: d.int(b1(true)),
        y: d.int(d.any(d.dynoBvec2([false, false]))),
        z: d.int(d.all(d.dynoBvec2([true, true]))),
      } as never),
    [1, 0, 1],
  ],
  [
    "uniform mat2x3",
    "vec3",
    () => d.mul(d.dynoMat2x3(new Float32Array([1, 2, 3, 4, 5, 6])), v2(1, 10)),
    [41, 52, 63],
  ],
  [
    "uniform mat3x2",
    "vec2",
    () =>
      d.mul(d.dynoMat3x2(new Float32Array([1, 2, 3, 4, 5, 6])), v3(1, 10, 100)),
    [531, 642],
  ],
  [
    "uniform array",
    "vec4",
    () =>
      new d.Dyno({
        inTypes: { arr: "vec4" },
        outTypes: { out: "vec4" },
        inputs: {
          arr: new d.DynoUniform({
            key: "arr",
            type: "vec4",
            count: 3,
            value: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
          }),
        },
        statements: ({ inputs, outputs }) => [
          `${outputs.out} = ${inputs.arr}[0] + ${inputs.arr}[2];`,
        ],
      }).outputs.out as Val,
    [10, 12, 14, 16],
  ],
  // Arrays of elements under 16 bytes, padded to vec4s in WGSL.
  [
    "uniform float, vec2 and bool arrays",
    "vec2",
    () =>
      new d.Dyno({
        inTypes: { f: "float", v: "vec2", b: "bool" },
        outTypes: { out: "vec2" },
        inputs: {
          f: new d.DynoUniform({
            key: "f",
            type: "float",
            count: 3,
            value: new Float32Array([1, 2, 3]),
          }),
          v: new d.DynoUniform({
            key: "v",
            type: "vec2",
            count: 2,
            value: new Float32Array([10, 20, 30, 40]),
          }),
          b: new d.DynoUniform({
            key: "b",
            type: "bool",
            count: 2,
            value: [false, true],
          }),
        },
        statements: ({ inputs, outputs }) => [
          `${outputs.out} = ${inputs.v}[1] + ${inputs.f}[2] + float(${inputs.b}[1]) * 100.0 + float(${inputs.b}[0]) * 1000.0;`,
        ],
      }).outputs.out as Val,
    [133, 143],
  ],
  [
    "uniform index",
    "int",
    () =>
      graph("int", (index) => d.add(index as never, i1(5))).apply({
        index: i1(2),
      }).value as Val,
    [7],
  ],
];

// Gsplat ops on one splat, in object space.
const splat = () =>
  d.combineGsplat({
    flags: d.dynoLiteral("uint", "GSPLAT_FLAG_ACTIVE"),
    index: i1(3),
    center: v3(1, 2, 3),
    scales: v3(0.5, 0.1, 0.25),
    quaternion: d.dynoVec4(q),
    rgba: v4(0.2, 0.4, 0.6, 0.8),
  });
export const splatCases: [string, d.DynoType, () => Val, Expected][] = [
  [
    "split/combine gsplat",
    "vec4",
    () => {
      const s = d.splitGsplat(
        d.combineGsplat({ gsplat: splat(), r: f1(1), opacity: f1(0.5) }),
      ).outputs;
      return d.extendVec(d.add(s.center, s.rgb), s.opacity);
    },
    [2, 2.4, 3.6, 0.5],
  ],
  [
    "combine gsplat rgb",
    "vec4",
    () => {
      const s = d.splitGsplat(
        d.combineGsplat({
          gsplat: splat(),
          rgb: d.dynoConst("vec3", [0.25, 0.5, 0.75]),
        }),
      ).outputs;
      return d.extendVec(s.rgb, s.opacity);
    },
    (v) => expect(v.slice(0, 3)).toEqual([0.25, 0.5, 0.75]),
  ],
  [
    "combine covsplat rgb",
    "vec3",
    () =>
      d.splitCovSplat(
        d.combineCovSplat({ rgb: d.dynoConst("vec3", [0.25, 0.5, 0.75]) }),
      ).outputs.rgb,
    [0.25, 0.5, 0.75],
  ],
  [
    "gsplat active/index",
    "ivec2",
    () => {
      const s = d.splitGsplat(splat()).outputs;
      return d.combine({
        vectorType: "ivec2",
        x: d.int(s.active),
        y: s.index,
      } as never);
    },
    [1, 3],
  ],
  [
    "gsplatNormal",
    "vec3",
    () => d.gsplatNormal(splat()),
    new THREE.Vector3(0, 1, 0).applyQuaternion(q).toArray(),
  ],
  [
    "gsplat axes",
    "vec3",
    () =>
      d.add(
        d.add(d.gsplatXAxis(splat()), d.gsplatYAxis(splat())),
        d.gsplatZAxis(splat()),
      ),
    new THREE.Vector3(1, 1, 1).applyQuaternion(q).toArray(),
  ],
  [
    "transformGsplat",
    "vec4",
    () => {
      const s = d.splitGsplat(
        d.transformGsplat(splat(), {
          scale: f1(2),
          translate: v3(1, 1, 1),
          recolor: v4(1, 1, 1, 0.5),
        }),
      ).outputs;
      return d.extendVec(s.center, s.opacity);
    },
    [3, 5, 7, 0.4],
  ],
  [
    "gsplatToCovSplat",
    "vec3",
    () =>
      d.splitCovSplat(
        d.combineCovSplat({ covsplat: d.gsplatToCovSplat(splat()), r: f1(1) }),
      ).outputs.rgb,
    [1, 0.4, 0.6],
  ],
  [
    "splatTexCoord",
    "ivec3",
    () => d.splatTexCoord(i1(2048 * 2048 * 2 + 2048 * 3 + 5)),
    [5, 3, 2],
  ],
  [
    "pagedSplatTexCoord",
    "ivec3",
    () => d.pagedSplatTexCoord(i1(65536 * 2 + 256 * 3 + 5)),
    [5, 3, 2],
  ],
];

// Sampler uniforms: texture bindings read with textureLoad/textureSampleLevel.
function dataTexture(
  data: ArrayBufferView,
  width: number,
  height: number,
  format: THREE.AnyPixelFormat,
  type: THREE.TextureDataType,
) {
  const t = new THREE.DataTexture(data as never, width, height, format, type);
  t.needsUpdate = true;
  return t;
}
export const uTex = dataTexture(
  Uint32Array.from({ length: 4 * 4 * 3 }, (_, i) => i * 3),
  4,
  3,
  THREE.RGBAIntegerFormat,
  THREE.UnsignedIntType,
);
const fTex = dataTexture(
  Float32Array.from({ length: 2 * 2 * 4 }, (_, i) => i * 0.5),
  2,
  2,
  THREE.RGBAFormat,
  THREE.FloatType,
);
// Two texels, 0 and 255 in red, linearly filtered.
export const bTex = dataTexture(
  new Uint8Array([0, 0, 0, 255, 255, 0, 0, 255]),
  2,
  1,
  THREE.RGBAFormat,
  THREE.UnsignedByteType,
);
bTex.magFilter = THREE.LinearFilter;
bTex.minFilter = THREE.LinearFilter;
const arrayTex = new THREE.DataArrayTexture(
  Uint32Array.from({ length: 2 * 2 * 3 * 4 }, (_, i) => i),
  2,
  2,
  3,
);
arrayTex.format = THREE.RGBAIntegerFormat;
arrayTex.type = THREE.UnsignedIntType;
arrayTex.needsUpdate = true;
// An 8-bit sRGB texture reads back decoded, as WebGL's SRGB8_ALPHA8.
const srgbTex = dataTexture(
  new Uint8Array([128, 0, 255, 128]),
  1,
  1,
  THREE.RGBAFormat,
  THREE.UnsignedByteType,
);
srgbTex.colorSpace = THREE.SRGBColorSpace;
// Two 1x1 layers, red 10 and 200: texture() picks the layer floor(z + 0.5).
const layersTex = new THREE.DataArrayTexture(
  new Uint8Array([10, 0, 0, 255, 200, 0, 0, 255]),
  1,
  1,
  2,
);
layersTex.needsUpdate = true;

export const matrixCases: [string, d.DynoType, () => Val, Expected][] = [
  [
    "mat2 + float",
    "mat2",
    () => d.add(d.dynoMat2(new Float32Array([1, 2, 3, 4]) as never), f1(10)),
    [11, 12, 13, 14],
  ],
  [
    "float - mat2",
    "mat2",
    () => d.sub(f1(10), d.dynoMat2(new Float32Array([1, 2, 3, 4]) as never)),
    [9, 8, 7, 6],
  ],
  [
    "mat2 / float",
    "mat2",
    () => d.div(d.dynoMat2(new Float32Array([2, 4, 6, 8]) as never), f1(2)),
    [1, 2, 3, 4],
  ],
  [
    "-mat2",
    "mat2",
    () => d.neg(d.dynoMat2(new Float32Array([1, -2, 3, 4]) as never)),
    [-1, 2, -3, -4],
  ],
];

export const textureCases: [string, d.DynoType, () => Val, Expected][] = [
  [
    "texelFetch usampler2D",
    "uvec4",
    () => d.texelFetch(d.dynoUsampler2D(uTex), d.dynoIvec2([1, 2])),
    [108, 111, 114, 117],
  ],
  [
    "texelFetch sampler2D float",
    "vec4",
    () => d.texelFetch(d.dynoSampler2D(fTex), d.dynoIvec2([1, 1])),
    [6, 6.5, 7, 7.5],
  ],
  [
    "texture sampler2D linear",
    "vec4",
    () => d.texture(d.dynoSampler2D(bTex), v2(0.5, 0.5)),
    [0.5, 0, 0, 1],
  ],
  ["textureSize", "ivec2", () => d.textureSize(d.dynoUsampler2D(uTex)), [4, 3]],
  [
    "texelFetch usampler2DArray",
    "uvec4",
    () => d.texelFetch(d.dynoUsampler2DArray(arrayTex), iv3(1, 0, 2)),
    [36, 37, 38, 39],
  ],
  [
    "texelFetch sRGB sampler2D decodes",
    "vec4",
    () => d.texelFetch(d.dynoSampler2D(srgbTex), d.dynoIvec2([0, 0])),
    (v) => {
      const lin = (c: number) => ((c + 0.055) / 1.055) ** 2.4;
      expect(v[0]).toBeCloseTo(lin(128 / 255), 3);
      expect(v[2]).toBeCloseTo(1, 5);
      expect(v[3]).toBeCloseTo(128 / 255, 5); // alpha stays linear
    },
  ],
  [
    "texture sampler2DArray rounds the layer",
    "vec4",
    () =>
      d.texture(d.dynoSampler2DArray(layersTex), d.dynoVec3([0.5, 0.5, 0.7])),
    [200 / 255, 0, 0, 1],
  ],
  [
    "textureSize 2DArray",
    "ivec3",
    () => d.textureSize(d.dynoUsampler2DArray(arrayTex)),
    [2, 2, 3],
  ],
];

// A struct uniform with a texture field (as SplatEdit's SdfArray).
const TestStruct = { type: "TestStruct" } as { type: "TestStruct" };
d.registerWgslStruct(TestStruct, {
  scale: "float",
  offset: "vec3",
  flag: "bool",
  tex: "usampler2D",
  count: "int",
});
textureCases.push([
  "struct uniform",
  "vec4",
  () =>
    new d.Dyno({
      inTypes: { s: TestStruct },
      outTypes: { out: "vec4" },
      inputs: {
        s: new d.DynoUniform({
          key: "s",
          type: TestStruct,
          value: {
            scale: 2,
            offset: [1, 2, 3],
            flag: true,
            tex: uTex,
            count: 7,
          },
        }),
      },
      wgsl: {
        statements: ({ inputs, outputs }) => {
          const s = inputs.s as string;
          const tex = d.wgslStructTexture(s, "tex");
          return [
            `let texel = f32(textureLoad(${tex}, vec2i(0, 0), 0).y);`,
            `${outputs.out} = vec4f(${s}.offset * ${s}.scale, texel + f32(${s}.flag) + f32(${s}.count));`,
          ];
        },
      },
    }).outputs.out as Val,
  [2, 4, 6, 3 + 1 + 7],
]);

// readPackedSplat from another PackedSplats (a struct uniform with its
// packed texture), checked against the CPU decode of the same splat.
const packed = new PackedSplats();
packed.pushSplat(
  new THREE.Vector3(0.5, -1, 2),
  new THREE.Vector3(0.1, 0.02, 0.3),
  new THREE.Quaternion(0.2, 0.4, -0.1, 0.9).normalize(),
  0.25,
  new THREE.Color(0.2, 0.6, 0.9),
);
packed.pushSplat(
  new THREE.Vector3(-3, 0.25, 1),
  new THREE.Vector3(0, 0, 0),
  new THREE.Quaternion(),
  1,
  new THREE.Color(1, 1, 1),
);
const near = (expected: number[], tolerance: number) => (v: number[]) => {
  expect(v.length).toBe(expected.length);
  v.forEach((a, i) =>
    expect(Math.abs(a - expected[i]), `component ${i}`).toBeLessThan(tolerance),
  );
};
const readPacked = (index: number) =>
  d.splitGsplat(d.readPackedSplat(packed.dyno, i1(index))).outputs;
const cpu = packed.getSplat(0);
textureCases.push(
  [
    "readPackedSplat center, opacity",
    "vec4",
    () => {
      const s = readPacked(0);
      return d.extendVec(s.center, s.opacity);
    },
    near([...cpu.center.toArray(), cpu.opacity], 1e-3),
  ],
  [
    "readPackedSplat scales, rgb",
    "vec4",
    () => {
      const s = readPacked(0);
      return d.extendVec(d.add(s.scales, s.rgb), s.opacity);
    },
    near(
      [
        cpu.scales.x + cpu.color.r,
        cpu.scales.y + cpu.color.g,
        cpu.scales.z + cpu.color.b,
        cpu.opacity,
      ],
      1e-3,
    ),
  ],
  [
    "readPackedSplat quaternion",
    "vec4",
    () => readPacked(0).quaternion,
    near(cpu.quaternion.toArray(), 1e-3),
  ],
  [
    "readPackedSplat active, index",
    "ivec4",
    () => {
      const zero = readPacked(1);
      const out = readPacked(5);
      return d.combine({
        vectorType: "ivec4",
        x: d.int(readPacked(0).active),
        y: d.int(zero.active),
        z: d.int(out.active),
        w: zero.index,
      } as never);
    },
    [1, 0, 0, 1],
  ],
);

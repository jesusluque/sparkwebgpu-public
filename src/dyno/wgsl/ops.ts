// WGSL code for the dyno ops whose GLSL isn't valid WGSL. Ops not listed here
// (arithmetic, most builtins, swizzles, splats, transforms, blocks) use their
// GLSL statements with constructor names mapped; see backend.ts.

import { unindent, unindentLines } from "../base";
import {
  BVec2,
  BVec3,
  BVec4,
  Bool,
  Float,
  FloatBitsToInt,
  FloatBitsToUint,
  IVec2,
  IVec3,
  IVec4,
  Int,
  IntBitsToFloat,
  Mat2,
  Mat3,
  Mat4,
  PackHalf2x16,
  PackSnorm2x16,
  PackUnorm2x16,
  SimpleCast,
  UVec2,
  UVec3,
  UVec4,
  Uint,
  UintBitsToFloat,
  UintToRgba8,
  UnpackHalf2x16,
  UnpackSnorm2x16,
  UnpackUnorm2x16,
  Vec2,
  Vec3,
  Vec4,
} from "../convert";
import {
  CompXor,
  Equal,
  GreaterThan,
  GreaterThanEqual,
  LessThan,
  LessThanEqual,
  Not,
  NotEqual,
  Select,
  Shl,
  Shr,
  Xor,
} from "../logic";
import {
  Add,
  Clamp,
  Div,
  InverseSqrt,
  IsInf,
  IsNan,
  Max,
  Min,
  Mix,
  Mod,
  Modf,
  Neg,
  Smoothstep,
  Step,
  Sub,
} from "../math";
import {
  OutputCovSplat,
  OutputExtCovSplat,
  OutputExtendedSplat,
  OutputPackedSplat,
  OutputRgba8,
} from "../output";
import {
  GsplatToCovSplat,
  NumCovSplats,
  NumExtSplats,
  ReadCovSplat,
  ReadExtSplat,
  ReadPackedSplat,
  ReadPackedSplatRange,
  TPackedSplats,
  definePackedSplats,
} from "../splats";
import { TexelFetch, Texture, TextureSize } from "../texture";
import type { DynoType } from "../types";
import { typeLiteral } from "../types";
import { DynoUniform } from "../uniforms";
import {
  DebugColorHue,
  NormalizedDepth,
  PcgHash,
  PcgMix,
  PcgNext,
} from "../util";
import { Combine, CompMult, FaceForward, Inverse, Outer } from "../vecmat";
import {
  type WgslEmitter,
  registerWgsl,
  registerWgslStruct,
  wgslStructTexture,
} from "./backend";
import { registerWgslGlobal } from "./prelude";
import {
  type TypeShape,
  shapeType,
  splat,
  textureType,
  typeShape,
  wgslType,
} from "./types";

type Op = {
  inTypes: Record<string, DynoType>;
  outTypes: Record<string, DynoType>;
  outKey: string;
};

function shapeOf(type: DynoType): TypeShape {
  const shape = typeShape(type);
  if (!shape) throw new Error(`Not a value type: ${typeLiteral(type)}`);
  return shape;
}

/** `out = expr` for a single-output op, from its inputs and op. */
function assign(
  expr: (inputs: Record<string, string>, op: Op) => string,
): WgslEmitter {
  return (op: Op, { inputs, outputs }) => {
    const out = (outputs as Record<string, string>)[op.outKey];
    if (!out) return {};
    return {
      statements: [`${out} = ${expr(inputs as Record<string, string>, op)};`],
    };
  };
}

function outType(op: Op): DynoType {
  return op.outTypes[op.outKey];
}

// math

// WGSL matrices lack GLSL's matrix +- scalar, matrix / scalar and -matrix:
// spelled out per column, or as a multiply.
for (const [cls, operator] of [
  [Add, "+"],
  [Sub, "-"],
  [Div, "/"],
] as const) {
  registerWgsl(
    cls,
    assign(({ a, b }, op) => {
      const sa = typeShape(op.inTypes.a);
      const sb = typeShape(op.inTypes.b);
      const t = shapeOf(outType(op));
      if (t.cols > 1 && (sa?.cols === 1 || sb?.cols === 1)) {
        if (operator === "/" && sb?.rows === 1) return `${a} * (1.0 / ${b})`;
        const col = (x: string, s: TypeShape | null, c: number) =>
          s && s.cols > 1 ? `${x}[${c}]` : x;
        const cols = Array.from(
          { length: t.cols },
          (_, c) => `${col(a, sa, c)} ${operator} ${col(b, sb, c)}`,
        );
        return `${shapeType(t)}(${cols.join(", ")})`;
      }
      if (operator === "/" && t.cols > 1) {
        throw new Error("WGSL has no matrix / matrix");
      }
      return `${a} ${operator} ${b}`;
    }),
  );
}

registerWgsl(
  Neg,
  assign(({ a }, op) =>
    shapeOf(op.inTypes.a).cols > 1 ? `${a} * -1.0` : `-${a}`,
  ),
);

registerWgsl(
  Mod,
  assign(({ a, b }, op) => {
    const t = outType(op);
    const x = splat(a, op.inTypes.a, t);
    const y = splat(b, op.inTypes.b, t);
    return `${x} - ${y} * floor(${x} / ${y})`;
  }),
);

registerWgsl(Modf, (_op, { inputs, outputs }) => {
  const { fract, integer } = outputs as Record<string, string>;
  return {
    statements: [
      `let parts = modf(${inputs.a});`,
      fract ? `${fract} = parts.fract;` : "",
      integer ? `${integer} = parts.whole;` : "",
    ].filter(Boolean),
  };
});

registerWgsl(
  InverseSqrt,
  assign(({ a }) => `inverseSqrt(${a})`),
);

for (const [cls, name] of [
  [Min, "min"],
  [Max, "max"],
] as const) {
  registerWgsl(
    cls,
    assign(({ a, b }, op) => {
      const t = outType(op);
      return `${name}(${splat(a, op.inTypes.a, t)}, ${splat(b, op.inTypes.b, t)})`;
    }),
  );
}

registerWgsl(
  Clamp,
  assign(({ a, b, c }, op) => {
    const t = outType(op);
    return `clamp(${splat(a, op.inTypes.a, t)}, ${splat(b, op.inTypes.b, t)}, ${splat(c, op.inTypes.c, t)})`;
  }),
);

registerWgsl(
  Mix,
  assign(({ a, b, c }, op) => {
    // GLSL mix with a bool selects; WGSL spells that select(f, t, cond).
    if (shapeOf(op.inTypes.c).scalar === "bool") {
      return `select(${a}, ${b}, ${c})`;
    }
    return `mix(${a}, ${b}, ${c})`;
  }),
);

registerWgsl(
  Step,
  assign(({ a, b }, op) => {
    const t = outType(op);
    return `step(${splat(a, op.inTypes.a, t)}, ${splat(b, op.inTypes.b, t)})`;
  }),
);

registerWgsl(
  Smoothstep,
  assign(({ a, b, c }, op) => {
    const t = outType(op);
    return `smoothstep(${splat(a, op.inTypes.a, t)}, ${splat(b, op.inTypes.b, t)}, ${splat(c, op.inTypes.c, t)})`;
  }),
);

// WGSL has no isnan/isinf, and may assume x == x: test the bits.
function floatBits(a: string, type: DynoType) {
  const { rows } = shapeOf(type);
  const u = rows === 1 ? "u32" : `vec${rows}u`;
  const k = (v: string) => (rows === 1 ? v : `${u}(${v})`);
  return { bits: `(bitcast<${u}>(${a}) & ${k("0x7fffffffu")})`, k };
}
registerWgsl(
  IsNan,
  assign(({ a }, op) => {
    const { bits, k } = floatBits(a, op.inTypes.a);
    return `${bits} > ${k("0x7f800000u")}`;
  }),
);
registerWgsl(
  IsInf,
  assign(({ a }, op) => {
    const { bits, k } = floatBits(a, op.inTypes.a);
    return `${bits} == ${k("0x7f800000u")}`;
  }),
);

// logic

registerWgsl(
  Xor,
  assign(({ a, b }, op) =>
    shapeOf(op.inTypes.a).scalar === "bool" ? `${a} != ${b}` : `${a} ^ ${b}`,
  ),
);
registerWgsl(
  Not,
  assign(({ a }, op) =>
    shapeOf(op.inTypes.a).scalar === "bool" ? `!${a}` : `~${a}`,
  ),
);
for (const [cls, operator] of [
  [LessThan, "<"],
  [LessThanEqual, "<="],
  [GreaterThan, ">"],
  [GreaterThanEqual, ">="],
  [Equal, "=="],
  [NotEqual, "!="],
] as const) {
  // Componentwise on vectors in WGSL, like GLSL's lessThan() etc.
  registerWgsl(
    cls,
    assign(({ a, b }) => `${a} ${operator} ${b}`),
  );
}

registerWgsl(Select, (op: Op, { inputs, outputs }) => {
  const out = outputs.select;
  if (!out) return {};
  const { a: cond, b: t, c: f } = inputs;
  const shape = typeShape(outType(op));
  if (shape && shape.cols === 1) {
    return { statements: [`${out} = select(${f}, ${t}, ${cond});`] };
  }
  // select() takes only scalars and vectors.
  return {
    statements: [
      `if (${cond}) {`,
      `    ${out} = ${t};`,
      "} else {",
      `    ${out} = ${f};`,
      "}",
    ],
  };
});

registerWgsl(
  CompXor,
  assign(({ a }, op) => {
    const shape = shapeOf(op.inTypes.a);
    if (shape.rows === 1) return a;
    const parts = ["x", "y", "z", "w"]
      .slice(0, shape.rows)
      .map((c) => `${a}.${c}`);
    const operator = shape.scalar === "bool" ? "!=" : "^";
    // WGSL doesn't chain comparisons: parenthesize left to right.
    return parts.reduce((acc, p) => `(${acc} ${operator} ${p})`);
  }),
);

// Shift amounts are unsigned in WGSL.
function shiftAmount(b: string, type: DynoType) {
  const { rows } = shapeOf(type);
  return rows === 1 ? `u32(${b})` : `vec${rows}u(${b})`;
}
registerWgsl(
  Shr,
  assign(({ a, b }, op) => `${a} >> ${shiftAmount(b, op.inTypes.b)}`),
);
registerWgsl(
  Shl,
  assign(({ a, b }, op) => `${a} << ${shiftAmount(b, op.inTypes.b)}`),
);

// convert

const XYZW = "xyzw";

/** GLSL constructor-style conversion of `a` from type `from` to `to`. */
export function convert(a: string, from: DynoType, to: DynoType): string {
  const f = shapeOf(from);
  const t = shapeOf(to);
  const target = shapeType(t);
  if (t.cols > 1) {
    if (f.cols === 1) {
      // mat(x): x on the diagonal.
      const cols = Array.from({ length: t.cols }, (_, c) =>
        Array.from({ length: t.rows }, (_, r) =>
          r === c ? `f32(${a})` : "0.0",
        ).join(", "),
      );
      return `${target}(${cols.join(", ")})`;
    }
    // mat(m): the overlap, identity elsewhere.
    const cols = Array.from({ length: t.cols }, (_, c) =>
      Array.from({ length: t.rows }, (_, r) =>
        c < f.cols && r < f.rows ? `${a}[${c}][${r}]` : r === c ? "1.0" : "0.0",
      ).join(", "),
    );
    return `${target}(${cols.join(", ")})`;
  }
  if (f.rows > t.rows) {
    return t.rows === 1
      ? `${target}(${a}.x)`
      : `${target}(${a}.${XYZW.slice(0, t.rows)})`;
  }
  if (f.rows < t.rows && f.rows > 1) {
    throw new Error(
      `Can't convert ${typeLiteral(from)} to ${typeLiteral(to)}: too few components`,
    );
  }
  return `${target}(${a})`;
}

for (const cls of [
  SimpleCast,
  Bool,
  Int,
  Uint,
  Float,
  BVec2,
  BVec3,
  BVec4,
  IVec2,
  IVec3,
  IVec4,
  UVec2,
  UVec3,
  UVec4,
  Vec2,
  Vec3,
  Vec4,
  Mat2,
  Mat3,
  Mat4,
]) {
  registerWgsl(
    cls,
    assign(({ a }, op) => convert(a, op.inTypes.a, outType(op))),
  );
}

for (const [cls, fn] of [
  [FloatBitsToInt, "bitcast<i32>"],
  [FloatBitsToUint, "bitcast<u32>"],
  [IntBitsToFloat, "bitcast<f32>"],
  [UintBitsToFloat, "bitcast<f32>"],
  [PackSnorm2x16, "pack2x16snorm"],
  [UnpackSnorm2x16, "unpack2x16snorm"],
  [PackUnorm2x16, "pack2x16unorm"],
  [UnpackUnorm2x16, "unpack2x16unorm"],
  [PackHalf2x16, "pack2x16float"],
  [UnpackHalf2x16, "unpack2x16float"],
  [UintToRgba8, "unpack4x8unorm"],
] as const) {
  registerWgsl(
    cls,
    assign(({ a }) => `${fn}(${a})`),
  );
}

// vecmat

registerWgsl(
  FaceForward,
  assign(({ a, b, c }) => `faceForward(${a}, ${b}, ${c})`),
);

registerWgsl(
  CompMult,
  assign(({ a, b }, op) => {
    const t = shapeOf(outType(op));
    const cols = Array.from(
      { length: t.cols },
      (_, c) => `${a}[${c}] * ${b}[${c}]`,
    );
    return `${shapeType(t)}(${cols.join(", ")})`;
  }),
);

// outerProduct(c, r) = c * transpose(r): column j is c * r[j].
registerWgsl(
  Outer,
  assign(({ a, b }, op) => {
    const t = shapeOf(outType(op));
    const cols = Array.from({ length: t.cols }, (_, j) => `${a} * ${b}[${j}]`);
    return `${shapeType(t)}(${cols.join(", ")})`;
  }),
);

export const INVERSE: Record<number, string> = {
  2: unindent(/* wgsl */ `
    fn dyno_inverse2(m: mat2x2f) -> mat2x2f {
      return mat2x2f(m[1][1], -m[0][1], -m[1][0], m[0][0]) * (1.0 / determinant(m));
    }
  `),
  3: unindent(/* wgsl */ `
    fn dyno_inverse3(m: mat3x3f) -> mat3x3f {
      let c0 = cross(m[1], m[2]);
      let c1 = cross(m[2], m[0]);
      let c2 = cross(m[0], m[1]);
      return transpose(mat3x3f(c0, c1, c2)) * (1.0 / dot(m[0], c0));
    }
  `),
  4: unindent(/* wgsl */ `
    fn dyno_inverse4(m: mat4x4f) -> mat4x4f {
      let a00 = m[0][0]; let a01 = m[0][1]; let a02 = m[0][2]; let a03 = m[0][3];
      let a10 = m[1][0]; let a11 = m[1][1]; let a12 = m[1][2]; let a13 = m[1][3];
      let a20 = m[2][0]; let a21 = m[2][1]; let a22 = m[2][2]; let a23 = m[2][3];
      let a30 = m[3][0]; let a31 = m[3][1]; let a32 = m[3][2]; let a33 = m[3][3];
      let b00 = a00 * a11 - a01 * a10; let b01 = a00 * a12 - a02 * a10;
      let b02 = a00 * a13 - a03 * a10; let b03 = a01 * a12 - a02 * a11;
      let b04 = a01 * a13 - a03 * a11; let b05 = a02 * a13 - a03 * a12;
      let b06 = a20 * a31 - a21 * a30; let b07 = a20 * a32 - a22 * a30;
      let b08 = a20 * a33 - a23 * a30; let b09 = a21 * a32 - a22 * a31;
      let b10 = a21 * a33 - a23 * a31; let b11 = a22 * a33 - a23 * a32;
      let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
      return mat4x4f(
        a11 * b11 - a12 * b10 + a13 * b09, a02 * b10 - a01 * b11 - a03 * b09,
        a31 * b05 - a32 * b04 + a33 * b03, a22 * b04 - a21 * b05 - a23 * b03,
        a12 * b08 - a10 * b11 - a13 * b07, a00 * b11 - a02 * b08 + a03 * b07,
        a32 * b02 - a30 * b05 - a33 * b01, a20 * b05 - a22 * b02 + a23 * b01,
        a10 * b10 - a11 * b08 + a13 * b06, a01 * b08 - a00 * b10 - a03 * b06,
        a30 * b04 - a31 * b02 + a33 * b00, a21 * b02 - a20 * b04 - a23 * b00,
        a11 * b07 - a10 * b09 - a12 * b06, a00 * b09 - a01 * b07 + a02 * b06,
        a31 * b01 - a30 * b03 - a32 * b00, a20 * b03 - a21 * b01 + a22 * b00
      ) * (1.0 / det);
    }
  `),
};
registerWgsl(Inverse, (op: Op, { inputs, outputs }) => {
  const out = outputs.inverse;
  if (!out) return {};
  const n = shapeOf(op.inTypes.a).cols;
  return {
    globals: [INVERSE[n]],
    statements: [`${out} = dyno_inverse${n}(${inputs.a});`],
  };
});

registerWgsl(Combine, (op: Op, { inputs, outputs }) => {
  const out = (outputs as Record<string, string>).vector;
  if (!out) return {};
  const ins = inputs as Record<string, string | undefined>;
  const shape = shapeOf(op.outTypes.vector);
  const elZero = `${shape.scalar}()`;
  const alias = ["r", "g", "b", "a"];
  const statements = [];
  for (let i = 0; i < shape.rows; i++) {
    const c = XYZW[i];
    const v =
      ins[c] ?? ins[alias[i]] ?? (ins.vector ? `${ins.vector}.${c}` : elZero);
    statements.push(`${out}.${c} = ${v};`);
  }
  return { statements };
});

// util

const PCG = unindent(/* wgsl */ `
  fn pcg_next(state: u32) -> u32 {
    return state * 747796405u + 2891336453u;
  }

  fn pcg_hash(state: u32) -> u32 {
    let hash = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (hash >> 22u) ^ hash;
  }

  fn pcg_mix1(value: u32) -> u32 {
    return value;
  }

  fn pcg_mix2(value: vec2u) -> u32 {
    return value.x + 0x9e3779b9u * value.y;
  }

  fn pcg_mix3(value: vec3u) -> u32 {
    return value.x + 0x9e3779b9u * value.y + 0x85ebca6bu * value.z;
  }

  fn pcg_mix4(value: vec4u) -> u32 {
    return value.x + 0x9e3779b9u * value.y + 0x85ebca6bu * value.z + 0xc2b2ae35u * value.w;
  }
`);

/** `a` as unsigned bits: uint as is, int converted, float reinterpreted. */
function toUintBits(a: string, type: DynoType) {
  const shape = shapeOf(type);
  const u = shape.rows === 1 ? "u32" : `vec${shape.rows}u`;
  if (shape.scalar === "u32") return a;
  if (shape.scalar === "f32") return `bitcast<${u}>(${a})`;
  return `${u}(${a})`;
}

registerWgsl(PcgNext, (op: Op, { inputs, outputs }) => ({
  globals: [PCG],
  statements: [
    `${outputs.state} = pcg_next(${toUintBits(inputs.state as string, op.inTypes.state)});`,
  ],
}));
registerWgsl(PcgHash, (_op: Op, { inputs, outputs }) => ({
  globals: [PCG],
  statements: [`${outputs.hash} = pcg_hash(${inputs.state});`],
}));
registerWgsl(PcgMix, (op: Op, { inputs, outputs }) => ({
  globals: [PCG],
  statements: [
    `${outputs.state} = pcg_mix${shapeOf(op.inTypes.value).rows}(${toUintBits(inputs.value as string, op.inTypes.value)});`,
  ],
}));

registerWgsl(NormalizedDepth, (_op: Op, { inputs, outputs }) => {
  const { z, zNear, zFar } = inputs as Record<string, string>;
  return {
    statements: [
      `let clamped = clamp(${z}, ${zNear}, ${zFar});`,
      `${outputs.depth} = (log2(clamped + 1.0) - log2(${zNear} + 1.0)) / (log2(${zFar} + 1.0) - log2(${zNear} + 1.0));`,
    ],
  };
});

registerWgsl(DebugColorHue, (_op: Op, { inputs, outputs }) => ({
  statements: [`${outputs.color} = debugColorHue(u32(${inputs.index}));`],
}));

// splats

registerWgsl(GsplatToCovSplat, (_op: Op, { inputs, outputs }) => {
  const { gsplat } = inputs as Record<string, string>;
  const { covsplat } = outputs as Record<string, string>;
  if (!gsplat) return { statements: [`${covsplat}.flags = 0u;`] };
  return {
    statements: unindent(/* wgsl */ `
      ${covsplat}.flags = 0u;
      if (isGsplatActive(${gsplat}.flags)) {
        ${covsplat}.flags = ${gsplat}.flags;
        ${covsplat}.index = ${gsplat}.index;
        ${covsplat}.rgba = ${gsplat}.rgba;
        ${covsplat}.center = ${gsplat}.center;
        var m = scaleQuaternionToMatrix(${gsplat}.scales, ${gsplat}.quaternion);
        m = m * transpose(m);
        ${covsplat}.xxyyzz = vec3f(m[0][0], m[1][1], m[2][2]);
        ${covsplat}.xyxzyz = vec3f(m[0][1], m[0][2], m[1][2]);
      }
    `).split("\n"),
  };
});

// The WebGPU kernels read and write a mesh's own splats themselves
// (slang/kernels), so these GLSL texture-based ops have no WGSL form.
for (const cls of [
  ReadPackedSplatRange,
  NumExtSplats,
  ReadExtSplat,
  NumCovSplats,
  ReadCovSplat,
  OutputPackedSplat,
  OutputCovSplat,
  OutputExtendedSplat,
  OutputExtCovSplat,
  OutputRgba8,
]) {
  registerWgsl(cls, (op) => {
    throw new Error(
      `${op.constructor.name} has no WGSL version: on WebGPU the kernels read and write splats`,
    );
  });
}

// Another PackedSplats read from its packed texture, as GLSL's
// readPackedArray does: e.g. to blend a mesh into other splats
// (examples/lofi's world transitions).

registerWgslGlobal(definePackedSplats, "");
registerWgslStruct(TPackedSplats, {
  textureArray: "usampler2DArray",
  numSplats: "int",
  rgbMinMaxLnScaleMinMax: "vec4",
  lodOpacity: "bool",
});

const WGSL_READ_PACKED_SPLAT = unindent(/* wgsl */ `
  fn dyno_decodeQuatOctXy88R8(encoded: u32) -> vec4f {
    let f = vec2f(f32(encoded & 0xffu), f32((encoded >> 8u) & 0xffu)) / 255.0 * 2.0 - 1.0;
    var axis = vec3f(f, 1.0 - abs(f.x) - abs(f.y));
    let t = max(-axis.z, 0.0);
    axis.x += select(t, -t, axis.x >= 0.0);
    axis.y += select(t, -t, axis.y >= 0.0);
    axis = normalize(axis);
    let halfTheta = (f32(encoded >> 16u) / 255.0) * 3.14159265359 * 0.5;
    return vec4f(axis * sin(halfTheta), cos(halfTheta));
  }

  fn dyno_readPackedSplat(
    texture: texture_2d_array<u32>, numSplats: i32, rgbMinMaxLnScaleMinMax: vec4f,
    index: i32, gsplat: ptr<function, Gsplat>
  ) -> bool {
    if ((index < 0) || (index >= numSplats)) {
      return false;
    }
    let coord = splatTexCoord(index);
    let w = textureLoad(texture, coord.xy, coord.z, 0);
    let rgbMin = rgbMinMaxLnScaleMinMax.x;
    let rgbMax = rgbMinMaxLnScaleMinMax.y;
    let rgba = unpack4x8unorm(w.x);
    (*gsplat).rgba = vec4f(rgba.rgb * (rgbMax - rgbMin) + rgbMin, rgba.a);
    (*gsplat).center = vec3f(unpack2x16float(w.y), unpack2x16float(w.z & 0xffffu).x);
    let uScales = vec3u(w.w & 0xffu, (w.w >> 8u) & 0xffu, (w.w >> 16u) & 0xffu);
    let lnScaleMin = rgbMinMaxLnScaleMinMax.z;
    let lnScaleScale = (rgbMinMaxLnScaleMinMax.w - lnScaleMin) / 254.0;
    let scales = exp(lnScaleMin + vec3f(max(uScales, vec3u(1u)) - 1u) * lnScaleScale);
    (*gsplat).scales = select(scales, vec3f(0.0), uScales == vec3u(0u));
    (*gsplat).quaternion = dyno_decodeQuatOctXy88R8(((w.z >> 16u) & 0xffffu) | ((w.w >> 8u) & 0xff0000u));
    return true;
  }
`);

registerWgsl(ReadPackedSplat, (_op, { inputs, outputs }) => {
  const { gsplat } = outputs as Record<string, string>;
  if (!gsplat) return {};
  const { packedSplats, index } = inputs as Record<string, string>;
  if (!packedSplats || !index) {
    return {
      statements: [
        `${gsplat}.flags = 0u;`,
        `${gsplat}.index = ${index ?? "0"};`,
      ],
    };
  }
  const texture = wgslStructTexture(packedSplats, "textureArray");
  return {
    globals: [WGSL_READ_PACKED_SPLAT],
    statements: unindentLines(/* wgsl */ `
      ${gsplat}.flags = 0u;
      if (dyno_readPackedSplat(${texture}, ${packedSplats}.numSplats, ${packedSplats}.rgbMinMaxLnScaleMinMax, ${index}, &${gsplat})) {
        if (${packedSplats}.lodOpacity != 0u) {
          ${gsplat}.rgba.a = 2.0 * ${gsplat}.rgba.a;
        }
        let zeroSize = all(${gsplat}.scales == vec3f(0.0));
        ${gsplat}.flags = select(GSPLAT_FLAG_ACTIVE, 0u, zeroSize);
      }
      ${gsplat}.index = ${index};
    `),
  };
});

// uniforms

registerWgsl(
  DynoUniform,
  (op: DynoUniform<DynoType, string>, context, backend) => {
    const name = (context.outputs as Record<string, string>)[op.outKey];
    if (!name) return {};
    // The GLSL generate gives any extra globals plus the uniform declaration,
    // which WGSL replaces with a field of the uniform block or a binding.
    const decl = `uniform ${typeLiteral(op.type)} ${name}${op.count != null ? `[${op.count}]` : ""};`;
    const globals = (op.generate(context).globals ?? []).filter(
      (g) => g !== decl,
    );
    backend.addUniform(name, op.type, op.uniform, op.count);
    return { globals, uniforms: { [name]: op.uniform } };
  },
);

// textures: textureLoad / textureSampleLevel on texture bindings. Compute
// shaders have no derivatives, so texture() samples level 0 (or the bias as
// an explicit level).

function textureArgs(texture: string, coord: string, type: DynoType) {
  const t = typeLiteral(type);
  if (t.includes("2DArray")) {
    return `${texture}, ${coord}.xy, i32(${coord}.z)`;
  }
  return `${texture}, ${coord}`;
}

registerWgsl(TextureSize, (op: Op, { inputs, outputs }) => {
  const t = typeLiteral(op.inTypes.texture);
  const size = `textureDimensions(${inputs.texture}, ${inputs.lod ?? "0"})`;
  const expr = t.includes("2DArray")
    ? `vec3i(vec3u(${size}, textureNumLayers(${inputs.texture})))`
    : `${wgslType(op.outTypes.size)}(${size})`;
  return { statements: [`${outputs.size} = ${expr};`] };
});

registerWgsl(TexelFetch, (op: Op, { inputs, outputs }) => {
  const type = op.inTypes.texture;
  const args = textureArgs(
    inputs.texture as string,
    inputs.coord as string,
    type,
  );
  return {
    statements: [
      `${outputs.texel} = textureLoad(${args}, ${inputs.lod ?? "0"});`,
    ],
  };
});

registerWgsl(Texture, (op: Op, { inputs, outputs }) => {
  const type = op.inTypes.texture;
  if (
    !typeLiteral(type).startsWith("sampler") ||
    !textureType(type)?.endsWith("<f32>")
  ) {
    throw new Error(
      `texture() on ${typeLiteral(type)}: WGSL samples only float textures; use texelFetch`,
    );
  }
  const texture = inputs.texture as string;
  const coord = inputs.coord as string;
  const t = typeLiteral(type);
  // GLSL picks the layer nearest z, floor(z + 0.5); i32() would truncate.
  const args = t.includes("2DArray")
    ? `${texture}, ${texture}_sampler, ${coord}.xy, i32(floor(${coord}.z + 0.5))`
    : `${texture}, ${texture}_sampler, ${coord}`;
  return {
    statements: [
      `${outputs.sample} = textureSampleLevel(${args}, ${inputs.bias ?? "0.0"});`,
    ],
  };
});

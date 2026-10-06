// Translates the GLSL ES 3.0 of user dynos (`globals` and `statements`
// strings) to WGSL, so a dyno written for WebGL runs on WebGPU unchanged.
//
// It tracks types to bridge what WGSL spells differently: explicit
// conversions, scalars splatted where GLSL builtins take them for vectors,
// ?: as select(), vector == as all(), out parameters as pointers, mutable
// parameters as local copies, and swizzle assignments (which WGSL lacks, and
// naga rejects) as per-component writes. GLSL integer literals stay abstract
// WGSL literals, which converts them to float where GLSL code relies on it.
// What it doesn't cover is an error with the line and column, never a guess.

import type { Dyno, DynoGenerated, GenerateContext, IOTypes } from "../base";
import type { DynoType } from "../types";
import { typeLiteral } from "../types";
import { DynoOutput, DynoValue } from "../value";
import {
  type WgslBackend,
  registerGlslTranslator,
  wgslStructFields,
  wgslStructTexture,
} from "./backend";
import {
  type Expr,
  type Global,
  GlslError,
  type Param,
  type Pos,
  type Stmt,
  type TypeSpec,
  parseGlobals,
  parseStatements,
} from "./glslParse";
import { INVERSE } from "./ops";
import { wgslGlobals } from "./prelude";
import { type WgslScalar, shapeType, textureType, typeShape } from "./types";

// Types

export type Ty =
  | { k: "num"; s: WgslScalar; rows: number; cols: number }
  | { k: "struct"; name: string }
  | { k: "array"; of: Ty; n: number | string }
  | { k: "tex"; glsl: string }
  | { k: "void" };

type Num = Extract<Ty, { k: "num" }>;

const num = (s: WgslScalar, rows = 1, cols = 1): Num => ({
  k: "num",
  s,
  rows,
  cols,
});
const F32 = num("f32");
const I32 = num("i32");
const U32 = num("u32");
const BOOL = num("bool");
const VOID: Ty = { k: "void" };

const isNum = (t: Ty): t is Num => t.k === "num";
// Not type predicates: their false branch says nothing about the kind.
const isScalar = (t: Ty): boolean => isNum(t) && t.rows === 1 && t.cols === 1;
const isVector = (t: Ty): boolean => isNum(t) && t.cols === 1 && t.rows > 1;
const isMatrix = (t: Ty): boolean => isNum(t) && t.cols > 1;
const isInt = (t: Ty) => isNum(t) && (t.s === "i32" || t.s === "u32");

function tyEq(a: Ty, b: Ty): boolean {
  if (a.k !== b.k) return false;
  switch (a.k) {
    case "num": {
      const n = b as Num;
      return a.s === n.s && a.rows === n.rows && a.cols === n.cols;
    }
    case "struct":
      return a.name === (b as typeof a).name;
    case "array": {
      const arr = b as typeof a;
      return String(a.n) === String(arr.n) && tyEq(a.of, arr.of);
    }
    case "tex":
      return a.glsl === (b as typeof a).glsl;
    default:
      return true;
  }
}

const GLSL_SCALAR: Record<WgslScalar, string> = {
  f32: "float",
  i32: "int",
  u32: "uint",
  bool: "bool",
};

/** GLSL spelling, for messages. */
function glslName(t: Ty): string {
  switch (t.k) {
    case "num": {
      if (t.cols > 1) {
        return t.cols === t.rows ? `mat${t.cols}` : `mat${t.cols}x${t.rows}`;
      }
      if (t.rows === 1) return GLSL_SCALAR[t.s];
      return `${t.s === "f32" ? "" : GLSL_SCALAR[t.s][0]}vec${t.rows}`;
    }
    case "struct":
      return t.name;
    case "array":
      return `${glslName(t.of)}[${t.n}]`;
    case "tex":
      return t.glsl;
    default:
      return "void";
  }
}

// Values and names that would clash in WGSL: its keywords and reserved
// words, and the builtins and types the translation emits.
const WGSL_RESERVED = new Set(
  `alias break case const const_assert continue continuing default diagnostic
  discard else enable false fn for if let loop override requires return struct
  switch true var while NULL Self abstract active alignas alignof as asm
  asm_fragment async attribute auto await become cast catch class co_await
  co_return co_yield coherent column_major common compile compile_fragment
  concept const_cast consteval constexpr constinit crate debugger decltype
  delete demote demote_to_helper do dynamic_cast enum explicit export extends
  extern external fallthrough filter final finally friend from fxgroup get goto
  groupshared highp impl implements import inline instanceof interface layout
  lowp macro macro_rules match mediump meta mod module move mut mutable
  namespace new nil noexcept noinline nointerpolation non_coherent noncoherent
  noperspective null nullptr of operator package packoffset partition pass
  patch pixelfragment precise precision premerge priv protected pub public
  readonly ref regardless register reinterpret_cast require resource restrict
  self set shared sizeof smooth snorm static static_assert static_cast std
  subroutine super target template this thread_local throw trait try type
  typedef typeid typename typeof union unless unorm unsafe unsized use using
  varying virtual volatile wgsl where with writeonly yield
  f16 f32 i32 u32 array atomic ptr sampler sampler_comparison
  vec2f vec3f vec4f vec2i vec3i vec4i vec2u vec3u vec4u vec2h vec3h vec4h
  mat2x2f mat2x3f mat2x4f mat3x2f mat3x3f mat3x4f mat4x2f mat4x3f mat4x4f
  select all any bitcast atan2 inverseSqrt faceForward textureSampleLevel
  textureLoad textureDimensions textureNumLayers countOneBits firstTrailingBit
  firstLeadingBit extractBits insertBits reverseBits pack2x16float
  unpack2x16float pack2x16unorm unpack2x16unorm pack2x16snorm unpack2x16snorm
  pack4x8unorm unpack4x8unorm pack4x8snorm unpack4x8snorm saturate quantizeToF16
  arrayLength dyno_uniforms dyno_inf`.split(/\s+/),
);

/** A GLSL name that is valid and free in WGSL. */
function safeName(name: string, moduleScope = false): string {
  if (name.startsWith("__")) return `u${name}`;
  if (name === "_" || WGSL_RESERVED.has(name)) return `${name}_`;
  // Slang's names in the kernel the dyno code joins end in _<number>.
  if (moduleScope && /_\d+$/.test(name)) return `${name}_`;
  return name;
}

interface StructInfo {
  wgsl: string;
  fields: Map<string, { ty: Ty; wgsl: string }>;
}

function builtinStruct(name: string, fields: [string, Ty][]): StructInfo {
  return {
    wgsl: name,
    fields: new Map(fields.map(([f, ty]) => [f, { ty, wgsl: f }])),
  };
}

const V3 = num("f32", 3);
const V4 = num("f32", 4);

// A function signature: GLSL parameter types and qualifiers, the WGSL name.
interface FnSig {
  params: { ty: Ty; qual: Param["qual"] }[];
  ret: Ty;
  wgsl: string;
}

// The splatDefines functions the WGSL prelude (prelude.ts) has.
const PRELUDE_FUNCTIONS: [string, Ty[], Ty][] = [
  ["isGsplatActive", [U32], BOOL],
  ["isCovSplatActive", [U32], BOOL],
  ["covSplatMatrix", [V3, V3], num("f32", 3, 3)],
  ["sqr", [F32], F32],
  ["pow4", [F32], F32],
  ["pow8", [F32], F32],
  ["srgbToLinear", [V3], V3],
  ["linearToSrgb", [V3], V3],
  ["quatVec", [V4, V3], V3],
  ["quatQuat", [V4, V4], V4],
  ["quaternionToMatrix", [V4], num("f32", 3, 3)],
  ["scaleQuaternionToMatrix", [V3, V4], num("f32", 3, 3)],
  ["slerp", [V4, V4, F32], V4],
  ["splatTexCoord", [I32], num("i32", 3)],
  ["pagedSplatTexCoord", [I32], num("i32", 3)],
  ["uintToVec4", [U32], V4],
  ["floatToVec4", [F32], V4],
  ["debugColorHue", [U32], V3],
  ["gsplatNormal", [V3, V4], V3],
];

const PRELUDE_CONSTANTS: [string, Ty, string][] = [
  ["PI", F32, "PI"],
  ["LN_SCALE_MIN", F32, "LN_SCALE_MIN"],
  ["LN_SCALE_MAX", F32, "LN_SCALE_MAX"],
  ["SPLAT_TEX_WIDTH_BITS", U32, "SPLAT_TEX_WIDTH_BITS"],
  ["SPLAT_TEX_HEIGHT_BITS", U32, "SPLAT_TEX_HEIGHT_BITS"],
  ["SPLAT_TEX_LAYER_BITS", U32, "SPLAT_TEX_LAYER_BITS"],
  ["SPLAT_TEX_WIDTH_MASK", U32, "SPLAT_TEX_WIDTH_MASK"],
  ["SPLAT_TEX_HEIGHT_MASK", U32, "SPLAT_TEX_HEIGHT_MASK"],
  ["GSPLAT_FLAG_ACTIVE", U32, "GSPLAT_FLAG_ACTIVE"],
  ["INFINITY", F32, "dyno_inf()"],
  ["NEG_INFINITY", F32, "(-dyno_inf())"],
];

// Expressions

// Operator classes, for WGSL's parenthesization rules: it doesn't chain
// comparisons, mix && with || or bitwise operators with others, and shift
// operands must be unary expressions.
type Cls =
  | "primary"
  | "unary"
  | "mul"
  | "add"
  | "shift"
  | "rel"
  | "and"
  | "or"
  | "bitand"
  | "bitor"
  | "bitxor";

interface Val {
  code: string;
  ty: Ty;
  cls: Cls;
  /** An abstract-int or abstract-float literal (expression). */
  lit?: boolean;
  /** For texture values: the binding's sampler, if it has one. */
  sampler?: string;
}

// Per operator class, the operand classes that need no parentheses on its
// left and right.
const UNARY: Cls[] = ["primary", "unary"];
const ARITH: Cls[] = [...UNARY, "mul", "add", "shift"];
const ALLOWED: Record<string, [Cls[], Cls[]]> = {
  mul: [[...UNARY, "mul"], UNARY],
  add: [
    [...UNARY, "mul", "add"],
    [...UNARY, "mul"],
  ],
  shift: [UNARY, UNARY],
  rel: [ARITH, ARITH],
  and: [
    [...ARITH, "rel", "and"],
    [...ARITH, "rel"],
  ],
  or: [
    [...ARITH, "rel", "or"],
    [...ARITH, "rel"],
  ],
  bitand: [[...UNARY, "bitand"], UNARY],
  bitor: [[...UNARY, "bitor"], UNARY],
  bitxor: [[...UNARY, "bitxor"], UNARY],
};

const OP_CLASS: Record<string, Cls> = {
  "*": "mul",
  "/": "mul",
  "%": "mul",
  "+": "add",
  "-": "add",
  "<<": "shift",
  ">>": "shift",
  "<": "rel",
  ">": "rel",
  "<=": "rel",
  ">=": "rel",
  "==": "rel",
  "!=": "rel",
  "&&": "and",
  "||": "or",
  "&": "bitand",
  "|": "bitor",
  "^": "bitxor",
};

function paren(v: Val, allowed: Cls[]): string {
  return allowed.includes(v.cls) ? v.code : `(${v.code})`;
}
const primary = (v: Val) => paren(v, ["primary"]);
const operand = (v: Val) => {
  const code = paren(v, ["primary", "unary"]);
  // `- -x` must not become the decrement `--x`.
  return code.startsWith("-") ? `(${code})` : code;
};

const SWIZZLE_SETS = ["xyzw", "rgba", "stpq"];

function floatLiteral(text: string): string {
  let t = text.replace(/[fF]$/, "");
  if (t.startsWith(".")) t = `0${t}`;
  if (/\.$/.test(t)) t += "0";
  t = t.replace(/\.([eE])/, ".0$1");
  if (!/[.eE]/.test(t)) t += ".0";
  return t;
}

/** Scope entries: what a name is and the WGSL that reads it. */
interface Sym {
  ty: Ty;
  code: string;
  kind: "var" | "const" | "param" | "ptr" | "input";
  /** Texture inputs: the sampler binding. */
  sampler?: string;
  /** Struct uniform inputs (registerWgslStruct): the uniform's name. */
  uniformStruct?: string;
}

export interface GlslInput {
  type: Ty;
  /** The WGSL expression for the value (as the backend resolved it). */
  expr: string;
  /** Texture inputs with a sampler binding. */
  sampler?: string;
  /** Struct uniforms: the uniform's name. */
  uniformStruct?: string;
}

/** Hooks into the WGSL backend for struct uniforms and their textures. */
export interface GlslHost {
  /** Fields of a struct type that uniforms may have (registerWgslStruct). */
  uniformStruct?(name: string): Record<string, DynoType> | undefined;
  /** The binding of a struct uniform's texture field. */
  structTexture?(uniform: string, field: string): string;
}

type Ctx = {
  /** Return type of the function being translated, null for dyno code. */
  ret: Ty | null;
  /** What dyno code's bare `return;` returns, if it may return. */
  returnValue?: string;
};

export class GlslTranslator {
  private structs = new Map<string, StructInfo>([
    [
      "Gsplat",
      builtinStruct("Gsplat", [
        ["center", V3],
        ["flags", U32],
        ["scales", V3],
        ["index", I32],
        ["quaternion", V4],
        ["rgba", V4],
      ]),
    ],
    [
      "CovSplat",
      builtinStruct("CovSplat", [
        ["center", V3],
        ["flags", U32],
        ["rgba", V4],
        ["xxyyzz", V3],
        ["index", I32],
        ["xyxzyz", V3],
      ]),
    ],
  ]);
  private functions = new Map<string, FnSig[]>();
  private globalScope = new Map<string, Sym>();
  private scopes: Map<string, Sym>[] = [];
  private helpers = new Map<string, string>();
  private usedHelpers = new Set<string>();
  private globalsCache = new Map<string, string[]>();
  private definitions = new Map<string, string>();
  private src = "";
  private temp = 0;

  constructor(private host: GlslHost = {}) {
    for (const [name, params, ret] of PRELUDE_FUNCTIONS) {
      this.functions.set(name, [
        { params: params.map((ty) => ({ ty, qual: "in" })), ret, wgsl: name },
      ]);
    }
    for (const [name, ty, code] of PRELUDE_CONSTANTS) {
      this.globalScope.set(name, { ty, code, kind: "const" });
    }
  }

  /** WGSL helper functions the translations so far used. */
  takeHelpers(): string[] {
    const used = [...this.usedHelpers].map(
      (h) => this.helpers.get(h) as string,
    );
    this.usedHelpers.clear();
    return used;
  }

  private error(message: string, pos: Pos): never {
    throw new GlslError(message, pos, this.src);
  }

  // Types

  private structTy(name: string): StructInfo | undefined {
    const info = this.structs.get(name);
    if (info) return info;
    const fields = this.host.uniformStruct?.(name);
    if (!fields) return undefined;
    const struct: StructInfo = { wgsl: `DynoU_${name}`, fields: new Map() };
    for (const [f, type] of Object.entries(fields)) {
      struct.fields.set(f, { ty: dynoTy(type), wgsl: f });
    }
    this.structs.set(name, struct);
    return struct;
  }

  private baseTy(name: string, pos: Pos): Ty {
    if (name === "void") return VOID;
    const shape = typeShape(name as DynoType);
    if (shape) return num(shape.scalar, shape.rows, shape.cols);
    if (textureType(name as DynoType)) return { k: "tex", glsl: name };
    if (this.structTy(name)) return { k: "struct", name };
    this.error(`unknown type '${name}'`, pos);
  }

  private ty(spec: TypeSpec, array?: Expr | null, init?: Val): Ty {
    let ty = this.baseTy(spec.name, spec.pos);
    for (const size of [spec.array, array]) {
      if (size === undefined) continue;
      if (size === null) {
        if (init?.ty.k !== "array") {
          this.error("an unsized array needs an array initializer", spec.pos);
        }
        ty = { k: "array", of: ty, n: init.ty.n };
      } else {
        ty = { k: "array", of: ty, n: this.arraySize(size) };
      }
    }
    return ty;
  }

  private arraySize(e: Expr): number | string {
    if (e.k === "num" && /^\d+[uU]?$/.test(e.text))
      return Number.parseInt(e.text);
    const v = this.expr(e);
    if (!isScalar(v.ty) || !isInt(v.ty)) {
      this.error("array size must be an integer constant", e.pos);
    }
    return v.code;
  }

  wgslTy(t: Ty): string {
    switch (t.k) {
      case "num":
        return shapeType({ scalar: t.s, rows: t.rows, cols: t.cols });
      case "struct":
        return (this.structs.get(t.name) as StructInfo).wgsl;
      case "array":
        return `array<${this.wgslTy(t.of)}, ${t.n}>`;
      case "tex":
        return textureType(t.glsl as DynoType) as string;
      default:
        return "";
    }
  }

  // Scopes

  private lookup(name: string): Sym | undefined {
    for (let i = this.scopes.length - 1; i >= 0; i--) {
      const sym = this.scopes[i].get(name);
      if (sym) return sym;
    }
    return this.globalScope.get(name);
  }

  private declare(name: string, sym: Sym) {
    (this.scopes[this.scopes.length - 1] ?? this.globalScope).set(name, sym);
  }

  private withScope<T>(f: () => T): T {
    this.scopes.push(new Map());
    try {
      return f();
    } finally {
      this.scopes.pop();
    }
  }

  private helper(name: string, code: () => string): string {
    if (!this.helpers.has(name)) this.helpers.set(name, code());
    this.usedHelpers.add(name);
    return name;
  }

  // Conversions

  /** `v` converted to `to`, as GLSL constructors and literals allow. */
  private convert(v: Val, to: Ty, pos: Pos): Val {
    if (tyEq(v.ty, to)) return v;
    if (
      isNum(v.ty) &&
      isNum(to) &&
      v.ty.rows === to.rows &&
      v.ty.cols === to.cols
    ) {
      // Abstract literals convert by themselves (to a float, or a uint),
      // except negative ones to uint, which wrap around as GLSL's do.
      const negative = v.code.startsWith("-");
      if (
        v.lit &&
        (to.s === "f32" ||
          (v.ty.s !== "f32" &&
            to.s !== "bool" &&
            !(negative && to.s === "u32")))
      ) {
        return { ...v, ty: to };
      }
      if (v.lit && negative && to.s === "u32" && v.ty.s === "i32") {
        const i = this.wgslTy(num("i32", to.rows));
        return {
          code: `${this.wgslTy(to)}(${i}(${v.code}))`,
          ty: to,
          cls: "primary",
        };
      }
      if (isMatrix(to)) this.error("matrices have only float components", pos);
      return { code: `${this.wgslTy(to)}(${v.code})`, ty: to, cls: "primary" };
    }
    this.error(`can't convert ${glslName(v.ty)} to ${glslName(to)}`, pos);
  }

  /** A scalar `v` widened to the vector shape of `like`. */
  private splat(v: Val, like: Ty, pos: Pos): Val {
    if (!isNum(like) || !isNum(v.ty)) return v;
    const target = num(like.s, like.rows, like.cols);
    if (isScalar(like) || !isScalar(v.ty)) return this.convert(v, target, pos);
    const s = this.convert(v, num(like.s), pos);
    return {
      code: `${this.wgslTy(target)}(${s.code})`,
      ty: target,
      cls: "primary",
    };
  }

  /** Converts `a` and `b` to a common scalar type. */
  private unify(a: Val, b: Val, pos: Pos): [Val, Val] {
    if (!isNum(a.ty) || !isNum(b.ty)) return [a, b];
    if (a.ty.s === b.ty.s) return [a, b];
    const to = (v: Val, s: WgslScalar) =>
      this.convert(v, num(s, (v.ty as Num).rows, (v.ty as Num).cols), pos);
    // A literal takes the other side's type, unless that would truncate it.
    if (a.lit && !b.lit && (a.ty.s !== "f32" || b.ty.s === "f32")) {
      return [to(a, b.ty.s), b];
    }
    if (b.lit && !a.lit && (b.ty.s !== "f32" || a.ty.s === "f32")) {
      return [a, to(b, a.ty.s)];
    }
    if (a.ty.s === "f32" || b.ty.s === "f32") {
      return [to(a, "f32"), to(b, "f32")];
    }
    if (a.lit && b.lit) return [a, to(b, a.ty.s)];
    this.error(`mismatched types ${glslName(a.ty)} and ${glslName(b.ty)}`, pos);
  }

  private toFloat(v: Val, pos: Pos): Val {
    if (!isNum(v.ty))
      this.error(`expected a float, not ${glslName(v.ty)}`, pos);
    return this.convert(v, num("f32", v.ty.rows, v.ty.cols), pos);
  }

  // Expressions

  expr(e: Expr): Val {
    switch (e.k) {
      case "num": {
        const t = e.text;
        if (/^0[xX]/.test(t)) {
          const u = /[uU]$/.test(t);
          return {
            code: u ? `${t.slice(0, -1)}u` : t,
            ty: u ? U32 : I32,
            cls: "primary",
            lit: !u,
          };
        }
        if (/[.eE]/.test(t) || /[fF]$/.test(t)) {
          return { code: floatLiteral(t), ty: F32, cls: "primary", lit: true };
        }
        const u = /[uU]$/.test(t);
        let digits = u ? t.slice(0, -1) : t;
        // GLSL octal
        if (/^0\d+$/.test(digits)) digits = String(Number.parseInt(digits, 8));
        return {
          code: u ? `${digits}u` : digits,
          ty: u ? U32 : I32,
          cls: "primary",
          lit: !u,
        };
      }
      case "bool":
        return { code: String(e.value), ty: BOOL, cls: "primary" };
      case "id": {
        const sym = this.lookup(e.name);
        if (!sym) this.error(`unknown name '${e.name}'`, e.pos);
        if (sym.ty.k === "tex") {
          return {
            code: sym.code,
            ty: sym.ty,
            cls: "primary",
            sampler: sym.sampler,
          };
        }
        return { code: sym.code, ty: sym.ty, cls: "primary" };
      }
      case "member":
        return this.member(e);
      case "index": {
        const obj = this.expr(e.obj);
        const index = this.expr(e.index);
        if (!isScalar(index.ty) || !isInt(index.ty)) {
          this.error("index must be an int or uint", e.index.pos);
        }
        let ty: Ty;
        if (obj.ty.k === "array") ty = obj.ty.of;
        else if (isMatrix(obj.ty)) ty = num("f32", (obj.ty as Num).rows);
        else if (isVector(obj.ty)) ty = num((obj.ty as Num).s);
        else this.error(`can't index ${glslName(obj.ty)}`, e.pos);
        return { code: `${primary(obj)}[${index.code}]`, ty, cls: "primary" };
      }
      case "call":
        return this.call(e);
      case "method": {
        if (e.name !== "length" || e.args.length) {
          this.error(`unknown method '${e.name}'`, e.pos);
        }
        const obj = this.expr(e.obj);
        let n: number | string;
        if (obj.ty.k === "array") n = obj.ty.n;
        else if (isMatrix(obj.ty)) n = (obj.ty as Num).cols;
        else if (isVector(obj.ty)) n = (obj.ty as Num).rows;
        else this.error(`${glslName(obj.ty)} has no length()`, e.pos);
        return {
          code: typeof n === "number" ? String(n) : `i32(${n})`,
          ty: I32,
          cls: "primary",
          lit: typeof n === "number",
        };
      }
      case "unary":
        return this.unary(e);
      case "postfix":
      case "assign":
        this.error(
          `${e.k === "assign" ? "assignment" : `'${e.op}'`} inside an expression is not supported`,
          e.pos,
        );
        break;
      case "binary":
        return this.binary(e.op, this.expr(e.left), this.expr(e.right), e.pos);
      case "cond": {
        const test = this.expr(e.test);
        if (!tyEq(test.ty, BOOL))
          this.error("?: needs a bool condition", e.pos);
        let [a, b] = this.unify(this.expr(e.then), this.expr(e.else), e.pos);
        if (isNum(a.ty) && isNum(b.ty) && !tyEq(a.ty, b.ty)) {
          if (isScalar(a.ty)) a = this.splat(a, b.ty, e.pos);
          else if (isScalar(b.ty)) b = this.splat(b, a.ty, e.pos);
        }
        if (!tyEq(a.ty, b.ty)) {
          this.error(
            `?: branches differ: ${glslName(a.ty)} and ${glslName(b.ty)}`,
            e.pos,
          );
        }
        if (!isNum(a.ty) || isMatrix(a.ty)) {
          this.error(
            `?: on ${glslName(a.ty)} is not supported (WGSL select() takes scalars and vectors)`,
            e.pos,
          );
        }
        return {
          code: `select(${b.code}, ${a.code}, ${test.code})`,
          ty: a.lit && b.lit ? a.ty : a.lit ? b.ty : a.ty,
          cls: "primary",
          lit: a.lit && b.lit,
        };
      }
      case "comma":
        this.error("the comma operator is not supported here", e.pos);
    }
    this.error("unsupported expression", e.pos);
  }

  private member(e: Extract<Expr, { k: "member" }>): Val {
    // Struct uniforms keep their textures as bindings of their own.
    if (e.obj.k === "id") {
      const sym = this.lookup(e.obj.name);
      if (sym?.uniformStruct && sym.ty.k === "struct") {
        const field = this.structTy(sym.ty.name)?.fields.get(e.name);
        if (!field) {
          this.error(`${sym.ty.name} has no field '${e.name}'`, e.pos);
        }
        if (field.ty.k === "tex") {
          const binding =
            this.host.structTexture?.(sym.uniformStruct, e.name) ??
            this.error("struct uniform textures need the WGSL backend", e.pos);
          const sampled = field.ty.glsl.startsWith("sampler");
          return {
            code: binding,
            ty: field.ty,
            cls: "primary",
            sampler: sampled ? `${binding}_sampler` : undefined,
          };
        }
        const code = `${sym.code}.${field.wgsl}`;
        if (isNum(field.ty) && field.ty.s === "bool") {
          return { code: `(${code} != 0u)`, ty: field.ty, cls: "primary" };
        }
        return { code, ty: field.ty, cls: "primary" };
      }
    }
    const obj = this.expr(e.obj);
    if (obj.ty.k === "struct") {
      const field = this.structTy(obj.ty.name)?.fields.get(e.name);
      if (!field) this.error(`${obj.ty.name} has no field '${e.name}'`, e.pos);
      return {
        code: `${primary(obj)}.${field.wgsl}`,
        ty: field.ty,
        cls: "primary",
      };
    }
    if (isNum(obj.ty) && isVector(obj.ty)) {
      const swz = this.swizzle(e.name, obj.ty.rows, e.pos);
      return {
        code: `${primary(obj)}.${swz}`,
        ty: num(obj.ty.s, swz.length),
        cls: "primary",
      };
    }
    this.error(`'.${e.name}' on ${glslName(obj.ty)}`, e.pos);
  }

  /** A GLSL swizzle in WGSL's letters, checked against the vector size. */
  private swizzle(name: string, size: number, pos: Pos): string {
    const set = SWIZZLE_SETS.find((s) => [...name].every((c) => s.includes(c)));
    if (!set || name.length > 4) this.error(`bad swizzle '.${name}'`, pos);
    const idx = [...name].map((c) => set.indexOf(c));
    if (idx.some((i) => i >= size)) {
      this.error(`swizzle '.${name}' out of range for vec${size}`, pos);
    }
    return idx.map((i) => (set === "rgba" ? "rgba" : "xyzw")[i]).join("");
  }

  private unary(e: Extract<Expr, { k: "unary" }>): Val {
    if (e.op === "++" || e.op === "--") {
      this.error(`'${e.op}' inside an expression is not supported`, e.pos);
    }
    const a = this.expr(e.arg);
    if (!isNum(a.ty)) this.error(`'${e.op}' on ${glslName(a.ty)}`, e.pos);
    switch (e.op) {
      case "+":
        return a;
      case "-":
        if (isMatrix(a.ty)) {
          return { code: `${operand(a)} * -1.0`, ty: a.ty, cls: "mul" };
        }
        return { code: `-${operand(a)}`, ty: a.ty, cls: "unary", lit: a.lit };
      case "!":
        if (!tyEq(a.ty, BOOL)) this.error("'!' needs a bool", e.pos);
        return { code: `!${operand(a)}`, ty: a.ty, cls: "unary" };
      default:
        if (!isInt(a.ty)) this.error("'~' needs an int or uint", e.pos);
        return { code: `~${operand(a)}`, ty: a.ty, cls: "unary", lit: a.lit };
    }
  }

  binary(op: string, left: Val, right: Val, pos: Pos): Val {
    let a = left;
    let b = right;
    if (!isNum(a.ty) || !isNum(b.ty)) {
      this.error(
        `'${op}' on ${glslName(a.ty)} and ${glslName(b.ty)} is not supported`,
        pos,
      );
    }
    const make = (code: string, ty: Ty, cls: Cls, lit?: boolean): Val => ({
      code,
      ty,
      cls,
      lit,
    });
    const join = (o: string, x: Val, y: Val, ty: Ty, cls = OP_CLASS[o]) => {
      const [l, r] = ALLOWED[cls];
      return make(
        `${paren(x, l)} ${o} ${paren(y, r)}`,
        ty,
        cls,
        x.lit && y.lit,
      );
    };
    if (op === "&&" || op === "||" || op === "^^") {
      if (!tyEq(a.ty, BOOL) || !tyEq(b.ty, BOOL)) {
        this.error(`'${op}' needs bools`, pos);
      }
      if (op === "^^") return join("!=", a, b, BOOL);
      return join(op, a, b, BOOL);
    }
    if (op === "<<" || op === ">>") {
      if (!isInt(a.ty) || !isInt(b.ty))
        this.error(`'${op}' needs integers`, pos);
      const amount = this.splat(
        this.convert(b, num("u32", (b.ty as Num).rows), pos),
        num("u32", a.ty.rows),
        pos,
      );
      return join(op, a, amount, a.ty);
    }
    [a, b] = this.unify(a, b, pos);
    const at = a.ty as Num;
    const bt = b.ty as Num;
    if (["<", ">", "<=", ">="].includes(op)) {
      if (!isScalar(at) || !isScalar(bt)) {
        this.error(`'${op}' compares scalars (use lessThan() etc.)`, pos);
      }
      return join(op, a, b, BOOL);
    }
    if (op === "==" || op === "!=") {
      if (!tyEq(at, bt)) {
        if (isScalar(at) || isScalar(bt)) {
          this.error(`'${op}' on ${glslName(at)} and ${glslName(bt)}`, pos);
        }
      }
      if (isScalar(at)) return join(op, a, b, BOOL);
      const all = op === "==" ? "all" : "any";
      if (isMatrix(at)) {
        const cols = Array.from(
          { length: at.cols },
          (_, c) => `${all}(${primary(a)}[${c}] ${op} ${primary(b)}[${c}])`,
        );
        return make(
          cols.join(op === "==" ? " && " : " || "),
          BOOL,
          op === "==" ? "and" : "or",
        );
      }
      return make(`${all}(${a.code} ${op} ${b.code})`, BOOL, "primary");
    }
    if (op === "&" || op === "|" || op === "^") {
      if (!isInt(at) || !isInt(bt)) this.error(`'${op}' needs integers`, pos);
      if (isScalar(at) && !isScalar(bt)) a = this.splat(a, bt, pos);
      if (isScalar(bt) && !isScalar(at)) b = this.splat(b, at, pos);
      return join(op, a, b, a.ty);
    }
    // + - * / %
    if (
      op === "/" &&
      a.lit &&
      b.lit &&
      /^[0.]+$/.test(b.code) &&
      at.s === "f32"
    ) {
      // 1.0 / 0.0 (GLSL's infinity) is an error as a WGSL constant: divide
      // at run time.
      const fn = this.helper(
        "glsl_div_f32",
        () => "fn glsl_div_f32(x: f32, y: f32) -> f32 {\n    return x / y;\n}",
      );
      return make(`${fn}(${a.code}, ${b.code})`, F32, "primary");
    }
    if (op === "%" && (at.s === "f32" || bt.s === "f32")) {
      this.error("'%' needs integers (use mod())", pos);
    }
    if (isMatrix(at) || isMatrix(bt)) {
      if (op === "*") {
        if (isMatrix(at) && isMatrix(bt) && at.cols === bt.rows) {
          return join("*", a, b, num("f32", at.rows, bt.cols));
        }
        if (isMatrix(at) && isVector(bt) && at.cols === bt.rows) {
          return join("*", a, b, num("f32", at.rows));
        }
        if (isVector(at) && isMatrix(bt) && at.rows === bt.rows) {
          return join("*", a, b, num("f32", bt.cols));
        }
        if (isScalar(at) || isScalar(bt)) {
          return join("*", a, b, isMatrix(at) ? at : bt);
        }
      } else if (tyEq(at, bt) && op !== "/") {
        return join(op, a, b, at);
      } else if (op === "/" && isMatrix(at) && isScalar(bt)) {
        return join("*", a, make(`1.0 / ${operand(b)}`, F32, "mul"), at);
      } else if (isScalar(at) || isScalar(bt)) {
        // Componentwise matrix-scalar ops, spelled out per column.
        const m = isMatrix(at) ? at : bt;
        const col = (v: Val, c: number) =>
          isMatrix(v.ty)
            ? make(`${primary(v)}[${c}]`, num("f32", m.rows), "primary")
            : v;
        const cols = Array.from(
          { length: m.cols },
          (_, c) => this.binary(op, col(a, c), col(b, c), pos).code,
        );
        return make(`${this.wgslTy(m)}(${cols.join(", ")})`, m, "primary");
      }
      this.error(`'${op}' on ${glslName(at)} and ${glslName(bt)}`, pos);
    }
    if (!isScalar(at) && !isScalar(bt) && at.rows !== bt.rows) {
      this.error(`'${op}' on ${glslName(at)} and ${glslName(bt)}`, pos);
    }
    return join(op, a, b, isScalar(at) ? bt : at);
  }

  // Calls

  private call(e: Extract<Expr, { k: "call" }>): Val {
    const name = e.callee.name;
    const args = () => e.args.map((a) => this.expr(a));
    if (e.callee.array !== undefined) {
      const vals = args();
      const of = this.baseTy(name, e.pos);
      const n =
        e.callee.array === null ? vals.length : this.arraySize(e.callee.array);
      if (typeof n === "number" && n !== vals.length) {
        this.error(
          `array of ${n} constructed from ${vals.length} values`,
          e.pos,
        );
      }
      const ty: Ty = { k: "array", of, n };
      const items = vals.map((v, i) => this.convert(v, of, e.args[i].pos).code);
      return {
        code: `${this.wgslTy(ty)}(${items.join(", ")})`,
        ty,
        cls: "primary",
      };
    }
    const shape = typeShape(name as DynoType);
    if (shape)
      return this.construct(
        num(shape.scalar, shape.rows, shape.cols),
        args(),
        e,
      );
    const struct = this.structs.get(name) ?? this.structTy(name);
    if (struct && !this.functions.has(name)) {
      const vals = args();
      const fields = [...struct.fields.values()];
      if (vals.length !== fields.length) {
        this.error(`${name}() takes ${fields.length} values`, e.pos);
      }
      const items = vals.map(
        (v, i) => this.convert(v, fields[i].ty, e.args[i].pos).code,
      );
      return {
        code: `${struct.wgsl}(${items.join(", ")})`,
        ty: { k: "struct", name },
        cls: "primary",
      };
    }
    const sigs = this.functions.get(name);
    if (sigs) return this.userCall(name, sigs, e);
    return this.builtin(name, e);
  }

  /** Scalar, vector and matrix constructors, with GLSL's conversions. */
  private construct(to: Num, vals: Val[], e: Expr): Val {
    const pos = e.pos;
    const wgsl = this.wgslTy(to);
    const prim = (code: string): Val => ({ code, ty: to, cls: "primary" });
    if (!vals.length) return prim(`${wgsl}()`);
    for (const v of vals) {
      if (!isNum(v.ty))
        this.error(
          `can't construct ${glslName(to)} from ${glslName(v.ty)}`,
          pos,
        );
    }
    const first = vals[0];
    const ft = first.ty as Num;
    if (isScalar(to)) {
      if (vals.length > 1) this.error(`${glslName(to)}() takes one value`, pos);
      const s = isScalar(ft)
        ? first
        : {
            ...first,
            code: `${primary(first)}${isMatrix(ft) ? "[0].x" : ".x"}`,
            ty: num(ft.s),
            cls: "primary" as Cls,
          };
      if (s.lit && tyEq(s.ty, to)) return s;
      if (s.lit && to.s === "f32") return { ...s, ty: to };
      if (tyEq(s.ty, to)) return s;
      return prim(`${wgsl}(${s.code})`);
    }
    if (isMatrix(to)) {
      if (vals.length === 1 && isScalar(ft)) {
        // A diagonal matrix.
        const s = this.convert(first, F32, pos);
        const comps: string[] = [];
        for (let c = 0; c < to.cols; c++) {
          for (let r = 0; r < to.rows; r++)
            comps.push(c === r ? s.code : "0.0");
        }
        return prim(`${wgsl}(${comps.join(", ")})`);
      }
      if (vals.length === 1 && isMatrix(ft)) {
        if (ft.rows === to.rows && ft.cols === to.cols) return first;
        // The top-left part, identity elsewhere.
        const cols: string[] = [];
        for (let c = 0; c < to.cols; c++) {
          const comps: string[] = [];
          for (let r = 0; r < to.rows; r++) {
            comps.push(
              c < ft.cols && r < ft.rows
                ? `${primary(first)}[${c}][${r}]`
                : c === r
                  ? "1.0"
                  : "0.0",
            );
          }
          cols.push(`vec${to.rows}f(${comps.join(", ")})`);
        }
        return prim(`${wgsl}(${cols.join(", ")})`);
      }
    }
    if (vals.length === 1 && isVector(ft) && ft.rows === to.rows) {
      return this.convert(first, to, pos);
    }
    // Components from the values in order; the last may have extra ones.
    const need = to.rows * to.cols;
    const parts: string[] = [];
    let count = 0;
    vals.forEach((v, i) => {
      const t = v.ty as Num;
      if (isMatrix(t))
        this.error(
          `can't construct ${glslName(to)} from a matrix and other values`,
          pos,
        );
      if (count >= need)
        this.error(`too many values for ${glslName(to)}`, e.pos);
      let c = this.convert(v, num(to.s, t.rows), pos);
      if (count + t.rows > need) {
        if (i !== vals.length - 1)
          this.error(`too many values for ${glslName(to)}`, pos);
        const keep = need - count;
        c = {
          code: `${primary(c)}.${"xyzw".slice(0, keep)}`,
          ty: num(to.s, keep),
          cls: "primary",
        };
        count = need;
      } else {
        count += t.rows;
      }
      parts.push(c.code);
    });
    if (vals.length === 1 && isScalar(ft) && isVector(to)) {
      return prim(`${wgsl}(${parts[0]})`);
    }
    if (count < need) this.error(`not enough values for ${glslName(to)}`, pos);
    if (isMatrix(to)) {
      // WGSL takes columns or scalars, not a mix: split vectors.
      const allCols = vals.every((v) => (v.ty as Num).rows === to.rows);
      const allScalars = vals.every((v) => isScalar(v.ty));
      if (!allCols && !allScalars) {
        const scalars: string[] = [];
        for (const v of vals) {
          const t = v.ty as Num;
          const c = this.convert(v, num("f32", t.rows), pos);
          if (t.rows === 1) scalars.push(c.code);
          else
            for (let k = 0; k < t.rows; k++)
              scalars.push(`${primary(c)}[${k}]`);
        }
        return prim(`${wgsl}(${scalars.slice(0, need).join(", ")})`);
      }
    }
    return prim(`${wgsl}(${parts.join(", ")})`);
  }

  private userCall(
    name: string,
    sigs: FnSig[],
    e: Extract<Expr, { k: "call" }>,
  ): Val {
    const vals = e.args.map((a) => this.expr(a));
    const matches = (exact: boolean) =>
      sigs.filter(
        (s) =>
          s.params.length === vals.length &&
          s.params.every((p, i) => {
            const v = vals[i];
            if (tyEq(p.ty, v.ty)) return true;
            if (exact || !isNum(p.ty) || !isNum(v.ty)) return false;
            return (
              p.qual === "in" &&
              p.ty.rows === v.ty.rows &&
              p.ty.cols === v.ty.cols &&
              (v.lit || p.ty.s === "f32")
            );
          }),
      );
    const exact = matches(true);
    const found = exact.length ? exact : matches(false);
    if (found.length !== 1) {
      this.error(
        `${found.length ? "ambiguous" : "no matching"} call ${name}(${vals.map((v) => glslName(v.ty)).join(", ")})`,
        e.pos,
      );
    }
    const sig = found[0];
    const args = sig.params.map((p, i) =>
      p.qual === "in"
        ? this.convert(vals[i], p.ty, e.args[i].pos).code
        : this.pointerArg(e.args[i]),
    );
    return {
      code: `${sig.wgsl}(${args.join(", ")})`,
      ty: sig.ret,
      cls: "primary",
    };
  }

  /** `&x` for an out/inout argument, which must be a variable. */
  private pointerArg(arg: Expr): string {
    if (arg.k === "id") {
      const sym = this.lookup(arg.name);
      if (sym?.kind === "ptr") return sym.code.slice(2, -1);
      if (sym?.kind === "var") return `&${sym.code}`;
    }
    this.error("an out or inout argument must be a local variable", arg.pos);
  }

  private builtin(name: string, e: Extract<Expr, { k: "call" }>): Val {
    const pos = e.pos;
    const vals = e.args.map((a) => this.expr(a));
    const n = vals.length;
    const want = (...counts: number[]) => {
      if (!counts.includes(n)) {
        this.error(
          `${name}() takes ${counts.join(" or ")} arguments, not ${n}`,
          pos,
        );
      }
    };
    const call = (fn: string, args: Val[], ty: Ty): Val => ({
      code: `${fn}(${args.map((a) => a.code).join(", ")})`,
      ty,
      cls: "primary",
    });
    const numeric = (v: Val, i = 0): Num => {
      if (!isNum(v.ty))
        this.error(
          `${name}() argument ${i + 1} is ${glslName(v.ty)}`,
          e.args[i]?.pos ?? pos,
        );
      return v.ty;
    };
    const floats = () => vals.map((v, i) => this.toFloat(v, e.args[i].pos));
    // genType f(genType...), scalars widened to the first argument's shape.
    const generic = (fn: string, count: number[], allowInt = false) => {
      want(...count);
      let args = allowInt ? vals : floats();
      args.forEach(numeric);
      const shapeOf = args.find((a) => !isScalar(a.ty))?.ty ?? args[0].ty;
      if (allowInt) {
        const [a] = args;
        args = args.map((v, i) => (i ? this.unify(a, v, e.args[i].pos)[1] : v));
        const s = (args[0].ty as Num).s;
        args = args.map((v, i) =>
          this.convert(v, num(s, (v.ty as Num).rows), e.args[i].pos),
        );
      }
      args = args.map((v, i) => this.splat(v, shapeOf, e.args[i].pos));
      return call(fn, args, args[0].ty);
    };

    if (FLOAT_FUNCTIONS.has(name)) return generic(name, [1]);
    switch (name) {
      case "normalize":
        return generic(isScalar(vals[0]?.ty ?? VOID) ? "sign" : name, [1]);
      case "roundEven":
        return generic("round", [1]);
      case "inversesqrt":
        return generic("inverseSqrt", [1]);
      case "abs":
      case "sign":
        return generic(name, [1], true);
      case "atan":
        return n === 2 ? generic("atan2", [2]) : generic("atan", [1]);
      case "pow":
        return generic("pow", [2]);
      case "min":
      case "max":
      case "clamp":
        return generic(name, name === "clamp" ? [3] : [2], true);
      case "mix": {
        want(3);
        const last = vals[2];
        if (isNum(last.ty) && last.ty.s === "bool") {
          const [a, b] = this.unify(vals[0], vals[1], pos);
          return call("select", [a, b, last], a.ty);
        }
        const [x, y, a] = floats();
        const t = isScalar(x.ty) ? y.ty : x.ty;
        return call(
          "mix",
          [this.splat(x, t, pos), this.splat(y, t, pos), a],
          t,
        );
      }
      case "step": {
        want(2);
        const [edge, x] = floats();
        const t = isScalar(x.ty) ? edge.ty : x.ty;
        return call(
          "step",
          [this.splat(edge, t, pos), this.splat(x, t, pos)],
          t,
        );
      }
      case "smoothstep": {
        want(3);
        const [a, b, x] = floats();
        const t = [x, a, b].find((v) => !isScalar(v.ty))?.ty ?? x.ty;
        // GLSL's formula, which code relies on for edge0 > edge1 (where
        // WGSL implementations may differ).
        const w = this.wgslTy(t);
        const fn = this.helper(`glsl_smoothstep_${w}`, () =>
          [
            `fn glsl_smoothstep_${w}(e0: ${w}, e1: ${w}, x: ${w}) -> ${w} {`,
            `    let t = clamp((x - e0) / (e1 - e0), ${w}(0.0), ${w}(1.0));`,
            "    return t * t * (3.0 - 2.0 * t);",
            "}",
          ].join("\n"),
        );
        return call(
          fn,
          [a, b, x].map((v) => this.splat(v, t, pos)),
          t,
        );
      }
      case "mod": {
        want(2);
        const [x, y] = floats();
        const t = isScalar(x.ty) ? y.ty : x.ty;
        const w = this.wgslTy(t);
        const fn = this.helper(`glsl_mod_${w}`, () =>
          [
            `// GLSL mod: x - y * floor(x / y), unlike WGSL's %.`,
            `fn glsl_mod_${w}(x: ${w}, y: ${w}) -> ${w} {`,
            "    return x - y * floor(x / y);",
            "}",
          ].join("\n"),
        );
        return call(fn, [this.splat(x, t, pos), this.splat(y, t, pos)], t);
      }
      case "modf": {
        want(2);
        const [x] = floats();
        const w = this.wgslTy(x.ty);
        const fn = this.helper(`glsl_modf_${w}`, () =>
          [
            `fn glsl_modf_${w}(x: ${w}, whole: ptr<function, ${w}>) -> ${w} {`,
            "    let parts = modf(x);",
            "    *whole = parts.whole;",
            "    return parts.fract;",
            "}",
          ].join("\n"),
        );
        const i = this.expr(e.args[1]);
        if (!tyEq(i.ty, x.ty))
          this.error("modf() parts must have the same type", pos);
        return {
          code: `${fn}(${x.code}, ${this.pointerArg(e.args[1])})`,
          ty: x.ty,
          cls: "primary",
        };
      }
      case "length":
      case "distance":
      case "dot": {
        want(name === "length" ? 1 : 2);
        const [x, y] = floats();
        // WGSL has these for vectors only.
        if (isScalar(x.ty)) {
          if (name === "dot") return this.binary("*", x, y, pos);
          const diff = name === "length" ? x : this.binary("-", x, y, pos);
          return call("abs", [diff], F32);
        }
        return call(name, floats(), F32);
      }
      case "cross":
        want(2);
        return call("cross", floats(), V3);
      case "reflect":
        want(2);
        return call("reflect", floats(), vals[0].ty);
      case "refract":
        want(3);
        return call("refract", floats(), vals[0].ty);
      case "faceforward":
        want(3);
        return call("faceForward", floats(), vals[0].ty);
      case "fma":
        return generic("fma", [3]);
      case "ldexp": {
        want(2);
        return call("ldexp", [this.toFloat(vals[0], pos), vals[1]], vals[0].ty);
      }
      case "determinant":
        want(1);
        return call("determinant", vals, F32);
      case "transpose": {
        want(1);
        const t = numeric(vals[0]);
        return call("transpose", vals, num("f32", t.cols, t.rows));
      }
      case "inverse": {
        want(1);
        const t = numeric(vals[0]);
        if (!isMatrix(t) || t.rows !== t.cols)
          this.error("inverse() takes a square matrix", pos);
        const fn = this.helper(`dyno_inverse${t.cols}`, () => INVERSE[t.cols]);
        return call(fn, vals, t);
      }
      case "matrixCompMult": {
        want(2);
        const t = numeric(vals[0]);
        const cols = Array.from(
          { length: t.cols },
          (_, c) => `${primary(vals[0])}[${c}] * ${primary(vals[1])}[${c}]`,
        );
        return {
          code: `${this.wgslTy(t)}(${cols.join(", ")})`,
          ty: t,
          cls: "primary",
        };
      }
      case "outerProduct": {
        want(2);
        const [c, r] = floats();
        const t = num("f32", (c.ty as Num).rows, (r.ty as Num).rows);
        const cols = Array.from(
          { length: t.cols },
          (_, j) => `${operand(c)} * ${primary(r)}[${j}]`,
        );
        return {
          code: `${this.wgslTy(t)}(${cols.join(", ")})`,
          ty: t,
          cls: "primary",
        };
      }
      case "lessThan":
      case "lessThanEqual":
      case "greaterThan":
      case "greaterThanEqual":
      case "equal":
      case "notEqual": {
        want(2);
        const op = {
          lessThan: "<",
          lessThanEqual: "<=",
          greaterThan: ">",
          greaterThanEqual: ">=",
          equal: "==",
          notEqual: "!=",
        }[name];
        const [a, b] = this.unify(vals[0], vals[1], pos);
        const t = numeric(a);
        return {
          code: `${operand(a)} ${op} ${operand(b)}`,
          ty: num("bool", t.rows),
          cls: "rel",
        };
      }
      case "any":
      case "all":
        want(1);
        return call(name, vals, BOOL);
      case "not":
        want(1);
        return { code: `!${operand(vals[0])}`, ty: vals[0].ty, cls: "unary" };
      case "isnan":
      case "isinf": {
        want(1);
        const t = numeric(vals[0]);
        const u = t.rows === 1 ? "u32" : `vec${t.rows}u`;
        const k = (v: string) => (t.rows === 1 ? v : `${u}(${v})`);
        const bits = `(bitcast<${u}>(${vals[0].code}) & ${k("0x7fffffffu")})`;
        return {
          code: `${bits} ${name === "isnan" ? ">" : "=="} ${k("0x7f800000u")}`,
          ty: num("bool", t.rows),
          cls: "rel",
        };
      }
      case "floatBitsToInt":
      case "floatBitsToUint":
      case "intBitsToFloat":
      case "uintBitsToFloat": {
        want(1);
        const t = numeric(vals[0]);
        const s: WgslScalar =
          name === "floatBitsToInt"
            ? "i32"
            : name === "floatBitsToUint"
              ? "u32"
              : "f32";
        const to = num(s, t.rows);
        const from = name.startsWith("float")
          ? "f32"
          : name.startsWith("int")
            ? "i32"
            : "u32";
        const arg = this.convert(vals[0], num(from, t.rows), pos);
        return {
          code: `bitcast<${this.wgslTy(to)}>(${arg.code})`,
          ty: to,
          cls: "primary",
        };
      }
      case "packHalf2x16":
      case "packUnorm2x16":
      case "packSnorm2x16":
      case "packUnorm4x8":
      case "packSnorm4x8": {
        want(1);
        const m = /pack(\w+?)(\dx\d+)/.exec(name) as RegExpExecArray;
        const kind = m[1] === "Half" ? "float" : m[1].toLowerCase();
        return call(`pack${m[2]}${kind}`, floats(), U32);
      }
      case "unpackHalf2x16":
      case "unpackUnorm2x16":
      case "unpackSnorm2x16":
      case "unpackUnorm4x8":
      case "unpackSnorm4x8": {
        want(1);
        const m = /unpack(\w+?)(\dx\d+)/.exec(name) as RegExpExecArray;
        const kind = m[1] === "Half" ? "float" : m[1].toLowerCase();
        const arg = this.convert(vals[0], U32, pos);
        return call(
          `unpack${m[2]}${kind}`,
          [arg],
          num("f32", m[2] === "2x16" ? 2 : 4),
        );
      }
      case "bitCount":
      case "findLSB":
      case "findMSB": {
        want(1);
        const t = numeric(vals[0]);
        const fn = {
          bitCount: "countOneBits",
          findLSB: "firstTrailingBit",
          findMSB: "firstLeadingBit",
        }[name];
        const r = call(fn, vals, t);
        // GLSL returns int(s) for uint too.
        return t.s === "u32" ? this.convert(r, num("i32", t.rows), pos) : r;
      }
      case "bitfieldReverse":
        want(1);
        return call("reverseBits", vals, vals[0].ty);
      case "bitfieldExtract": {
        want(3);
        const [v, o, b] = vals;
        return call(
          "extractBits",
          [v, this.convert(o, U32, pos), this.convert(b, U32, pos)],
          v.ty,
        );
      }
      case "bitfieldInsert": {
        want(4);
        const [v, ins, o, b] = vals;
        return call(
          "insertBits",
          [v, ins, this.convert(o, U32, pos), this.convert(b, U32, pos)],
          v.ty,
        );
      }
      case "texture":
      case "textureLod":
        return this.textureSample(name, vals, e);
      case "texelFetch":
        return this.texelFetch(vals, e);
      case "textureSize": {
        want(1, 2);
        const tex = this.textureArg(vals[0], e.args[0].pos);
        const lod = n > 1 ? this.convert(vals[1], I32, pos).code : "0";
        const dims = `textureDimensions(${tex.code}, ${lod})`;
        if (tex.glsl.includes("2DArray")) {
          return {
            code: `vec3i(vec3u(${dims}, textureNumLayers(${tex.code})))`,
            ty: num("i32", 3),
            cls: "primary",
          };
        }
        const rows = tex.glsl.includes("3D") ? 3 : 2;
        return {
          code: `vec${rows}i(${dims})`,
          ty: num("i32", rows),
          cls: "primary",
        };
      }
      case "dFdx":
      case "dFdy":
      case "fwidth":
        this.error(
          `${name}() is not available: dynos run in compute shaders, which have no derivatives`,
          pos,
        );
        break;
      case "textureGrad":
      case "textureProj":
      case "textureOffset":
      case "texelFetchOffset":
      case "textureLodOffset":
        this.error(
          `${name}() is not supported (use texture, textureLod or texelFetch)`,
          pos,
        );
        break;
    }
    this.error(`unknown function '${name}'`, pos);
  }

  private textureArg(
    v: Val,
    pos: Pos,
  ): { code: string; glsl: string; sampler?: string; value: string } {
    if (v.ty.k !== "tex")
      this.error(`expected a sampler, not ${glslName(v.ty)}`, pos);
    const value =
      (textureType(v.ty.glsl as DynoType) as string).match(/<(\w+)>/)?.[1] ??
      "f32";
    return { code: v.code, glsl: v.ty.glsl, sampler: v.sampler, value };
  }

  private texelType(value: string): Num {
    return num(value as WgslScalar, 4);
  }

  private textureSample(
    name: string,
    vals: Val[],
    e: Extract<Expr, { k: "call" }>,
  ): Val {
    const pos = e.pos;
    if (vals.length < 2 || vals.length > 3)
      this.error(`${name}() takes 2 or 3 arguments`, pos);
    const tex = this.textureArg(vals[0], e.args[0].pos);
    if (tex.value !== "f32" || tex.glsl.includes("Shadow")) {
      this.error(
        `${name}() on ${tex.glsl}: WGSL samples only float textures; use texelFetch`,
        pos,
      );
    }
    if (!tex.sampler) {
      this.error(
        `${name}() needs a sampler: pass the texture as a dyno input`,
        pos,
      );
    }
    const coord = this.toFloat(vals[1], e.args[1].pos);
    // Compute shaders have no derivatives: texture() samples level 0, with
    // a bias as the level (as the WGSL Texture op does).
    const level = vals[2] ? this.toFloat(vals[2], e.args[2].pos).code : "0.0";
    const c = primary(coord);
    const args = tex.glsl.includes("2DArray")
      ? `${tex.code}, ${tex.sampler}, ${c}.xy, i32(floor(${c}.z + 0.5))`
      : `${tex.code}, ${tex.sampler}, ${coord.code}`;
    return {
      code: `textureSampleLevel(${args}, ${level})`,
      ty: V4,
      cls: "primary",
    };
  }

  private texelFetch(vals: Val[], e: Extract<Expr, { k: "call" }>): Val {
    const pos = e.pos;
    if (vals.length !== 3) this.error("texelFetch() takes 3 arguments", pos);
    const tex = this.textureArg(vals[0], e.args[0].pos);
    const coord = vals[1];
    if (!isNum(coord.ty) || coord.ty.s !== "i32") {
      this.error("texelFetch() coordinates must be ivec", e.args[1].pos);
    }
    const lod = this.convert(vals[2], I32, e.args[2].pos).code;
    const c = primary(coord);
    const args = tex.glsl.includes("2DArray")
      ? `${tex.code}, ${c}.xy, ${c}.z`
      : `${tex.code}, ${coord.code}`;
    return {
      code: `textureLoad(${args}, ${lod})`,
      ty: this.texelType(tex.value),
      cls: "primary",
    };
  }

  // Statements

  private lvalue(e: Expr): { code: string; ty: Ty } {
    let root = e;
    while (root.k === "member" || root.k === "index") root = root.obj;
    if (root.k !== "id") this.error("can't assign to this expression", e.pos);
    const sym = this.lookup(root.name);
    if (!sym) this.error(`unknown name '${root.name}'`, root.pos);
    if (sym.kind === "const")
      this.error(`can't assign to constant '${root.name}'`, e.pos);
    if (sym.kind === "input")
      this.error(`can't assign to dyno input '${root.name}'`, e.pos);
    if (sym.kind === "param")
      this.error(`can't assign to parameter '${root.name}'`, e.pos);
    const v = this.expr(e);
    return { code: v.code, ty: v.ty };
  }

  /** An assignment (or ++/--) as WGSL statements. */
  private assign(
    op: string,
    target: Expr,
    value: Val | null,
    pos: Pos,
  ): string[] {
    // A multi-component swizzle: per-component writes.
    if (target.k === "member") {
      const base = this.expr(target.obj);
      if (isNum(base.ty) && isVector(base.ty) && target.name.length > 1) {
        this.lvalue(target.obj);
        const swz = this.swizzle(target.name, base.ty.rows, target.pos);
        if (new Set(swz).size !== swz.length)
          this.error("swizzle assignment repeats a component", target.pos);
        const ty = num(base.ty.s, swz.length);
        const rhs = this.assignedValue(op, this.expr(target), ty, value, pos);
        if (
          swz === "xyzw".slice(0, base.ty.rows) ||
          swz === "rgba".slice(0, base.ty.rows)
        ) {
          return [`${base.code} = ${rhs.code};`];
        }
        const tmp = `swz_${this.temp++}`;
        const b = primary(base);
        return [
          "{",
          `    let ${tmp}: ${this.wgslTy(ty)} = ${rhs.code};`,
          ...[...swz].map((c, i) => `    ${b}.${c} = ${tmp}.${"xyzw"[i]};`),
          "}",
        ];
      }
    }
    const lhs = this.lvalue(target);
    if (value === null) {
      // ++ and --: WGSL has them for integer scalars only.
      if (!isNum(lhs.ty) || isMatrix(lhs.ty))
        this.error(`'${op}' on ${glslName(lhs.ty)}`, pos);
      if (isScalar(lhs.ty) && isInt(lhs.ty)) return [`${lhs.code}${op};`];
      const one = lhs.ty.s === "f32" ? "1.0" : "1";
      return [`${lhs.code} ${op[0]}= ${one};`];
    }
    if (op === "=") {
      return [`${lhs.code} = ${this.convert(value, lhs.ty, pos).code};`];
    }
    const rhs = this.assignedValue(
      op,
      { code: lhs.code, ty: lhs.ty, cls: "primary" },
      lhs.ty,
      value,
      pos,
    );
    // A plain `lhs op rest` keeps WGSL's compound assignment.
    const prefix = `${lhs.code} ${op.slice(0, -1)} `;
    if (rhs.code.startsWith(prefix) && rhs.cls === OP_CLASS[op.slice(0, -1)]) {
      return [`${lhs.code} ${op} ${rhs.code.slice(prefix.length)};`];
    }
    return [`${lhs.code} = ${rhs.code};`];
  }

  private assignedValue(
    op: string,
    current: Val,
    ty: Ty,
    value: Val | null,
    pos: Pos,
  ): Val {
    if (value === null) {
      return this.binary(
        op[0],
        current,
        { code: "1", ty: I32, cls: "primary", lit: true },
        pos,
      );
    }
    if (op === "=") return this.convert(value, ty, pos);
    const r = this.binary(op.slice(0, -1), current, value, pos);
    if (!tyEq(r.ty, ty)) {
      this.error(`'${op}' gives ${glslName(r.ty)}, not ${glslName(ty)}`, pos);
    }
    return r;
  }

  /** An expression statement (or for-loop update) as WGSL statements. */
  private exprStatement(e: Expr): string[] {
    switch (e.k) {
      case "assign":
        return this.assign(e.op, e.target, this.expr(e.value), e.pos);
      case "postfix":
      case "unary":
        if (e.op === "++" || e.op === "--")
          return this.assign(e.op, e.arg, null, e.pos);
        break;
      case "comma":
        return e.exprs.flatMap((x) => this.exprStatement(x));
      case "call": {
        const v = this.expr(e);
        if (this.functions.has(e.callee.name)) return [`${v.code};`];
        return [`_ = ${v.code};`];
      }
    }
    const v = this.expr(e);
    return [`_ = ${v.code};`];
  }

  private declaration(
    s: Extract<Stmt, { k: "decl" }>,
    global: boolean,
  ): string[] {
    const out: string[] = [];
    for (const d of s.vars) {
      const init = d.init ? this.expr(d.init) : undefined;
      const ty = this.ty(s.type, d.array, init);
      if (ty.k === "void" || ty.k === "tex")
        this.error(`can't declare a ${glslName(ty)} variable`, d.pos);
      const name = safeName(d.name, global);
      const value =
        init && d.init ? this.convert(init, ty, d.init.pos).code : undefined;
      const t = this.wgslTy(ty);
      if (s.isConst) {
        if (!value) this.error("a const needs a value", d.pos);
        out.push(`const ${name}: ${t} = ${value};`);
      } else if (global) {
        out.push(`var<private> ${name}: ${t}${value ? ` = ${value}` : ""};`);
      } else {
        out.push(`var ${name}: ${t}${value ? ` = ${value}` : ""};`);
      }
      this.declare(d.name, {
        ty,
        code: name,
        kind: s.isConst ? "const" : "var",
      });
    }
    return out;
  }

  private body(s: Stmt, ctx: Ctx): string[] {
    const stmts = s.k === "block" ? s.body : [s];
    return this.withScope(() =>
      stmts.flatMap((x) => this.statement(x, ctx)),
    ).map((l) => `    ${l}`);
  }

  private condition(e: Expr): string {
    const v = this.expr(e);
    if (!tyEq(v.ty, BOOL))
      this.error(`condition is ${glslName(v.ty)}, not bool`, e.pos);
    return v.code;
  }

  statement(s: Stmt, ctx: Ctx): string[] {
    switch (s.k) {
      case "decl":
        return this.declaration(s, false);
      case "expr":
        return this.exprStatement(s.expr);
      case "block":
        return ["{", ...this.body(s, ctx), "}"];
      case "empty":
        return [];
      case "if": {
        const lines = [
          `if (${this.condition(s.test)}) {`,
          ...this.body(s.then, ctx),
        ];
        if (s.else) {
          if (s.else.k === "if") {
            const [first, ...rest] = this.statement(s.else, ctx);
            lines.push(`} else ${first}`, ...rest);
            return lines;
          }
          lines.push("} else {", ...this.body(s.else, ctx));
        }
        lines.push("}");
        return lines;
      }
      case "for":
        return this.withScope(() => {
          const pre: string[] = [];
          let init = "";
          if (s.init) {
            const lines = this.statement(s.init, ctx);
            if (lines.length === 1) init = lines[0].replace(/;$/, "");
            else pre.push(...lines);
          }
          const test = s.test ? this.condition(s.test) : "";
          const update = s.update ? this.exprStatement(s.update) : [];
          const body = this.body(s.body, ctx);
          // WGSL's for takes one update statement; a loop's continuing
          // block (where continue goes) takes any.
          const loop =
            update.length > 1
              ? [
                  "loop {",
                  ...(test
                    ? [`    if !(${test}) {`, "        break;", "    }"]
                    : []),
                  ...body,
                  "    continuing {",
                  ...update.map((l) => `        ${l}`),
                  "    }",
                  "}",
                ]
              : [
                  `for (${init}; ${test}; ${(update[0] ?? "").replace(/;$/, "")}) {`,
                  ...body,
                  "}",
                ];
          if (update.length > 1 && init) pre.push(`${init};`);
          return pre.length
            ? ["{", ...[...pre, ...loop].map((l) => `    ${l}`), "}"]
            : loop;
        });
      case "while":
        return [
          `while (${this.condition(s.test)}) {`,
          ...this.body(s.body, ctx),
          "}",
        ];
      case "do": {
        // continue goes to the continuing block, which tests, as in GLSL.
        const body = this.body(s.body, ctx);
        const test = this.condition(s.test);
        return [
          "loop {",
          ...body,
          "    continuing {",
          `        break if !(${test});`,
          "    }",
          "}",
        ];
      }
      case "return": {
        if (ctx.ret === null) {
          if (s.value) this.error("dyno code can only 'return;'", s.pos);
          if (ctx.returnValue == null)
            this.error("'return' is not available here", s.pos);
          return [`return ${ctx.returnValue};`];
        }
        if (!s.value) {
          if (ctx.ret.k !== "void") this.error("missing return value", s.pos);
          return ["return;"];
        }
        return [
          `return ${this.convert(this.expr(s.value), ctx.ret, s.value.pos).code};`,
        ];
      }
      case "break":
      case "continue":
        return [`${s.k};`];
      case "discard":
        this.error(
          "discard is not available: dynos run in compute shaders",
          s.pos,
        );
        break;
      case "switch": {
        const disc = this.expr(s.disc);
        if (!isScalar(disc.ty) || !isInt(disc.ty))
          this.error("switch needs an int or uint", s.disc.pos);
        const lines = [`switch (${disc.code}) {`];
        let selectors: string[] = [];
        let hasDefault = false;
        s.cases.forEach((c, i) => {
          if (c.tests) {
            for (const t of c.tests)
              selectors.push(this.convert(this.expr(t), disc.ty, t.pos).code);
          } else {
            selectors.push("default");
            hasDefault = true;
          }
          if (!c.body.length && i < s.cases.length - 1) return;
          if (i < s.cases.length - 1 && !terminates(c.body)) {
            this.error(
              "switch fallthrough is not supported (end the case with break)",
              c.pos,
            );
          }
          const body = c.body.filter(
            (b, j) => !(b.k === "break" && j === c.body.length - 1),
          );
          lines.push(`    case ${selectors.join(", ")}: {`);
          const inner = this.withScope(() =>
            body.flatMap((b) => this.statement(b, ctx)),
          );
          lines.push(...inner.map((l) => `        ${l}`), "    }");
          selectors = [];
        });
        if (!hasDefault) lines.push("    default: {}");
        lines.push("}");
        return lines;
      }
    }
    this.error("unsupported statement", s.pos);
  }

  // Functions

  /** Parameters the body assigns to, which become local copies. */
  private assignedNames(stmts: Stmt[]): Set<string> {
    const names = new Set<string>();
    const root = (target: Expr): void => {
      let e = target;
      while (e.k === "member" || e.k === "index") e = e.obj;
      if (e.k === "id") names.add(e.name);
    };
    const visitExpr = (e: Expr | undefined): void => {
      if (!e) return;
      switch (e.k) {
        case "assign":
          root(e.target);
          visitExpr(e.target);
          visitExpr(e.value);
          return;
        case "unary":
        case "postfix":
          if (e.op === "++" || e.op === "--") root(e.arg);
          visitExpr(e.arg);
          return;
        case "call": {
          const sigs = this.functions.get(e.callee.name);
          e.args.forEach((a, i) => {
            if (
              (e.callee.name === "modf" && i === 1) ||
              sigs?.some((s) => s.params[i] && s.params[i].qual !== "in")
            ) {
              root(a);
            }
            visitExpr(a);
          });
          return;
        }
        case "method":
          visitExpr(e.obj);
          e.args.forEach(visitExpr);
          return;
        case "member":
          visitExpr(e.obj);
          return;
        case "index":
          visitExpr(e.obj);
          visitExpr(e.index);
          return;
        case "binary":
          visitExpr(e.left);
          visitExpr(e.right);
          return;
        case "cond":
          visitExpr(e.test);
          visitExpr(e.then);
          visitExpr(e.else);
          return;
        case "comma":
          e.exprs.forEach(visitExpr);
          return;
      }
    };
    const visit = (s: Stmt | undefined): void => {
      if (!s) return;
      switch (s.k) {
        case "decl":
          for (const d of s.vars) visitExpr(d.init);
          return;
        case "expr":
          visitExpr(s.expr);
          return;
        case "block":
          s.body.forEach(visit);
          return;
        case "if":
          visitExpr(s.test);
          visit(s.then);
          visit(s.else);
          return;
        case "for":
          visit(s.init);
          visitExpr(s.test);
          visitExpr(s.update);
          visit(s.body);
          return;
        case "while":
        case "do":
          visitExpr(s.test);
          visit(s.body);
          return;
        case "return":
          visitExpr(s.value);
          return;
        case "switch":
          visitExpr(s.disc);
          for (const c of s.cases) c.body.forEach(visit);
          return;
      }
    };
    stmts.forEach(visit);
    return names;
  }

  private func(g: Extract<Global, { k: "func" }>): string | null {
    const ret = this.ty(g.ret);
    const params = g.params.map((p) => ({ ...p, ty: this.ty(p.type) }));
    const sigs = this.functions.get(g.name) ?? [];
    const same = sigs.find(
      (s) =>
        s.params.length === params.length &&
        s.params.every((p, i) => tyEq(p.ty, params[i].ty)),
    );
    let sig = same;
    if (!sig) {
      if (this.isBuiltinName(g.name)) {
        this.error(
          `'${g.name}' is a GLSL builtin and can't be redefined`,
          g.pos,
        );
      }
      // GLSL overloads; WGSL needs distinct names.
      const base = safeName(g.name, true);
      const wgsl = sigs.length
        ? `${base}_${params.map((p) => glslName(p.ty).replace(/\W/g, "")).join("_")}`
        : base;
      sig = {
        params: params.map((p) => ({ ty: p.ty, qual: p.qual })),
        ret,
        wgsl,
      };
      sigs.push(sig);
      this.functions.set(g.name, sigs);
    } else if (!tyEq(sig.ret, ret)) {
      this.error(`'${g.name}' redeclared with another return type`, g.pos);
    }
    if (!g.body) return null;
    const body = g.body;
    const assigned = this.assignedNames(body);
    const fnSig = sig;
    return this.withScope(() => {
      const decls: string[] = [];
      const copies: string[] = [];
      for (const p of params) {
        const name = safeName(p.name);
        const t = this.wgslTy(p.ty);
        if (p.qual !== "in") {
          decls.push(`${name}: ptr<function, ${t}>`);
          this.declare(p.name, { ty: p.ty, code: `(*${name})`, kind: "ptr" });
        } else if (assigned.has(p.name)) {
          // WGSL parameters are immutable: copy those the body changes.
          decls.push(`${name}_in: ${t}`);
          copies.push(`    var ${name}: ${t} = ${name}_in;`);
          this.declare(p.name, { ty: p.ty, code: name, kind: "var" });
        } else {
          decls.push(`${name}: ${t}`);
          this.declare(p.name, {
            ty: p.ty,
            code: name,
            kind: "param",
          });
        }
      }
      const ctx: Ctx = { ret };
      const lines = this.withScope(() =>
        body.flatMap((s) => this.statement(s, ctx)),
      );
      // WGSL requires a return at the end of a function with a result.
      if (ret.k !== "void" && !endsInReturn(body)) {
        lines.push(`return ${this.wgslTy(ret)}();`);
      }
      return [
        `fn ${fnSig.wgsl}(${decls.join(", ")})${ret.k === "void" ? "" : ` -> ${this.wgslTy(ret)}`} {`,
        ...copies,
        ...lines.map((l) => `    ${l}`),
        "}",
      ].join("\n");
    });
  }

  private isBuiltinName(name: string): boolean {
    return BUILTIN_NAMES.has(name);
  }

  private struct(g: Extract<Global, { k: "struct" }>): string {
    if (this.structs.has(g.name))
      this.error(`struct '${g.name}' is already defined`, g.pos);
    const info: StructInfo = {
      wgsl: safeName(g.name, true),
      fields: new Map(),
    };
    const lines: string[] = [];
    for (const f of g.fields) {
      const ty = this.ty(f.type, f.array);
      const wgsl = safeName(f.name);
      info.fields.set(f.name, { ty, wgsl });
      lines.push(`    ${wgsl}: ${this.wgslTy(ty)},`);
    }
    this.structs.set(g.name, info);
    return `struct ${info.wgsl} {\n${lines.join("\n")}\n}`;
  }

  /** GLSL global definitions, as WGSL ones (one string each). */
  translateGlobals(src: string): string[] {
    const cached = this.globalsCache.get(src);
    if (cached) return cached;
    this.src = src;
    const out: string[] = [];
    for (const g of parseGlobals(src)) {
      let code: string | null;
      let key: string;
      let pos: Pos;
      if (g.k === "func") {
        code = this.func(g);
        key = `fn ${g.name}(${g.params.map((p) => p.type.name).join(",")})`;
        pos = g.pos;
      } else if (g.k === "struct") {
        code = this.struct(g);
        key = `struct ${g.name}`;
        pos = g.pos;
      } else {
        code = this.declaration(g.decl, true).join("\n");
        key = `var ${g.decl.vars.map((v) => v.name).join(",")}`;
        pos = g.decl.pos;
      }
      if (code == null) continue;
      const previous = this.definitions.get(key);
      if (previous != null && previous !== code)
        this.error(`${key} is already defined`, pos);
      this.definitions.set(key, code);
      out.push(code);
    }
    this.globalsCache.set(src, out);
    return out;
  }

  /** A dyno's GLSL statements as WGSL lines, its inputs and outputs by name. */
  translateStatements(
    src: string,
    names: Map<string, GlslInput & { output?: boolean }>,
    returnValue?: string,
  ): string[] {
    this.src = src;
    const stmts = parseStatements(src);
    return this.withScope(() => {
      for (const [name, v] of names) {
        const code = /^[A-Za-z_][\w.]*$/.test(v.expr) ? v.expr : `(${v.expr})`;
        this.declare(name, {
          ty: v.type,
          code,
          kind: v.output ? "var" : "input",
          sampler: v.sampler,
          uniformStruct: v.uniformStruct,
        });
      }
      const ctx: Ctx = { ret: null, returnValue };
      return this.withScope(() => stmts.flatMap((s) => this.statement(s, ctx)));
    });
  }
}

/** Whether a switch case's statements end by leaving it. */
function terminates(stmts: Stmt[]): boolean {
  const last = stmts[stmts.length - 1];
  if (!last) return false;
  if (["break", "return", "continue"].includes(last.k)) return true;
  if (last.k === "block") return terminates(last.body);
  if (last.k === "if" && last.else) {
    return terminates([last.then]) && terminates([last.else]);
  }
  return false;
}

function endsInReturn(stmts: Stmt[]): boolean {
  const last = stmts[stmts.length - 1];
  if (!last) return false;
  if (last.k === "return") return true;
  if (last.k === "block") return endsInReturn(last.body);
  if (last.k === "if" && last.else) {
    return endsInReturn([last.then]) && endsInReturn([last.else]);
  }
  return false;
}

// genType f(genType) builtins WGSL names the same.
const FLOAT_FUNCTIONS = new Set(
  `radians degrees sin cos tan asin acos sinh cosh tanh asinh acosh atanh exp
  log exp2 log2 sqrt floor ceil trunc fract round`.split(/\s+/),
);

const BUILTIN_NAMES = new Set(
  `radians degrees sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh
  pow exp log exp2 log2 sqrt inversesqrt abs sign floor ceil trunc round
  roundEven fract mod modf min max clamp mix step smoothstep isnan isinf
  floatBitsToInt floatBitsToUint intBitsToFloat uintBitsToFloat fma ldexp
  packHalf2x16 unpackHalf2x16 packUnorm2x16 unpackUnorm2x16 packSnorm2x16
  unpackSnorm2x16 packUnorm4x8 unpackUnorm4x8 packSnorm4x8 unpackSnorm4x8
  length distance dot cross normalize faceforward reflect refract
  matrixCompMult outerProduct transpose determinant inverse lessThan
  lessThanEqual greaterThan greaterThanEqual equal notEqual any all not
  bitCount findLSB findMSB bitfieldExtract bitfieldInsert bitfieldReverse
  texture textureLod texelFetch textureSize dFdx dFdy fwidth`.split(/\s+/),
);

/** The translator's type for a dyno type. */
export function dynoTy(type: DynoType): Ty {
  const name = typeLiteral(type);
  const shape = typeShape(type);
  if (shape) return num(shape.scalar, shape.rows, shape.cols);
  if (textureType(type)) return { k: "tex", glsl: name };
  return { k: "struct", name };
}

/** The uniform array count behind a dyno input, if it is one. */
export function inputCount(value: unknown): number | undefined {
  let v = value as { dynoOut?: () => unknown } | undefined;
  while (v && !(v instanceof DynoValue)) {
    if (typeof v.dynoOut !== "function") return undefined;
    v = v.dynoOut() as typeof v;
  }
  if (v instanceof DynoOutput) {
    const count = (v.dyno as { count?: number }).count;
    return typeof count === "number" ? count : undefined;
  }
  return undefined;
}

const WGSL_HOST: GlslHost = {
  uniformStruct: wgslStructFields,
  structTexture: wgslStructTexture,
};

// One translator per program: its functions and structs are shared by the
// program's dynos, as GLSL globals are.
const translators = new WeakMap<WgslBackend, GlslTranslator>();

const PREFIX_IN = "__dyno_in_";
const PREFIX_OUT = "__dyno_out_";

/**
 * A plain dyno's GLSL as WGSL: its generate() runs with placeholder names
 * for inputs and outputs, which the translation then types (from the dyno's
 * inTypes/outTypes, uniform arrays from their DynoUniform) and replaces with
 * the backend's expressions. Shared GLSL globals the backend maps
 * (wgslGlobals) and WGSL globals pass through.
 */
export function translateDyno(
  dyno: Dyno<IOTypes, IOTypes>,
  context: GenerateContext<IOTypes, IOTypes>,
  backend: WgslBackend,
): DynoGenerated {
  let translator = translators.get(backend);
  if (!translator) {
    translator = new GlslTranslator(WGSL_HOST);
    translators.set(backend, translator);
  }
  const passThrough = (g: string) =>
    wgslGlobals.has(g) || backend.isWgslGlobal(g);
  const host = WGSL_HOST;
  const inputs: Record<string, string> = {};
  const outputs: Record<string, string> = {};
  const names = new Map<string, GlslInput & { output?: boolean }>();
  for (const [key, expr] of Object.entries(context.inputs)) {
    if (expr == null) continue;
    const placeholder = PREFIX_IN + key;
    inputs[key] = placeholder;
    let type = dynoTy(dyno.inTypes[key]);
    const count = inputCount(dyno.inputs[key]);
    if (count != null) type = { k: "array", of: type, n: count };
    const glsl = typeLiteral(dyno.inTypes[key]);
    const isTexture = type.k === "tex";
    const struct = type.k === "struct" && host.uniformStruct?.(glsl);
    names.set(placeholder, {
      type,
      expr,
      sampler:
        isTexture && glsl.startsWith("sampler") ? `${expr}_sampler` : undefined,
      uniformStruct: struct ? expr : undefined,
    });
  }
  for (const [key, name] of Object.entries(context.outputs)) {
    if (name == null) continue;
    const placeholder = PREFIX_OUT + key;
    outputs[key] = placeholder;
    names.set(placeholder, {
      type: dynoTy(dyno.outTypes[key]),
      expr: name,
      output: true,
    });
  }
  const generated = dyno.generate({
    inputs,
    outputs,
    compile: context.compile,
  });
  if (generated.uniforms && Object.keys(generated.uniforms).length) {
    throw new Error(
      "Dyno GLSL: uniforms from generate() can't be translated to WGSL; pass DynoUniform inputs",
    );
  }
  const where = `${dyno.constructor.name} dyno`;
  try {
    const globals: string[] = [];
    for (const g of generated.globals ?? []) {
      if (passThrough(g)) globals.push(g);
      else globals.push(...translator.translateGlobals(g));
    }
    const statements = generated.statements?.length
      ? translator.translateStatements(
          generated.statements.join("\n"),
          names,
          backend.returnValue,
        )
      : undefined;
    const all = [...translator.takeHelpers(), ...globals];
    for (const g of all) if (!passThrough(g)) backend.addGlobal(g);
    return { globals: all, statements };
  } catch (e) {
    if (e instanceof GlslError) {
      e.message = `${where}: GLSL to WGSL: ${e.message}`;
    }
    throw e;
  }
}

registerGlslTranslator(translateDyno);

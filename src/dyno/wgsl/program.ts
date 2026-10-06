// Dyno graphs compiled to WGSL functions, with the uniform block they share.
//
// WGSL has no templates like DynoProgramTemplate's: a kernel calls the
// functions by name (see src/webgpu/dyno), so each graph becomes
//   fn <name>(<inputs>) -> <output type> { ... }
// All functions of one program share a compilation, so value and uniform
// names are unique across them, and one uniform block / bind group.

import { Compilation, type Dyno, type IOTypes } from "../base";
import type { DynoType } from "../types";
import { typeLiteral } from "../types";
import { WgslBackend, type WgslUniformField } from "./backend";
import "./ops";
import "./glslToWgsl";
import { WGSL_PRELUDE } from "./prelude";
import { typeShape, uniformLayout, wgslType } from "./types";

export interface WgslFunction {
  /** Name of the WGSL function. */
  name: string;
  graph: Dyno<IOTypes, IOTypes>;
  /** The graph's inputs, which become the parameters, in order. */
  inputs: Record<string, DynoType>;
  /** The graph output the function returns. */
  output: string;
}

const INDENT = "    ";

export class WgslDynoProgram {
  readonly code: string;
  readonly backend: WgslBackend;
  readonly updaters: (() => void)[];

  constructor({
    functions,
    group = 1,
    prelude = true,
  }: {
    functions: WgslFunction[];
    /** Bind group for the uniform block and textures. */
    group?: number;
    /** Include the Gsplat structs and splatDefines helpers (default true). */
    prelude?: boolean;
  }) {
    const backend = new WgslBackend(group);
    const compile = new Compilation({ indent: INDENT, backend });
    const bodies: string[] = [];
    for (const fn of functions) {
      const outType = fn.graph.outTypes[fn.output];
      if (!outType) {
        throw new Error(`${fn.name}: graph has no output '${fn.output}'`);
      }
      const inputs: Record<string, string> = {};
      const params: string[] = [];
      for (const [key, type] of Object.entries(fn.inputs)) {
        inputs[key] = `in_${key}`;
        params.push(`in_${key}: ${wgslType(type)}`);
      }
      const out = `out_${fn.output}`;
      compile.declares.add(out);
      // A bare `return;` in a dyno's GLSL returns what the graph has so far.
      backend.returnValue = out;
      const statements = fn.graph.compile({
        inputs,
        outputs: { [fn.output]: out },
        compile,
      });
      bodies.push(
        [
          `fn ${fn.name}(${params.join(", ")}) -> ${wgslType(outType)} {`,
          `${INDENT}var ${out}: ${wgslType(outType)};`,
          ...statements.map((s) => (s ? INDENT + s : s)),
          `${INDENT}return ${out};`,
          "}",
        ].join("\n"),
      );
    }
    this.backend = backend;
    this.updaters = compile.updaters;
    this.code = [
      prelude ? WGSL_PRELUDE.trim() : "",
      ...backend.bindingDeclarations(),
      ...backend.wgslGlobals(compile.globals),
      ...bodies,
    ]
      .filter(Boolean)
      .join("\n\n");
  }

  /** Runs the dyno update callbacks, refreshing the uniform values. */
  update() {
    for (const updater of this.updaters) updater();
  }

  get uniformBytes(): number {
    return this.backend.uniformBytes;
  }

  /** The uniform block from the uniforms' current values. */
  packUniforms(data = new ArrayBuffer(this.uniformBytes)): ArrayBuffer {
    const view = new DataView(data);
    for (const field of this.backend.fields) writeField(view, field);
    return data;
  }
}

/** A uniform value as numbers: scalars, vectors, matrices column-major. */
export function flattenValue(value: unknown): number[] {
  if (typeof value === "number") return [value];
  if (typeof value === "boolean") return [value ? 1 : 0];
  if (value == null) return [];
  const v = value as {
    elements?: ArrayLike<number>;
    toArray?: () => number[];
    isColor?: boolean;
  };
  if (v.elements) return Array.from(v.elements);
  if (Array.isArray(value) && value.some((x) => typeof x === "object")) {
    return value.flatMap(flattenValue);
  }
  if (typeof v.toArray === "function") return v.toArray();
  return Array.from(value as ArrayLike<number>, Number);
}

function writeField(view: DataView, field: WgslUniformField) {
  if (field.members) {
    const struct = (field.uniform.value ?? {}) as Record<string, unknown>;
    for (const m of field.members) {
      writeValue(
        view,
        m.type,
        field.offset + m.offset,
        flattenValue(struct[m.name]),
      );
    }
    return;
  }
  const values = flattenValue(field.uniform.value);
  const shape = typeShape(field.type);
  const perElement = shape ? shape.rows * shape.cols : 1;
  const count = field.count ?? 1;
  for (let e = 0; e < count; e++) {
    writeValue(
      view,
      field.type,
      field.offset + e * (field.stride ?? 0),
      values.slice(e * perElement, (e + 1) * perElement),
    );
  }
}

/** One scalar, vector or matrix at `offset`, columns at uniform stride. */
function writeValue(
  view: DataView,
  type: DynoType,
  offset: number,
  values: number[],
) {
  const shape = typeShape(type);
  if (!shape) throw new Error(`uniform of type ${typeLiteral(type)}`);
  const { rows, cols } = shape;
  const colStride = cols > 1 ? uniformLayout(type).size / cols : 0;
  for (let c = 0; c < cols; c++) {
    for (let r = 0; r < rows; r++) {
      const v = values[c * rows + r] ?? 0;
      const at = offset + c * colStride + 4 * r;
      if (shape.scalar === "f32") view.setFloat32(at, v, true);
      else if (shape.scalar === "i32") view.setInt32(at, v, true);
      else view.setUint32(at, v >>> 0, true);
    }
  }
}

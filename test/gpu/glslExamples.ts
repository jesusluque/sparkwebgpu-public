// The GLSL dynos of Spark's examples, read from the example sources so the
// tests run the code as written: each dyno's globals and statements
// templates (its /* glsl */ literals) and its inTypes, rebuilt as a Dyno.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import * as THREE from "three";
import * as d from "../../src/dyno";

const ROOT = join(import.meta.dirname, "../..");

/** The `/* glsl *\/` template literals of a source file, in order. */
export function glslLiterals(source: string): { text: string; at: number }[] {
  const out: { text: string; at: number }[] = [];
  const re = /\/\*\s*glsl\s*\*\/\s*`/g;
  for (let m = re.exec(source); m; m = re.exec(source)) {
    const start = m.index + m[0].length;
    const end = templateEnd(source, start);
    out.push({ text: source.slice(start, end), at: m.index });
    re.lastIndex = end;
  }
  return out;
}

// The closing backtick of a template literal starting at `i`, through
// ${...} expressions with braces, strings and nested templates.
function templateEnd(s: string, from: number): number {
  for (let i = from; i < s.length; i++) {
    if (s[i] === "\\") i++;
    else if (s[i] === "`") return i;
    else if (s[i] === "$" && s[i + 1] === "{") i = expressionEnd(s, i + 2);
  }
  throw new Error("unterminated template");
}

function expressionEnd(s: string, from: number): number {
  let depth = 1;
  for (let i = from; i < s.length; i++) {
    const c = s[i];
    if (c === "{") depth++;
    else if (c === "}" && --depth === 0) return i;
    else if (c === "`") i = templateEnd(s, i + 1);
    else if (c === '"' || c === "'") {
      for (i++; s[i] !== c; i++) if (s[i] === "\\") i++;
    }
  }
  throw new Error("unterminated ${");
}

/** The object literal after `key:` before `at`, evaluated. */
function objectBefore(source: string, at: number, key: string, scope: object) {
  const start = source.lastIndexOf(`${key}:`, at);
  const open = source.indexOf("{", start);
  if (
    start < 0 ||
    /[\w]/.test(source.slice(start + key.length + 1, open).trim())
  ) {
    return undefined;
  }
  const close = expressionEnd(source, open + 1);
  return evaluate(source.slice(open, close + 1), scope);
}

function evaluate(expr: string, scope: object) {
  const names = Object.keys(scope);
  return new Function(...names, `return (${expr});`)(
    ...names.map((n) => (scope as Record<string, unknown>)[n]),
  );
}

export interface ExampleDyno {
  name: string;
  inTypes: Record<string, d.DynoType>;
  globals: string[];
  /** The statements template, as a function of inputs and outputs. */
  statements: (
    inputs: Record<string, string | undefined>,
    outputs: Record<string, string | undefined>,
  ) => string;
}

interface Spec {
  file: string;
  /** Per dyno: indices of its globals literals and of its statements. */
  dynos: { globals: number[]; statements: number; inTypes?: object }[];
  /** Values of the other names the templates use. */
  scope?: Record<string, unknown>;
}

const centers = [
  { x: 0.5, y: -1.25, z: 2 },
  { x: -1, y: 0, z: 0.75 },
  { x: 0, y: 1.5, z: -3 },
];
const MAX_IMPULSES = 4;
const holesTypes: Record<string, string | object> = {
  gsplat: d.Gsplat,
  time: "float",
  explosionStrength: "float",
  gravity: "float",
  bounceDamping: "float",
  floorLevel: "float",
  friction: "float",
  shrinkSpeed: "float",
};
for (let i = 0; i < MAX_IMPULSES; i++) {
  holesTypes[`clickCenter${i}`] = "vec3";
  holesTypes[`clickRadius${i}`] = "float";
  holesTypes[`clickStart${i}`] = "float";
  holesTypes[`clickActive${i}`] = "float";
}

const SPECS: Spec[] = [
  {
    file: "examples/glsl/index.html",
    dynos: [
      { globals: [0], statements: 1 },
      { globals: [2], statements: 3 },
    ],
  },
  {
    file: "examples/interactive-holes/index.html",
    dynos: [{ globals: [0], statements: 1, inTypes: holesTypes }],
    scope: { MAX_IMPULSES },
  },
  {
    file: "examples/lofi/index.html",
    dynos: [
      { globals: [], statements: 0 },
      { globals: [1], statements: 2 },
      { globals: [], statements: 3 },
      { globals: [4], statements: 5 },
    ],
  },
  {
    file: "examples/splat-flow/index.html",
    dynos: [{ globals: [0], statements: 1 }],
    scope: { PAUSE_SECONDS: 2.0, centers },
  },
  {
    file: "examples/splat-shader-effects/index.html",
    dynos: [{ globals: [0], statements: 1 }],
  },
  {
    file: "examples/splat-dissolve-effects/index.html",
    dynos: [{ globals: [0], statements: 1 }],
  },
  {
    file: "examples/splat-reveal-effects/index.html",
    dynos: [{ globals: [0], statements: 1 }],
  },
  {
    file: "examples/render-cube-depth/index.html",
    dynos: [{ globals: [0], statements: 1 }],
    scope: { minAlpha: 0.1 },
  },
  {
    file: "examples/splat-transitions/effects/explosion.js",
    dynos: [
      { globals: [0], statements: 1 },
      { globals: [2], statements: 3 },
    ],
  },
  {
    file: "examples/splat-transitions/effects/flow.js",
    dynos: [{ globals: [0], statements: 1 }],
    scope: { PAUSE_SECONDS: 2.0, centers },
  },
  {
    file: "examples/splat-transitions/effects/morph.js",
    dynos: [{ globals: [0], statements: 1 }],
  },
  {
    file: "examples/splat-transitions/effects/spheric.js",
    dynos: [{ globals: [0], statements: 1 }],
  },
  {
    file: "examples/interactive-deform/main.js",
    dynos: [{ globals: [], statements: 0 }],
  },
];

export function exampleDynos(): ExampleDyno[] {
  const result: ExampleDyno[] = [];
  for (const spec of SPECS) {
    const source = readFileSync(join(ROOT, spec.file), "utf8");
    const literals = glslLiterals(source);
    const scope: Record<string, unknown> = { dyno: d, ...spec.scope };
    // splat-flow and flow.js pass a GLSL global built from `centers`.
    const centerGLSL = literals.find((l) =>
      l.text.includes("getCenterOfMass(int"),
    );
    if (centerGLSL) {
      scope.centerGLSL = evaluate(`\`${centerGLSL.text}\``, scope);
    }
    spec.dynos.forEach((dyno, i) => {
      const first = literals[dyno.globals[0] ?? dyno.statements];
      const inTypes =
        dyno.inTypes ?? objectBefore(source, first.at, "inTypes", scope);
      if (!inTypes) throw new Error(`${spec.file}: no inTypes for dyno ${i}`);
      const template = (text: string) => {
        const names = ["inputs", "outputs", ...Object.keys(scope)];
        return new Function(...names, `return \`${text}\`;`) as (
          ...args: unknown[]
        ) => string;
      };
      const globals = dyno.globals.map((g) =>
        d.unindent(template(literals[g].text)({}, {}, ...Object.values(scope))),
      );
      const statements = template(literals[dyno.statements].text);
      result.push({
        name: `${spec.file.replace(/^examples\//, "")}#${i}`,
        inTypes,
        globals,
        statements: (inputs, outputs) =>
          statements(inputs, outputs, ...Object.values(scope)),
      });
    });
  }
  return result;
}

/** A uniform of `type` with a sample value, for an example dyno's input. */
// lofi's uniform arrays.
const ARRAYS: Record<string, number> = {
  tips: 10,
  spotlightOrigins: 2,
  spotlightDirs: 2,
  spotlightFovs: 2,
};

export function sampleInput(
  name: string,
  type: d.DynoType,
): d.DynoVal<d.DynoType> {
  const t = d.typeLiteral(type);
  const count = ARRAYS[name];
  if (count) {
    const value = Array.from({ length: count }, (_, i) =>
      t === "float"
        ? 30 + i
        : t === "vec3"
          ? new THREE.Vector3(i, 1, -i)
          : new THREE.Vector4(i, 0.5, 0, 0.2 * i),
    );
    return new d.DynoUniform({ type, count, value } as never) as never;
  }
  const h = [...name].reduce((a, c) => a + c.charCodeAt(0), 0);
  const x = 0.3 + (h % 7) * 0.1;
  switch (t) {
    case "float":
      return d.dynoFloat(x);
    case "int":
      return d.dynoInt(1 + (h % 3));
    case "bool":
      return d.dynoBool(h % 2 === 0);
    case "vec3":
      return d.dynoVec3(new THREE.Vector3(x, -x, 0.5));
    case "vec4":
      return d.dynoVec4(new THREE.Vector4(x, 0.2, -0.3, 1));
    case "mat4":
      return d.dynoMat4(new THREE.Matrix4().makeTranslation(x, 0, -1));
  }
  throw new Error(`no sample input for ${t}`);
}

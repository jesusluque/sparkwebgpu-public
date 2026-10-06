// The GLSL to WGSL translation of dyno code (src/dyno/wgsl/glslToWgsl.ts):
// what constructs become, and the errors for what it doesn't cover. The GPU
// suite (test/gpu/glslDyno.test.ts) compiles and runs translations.

import { describe, expect, it } from "vitest";
import * as d from "../../src/dyno";
import { WgslDynoProgram } from "../../src/dyno/wgsl";
import {
  type GlslInput,
  GlslTranslator,
  dynoTy,
} from "../../src/dyno/wgsl/glslToWgsl";

const vars = (
  types: Record<string, string | { type: string }>,
): Map<string, GlslInput & { output?: boolean }> =>
  new Map(
    Object.entries(types).map(([name, type]) => [
      name,
      {
        type: dynoTy(type as d.DynoType),
        expr: name.startsWith("out") ? `o_${name}` : `i_${name}`,
        output: name.startsWith("out"),
      },
    ]),
  );

function statements(
  code: string,
  types: Record<string, string | { type: string }> = {},
  globals?: string,
) {
  const t = new GlslTranslator();
  const g = globals ? t.translateGlobals(globals) : [];
  const s = t.translateStatements(code, vars(types), "o_ret");
  return { code: s.join("\n"), globals: [...t.takeHelpers(), ...g].join("\n") };
}

const stmt = (
  code: string,
  types: Record<string, string | { type: string }> = {},
) => statements(code, types).code;

describe("GLSL to WGSL", () => {
  it("declares typed variables and constants", () => {
    expect(stmt("float x = 1.; int i = 2, j; const vec2 K = vec2(.5);")).toBe(
      [
        "var x: f32 = 1.0;",
        "var i: i32 = 2;",
        "var j: i32;",
        "const K: vec2f = vec2f(0.5);",
      ].join("\n"),
    );
  });

  it("maps types and constructors, converting components", () => {
    expect(
      stmt("ivec3 a = ivec3(1); vec3 b = vec3(a); mat2 m = mat2(1.0);"),
    ).toBe(
      [
        "var a: vec3i = vec3i(1);",
        "var b: vec3f = vec3f(a);",
        "var m: mat2x2f = mat2x2f(1.0, 0.0, 0.0, 1.0);",
      ].join("\n"),
    );
    expect(stmt("int i; vec3 v = vec3(i, 0, 1.5);")).toContain(
      "vec3f(f32(i), 0, 1.5)",
    );
    expect(stmt("vec4 v; vec3 a = vec3(v); float f = float(v);")).toContain(
      "var a: vec3f = vec3f(v.xyz);\nvar f: f32 = v.x;",
    );
  });

  it("keeps literals abstract, so ints promote where GLSL code relies on it", () => {
    expect(stmt("float x = 1; x = x * 2;")).toBe("var x: f32 = 1;\nx = x * 2;");
    expect(stmt("int i; float x = i * 0.5;")).toContain("f32(i) * 0.5");
  });

  it("turns ?: into select", () => {
    expect(stmt("float a; float b = a > 0.0 ? a : -a;", {})).toContain(
      "select(-a, a, a > 0.0)",
    );
  });

  it("writes multi-component swizzles per component", () => {
    expect(stmt("vec3 p; p.xz *= 2.0;")).toBe(
      [
        "var p: vec3f;",
        "{",
        "    let swz_0: vec2f = p.xz * 2.0;",
        "    p.x = swz_0.x;",
        "    p.z = swz_0.y;",
        "}",
      ].join("\n"),
    );
    expect(stmt("vec4 c; c.rgba = vec4(1.0); c.a += 1.0;")).toBe(
      "var c: vec4f;\nc = vec4f(1.0);\nc.a += 1.0;",
    );
  });

  it("splats scalars where GLSL builtins take them for vectors", () => {
    const code = stmt(
      "vec3 v; v = clamp(v, 0.0, 1.0); v = step(0.5, v); v = smoothstep(0.0, 1.0, v); v = min(v, 2.0);",
    );
    expect(code).toContain("clamp(v, vec3f(0.0), vec3f(1.0))");
    expect(code).toContain("step(vec3f(0.5), v)");
    expect(code).toContain("glsl_smoothstep_vec3f(vec3f(0.0), vec3f(1.0), v)");
    expect(code).toContain("min(v, vec3f(2.0))");
  });

  it("maps builtins WGSL names differently", () => {
    const { code, globals } = statements(
      "float y, x; vec2 v; float a = atan(y, x) + inversesqrt(x) + mod(x, 2.0); vec2 m = mod(v, 3.0); uint u = floatBitsToUint(x);",
    );
    expect(code).toContain(
      "atan2(y, x) + inverseSqrt(x) + glsl_mod_f32(x, 2.0)",
    );
    expect(code).toContain("glsl_mod_vec2f(v, vec2f(3.0))");
    expect(code).toContain("bitcast<u32>(x)");
    expect(globals).toContain("fn glsl_mod_f32(x: f32, y: f32) -> f32");
  });

  it("spells out dot() as Metal's fma chain, so hashes round the same", () => {
    const { code, globals } = statements(
      "vec3 p; vec4 q; float x; float a = dot(p, vec3(127.1, 311.7, 74.7)) + dot(q, q) + dot(x, 2.0);",
    );
    expect(code).toContain(
      "glsl_dot_vec3f(p, vec3f(127.1, 311.7, 74.7)) + glsl_dot_vec4f(q, q) + x * 2.0",
    );
    expect(globals).toContain(
      "return fma(a.z, b.z, fma(a.y, b.y, a.x * b.x));",
    );
    expect(globals).toContain(
      "return fma(a.w, b.w, fma(a.z, b.z, fma(a.y, b.y, a.x * b.x)));",
    );
  });

  it("compares vectors as GLSL does", () => {
    expect(stmt("vec2 a, b; bool e = a == b; bool n = a != b;")).toContain(
      "var e: bool = all(a == b);\nvar n: bool = any(a != b);",
    );
  });

  it("makes unsigned shift amounts", () => {
    expect(stmt("int i, j; int k = i << j; uint u = 1u << 3;")).toContain(
      "var k: i32 = i << u32(j);\nvar u: u32 = 1u << 3;",
    );
  });

  it("parenthesizes where WGSL needs it", () => {
    expect(stmt("bool a, b, c; bool d = a && b || c;")).toContain(
      "(a && b) || c",
    );
    expect(stmt("int a, b, c; int d = a & b | c;")).toContain("(a & b) | c");
    expect(stmt("float a, b; float c = -(a + b) * -a;")).toContain(
      "-(a + b) * -a",
    );
  });

  it("translates control flow", () => {
    expect(
      stmt(
        "int s = 0; for (int i = 0; i < 4; ++i) { if (i == 1) continue; else if (i == 3) break; s += i; } while (s < 10) s *= 2; do { s--; } while (s > 5);",
      ),
    ).toBe(
      [
        "var s: i32 = 0;",
        "for (var i: i32 = 0; i < 4; i++) {",
        "    if (i == 1) {",
        "        continue;",
        "    } else if (i == 3) {",
        "        break;",
        "    }",
        "    s += i;",
        "}",
        "while (s < 10) {",
        "    s *= 2;",
        "}",
        "loop {",
        "    s--;",
        "    continuing {",
        "        break if !(s > 5);",
        "    }",
        "}",
      ].join("\n"),
    );
  });

  it("turns a comma for-update into a loop's continuing block", () => {
    expect(stmt("for (int i = 0, j = 4; i < j; i++, j--) {}")).toBe(
      [
        "{",
        "    var i: i32 = 0;",
        "    var j: i32 = 4;",
        "    loop {",
        "        if !(i < j) {",
        "            break;",
        "        }",
        "        continuing {",
        "            i++;",
        "            j--;",
        "        }",
        "    }",
        "}",
      ].join("\n"),
    );
  });

  it("returns the graph's output from a bare return", () => {
    expect(stmt("return;")).toBe("return o_ret;");
  });

  it("resolves dyno inputs and outputs, and Gsplat fields", () => {
    expect(
      stmt("out = in; out.center.y += 1.0; out.rgba.rgb *= in.rgba.a;", {
        in: { type: "Gsplat" },
        out: { type: "Gsplat" },
      }),
    ).toBe(
      [
        "o_out = i_in;",
        "o_out.center.y += 1.0;",
        "{",
        "    let swz_0: vec3f = o_out.rgba.rgb * i_in.rgba.a;",
        "    o_out.rgba.r = swz_0.x;",
        "    o_out.rgba.g = swz_0.y;",
        "    o_out.rgba.b = swz_0.z;",
        "}",
      ].join("\n"),
    );
  });

  it("translates functions: mutable parameters, out parameters, overloads", () => {
    const t = new GlslTranslator();
    const globals = t.translateGlobals(`
      float f(float x) { x *= 2.0; return x; }
      void g(out float a, inout vec2 b) { a = 1.0; b.y = a; }
      float f(vec2 v) { return v.x; }
    `);
    expect(globals).toEqual([
      "fn f(x_in: f32) -> f32 {\n    var x: f32 = x_in;\n    x *= 2.0;\n    return x;\n}",
      "fn g(a: ptr<function, f32>, b: ptr<function, vec2f>) {\n    (*a) = 1.0;\n    (*b).y = (*a);\n}",
      "fn f_vec2(v: vec2f) -> f32 {\n    return v.x;\n}",
    ]);
    expect(
      t.translateStatements(
        "float a; vec2 b; g(a, b); a = f(b) + f(a);",
        new Map(),
      ),
    ).toEqual([
      "var a: f32;",
      "var b: vec2f;",
      "g(&a, &b);",
      "a = f_vec2(b) + f(a);",
    ]);
  });

  it("renames names WGSL reserves", () => {
    expect(stmt("float target = 1.0; float select = target;")).toBe(
      "var target_: f32 = 1.0;\nvar select_: f32 = target_;",
    );
  });

  it("translates structs and arrays", () => {
    const t = new GlslTranslator();
    expect(
      t.translateGlobals(
        "struct S { vec3 p; float w[2]; }; const float K[2] = float[](1.0, 2.0);",
      ),
    ).toEqual([
      "struct S {\n    p: vec3f,\n    w: array<f32, 2>,\n}",
      "const K: array<f32, 2> = array<f32, 2>(1.0, 2.0);",
    ]);
    expect(
      t.translateStatements(
        "S s = S(vec3(0.0), K); s.w[1] = K.length();",
        new Map(),
      ),
    ).toEqual(["var s: S = S(vec3f(0.0), K);", "s.w[1] = 2;"]);
  });

  it("reuses a global's translation, and rejects conflicting redefinitions", () => {
    const t = new GlslTranslator();
    const g = "float h(float x) { return x; }";
    expect(t.translateGlobals(g)).toBe(t.translateGlobals(g));
    expect(() => t.translateGlobals("float h(float y) { return -y; }")).toThrow(
      /already defined/,
    );
  });
});

describe("GLSL to WGSL errors", () => {
  const cases: [string, string, RegExp][] = [
    [
      "derivatives",
      "float x;\nfloat y = dFdx(x);",
      /GLSL 2:11: dFdx\(\) is not available/,
    ],
    [
      "discard",
      "if (true) {\n  discard;\n}",
      /GLSL 2:3: discard is not available/,
    ],
    [
      "unknown functions",
      "float y = noise(1.0);",
      /GLSL 1:11: unknown function 'noise'/,
    ],
    ["unknown names", "float y = z;", /GLSL 1:11: unknown name 'z'/],
    [
      "assignments in expressions",
      "float a, b; a = (b = 1.0);",
      /GLSL 1:20: assignment inside an expression/,
    ],
    [
      "++ in expressions",
      "int i, j; j = i++;",
      /GLSL 1:16: '\+\+' inside an expression/,
    ],
    [
      "uniform declarations",
      "uniform float u;",
      /GLSL 1:1: 'uniform' declarations are not supported/,
    ],
    [
      "preprocessor conditionals",
      "#ifdef X\n#endif",
      /GLSL 1:1: preprocessor directive #ifdef/,
    ],
    [
      "switch fallthrough",
      "int i; switch (i) { case 0: i = 1; case 1: break; }",
      /fallthrough is not supported/,
    ],
    ["syntax errors", "float x = ;", /GLSL 1:11: unexpected ';'/],
    ["assigning inputs", "x = 1.0;", /can't assign to dyno input 'x'/],
    [
      "type mismatches",
      "vec3 v; vec2 w = v + vec2(1.0);",
      /'\+' on vec3 and vec2/,
    ],
    [
      "out arguments that aren't variables",
      "vec2 v; float f = modf(1.5, v.x);",
      /must be a local variable/,
    ],
  ];
  for (const [name, code, message] of cases) {
    it(`reports ${name} with the line and column`, () => {
      expect(() => stmt(code, { x: "float" })).toThrow(message);
    });
  }

  it("shows the source line", () => {
    expect(() => stmt("float a = 1.0;\nfloat b = a +;")).toThrow(
      /GLSL 2:14: unexpected ';'\n {4}float b = a \+;/,
    );
  });
});

describe("dyno GLSL in WgslDynoProgram", () => {
  it("translates plain dynos and names the dyno in errors", () => {
    const make = (statements: string) =>
      new WgslDynoProgram({
        functions: [
          {
            name: "f",
            graph: d.dynoBlock({ x: "float" }, { y: "float" }, ({ x }) => ({
              y: new d.Dyno({
                inTypes: { x: "float" },
                outTypes: { y: "float" },
                inputs: { x },
                statements: ({ inputs, outputs }) => [
                  statements
                    .replace("$x", inputs.x as string)
                    .replace("$y", outputs.y as string),
                ],
              }).outputs.y,
            })),
            inputs: { x: "float" },
            output: "y",
          },
        ],
      });
    expect(make("$y = mod($x, 2.0);").code).toContain(
      "out_y = glsl_mod_f32(in_x, 2.0);",
    );
    expect(() => make("$y = dFdy($x);")).toThrow(
      /Dyno dyno: GLSL to WGSL: GLSL 1:/,
    );
  });

  it("leaves dynos with a wgsl option alone", () => {
    const program = new WgslDynoProgram({
      functions: [
        {
          name: "f",
          graph: d.dynoBlock({}, { y: "float" }, () => ({
            y: new d.Dyno({
              inTypes: {},
              outTypes: { y: "float" },
              statements: ({ outputs }) => [`${outputs.y} = dFdx(1.0);`],
              wgsl: { statements: ({ outputs }) => [`${outputs.y} = 2.0;`] },
            }).outputs.y,
          })),
          inputs: {},
          output: "y",
        },
      ],
    });
    expect(program.code).toContain("out_y = 2.0;");
  });
});

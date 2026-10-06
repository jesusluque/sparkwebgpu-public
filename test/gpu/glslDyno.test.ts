// User dynos in GLSL, translated to WGSL (src/dyno/wgsl/glslToWgsl.ts):
// the examples' dynos as written compile with Dawn and naga, and snippets
// compute on the GPU what their GLSL means (computed in JS).

import * as THREE from "three";
import { describe, expect, it } from "vitest";
import * as d from "../../src/dyno";
import { WgslDynoProgram } from "../../src/dyno/wgsl";
import { device } from "./device";
import { bTex, graph, uTex } from "./dynoCases";
import { compileErrors, evalDyno } from "./dynoHarness";
import { exampleDynos, sampleInput } from "./glslExamples";
import { naga, nagaErrors } from "./naga";

describe("example GLSL dynos in WGSL", () => {
  for (const ex of exampleDynos()) {
    const program = () => {
      const graph = d.dynoBlock(
        { gsplat: d.Gsplat },
        { gsplat: d.Gsplat },
        ({ gsplat }) => {
          const dyno = new d.Dyno({
            inTypes: ex.inTypes,
            outTypes: { gsplat: d.Gsplat },
            globals: () => ex.globals,
            statements: ({ inputs, outputs }) =>
              d.unindentLines(ex.statements(inputs, outputs)),
          });
          const inputs: Record<string, d.DynoVal<d.DynoType>> = {};
          for (const [key, type] of Object.entries(ex.inTypes)) {
            inputs[key] =
              d.typeLiteral(type) === "Gsplat"
                ? (gsplat as d.DynoVal<d.DynoType>)
                : sampleInput(key, type);
          }
          return { gsplat: dyno.apply(inputs as never).gsplat };
        },
      );
      return new WgslDynoProgram({
        functions: [
          {
            name: "dyno_test",
            graph,
            inputs: { gsplat: d.Gsplat },
            output: "gsplat",
          },
        ],
      });
    };

    it(`${ex.name} translates`, () => {
      expect(program().code).toContain("fn dyno_test(");
    });

    it.skipIf(!device)(`${ex.name} compiles with Dawn`, async () => {
      const { code } = program();
      expect(await compileErrors(code), code).toEqual([]);
    });

    it.skipIf(!naga)(`${ex.name} validates with naga`, () => {
      const { code } = program();
      expect(nagaErrors(code), code).toBe("");
    });
  }
});

// A GLSL dyno computing `value`; $name in the code is the input or output.
function glsl(
  outType: d.DynoType,
  code: { globals?: string; statements: string },
  inputs: Record<string, d.DynoVal<d.DynoType>> = {},
) {
  const inTypes: Record<string, d.DynoType> = {};
  for (const [k, v] of Object.entries(inputs)) inTypes[k] = d.valType(v);
  return () =>
    new d.Dyno({
      inTypes,
      outTypes: { value: outType },
      inputs,
      globals: () => (code.globals ? [d.unindent(code.globals)] : []),
      statements: ({ inputs, outputs }) =>
        d.unindentLines(
          code.statements.replace(/\$(\w+)/g, (_, k: string) =>
            String(k === "value" ? outputs.value : inputs[k]),
          ),
        ),
    }).outputs.value as d.DynoVal<d.DynoType>;
}

const f1 = (v: number) => d.dynoFloat(v);
const i1 = (v: number) => d.dynoInt(v);
const rotXZ = (x: number, z: number, a: number) => {
  // GLSL v * m is a row vector times m: component j is dot(v, column j).
  const c = Math.cos(a);
  const s = Math.sin(a);
  return [x * c - z * s, x * s + z * c];
};
const [rx, rz] = rotXZ(1, 3, 0.5);
const smooth = (e0: number, e1: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
};

const parity: [string, d.DynoType, () => d.DynoVal<d.DynoType>, number[]][] = [
  [
    "mod floors like GLSL",
    "vec3",
    glsl(
      "vec3",
      {
        statements:
          "$value = vec3(mod(-1.5, 1.0), mod($a, 0.75), mod(7.0, -2.0));",
      },
      { a: f1(2) },
    ),
    [0.5, 0.5, -1],
  ],
  [
    "?:, int to float, compound assignment, ++, <<=",
    "vec2",
    glsl(
      "vec2",
      {
        statements: `
          int i = $n;
          float x = i > 2 ? 1.5 : float(i);
          x *= 2.0;
          x += float(i / 2);
          i++;
          i <<= 1;
          $value = vec2(x, i);
        `,
      },
      { n: i1(3) },
    ),
    [4, 8],
  ],
  [
    "swizzle assignment, vec2 * mat2",
    "vec3",
    glsl(
      "vec3",
      {
        globals: `
          mat2 rot(float a) {
            float s = sin(a), c = cos(a);
            return mat2(c, -s, s, c);
          }
        `,
        statements: `
          vec3 p = vec3(1.0, 2.0, 3.0);
          p.xz *= rot($a);
          p.yx += vec2(10.0, 20.0);
          $value = p;
        `,
      },
      { a: f1(0.5) },
    ),
    [rx + 20, 12, rz],
  ],
  [
    "out and inout parameters, modf",
    "vec3",
    glsl("vec3", {
      globals: `
        void addTo(inout float a, float b) { a += b; }
        float split(float x, out float whole) { return modf(x, whole); }
      `,
      statements: `
        float w;
        float f = split(2.75, w);
        float acc = 1.0;
        addTo(acc, w);
        $value = vec3(f, w, acc);
      `,
    }),
    [0.75, 2, 3],
  ],
  [
    "for, while, do-while, switch, break, continue",
    "ivec4",
    glsl("ivec4", {
      statements: `
        int sum = 0;
        for (int i = 0; i < 10; i++) {
          if (i % 2 == 1) continue;
          if (i > 7) break;
          sum += i;
        }
        int j = 0;
        while (j < 5) { j += 2; }
        int k = 0;
        do { k++; } while (k < 3);
        int s = 0;
        switch (k) {
          case 1:
          case 2:
            s = 10;
            break;
          case 3: {
            s = 30;
            break;
          }
          default:
            s = -1;
        }
        $value = ivec4(sum, j, k, s);
      `,
    }),
    [12, 6, 3, 30],
  ],
  [
    "structs, arrays, const arrays indexed at run time",
    "vec4",
    glsl(
      "vec4",
      {
        globals: `
          struct Pair { vec2 a; float b[2]; };
          const float W[3] = float[3](0.25, 0.5, 2.0);
          Pair makePair(float x) {
            Pair p = Pair(vec2(x, 2.0 * x), float[2](3.0, 4.0));
            p.b[1] += p.a.y;
            return p;
          }
        `,
        statements: `
          Pair p = makePair(1.0);
          int idx = $n - 1;
          $value = vec4(p.a, p.b[1], W[idx]);
        `,
      },
      { n: i1(3) },
    ),
    [1, 2, 6, 2],
  ],
  [
    "atan(y, x), inversesqrt, smoothstep, mix(bool), step, clamp",
    "vec4",
    glsl("vec4", {
      statements: `
        vec3 v = vec3(0.2, 0.5, 0.9);
        $value = vec4(
          atan(1.0, -1.0),
          inversesqrt(4.0),
          smoothstep(0.0, 1.0, v * 0.8).y,
          mix(1.0, 2.0, true) + step(0.5, v).x + clamp(v, 0.3, 0.6).x
        );
      `,
    }),
    [Math.atan2(1, -1), 0.5, smooth(0, 1, 0.4), 2.3],
  ],
  [
    "bit casts and bit counts",
    "uvec4",
    glsl("uvec4", {
      statements: `
        $value = uvec4(
          floatBitsToUint(1.0),
          uint(bitCount(7u)),
          uint(findMSB(256)),
          uint(uintBitsToFloat(0x40000000u))
        );
      `,
    }),
    [0x3f800000, 3, 8, 2],
  ],
  [
    "uniforms: float, bool, arrays",
    "vec4",
    glsl(
      "vec4",
      {
        statements: `
          float s = 0.0;
          for (int i = 0; i < 3; i++) s += $arr[i];
          $value = vec4(s, $f, $flag ? 1.0 : 0.0, $v[1].y);
        `,
      },
      {
        f: f1(1.25),
        flag: d.dynoBool(true),
        arr: new d.DynoUniform({ type: "float", count: 3, value: [1, 2, 4] }),
        v: new d.DynoUniform({
          type: "vec3",
          count: 2,
          value: [new THREE.Vector3(1, 2, 3), new THREE.Vector3(4, 5, 6)],
        }),
      } as never,
    ),
    [7, 1.25, 1, 5],
  ],
  [
    "vector comparisons",
    "ivec4",
    glsl("ivec4", {
      statements: `
        bvec3 lt = lessThan(vec3(1, 2, 3), vec3(2, 2, 2));
        bool eq = vec2(1.0, 2.0) == vec2(1.0, 2.0);
        bool ne = ivec2(1, 2) != ivec2(1, 2);
        $value = ivec4(int(any(lt)), int(all(lt)), int(eq), int(ne || all(not(lt.yz))));
      `,
    }),
    [1, 0, 1, 1],
  ],
  [
    "return in dyno code",
    "float",
    glsl(
      "float",
      {
        statements: `
          $value = 1.0;
          if ($f > 0.0) {
            $value = 2.0;
            return;
          }
          $value = 3.0;
        `,
      },
      { f: f1(1) },
    ),
    [2],
  ],
  [
    "matrices: resize, matrix + scalar, inverse, determinant",
    "vec4",
    glsl("vec4", {
      statements: `
        mat4 m4 = mat4(2.0);
        mat3 m = mat3(m4);
        m = m + 1.0;
        vec3 r = m * vec3(1.0, 0.0, 0.0);
        float det = determinant(inverse(mat2(2.0, 0.0, 0.0, 4.0)));
        $value = vec4(r, det);
      `,
    }),
    [3, 1, 1, 0.125],
  ],
  [
    "#define, private globals, overloads",
    "vec3",
    glsl("vec3", {
      globals: `
        #define SCALE 2.0
        float g = 1.0;
        float f(float x) { return x * SCALE; }
        float f(vec2 v) { g += 1.0; return v.x + v.y; }
      `,
      statements: `
        float a = f(1.5);
        float b = f(vec2(1.0, 2.0));
        $value = vec3(a, b, g);
      `,
    }),
    [3, 3, 2],
  ],
  [
    "swizzle compound assignment on an output",
    "vec4",
    glsl("vec4", {
      statements: `
        $value = vec4(1.0);
        $value.rgb *= 0.5;
        $value.a -= 0.25;
        $value.zx = vec2(2, 3);
      `,
    }),
    [3, 0.5, 2, 0.75],
  ],
  [
    "textures: texelFetch, texture, textureSize",
    "vec4",
    glsl(
      "vec4",
      {
        statements: `
          uvec4 t = texelFetch($u, ivec2(1, 2), 0);
          vec4 s = texture($b, vec2(0.5, 0.5));
          ivec2 size = textureSize($u, 0);
          $value = vec4(float(t.x), s.r, float(size.x), float(size.y));
        `,
      },
      { u: d.dynoUsampler2D(uTex), b: d.dynoSampler2D(bTex) } as never,
    ),
    [108, 0.5, 4, 3],
  ],
];

describe.skipIf(!device)("GLSL dynos compute what GLSL means", () => {
  for (const [name, type, f, expected] of parity) {
    it(name, async () => {
      const actual = await evalDyno(graph(type, f as never));
      expect(actual.length).toBe(expected.length);
      actual.forEach((a, i) => {
        expect(
          Math.abs(a - expected[i]),
          `component ${i}: ${a} vs ${expected[i]}`,
        ).toBeLessThan(1e-4 * Math.max(1, Math.abs(expected[i])));
      });
    });
  }
});

describe.skipIf(!naga)("GLSL dyno WGSL validates with naga", () => {
  for (const [name, type, f] of parity) {
    it(name, () => {
      const program = new WgslDynoProgram({
        functions: [
          {
            name: "dyno_test",
            graph: graph(type, f as never),
            inputs: { index: "int" },
            output: "value",
          },
        ],
      });
      expect(nagaErrors(program.code), program.code).toBe("");
    });
  }
});

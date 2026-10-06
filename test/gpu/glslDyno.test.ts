// User dynos in GLSL, translated to WGSL (src/dyno/wgsl/glslToWgsl.ts):
// the examples' dynos as written compile with Dawn and naga, and snippets
// compute on the GPU what their GLSL means (computed in JS).

import { describe, expect, it } from "vitest";
import * as d from "../../src/dyno";
import { WgslDynoProgram } from "../../src/dyno/wgsl";
import { device } from "./device";
import { compileErrors } from "./dynoHarness";
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

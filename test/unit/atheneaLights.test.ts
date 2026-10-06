// The light sidecar's .usda half (src/webgpu/athenea/usda.ts,
// lightSidecar.ts): 067's draft for the Corvette read as written, and a
// hand-written sidecar with every curve, rule and sequence of 062 §7 / 066,
// evaluated into w_k(t).

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  LightRig,
  blackbody,
  parseLightSidecar,
} from "../../src/webgpu/athenea/lightSidecar";
import { parseUsda, walkPrims } from "../../src/webgpu/athenea/usda";

const CORVETTE = readFileSync(
  new URL(
    "../fixtures/athenea-lights/corvette-067.lights.usda",
    import.meta.url,
  ),
  "utf8",
);

const SHOW = `#usda 1.0
(
    doc = """A sidecar with every behaviour:
    curves, blinkers, rules and a sequence."""
    customLayerData = { string source = "hand-written" }
)

def Scope "Lights" (
    prepend apiSchemas = ["AtheneaLightSidecarAPI"]
)
{
    string athenea:lightSidecar:cloudHash = "fnv1a64:0123456789abcdef"
    float athenea:lightSidecar:nitsPerUnit = 100
    asset athenea:lightSidecar:athl = @./show.lights.athl@
    token athenea:lightSidecar:defaultState = "aparcado"

    def Scope "LightGroups"
    {
        def Scope "cruce" (prepend apiSchemas = ["AtheneaLightGroupAPI"])
        {
            token athenea:lightGroup:technology = "xenon"
            float athenea:lightGroup:radiance = 200
            float athenea:lightGroup:temperatureK = 6000
            float athenea:lightGroup:riseSeconds = 4
            int2[] athenea:lightGroup:splatRanges = [(10, 20), (40, 41)]
            asset shaping:ies:file = @./cruce.ies@
        }
        def Scope "halogena"
        {
            token athenea:lightGroup:technology = "halogen"
            float athenea:lightGroup:temperatureK = 3000
            float athenea:lightGroup:riseSeconds = 0.1
        }
        def Scope "drl_izq"
        {
            token athenea:lightGroup:side = "left"
            token athenea:lightGroup:function = "daytimeRunning"
            float athenea:lightGroup:riseSeconds = 0
            float athenea:lightGroup:fallSeconds = 0
        }
        def Scope "intermitente_izq"
        {
            token athenea:lightGroup:function = "indicator"
            color3f athenea:lightGroup:color = (1, 0.5, 0)
            float athenea:lightGroup:riseSeconds = 0
            float athenea:lightGroup:fallSeconds = 0
        }
        def Scope "repetidor_izq"
        {
            token athenea:lightGroup:technology = "blinker"
            float athenea:lightGroup:riseSeconds = 0
            float athenea:lightGroup:fallSeconds = 0
            color3f athenea:lightGroup:lensTint = (1, 0.4, 0.05)
        }
        def Scope "segmento_2"
        {
            float athenea:lightGroup:delaySeconds = 0.25
            float athenea:lightGroup:riseSeconds = 0
        }
    }

    def Scope "LightStates"
    {
        def Scope "aparcado" { dictionary athenea:lightState:targets = {} }
        def Scope "diurno"   { dictionary athenea:lightState:targets = { double drl_izq = 1 } }
        def Scope "intermitente" {
            token athenea:lightState:base = "diurno"
            dictionary athenea:lightState:targets = { double intermitente_izq = 1  double repetidor_izq = 1 }
        }
        def Scope "noche" { dictionary athenea:lightState:targets = { double cruce = 1  double halogena = 1  double drl_izq = 1  double segmento_2 = 1 } }
    }

    def Scope "LightRules"
    {
        def Scope "drl_off_with_low_beam" {
            token athenea:lightRule:when = "cruce"
            token athenea:lightRule:target = "drl_izq"
            double athenea:lightRule:scale = 0
        }
        def Scope "drl_dim_with_indicator" {
            token athenea:lightRule:when = "intermitente_izq && !cruce"
            token athenea:lightRule:target = "drl_izq"
            double athenea:lightRule:scale = 0.25
        }
        def Scope "repeaters_in_phase" {
            token athenea:lightRule:when = "intermitente_izq"
            token[] athenea:lightRule:target = ["repetidor_izq"]
            token athenea:lightRule:mode = "sync"
        }
    }

    def Scope "Sequences"
    {
        def Scope "bienvenida" {
            double[] athenea:lightSequence:times = [0, 1, 2]
            token[] athenea:lightSequence:states = ["aparcado", "diurno", "noche"]
            double athenea:lightSequence:level:halogena.timeSamples = { 0: 0, 0.5: 1, 1: 0.5, }
            bool athenea:lightSequence:loop = true
            double athenea:lightSequence:duration = 3
        }
    }
}
`;

describe("usda reader", () => {
  it("reads 067's draft sidecar as written", () => {
    const layer = parseUsda(CORVETTE);
    expect(layer.version).toBe("1.0");
    expect(layer.metadata.subLayers).toEqual([{ asset: "./corvette_c7.usda" }]);
    const prims = [...walkPrims(layer.prims)];
    const light = prims.find((p) => p.path === "/Corvette/Looks/Light/OpenPBR");
    expect(light?.specifier).toBe("over");
    expect(light?.properties.get("inputs:emission_luminance")?.value).toBe(
      1000,
    );
    const rear = prims.find(
      (p) => p.path === "/Corvette/Looks/Light_Rear_Red/OpenPBR",
    );
    expect(rear?.properties.get("inputs:emission_color")).toMatchObject({
      type: "color3f",
      value: [0.381, 0.003, 0],
    });
    const pilotos = prims.find((p) => p.name === "pilotos");
    expect(
      pilotos?.properties.get("athenea:lightGroup:members")?.value,
    ).toEqual([
      { path: "/Corvette/Rear_Light_Relfectors" },
      { path: "/Corvette/Rear_Light_Relfectors_001" },
    ]);
  });

  it("reads metadata, list ops, triple-quoted strings, dictionaries and time samples", () => {
    const layer = parseUsda(SHOW);
    expect(layer.metadata.doc).toContain("curves, blinkers");
    expect(layer.metadata.customLayerData).toEqual({ source: "hand-written" });
    const prims = [...walkPrims(layer.prims)];
    expect(prims[0].metadata.apiSchemas).toEqual(["AtheneaLightSidecarAPI"]);
    const seq = prims.find((p) => p.name === "bienvenida");
    expect(
      seq?.properties.get("athenea:lightSequence:level:halogena")?.timeSamples,
    ).toEqual([
      [0, 0],
      [0.5, 1],
      [1, 0.5],
    ]);
  });

  it("says where it fails", () => {
    expect(() =>
      parseUsda('#usda 1.0\ndef Scope "a" {\n  float x = \n}'),
    ).toThrow(/line 4/);
    expect(() => parseUsda("not usd")).toThrow(/#usda/);
  });
});

describe("light sidecar (066/067)", () => {
  const corvette = parseLightSidecar(CORVETTE);

  it("finds 067's groups, states and rules", () => {
    expect(corvette.groups.map((g) => g.name)).toEqual([
      "cruce",
      "largas",
      "drl",
      "pilotos",
      "tubo_trasero",
    ]);
    const cruce = corvette.groups[0];
    expect(cruce).toMatchObject({
      function: "lowBeam",
      technology: "led",
      riseSeconds: 0.15,
      temperatureK: 5500,
      members: ["/Corvette/Main_Lens_Reflecotr"],
    });
    expect(corvette.groups[4].technology).toBe("lightGuide");
    expect(corvette.states.map((s) => s.name)).toEqual([
      "aparcado",
      "diurno",
      "noche_ciudad",
      "noche_carretera",
      "frenando",
    ]);
    expect(corvette.states[4].targets).toEqual({ pilotos: 4, tubo_trasero: 1 });
    expect(corvette.rules.map((r) => [r.when, r.targets, r.scale])).toEqual([
      ["cruce", ["drl"], 0],
      ["!cruce", ["largas"], 0],
    ]);
    expect(corvette.warnings).toEqual([]);
  });

  it("evaluates states, rules and the LED ramp", () => {
    const rig = new LightRig(corvette, 0);
    let w = rig.evaluate(0);
    expect([...w.targets]).toEqual([0, 0, 0, 0, 0]);
    rig.setState("diurno", 1);
    // drl has no rise: on at once (riseSeconds defaults to the LED's 0.05).
    w = rig.evaluate(1.025);
    expect(w.weights[4 * 2 + 3]).toBeCloseTo(0.5, 5);
    w = rig.evaluate(2);
    expect(w.weights[4 * 2 + 3]).toBe(1);
    // Low beam on: the rule puts the DRL out, the beam ramps over 0.15 s.
    rig.setState("noche_ciudad", 3);
    rig.setLevel("drl", 1, 3);
    w = rig.evaluate(3.075);
    expect(w.targets[2]).toBe(0);
    expect(w.weights[3]).toBeCloseTo(0.5, 5);
    // The colour is the blackbody of 5500 K at luminance 1.
    const bb = blackbody(5500);
    expect(w.weights[0] / w.weights[3]).toBeCloseTo(bb[0], 5);
    expect(0.2126 * bb[0] + 0.7152 * bb[1] + 0.0722 * bb[2]).toBeCloseTo(1, 5);
    expect(bb[2]).toBeGreaterThan(bb[0] * 0.7); // near daylight, not tungsten
    // High beam without low beam is held off.
    rig.setState("aparcado", 4);
    rig.setLevel("drl", null, 4);
    rig.setLevel("largas", 1, 4);
    expect(rig.evaluate(5).targets[1]).toBe(0);
    // Braking: the tail lamps at 4x.
    rig.clearLevels(6);
    rig.setState("frenando", 6);
    w = rig.evaluate(7);
    expect(w.weights[4 * 3 + 3]).toBe(4);
    expect(w.weights[4 * 3]).toBe(4); // white, 1 nit a unit
  });

  const show = parseLightSidecar(SHOW);

  it("reads the sidecar's own attributes and the rest of the schema", () => {
    expect(show.warnings).toEqual([]);
    expect(show).toMatchObject({
      cloudHash: "fnv1a64:0123456789abcdef",
      nitsPerUnit: 100,
      athl: "./show.lights.athl",
      defaultState: "aparcado",
    });
    expect(show.groups[0]).toMatchObject({
      technology: "xenon",
      splatRanges: [
        [10, 20],
        [40, 41],
      ],
      iesFile: "./cruce.ies",
    });
    // An indicator is a blinker by function.
    expect(show.groups[3].technology).toBe("blinker");
    // A state on another: diurno's targets, then its own.
    expect(show.states[2].targets).toEqual({
      drl_izq: 1,
      intermitente_izq: 1,
      repetidor_izq: 1,
    });
    expect(show.rules[2].mode).toBe("sync");
    expect(show.sequences[0]).toMatchObject({ loop: true, duration: 3 });
  });

  it("blinks at 90 a minute, repeaters in phase, DRL dimmed on that side", () => {
    const rig = new LightRig(show, 0);
    rig.setState("diurno", 0);
    rig.evaluate(0.5);
    rig.setState("intermitente", 1);
    const level = (t: number, k: number) => rig.evaluate(t).weights[4 * k + 3];
    // 1.5 Hz, duty 0.5: on for 1/3 s from the command, then off.
    expect(level(1.1, 3)).toBe(1);
    expect(level(1.1, 4)).toBe(1);
    expect(level(1.4, 3)).toBe(0);
    expect(level(1.4, 4)).toBe(0);
    expect(level(1.7, 3)).toBe(1);
    // Amber, nits / nitsPerUnit.
    const w = rig.evaluate(1.7).weights;
    expect(w[12]).toBeCloseTo(0.01, 7);
    expect(w[13]).toBeCloseTo(0.005, 7);
    expect(w[14]).toBe(0);
    // The DRL on the blinking side at a quarter.
    expect(level(1.8, 2)).toBe(0.25);
    expect(rig.animating(1.8)).toBe(true);
  });

  it("warms a xenon lamp from cold to white and a halogen through red", () => {
    const rig = new LightRig(show, 0);
    rig.setState("noche", 10);
    let w = rig.evaluate(10.001);
    // The arc strikes at 30 % of the flux, at about its start temperature.
    expect(w.weights[3]).toBeCloseTo(0.3, 2);
    const cold = w.weights[2] / w.weights[0];
    w = rig.evaluate(14);
    expect(w.weights[3]).toBe(1);
    const warm = w.weights[2] / w.weights[0];
    expect(warm).toBeGreaterThan(cold); // bluer when warm (4300 K -> 6000 K)
    // radiance 200 / nitsPerUnit 100, luminance-1 colour.
    expect(
      0.2126 * w.weights[0] + 0.7152 * w.weights[1] + 0.0722 * w.weights[2],
    ).toBeCloseTo(2, 4);
    // Halogen: first order, tau 0.1 s.
    const rig2 = new LightRig(show, 0);
    rig2.setState("noche", 0);
    const h = rig2.evaluate(0.1).weights;
    expect(h[4 * 1 + 3]).toBeCloseTo(1 - Math.exp(-1), 5);
    expect(h[4 * 1] / h[4 * 1 + 2]).toBeGreaterThan(
      blackbody(3000)[0] / blackbody(3000)[2],
    ); // redder while it heats
    // The segment answers a quarter second late.
    expect(rig2.evaluate(0.2).weights[4 * 5 + 3]).toBe(0);
    expect(rig2.evaluate(0.3).weights[4 * 5 + 3]).toBe(1);
    // The low beam's rule turned the DRL out.
    expect(rig2.evaluate(0.3).targets[2]).toBe(0);
  });

  it("plays a looping sequence with a keyframed curve", () => {
    const rig = new LightRig(show, 0);
    rig.play("bienvenida", 100);
    const at = (t: number) => rig.evaluate(t);
    expect(rig.currentState).toBe("aparcado");
    expect(at(100.25).targets[1]).toBeCloseTo(0.5, 6); // the curve, not the state
    expect(at(101.2).targets[2]).toBe(1);
    expect(rig.currentState).toBe("diurno");
    expect(at(102.5).targets[0]).toBe(1);
    expect(rig.currentState).toBe("noche");
    // Looped: back to the start at 3 s.
    expect(at(103.5).targets[0]).toBe(0);
    rig.stop(103.6);
    expect(rig.playing).toBe(null);
    expect(rig.currentState).toBe("aparcado");
  });
});

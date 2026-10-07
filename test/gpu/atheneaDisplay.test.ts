// The athenea output stage (AtheneaDisplay, slang/athenea_adapter/
// display.slang) against athenea's own `displayColour` (the vendored
// technique/display.slang, run by slang/tests/athenea_display.slang), and
// against what athenea's tests hold its display transform to: Standard and
// AgX as their formulas (written again here in JS), and ACES 2.0's 18% grey
// at 0.0999993 of reference white at 100 nits (tests/technique/
// test_display.cpp).

import { describe, expect, it } from "vitest";
import {
  AtheneaDisplay,
  prepareAces2,
} from "../../src/webgpu/athenea/AtheneaDisplay";
import { type FxBuffer, FxGraph, Gpu, rect } from "../../src/webgpu/fx";
import refModule from "../../src/webgpu/generated/tests/athenea_display";
import { UniformWriter } from "../../src/webgpu/uniforms";
import { device, readBack, storage } from "./device";

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

/** n premultiplied linear colours from 2^-10 to 2^6, some uncovered. */
function hdrColours(n: number, seed = 7): Float32Array {
  const next = rng(seed);
  const out = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    const a = i % 4 === 0 ? 1 : next();
    const level = 2 ** (-10 + 16 * next());
    for (let c = 0; c < 3; c++) out[i * 4 + c] = level * next() * a;
    out[i * 4 + 3] = a;
  }
  // Neutrals across the range, opaque.
  for (let i = 0; i < Math.min(n, 32); i++) {
    const v = 2 ** (-10 + i / 2);
    out.set([v, v, v, 1], i * 4);
  }
  return out;
}

const srgbEncode = (x: number) =>
  x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055;
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

// display.slang's agx, from Wrensch's "Minimal AgX" (row vectors: v * M).
const INSET = [
  [0.842479062253094, 0.0423282422610123, 0.0423756549057051],
  [0.0784335999999992, 0.878468636469772, 0.0784336],
  [0.0792237451477643, 0.0791661274605434, 0.879142973793104],
];
const OUTSET = [
  [1.19687900512017, -0.0528968517574562, -0.0529716355144438],
  [-0.0980208811401368, 1.15190312990417, -0.0980434501171241],
  [-0.0990297440797205, -0.0989611768448433, 1.15107367264116],
];
const rowTimes = (v: number[], m: number[][]) =>
  [0, 1, 2].map((j) => v[0] * m[0][j] + v[1] * m[1][j] + v[2] * m[2][j]);
function agx(rgb: number[]): number[] {
  const minEv = -12.47393;
  const maxEv = 4.026069;
  let c = rowTimes(
    rgb.map((x) => Math.max(x, 0)),
    INSET,
  );
  c = c.map(
    (x) =>
      (Math.min(maxEv, Math.max(minEv, Math.log2(Math.max(x, 1e-10)))) -
        minEv) /
      (maxEv - minEv),
  );
  const y = c.map(
    (x) =>
      ((((15.5 * x - 40.14) * x + 31.96) * x - 6.868) * x + 0.4298) * x * x +
      0.1191 * x -
      0.00232,
  );
  return rowTimes(y, OUTSET).map((x) => Math.max(x, 0) ** 2.2);
}

describe.skipIf(!device)("athenea display transform", () => {
  const gpu = Gpu.forDevice(device as GPUDevice);
  const graph = new FxGraph(gpu);
  const n = 509;
  const colours = hdrColours(n);
  const frame = rect(0, 0, n, 1);
  const background = [0.05, 0.1, 0.2];

  function picture(data: Float32Array): FxBuffer {
    return {
      buffer: storage(data),
      width: data.length / 4,
      height: 1,
      stride: data.length / 4,
      rect: rect(0, 0, data.length / 4, 1),
    };
  }

  async function ours(params: Record<string, number | number[]>) {
    const out = graph.render(
      {
        effect: new AtheneaDisplay(),
        params,
        inputs: { Source: { picture: picture(colours) } },
      },
      frame,
    );
    gpu.flush();
    const p = out.picture as FxBuffer;
    return new Float32Array(await readBack(p.buffer)).slice(0, n * 4);
  }

  async function reference(
    view: number,
    display: number,
    exposure: number,
    peak = 100,
  ) {
    const aces = prepareAces2(gpu, peak, display >= 2 ? "p3d65" : "rec709");
    gpu.flush();
    const out = storage(n * 16);
    const dp = UniformWriter.for(refModule, "params").setAll({
      peakScale: peak / 100,
    });
    const rp = UniformWriter.for(refModule, "refParams").setAll({
      count: n,
      view,
      display,
      exposure,
      backgroundR: background[0],
      backgroundG: background[1],
      backgroundB: background[2],
    });
    gpu.load(refModule, "displayReference").run({
      grid: [n],
      buffers: {
        refSrc: storage(colours),
        refOut: out,
        acesParams: aces.params,
        acesTables: aces.tables,
      },
      uniforms: { params: dp.data, refParams: rp.data },
    });
    return new Float32Array(await readBack(out));
  }

  const worst = (a: Float32Array, b: Float32Array | number[][]) => {
    let w = 0;
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < 3; c++) {
        const want = Array.isArray(b) ? b[i][c] : b[i * 4 + c];
        const d = Math.abs(a[i * 4 + c] - want);
        if (!(d <= w)) w = Number.isNaN(d) ? Number.POSITIVE_INFINITY : d;
      }
    }
    return w;
  };

  // athenea's test_display.cpp cases, and ACES 2.0 on each display.
  const cases: [string, number, number, number][] = [
    ["Standard, sRGB", 0, 0, 0],
    ["Standard, Rec.709, +1.5", 0, 1, 1.5],
    ["Standard, P3, -2", 0, 2, -2],
    ["AgX, sRGB", 1, 0, 0],
    ["AgX, P3, +3", 1, 2, 3],
    ["AgX, Rec.709, -4", 1, 1, -4],
    ["ACES 2.0, sRGB", 2, 0, 0],
    ["ACES 2.0, P3, +1", 2, 2, 1],
    ["ACES 2.0, Rec.709, -2", 2, 1, -2],
  ];
  for (const [name, view, display, exposure] of cases) {
    it(`is athenea's displayColour: ${name}`, async () => {
      const got = await ours({
        view,
        display,
        exposure,
        background,
        encoded: 1,
      });
      const want = await reference(view, display, exposure);
      // ACES reads its parameters through a local copy (Safari's Metal
      // translation rejects them indexed in place), which rounds a few
      // operations differently: under 3e-6, a thousandth of an 8-bit step.
      expect(worst(got, want)).toBeLessThan(view === 2 ? 5e-6 : 2e-6);
    });
  }

  it("is the formulas: Standard and AgX onto sRGB (JS)", async () => {
    for (const view of [0, 1]) {
      const got = await ours({ view, display: 0, exposure: 0.5, background });
      const want: number[][] = [];
      for (let i = 0; i < n; i++) {
        const a = Math.min(1, Math.max(0, colours[i * 4 + 3]));
        let c = [0, 1, 2].map(
          (k) => (colours[i * 4 + k] + background[k] * (1 - a)) * 2 ** 0.5,
        );
        if (view === 1) c = agx(c);
        // forSrgbOutput: the code value with the sRGB curve undone, which
        // for an sRGB display is the clamped display-linear colour.
        want.push(c.map(clamp01));
      }
      // Display-linear values: 2e-5 is athenea's own tolerance on code values.
      expect(worst(got, want)).toBeLessThan(2e-5);
    }
  });

  it("ACES 2.0 shows 18% grey at 0.0999993 of reference white (100 nits)", async () => {
    const grey = new Float32Array([0.18, 0.18, 0.18, 1]);
    const out = graph.render(
      {
        effect: new AtheneaDisplay(),
        params: { view: 2, display: 0 },
        inputs: { Source: { picture: picture(grey) } },
      },
      rect(0, 0, 1, 1),
    );
    gpu.flush();
    const shown = new Float32Array(
      await readBack((out.picture as FxBuffer).buffer),
    );
    for (let c = 0; c < 3; c++) {
      expect(Math.abs(shown[c] - 0.0999993)).toBeLessThan(2e-4);
    }
    // And the code value an sRGB display gets for it.
    expect(srgbEncode(shown[0])).toBeCloseTo(srgbEncode(0.0999993), 3);
  });
});

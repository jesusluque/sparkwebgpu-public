// The aopenfx example effects (src/webgpu/fx) against CPU references on
// small pictures, run through the same FxGraph host the chain uses.

import { describe, expect, it } from "vitest";
import {
  Bloom,
  Blur,
  CheckerBoard,
  Constant,
  CornerPin,
  Crop,
  type Effect,
  type FxBuffer,
  FxChain,
  FxGraph,
  type FxNode,
  Gpu,
  Grade,
  ImageStatistics,
  Invert,
  Merge,
  type ParamValues,
  type Rect,
  Transform,
  rect,
} from "../../src/webgpu/fx";
import { pinMatrix, rectCorners } from "../../src/webgpu/fx/effects/CornerPin";
import invertModule from "../../src/webgpu/generated/fx/invert";
import { UniformWriter } from "../../src/webgpu/uniforms";
import { device, readBack, storage } from "./device";

/** A CPU picture: float RGBA rows, y up, placed at `rect`. */
interface Pic {
  rect: Rect;
  w: number;
  h: number;
  data: Float32Array;
}

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

/** Premultiplied random pixels, alpha in [0.2, 1]. */
function randomPic(r: Rect, seed = 1): Pic {
  const w = r.x2 - r.x1;
  const h = r.y2 - r.y1;
  const next = rng(seed);
  const data = new Float32Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const a = 0.2 + 0.8 * next();
    for (let c = 0; c < 3; c++) data[i * 4 + c] = next() * a;
    data[i * 4 + 3] = a;
  }
  return { rect: r, w, h, data };
}

function at(p: Pic, x: number, y: number): number[] {
  const px = x - p.rect.x1;
  const py = y - p.rect.y1;
  if (px < 0 || py < 0 || px >= p.w || py >= p.h) return [0, 0, 0, 0];
  const i = (py * p.w + px) * 4;
  return Array.from(p.data.subarray(i, i + 4));
}

function picFrom(r: Rect, f: (x: number, y: number) => number[]): Pic {
  const w = r.x2 - r.x1;
  const h = r.y2 - r.y1;
  const data = new Float32Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++)
      data.set(f(r.x1 + x, r.y1 + y), (y * w + x) * 4);
  }
  return { rect: r, w, h, data };
}

function upload(p: Pic): FxBuffer {
  return {
    buffer: storage(p.data),
    width: p.w,
    height: p.h,
    stride: p.w,
    rect: p.rect,
  };
}

async function download(b: FxBuffer): Promise<Pic> {
  const all = new Float32Array(await readBack(b.buffer));
  const data = new Float32Array(b.width * b.height * 4);
  for (let y = 0; y < b.height; y++) {
    data.set(
      all.subarray(y * b.stride * 4, (y * b.stride + b.width) * 4),
      y * b.width * 4,
    );
  }
  return { rect: b.rect, w: b.width, h: b.height, data };
}

function maxError(got: Pic, want: Pic) {
  expect(got.rect).toEqual(want.rect);
  let worst = 0;
  for (let i = 0; i < want.data.length; i++) {
    const d = Math.abs(got.data[i] - want.data[i]);
    if (!(d <= worst)) worst = Number.isNaN(d) ? Number.POSITIVE_INFINITY : d;
  }
  return worst;
}

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

describe.skipIf(!device)("aofx effects", () => {
  const gpu = Gpu.forDevice(device as GPUDevice);
  const graph = new FxGraph(gpu);

  async function render(
    effect: Effect,
    params: ParamValues,
    inputs: Record<string, Pic | FxNode | null>,
    window: Rect,
    node: Partial<FxNode> = {},
    project?: [number, number],
  ) {
    const wired: FxNode["inputs"] = {};
    for (const [k, v] of Object.entries(inputs)) {
      wired[k] = v && "data" in v ? { picture: upload(v) } : v;
    }
    const out = graph.render(
      { effect, params, inputs: wired, ...node },
      window,
      project ? { projectWidth: project[0], projectHeight: project[1] } : {},
    );
    gpu.flush();
    return {
      out,
      pic: out.picture ? await download(out.picture) : null,
    };
  }

  const frame = rect(0, 0, 37, 29);
  const src = randomPic(frame, 3);

  it("Invert: one minus, mixed, with (un)premult and a mask", async () => {
    const mask = randomPic(rect(5, 4, 30, 20), 9);
    const { pic } = await render(
      new Invert(),
      { mix: 0.7, premult: true },
      { Source: src, Mask: mask },
      frame,
    );
    const want = picFrom(frame, (x, y) => {
      const v = at(src, x, y);
      const a = v[3];
      const inv = v.map((c) => (1 - c / a) * a);
      const amount = 0.7 * at(mask, x, y)[3];
      return v.map((c, i) => lerp(c, inv[i], amount));
    });
    expect(maxError(pic as Pic, want)).toBeLessThan(1e-5);
  });

  it("Invert at mix 0 is an identity and renders nothing", async () => {
    const before = gpu.dispatches;
    const input = upload(src);
    const out = graph.render(
      {
        effect: new Invert(),
        params: { mix: 0 },
        inputs: { Source: { picture: input } },
      },
      frame,
    );
    expect(out.picture).toBe(input);
    expect(gpu.dispatches).toBe(before);
    gpu.flush();
  });

  it("Crop: blacks outside a soft box, or cuts the picture to it", async () => {
    const box = { corner1: [25.5, 20], corner2: [4, 3.25] };
    const soft = await render(
      new Crop(),
      { ...box, softness: 3 },
      { Source: src },
      frame,
    );
    const want = picFrom(frame, (x, y) => {
      const px = x + 0.5;
      const py = y + 0.5;
      const k =
        clamp01((px - 4) / 3) *
        clamp01((25.5 - px) / 3) *
        clamp01((py - 3.25) / 3) *
        clamp01((20 - py) / 3);
      return at(src, x, y).map((c) => c * k);
    });
    expect(maxError(soft.pic as Pic, want)).toBeLessThan(1e-5);

    const cut = await render(
      new Crop(),
      { ...box, cut: true },
      { Source: src },
      frame,
    );
    const r = rect(4, 3, 26, 20);
    expect(cut.out.rod).toEqual(r);
    expect(
      maxError(
        cut.pic as Pic,
        picFrom(r, (x, y) => at(src, x, y)),
      ),
    ).toBe(0);
  });

  it("Blur: separable gaussian, region grown by the reach", async () => {
    const size = [8, 5];
    const { pic, out } = await render(
      new Blur(),
      { size },
      { Source: src },
      rect(-50, -50, 100, 100),
    );
    const sig = size.map((s) => s / 2.4);
    const rad = sig.map((s) => Math.ceil(s * 3 - 1e-9));
    const grownRect = rect(-rad[0], -rad[1], 37 + rad[0], 29 + rad[1]);
    expect(out.rod).toEqual(grownRect);
    const taps = (s: number, r: number) => {
      const w = [];
      for (let t = -r; t <= r; t++) w.push(Math.exp(-(t * t) / (2 * s * s)));
      const total = w.reduce((a, b) => a + b, 0);
      return w.map((v) => v / total);
    };
    const wx = taps(sig[0], rad[0]);
    const wy = taps(sig[1], rad[1]);
    const h = picFrom(
      rect(
        grownRect.x1,
        grownRect.y1 - rad[1],
        grownRect.x2,
        grownRect.y2 + rad[1],
      ),
      (x, y) => {
        const s = [0, 0, 0, 0];
        wx.forEach((w, i) =>
          at(src, x + i - rad[0], y).forEach((c, k) => {
            s[k] += w * c;
          }),
        );
        return s;
      },
    );
    const want = picFrom(grownRect, (x, y) => {
      const s = [0, 0, 0, 0];
      wy.forEach((w, i) =>
        at(h, x, y + i - rad[1]).forEach((c, k) => {
          s[k] += w * c;
        }),
      );
      return s;
    });
    expect(maxError(pic as Pic, want)).toBeLessThan(1e-5);
  });

  it("Grade: the line, gamma, clamps and the clip matte", async () => {
    const params = {
      blackPoint: [0.05, 0, 0.1, 0],
      whitePoint: [0.9, 1, 0.8, 1],
      black: [0.02, 0.01, 0, 0],
      white: [1.3, 1.1, 0.9, 1],
      multiply: [1.2, 1, 1, 1],
      offset: [0, 0.05, -0.1, 0],
      gamma: [1.4, 0.8, 1, 1],
      clampBlack: false,
      clampWhite: false,
    };
    const { pic, out } = await render(
      new Grade(),
      params,
      { Source: src },
      frame,
      { planes: ["Clip"] },
    );
    const c4 = (n: keyof typeof params) => params[n] as number[];
    const graded = (v: number[]) =>
      v.map((x, c) => {
        const a =
          (c4("multiply")[c] * (c4("white")[c] - c4("black")[c])) /
          (c4("whitePoint")[c] - c4("blackPoint")[c]);
        const b = c4("offset")[c] + c4("black")[c] - a * c4("blackPoint")[c];
        const y = a * x + b;
        return y > 0 ? y ** (1 / c4("gamma")[c]) : y;
      });
    const want = picFrom(frame, (x, y) => graded(at(src, x, y)));
    expect(maxError(pic as Pic, want)).toBeLessThan(2e-5);
    const clip = await download(out.planes.get("Clip") as FxBuffer);
    const wantClip = picFrom(frame, (x, y) => {
      const o = graded(at(src, x, y)).map((v) => (v < 0 || v > 1 ? 1 : 0));
      return [o[0], o[1], o[2], Math.max(o[0], o[1], o[2])];
    });
    expect(maxError(clip, wantClip)).toBe(0);
  });

  it("Merge: over, multiply and screen of offset pictures; a missing A", async () => {
    const a = randomPic(rect(10, 5, 50, 20), 5);
    const b = randomPic(rect(0, 0, 30, 25), 6);
    const ops: Record<
      number,
      (A: number, B: number, aa: number, bb: number) => number
    > = {
      28: (A, B, aa) => A + B * (1 - aa),
      26: (A, B) => A * B,
      34: (A, B) => A + B - A * B,
    };
    for (const [op, f] of Object.entries(ops)) {
      const { pic, out } = await render(
        new Merge(),
        { operation: Number(op) },
        { A: a, B: b },
        rect(-100, -100, 100, 100),
      );
      expect(out.rod).toEqual(rect(0, 0, 50, 25));
      const want = picFrom(out.rod, (x, y) => {
        const A = at(a, x, y);
        const B = at(b, x, y);
        return A.map((v, i) => f(v, B[i], A[3], B[3]));
      });
      expect(maxError(pic as Pic, want)).toBeLessThan(1e-6);
    }
    const onlyB = await render(new Merge(), {}, { B: b }, rect(0, 0, 30, 25));
    expect(maxError(onlyB.pic as Pic, b)).toBe(0);
  });

  it("Transform: an impulse translate moves pixels exactly", async () => {
    const { pic, out } = await render(
      new Transform(),
      { translate: [5, -3], filter: 0 },
      { Source: src },
      rect(-100, -100, 100, 100),
    );
    expect(out.rod).toEqual(rect(4, -4, 43, 27));
    expect(
      maxError(
        pic as Pic,
        picFrom(out.rod, (x, y) => at(src, x - 5, y + 3)),
      ),
    ).toBe(0);
  });

  it("Transform: bilinear rotation about the centre against a CPU sampler", async () => {
    const angle = 30;
    const center = [12, 9];
    const { pic, out } = await render(
      new Transform(),
      { rotate: angle, center, filter: 2 },
      { Source: src },
      rect(-100, -100, 100, 100),
    );
    const r = (-angle * Math.PI) / 180;
    const bil = (u: number, v: number) => {
      const fu = Math.floor(u);
      const fv = Math.floor(v);
      const du = u - fu;
      const dv = v - fv;
      const p = (x: number, y: number) => at(src, x, y);
      return [0, 1, 2, 3].map((c) =>
        lerp(
          lerp(p(fu, fv)[c], p(fu + 1, fv)[c], du),
          lerp(p(fu, fv + 1)[c], p(fu + 1, fv + 1)[c], du),
          dv,
        ),
      );
    };
    const want = picFrom(out.rod, (x, y) => {
      // Back through the inverse rotation about the centre.
      const dx = x + 0.5 - center[0];
      const dy = y + 0.5 - center[1];
      const sx = center[0] + Math.cos(r) * dx - Math.sin(r) * dy;
      const sy = center[1] + Math.sin(r) * dx + Math.cos(r) * dy;
      return bil(sx - 0.5, sy - 0.5);
    });
    expect(maxError(pic as Pic, want)).toBeLessThan(1e-4);
  });

  it("CornerPin: a perspective quad against a CPU homography", async () => {
    const quad = [
      [3, 2],
      [40, 6],
      [33, 30],
      [8, 25],
    ];
    const params = {
      corner1: quad[0],
      corner2: quad[1],
      corner3: quad[2],
      corner4: quad[3],
      softness: 2,
    };
    const { pic, out } = await render(
      new CornerPin(),
      params,
      { Source: src },
      rect(0, 0, 64, 64),
    );
    expect(out.rod).toEqual(rect(2, 1, 41, 31));
    const h = pinMatrix(rectCorners(frame), quad as [number, number][]);
    const want = picFrom(out.rod, (x, y) => {
      const px = x + 0.5;
      const py = y + 0.5;
      const w = h[6] * px + h[7] * py + 1;
      const sx = (h[0] * px + h[1] * py + h[2]) / w;
      const sy = (h[3] * px + h[4] * py + h[5]) / w;
      const x0 = Math.floor(sx - 0.5);
      const y0 = Math.floor(sy - 0.5);
      const fx = sx - 0.5 - x0;
      const fy = sy - 0.5 - y0;
      const p = (i: number, j: number) => at(src, i, j);
      const edge = clamp01(
        Math.min(Math.min(sx, 37 - sx), Math.min(sy, 29 - sy)) / 2,
      );
      return [0, 1, 2, 3].map(
        (c) =>
          lerp(
            lerp(p(x0, y0)[c], p(x0 + 1, y0)[c], fx),
            lerp(p(x0, y0 + 1)[c], p(x0 + 1, y0 + 1)[c], fx),
            fy,
          ) * edge,
      );
    });
    expect(maxError(pic as Pic, want)).toBeLessThan(2e-3);
  });

  it("Bloom: below the threshold untouched; a point's energy spread, alpha kept", async () => {
    const r = rect(0, 0, 64, 48);
    const dim = picFrom(r, (x, y) => [0.3, 0.2 * ((x + y) % 2), 0.1, 0.8]);
    const quiet = await render(
      new Bloom(),
      { threshold: 1, knee: 0.5, intensity: 1, size: 0.5 },
      { Source: dim },
      r,
    );
    expect(maxError(quiet.pic as Pic, dim)).toBeLessThan(1e-6);

    // One pixel of light, dim enough for the Karis weights to be 1: every
    // level keeps its energy, so the added light is the intensity times it.
    const v = 0.01;
    const point = picFrom(r, (x, y) =>
      x === 32 && y === 24 ? [v, v / 2, v / 4, 1] : [0, 0, 0, 1],
    );
    const { pic } = await render(
      new Bloom(),
      { threshold: 0, knee: 0, intensity: 0.5, size: 0.25 },
      { Source: point },
      r,
    );
    const out = pic as Pic;
    const added = [0, 0, 0];
    for (let i = 0; i < out.w * out.h; i++) {
      for (let c = 0; c < 3; c++)
        added[c] += out.data[i * 4 + c] - point.data[i * 4 + c];
      expect(out.data[i * 4 + 3]).toBe(1);
    }
    expect(added[0] / (0.5 * v)).toBeCloseTo(1, 1);
    expect(added[1] / added[0]).toBeCloseTo(0.5, 3);
    // Spread: light far from the point, about as much either side (a
    // pyramid on a power-of-two grid is not quite shift invariant).
    const right = at(out, 32 + 6, 24)[0];
    const left = at(out, 32 - 6, 24)[0];
    expect(right).toBeGreaterThan(0);
    expect(Math.abs(right / left - 1)).toBeLessThan(0.25);

    // Intensity zero is the input, not rendered.
    const none = await render(
      new Bloom(),
      { intensity: 0 },
      { Source: point },
      r,
    );
    expect(maxError(none.pic as Pic, point)).toBe(0);
  });

  it("Constant fills the project, CheckerBoard its size", async () => {
    const win = rect(-4, -4, 40, 30);
    const c = await render(
      new Constant(),
      { color: [0.1, 0.2, 0.3, 0.5] },
      {},
      win,
      {},
      [40, 30],
    );
    // EVERYWHERE, clamped to the project as an aofx engine does.
    expect(c.out.rod).toEqual(rect(0, 0, 40, 30));
    expect(
      maxError(
        c.pic as Pic,
        picFrom(c.out.rod, () => [0.1, 0.2, 0.3, 0.5]),
      ),
    ).toBeLessThan(1e-7);
    const k = await render(
      new CheckerBoard(),
      { boxsize: [8, 5], size: [30, 20] },
      {},
      win,
    );
    expect(k.out.rod).toEqual(rect(0, 0, 30, 20));
    const cells = [
      [0.1, 0.1, 0.1, 1],
      [0.5, 0.5, 0.5, 1],
      [0.5, 0.5, 0.5, 1],
      [0.1, 0.1, 0.1, 1],
    ];
    const want = picFrom(
      rect(0, 0, 30, 20),
      (x, y) =>
        cells[
          (Math.floor((x + 0.5) / 8) & 1) + 2 * (Math.floor((y + 0.5) / 5) & 1)
        ],
    );
    expect(maxError(k.pic as Pic, want)).toBeLessThan(1e-7);
  });

  it("ImageStatistics measures and passes the picture through", async () => {
    const { pic, out } = await render(
      new ImageStatistics(),
      {},
      { Source: src },
      frame,
      { instance: "s" },
    );
    expect(maxError(pic as Pic, src)).toBe(0);
    const stats = await out.pending.get("stats.s");
    const n = src.w * src.h;
    for (let c = 0; c < 4; c++) {
      const v = Array.from({ length: n }, (_, i) => src.data[i * 4 + c]);
      const mean = v.reduce((a, b) => a + b, 0) / n;
      const sd = Math.sqrt(v.reduce((a, b) => a + b * b, 0) / n - mean * mean);
      expect(stats?.[c]).toBeCloseTo(Math.min(...v), 6);
      expect(stats?.[4 + c]).toBeCloseTo(Math.max(...v), 6);
      expect(stats?.[8 + c]).toBeCloseTo(mean, 5);
      expect(stats?.[12 + c]).toBeCloseTo(sd, 4);
    }
  });

  it("channel switches put back what the effect was not allowed to change", async () => {
    const { pic } = await render(new Invert(), {}, { Source: src }, frame, {
      channels: [true, false, true, false],
    });
    const want = picFrom(frame, (x, y) => {
      const v = at(src, x, y);
      return [1 - v[0], v[1], 1 - v[2], v[3]];
    });
    expect(maxError(pic as Pic, want)).toBeLessThan(1e-6);
  });

  it("Gpu.run refuses dispatches that disagree with the kernel", () => {
    const k = gpu.load(invertModule, "invertMain");
    const pic = upload(src);
    const out = gpu.scratch(src.w, src.h);
    const u = UniformWriter.for(invertModule);
    expect(() => gpu.run(k, [37, 29, 1], [pic, out], u)).toThrow(
      /2 buffers for the 3/,
    );
    expect(() =>
      gpu.run(k, [37, 29, 1], [pic, pic, out], new Uint8Array(64)),
    ).toThrow(/is 80 bytes, got 64/);
    expect(() => gpu.run(k, [37, 29, 1], [pic, out, out], u)).toThrow(
      /written and also bound/,
    );
    expect(() =>
      gpu.run(k, [37, 29, 1], [pic, pic, { ...out, height: 40 }], u),
    ).toThrow(/needs/);
    expect(() => gpu.run(k, [2e7, 1, 1], [pic, pic, out], u)).toThrow(
      /groups along x/,
    );
    gpu.flush();
  });

  for (const format of [
    "rgba8unorm",
    "rgba16float",
    "bgra8unorm",
  ] as GPUTextureFormat[]) {
    it(`a chain round-trips a ${format} texture (y flipped, in place)`, async () => {
      const d = device as GPUDevice;
      const [w, h] = [19, 11];
      const texture = d.createTexture({
        size: [w, h],
        format,
        usage:
          GPUTextureUsage.COPY_SRC |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.TEXTURE_BINDING,
      });
      // Texel (x, row) = (x/w, row/h, 0.5, 1): row 0 is the top.
      const eight = format !== "rgba16float";
      const bpr = Math.ceil((w * (eight ? 4 : 8)) / 256) * 256;
      const bytes = new Uint8Array(bpr * h);
      const half = (v: number) => {
        const f = new Float32Array([v]);
        const i = new Uint32Array(f.buffer)[0];
        const e = ((i >> 23) & 0xff) - 127 + 15;
        return v === 0
          ? 0
          : ((i >> 16) & 0x8000) | (e << 10) | ((i >> 13) & 0x3ff);
      };
      const value = (x: number, row: number) => [
        Math.round((x / w) * 255) / 255,
        Math.round((row / h) * 255) / 255,
        0.5,
        1,
      ];
      for (let row = 0; row < h; row++) {
        for (let x = 0; x < w; x++) {
          const v = value(x, row);
          if (eight) {
            const q = v.map((c) => Math.round(c * 255));
            const o = format === "bgra8unorm" ? [q[2], q[1], q[0], q[3]] : q;
            bytes.set(o, row * bpr + x * 4);
          } else {
            new DataView(bytes.buffer).setUint16(
              row * bpr + x * 8,
              half(v[0]),
              true,
            );
            new DataView(bytes.buffer).setUint16(
              row * bpr + x * 8 + 2,
              half(v[1]),
              true,
            );
            new DataView(bytes.buffer).setUint16(
              row * bpr + x * 8 + 4,
              half(v[2]),
              true,
            );
            new DataView(bytes.buffer).setUint16(
              row * bpr + x * 8 + 6,
              half(v[3]),
              true,
            );
          }
        }
      }
      d.queue.writeTexture({ texture }, bytes, { bytesPerRow: bpr }, [w, h]);

      const chain = new FxChain(gpu);
      chain.add(new Invert(), {}, { channels: [true, true, false, false] });
      chain.add(new Transform(), { translate: [2, 1], filter: 0 });
      // Read the picture the chain sees, then apply in place.
      const read = chain.bridge.read(texture);
      gpu.flush();
      const seen = await download(read);
      expect(at(seen, 3, 0)[1]).toBeCloseTo(value(3, h - 1)[1], 2);
      chain.apply(texture);

      const out = d.createBuffer({
        size: bpr * h,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const enc = d.createCommandEncoder();
      enc.copyTextureToBuffer({ texture }, { buffer: out, bytesPerRow: bpr }, [
        w,
        h,
      ]);
      d.queue.submit([enc.finish()]);
      await out.mapAsync(GPUMapMode.READ);
      const got = new Uint8Array(out.getMappedRange().slice(0));
      out.unmap();
      const texel = (x: number, row: number) => {
        if (eight) {
          const q = Array.from(
            got.subarray(row * bpr + x * 4, row * bpr + x * 4 + 4),
          ).map((c) => c / 255);
          return format === "bgra8unorm" ? [q[2], q[1], q[0], q[3]] : q;
        }
        const dv = new DataView(got.buffer);
        return [0, 1, 2, 3].map((c) => {
          const b = dv.getUint16(row * bpr + x * 8 + c * 2, true);
          const e = (b >> 10) & 0x1f;
          return e === 0 ? 0 : 2 ** (e - 15) * (1 + (b & 0x3ff) / 1024);
        });
      };
      // Picture (x, y) up = texture (x, h-1-y). Moved by (2, 1) up, so
      // texture row r now holds what row r+1 held, and x by 2.
      for (let row = 0; row < h; row++) {
        for (let x = 0; x < w; x++) {
          const fromX = x - 2;
          const fromRow = row + 1;
          const want =
            fromX < 0 || fromRow >= h
              ? [0, 0, 0, 0]
              : ((v) => [1 - v[0], 1 - v[1], v[2], v[3]])(
                  value(fromX, fromRow),
                );
          const g = texel(x, row);
          for (let c = 0; c < 4; c++) expect(g[c]).toBeCloseTo(want[c], 2);
        }
      }
      texture.destroy();
    });
  }
});

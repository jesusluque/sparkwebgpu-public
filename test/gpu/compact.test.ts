// The compact relit path (thread CB, src/webgpu/athenea/compact.ts,
// slang/athenea_adapter/compact.slang) on a synthetic TX cloud:
//   - the compact decode (CPU, as the shader) against the cloud it encoded;
//   - the in-draw relight against today's relight (atheneaRelightPlugin,
//     cpca8) run on the same cloud dequantized (compactToSource): per-splat
//     colours to a relMSE of 1e-5, so what is left is the path's halves;
//   - the draw against today's draw of that cloud (float images).

import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { encodeExtSplat } from "../../src/utils";
import {
  COMPACT_PIECE,
  compactToSource,
  decodeCompactSplat,
  encodeCompact,
  packNormal16,
} from "../../src/webgpu/athenea/compact";
import {
  packJointBasis,
  syntheticJointBasis,
  syntheticJointRecords,
} from "../../src/webgpu/athenea/cpcaJoint";
import {
  type Streams,
  compactRun,
  compare,
  placeCamera,
  referenceRun,
} from "./compactHarness";
import { wideDevice } from "./device";

const N = COMPACT_PIECE + 3000; // two pieces, three chunks

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

/** A synthetic TX cloud: a sphere shell of splats facing out, a few materials. */
function syntheticCloud(): Streams {
  const r = rng(3);
  const ext0 = new Uint32Array(4 * N);
  const ext1 = new Uint32Array(4 * N);
  const normalOct = new Uint32Array(N);
  const pbr = new Uint32Array(N);
  const lobes = new Uint32Array(3 * N);
  const curvature = new Uint32Array(2 * N);
  const shadowBits = new Uint32Array(8 * N);
  const half = (v: number) => {
    const f = new Float32Array([v]);
    const u = new Uint32Array(f.buffer)[0];
    const e = ((u >> 23) & 255) - 127 + 15;
    const m = (u >> 13) & 1023;
    return e <= 0 ? 0 : ((u >> 16) & 0x8000) | (e << 10) | m;
  };
  // pbr: metallic, roughness, transmission bytes, flags (thin 1 << 24).
  const materials = [
    {
      pbr: 0 | (128 << 8),
      lobes: [0xffffffff, 0x3aa60000 | (20 << 8) | 255, 0],
    },
    {
      pbr: 255 | (40 << 8) | (2 << 24),
      lobes: [0xffffffff, 0x409dffff, 0x80000000],
    },
    {
      pbr: (10 << 8) | (255 << 16) | (1 << 24),
      lobes: [0xffffffff, 0x3aa60000, 0],
    },
    { pbr: 200 | (90 << 8), lobes: [0xc0ffffff, 0x3aa6ff00, 0x80000000] },
  ];
  for (let i = 0; i < N; i++) {
    const z = 2 * r() - 1;
    const phi = 2 * Math.PI * r();
    const rad = Math.sqrt(1 - z * z);
    const n: [number, number, number] = [
      rad * Math.cos(phi),
      rad * Math.sin(phi),
      z,
    ];
    const p = n.map((v) => 0.5 * v);
    // A disc facing out: the quaternion turning +Z to n.
    const q = new THREE.Quaternion().setFromUnitVectors(
      new THREE.Vector3(0, 0, 1),
      new THREE.Vector3(...n),
    );
    const mode = process.env.COMPACT_ALPHA_MODE;
    const a0 = i % 7 === 0 ? 0.05 + 0.3 * r() : i % 3 === 0 ? 1 + r() : 0.9;
    const a =
      mode === "plain" ? 0.9 : mode === "lod" ? (i % 3 === 0 ? a0 : 0.9) : a0;
    encodeExtSplat(
      [ext0, ext1],
      i,
      p[0],
      p[1],
      p[2],
      0.01 + 0.01 * r(),
      0.01 + 0.01 * r(),
      0.001,
      q.x,
      q.y,
      q.z,
      q.w,
      a,
      r(),
      r(),
      r(),
    );
    normalOct[i] = packNormal16(n);
    const m = materials[Math.floor(i / 97) % materials.length];
    pbr[i] = m.pbr;
    lobes.set(m.lobes, 3 * i);
    const k = (r() - 0.5) * 8;
    curvature[2 * i] = half(2 + k) | (half(0.1 * k) << 16);
    curvature[2 * i + 1] = half(2 - k);
    // Open where the cell faces out (a hemisphere), a few closed at random.
    for (let w = 0; w < 8; w++)
      shadowBits[8 * i + w] = ((r() * 2 ** 32) >>> 0) | 0x0f0f0f0f;
  }
  return {
    ext0,
    ext1,
    normalOct,
    pbr,
    lobes,
    curvature,
    shadowBits,
    q8: syntheticJointRecords(N, 256),
  };
}

function sky() {
  const W = 64;
  const H = 32;
  const data = new Float32Array(W * H * 4);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const k = 4 * (y * W + x);
      const up = 1 - y / H;
      data[k] = 0.3 + up;
      data[k + 1] = 0.4 + 0.8 * up;
      data[k + 2] = 0.6 + 0.5 * up;
      data[k + 3] = 1;
      if (Math.abs(x - 20) < 2 && Math.abs(y - 8) < 2) data.fill(200, k, k + 3);
    }
  return { width: W, height: H, data };
}

describe.skipIf(!wideDevice).each([8, 4])(
  "the compact relit path, weights on %i bits",
  (bits) => {
    const d = wideDevice as GPUDevice;
    const s = syntheticCloud();
    const basis = syntheticJointBasis(256, 5);
    // 4 bits: the records put on the 7-level grid first, as
    // encodeJointCpca(..., 7) writes them.
    const weightBits = bits as 8 | 4;
    if (weightBits === 4) {
      for (let r = 0; r < N; r++) {
        for (let w = 0; w < 6; w++) {
          const at = 7 * r + w;
          let word = 0;
          for (let k = 0; k < 4; k++) {
            const q8 = ((s.q8[at] << (24 - 8 * k)) >> 24) as number;
            const q = Math.round((Math.round((q8 * 7) / 127) * 127) / 7);
            word |= (q & 0xff) << (8 * k);
          }
          s.q8[at] = word >>> 0;
        }
      }
    }
    const cloud = encodeCompact({
      count: N,
      ...s,
      basis: packJointBasis(basis),
      clusters: 256,
      weightBits,
    });

    it("decodes what it encoded", () => {
      const f = new Float32Array(s.ext0.buffer);
      let worstPos = 0;
      for (let i = 0; i < N; i += 37) {
        const g = decodeCompactSplat(cloud, i);
        for (let k = 0; k < 3; k++)
          worstPos = Math.max(worstPos, Math.abs(g.center[k] - f[4 * i + k]));
        expect(g.rotation).toBe(s.ext1[4 * i + 3]);
        expect(g.pbr).toBe(s.pbr[i]);
        expect([...g.cells]).toEqual([
          ...s.shadowBits.subarray(8 * i, 8 * i + 8),
        ]);
      }
      // A unit sphere's chunk box over 65535 steps.
      expect(worstPos).toBeLessThan(2e-5);
      expect(cloud.layout.pieces).toBe(2);
      expect(cloud.layout.materials).toBe(4);
    });

    it("relights as today's path does on the same values", async () => {
      const W = 320;
      const H = 240;
      const { camera, place } = placeCamera(
        W,
        H,
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(1.2, -1.0, 0.6),
        50,
        1,
      );
      const source = compactToSource(cloud, s.q8);
      const run = {
        W,
        H,
        sky: sky(),
        camera,
        place,
        colours: true,
        image: true,
      };
      const noFaint = process.env.COMPACT_NO_FAINT === "1";
      const ref = await referenceRun(d, { ...source }, N, basis, {
        ...run,
        shapes: true,
        renderer: noFaint ? { faintSplats: "drop" } : {},
      });
      const got = await compactRun(d, cloud, {
        ...run,
        path: { faint: !noFaint },
      });
      const colours = compare(
        got.colours as Float32Array,
        ref.colours as Float32Array,
      );
      let alphaDiff = 0;
      for (let i = 0; i < N; i++) {
        alphaDiff = Math.max(
          alphaDiff,
          Math.abs(
            (got.colours as Float32Array)[4 * i + 3] -
              (ref.colours as Float32Array)[4 * i + 3],
          ),
        );
      }
      const image = compare(
        got.image as Float32Array,
        ref.image as Float32Array,
      );
      {
        // The footprints, splat by splat: the worst relative differences.
        const a = got.shapes as Float32Array;
        const b = ref.shapes as Float32Array;
        const worst = [0, 0, 0, 0];
        let onlyRef = 0;
        let onlyGot = 0;
        const examples: unknown[] = [];
        for (let i = 0; i < N; i++) {
          const gz = a[4 * i + 3] === 0;
          const rz = b[4 * i + 3] === 0;
          if (gz !== rz) {
            if (gz) onlyRef++;
            else onlyGot++;
            continue;
          }
          if (gz) continue;
          for (let k = 0; k < 4; k++) {
            const e =
              Math.abs(a[4 * i + k] - b[4 * i + k]) /
              (Math.abs(b[4 * i + k]) + 1e-3);
            if (e > worst[k]) worst[k] = e;
            if (e > 0.05 && examples.length < 6)
              examples.push({
                i,
                k,
                got: [...a.subarray(4 * i, 4 * i + 4)],
                ref: [...b.subarray(4 * i, 4 * i + 4)],
              });
          }
        }
        const pa = got.places as Float32Array;
        const pb = ref.places as Float32Array;
        let centre = 0;
        let angle = 0;
        for (let i = 0; i < N; i++) {
          if (a[4 * i + 3] === 0 || b[4 * i + 3] === 0) continue;
          centre = Math.max(
            centre,
            Math.hypot(pa[4 * i] - pb[4 * i], pa[4 * i + 1] - pb[4 * i + 1]),
          );
          // Only where the axes differ (a round footprint has no angle).
          if (Math.abs(a[4 * i] - a[4 * i + 1]) > 0.1 * a[4 * i]) {
            let da = Math.abs(pa[4 * i + 2] - pb[4 * i + 2]);
            da = Math.min(da, Math.PI - da);
            angle = Math.max(angle, da);
          }
        }
        process.stderr.write(
          `COMPACT_SHAPES ${JSON.stringify({ worst, onlyRef, onlyGot, centre, angle, examples })}\n`,
        );
      }
      if (process.env.COMPACT_DUMP) {
        const { writeFileSync } = await import("node:fs");
        writeFileSync(
          `${process.env.COMPACT_DUMP}/ref.f32`,
          ref.image as Float32Array,
        );
        writeFileSync(
          `${process.env.COMPACT_DUMP}/got.f32`,
          got.image as Float32Array,
        );
      }
      const alphaImage = compare(
        got.image as Float32Array,
        ref.image as Float32Array,
        4,
        4,
      );
      process.stderr.write(`COMPACT_ALPHA ${JSON.stringify(alphaImage)}\n`);
      process.stderr.write(
        `COMPACT_SYNTHETIC ${JSON.stringify({ colours, alphaDiff, image, drawn: got.drawn })}\n`,
      );
      expect(colours.relMSE).toBeLessThan(1e-5);
      expect(alphaDiff).toBeLessThan(1e-3);
      expect(image.relMSE).toBeLessThan(1e-3);
    }, 120_000);
  },
);

describe.skipIf(!wideDevice || !process.env.COMPACT_SINGLE)("one splat", () => {
  const d = wideDevice as GPUDevice;
  it("draws as today's", async () => {
    const n = 1;
    const ext0 = new Uint32Array(4 * n);
    const ext1 = new Uint32Array(4 * n);
    const tilt = Number(process.env.COMPACT_SINGLE);
    const q = new THREE.Quaternion().setFromUnitVectors(
      new THREE.Vector3(0, 0, 1),
      new THREE.Vector3(Math.cos(tilt), Math.sin(tilt), 0),
    );
    encodeExtSplat(
      [ext0, ext1],
      0,
      0,
      0,
      0,
      0.05,
      0.03,
      0.001,
      q.x,
      q.y,
      q.z,
      q.w,
      0.9,
      0.5,
      0.5,
      0.5,
    );
    const s: Streams = {
      ext0,
      ext1,
      normalOct: new Uint32Array([
        packNormal16([Math.cos(tilt), Math.sin(tilt), 0]),
      ]),
      pbr: new Uint32Array([128 << 8]),
      lobes: new Uint32Array([0xffffffff, 0x3aa60000, 0]),
      curvature: null,
      shadowBits: new Uint32Array(8).fill(0xffffffff),
      q8: syntheticJointRecords(1, 256),
    };
    const basis = syntheticJointBasis(256, 5);
    const cloud = encodeCompact({
      count: n,
      ...s,
      basis: packJointBasis(basis),
      clusters: 256,
    });
    const W = 64;
    const H = 64;
    const { camera, place } = placeCamera(
      W,
      H,
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, -1, 0),
      20,
      1,
    );
    const run = { W, H, sky: sky(), camera, place, colours: true, image: true };
    const ref = await referenceRun(d, compactToSource(cloud, s.q8), n, basis, {
      ...run,
      shapes: true,
    });
    const got = await compactRun(d, cloud, run);
    const sum = (im: Float32Array) => {
      let a = 0;
      for (let k = 3; k < im.length; k += 4) a += im[k];
      return a;
    };
    process.stderr.write(
      `ONE ${JSON.stringify({ ref: sum(ref.image as Float32Array), got: sum(got.image as Float32Array), refShape: [...(ref.shapes as Float32Array)], gotShape: [...(got.shapes as Float32Array)], refPlace: [...(ref.places as Float32Array)], gotPlace: [...(got.places as Float32Array)] })}\n`,
    );
    const rows = (im: Float32Array) => {
      const out: string[] = [];
      for (let y = 26; y < 38; y++) {
        let line = "";
        for (let x = 20; x < 44; x++)
          line += Math.round(im[(y * W + x) * 4 + 3] * 9);
        out.push(line);
      }
      return out.join("\n");
    };
    process.stderr.write(
      `REF\n${rows(ref.image as Float32Array)}\nGOT\n${rows(got.image as Float32Array)}\n`,
    );
  });
});

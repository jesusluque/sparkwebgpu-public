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
  CompactRelitPath,
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
  readImage,
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
function syntheticCloud(N = COMPACT_PIECE + 3000, clusters = 256): Streams {
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
      // The material's own roughness 0.34 (athenea 35ae3cb's mark, bits
      // 26..31): the polish's field answers its closed part (polishField).
      pbr: (0 | (128 << 8) | (1 << 26) | (11 << 27)) >>> 0,
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
    // Its own roughness 0 (a chrome under its coat): no field.
    {
      pbr: 200 | (90 << 8) | (1 << 26),
      lobes: [0xc0ffffff, 0x3aa6ff00, 0x80000000],
    },
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
    q8: syntheticJointRecords(N, clusters),
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
        // The same sheet floor on both sides: the reference relight's
        // default floors a thin sheet at 0.1 BEFORE the draw's blur, the
        // compact path now floors it after (sheetFade, thread CS) unless
        // told CN's way (sheetFade < 0). Material 2 is a thin sheet.
        path: { faint: !noFaint, sheetFade: -1 },
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

    it("holds a fold's coat and reads the polish's field as today's path (athenea 35ae3cb)", async () => {
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
      const run = { W, H, sky: sky(), camera, place, colours: true };
      const rules = { coatFold: true, polishField: true };
      const ref = await referenceRun(d, { ...source }, N, basis, {
        ...run,
        relight: rules,
      });
      const got = await compactRun(d, cloud, {
        ...run,
        path: { sheetFade: -1, ...rules },
      });
      const plain = await compactRun(d, cloud, {
        ...run,
        path: { sheetFade: -1, polishField: false, coatFold: false },
      });
      const colours = compare(
        got.colours as Float32Array,
        ref.colours as Float32Array,
      );
      const moved = compare(
        got.colours as Float32Array,
        plain.colours as Float32Array,
      );
      process.stderr.write(
        `COMPACT_RULES ${JSON.stringify({ colours, moved })}\n`,
      );
      // The rules change the light ...
      expect(moved.relMSE).toBeGreaterThan(3e-7);
      // ... the same way on both paths.
      expect(colours.relMSE).toBeLessThan(1e-5);
    }, 120_000);

    it("floors thin sheets after the blur as the draw does (sheetFade)", async () => {
      // The compact path's default (thread CS) against today's path with
      // the floor in the draw (sheetFloor "draw", the renderer's sheetFade
      // 3, pieces.html's): the image (the per-splat colours differ by
      // design: the draw takes the sheet's scale, the compact shade too).
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
      const run = { W, H, sky: sky(), camera, place, image: true };
      const ref = await referenceRun(
        d,
        { ...compactToSource(cloud, s.q8) },
        N,
        basis,
        { ...run, relight: { sheetFloor: "draw" }, renderer: { sheetFade: 3 } },
      );
      const got = await compactRun(d, cloud, {
        ...run,
        path: { sheetFade: 3 },
      });
      const before = await compactRun(d, cloud, {
        ...run,
        path: { sheetFade: -1 },
      });
      const image = compare(
        got.image as Float32Array,
        ref.image as Float32Array,
      );
      const old = compare(
        before.image as Float32Array,
        ref.image as Float32Array,
      );
      process.stderr.write(
        `COMPACT_SHEET_AFTER_BLUR ${JSON.stringify({ image, before: old })}\n`,
      );
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
    const run = {
      W,
      H,
      sky: sky(),
      camera,
      place,
      colours: true,
      image: true,
      path: { stableOrder: true },
    };
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

// A draw list (setDrawList): the cull walks the listed gaussians only (a
// LoD cut chosen on the host), the list growing the frame buffer (and the
// projections redone), and every gaussian listed draws what no list draws.
describe.skipIf(!wideDevice)("the compact path's draw list", () => {
  const d = wideDevice as GPUDevice;
  it("draws the listed gaussians", async () => {
    const s = syntheticCloud();
    const cloud = encodeCompact({
      count: N,
      ...s,
      basis: packJointBasis(syntheticJointBasis(256, 5)),
      clusters: 256,
    });
    const W = 160;
    const H = 120;
    const { camera } = placeCamera(
      W,
      H,
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(3, -2, 1),
      40,
      1,
    );
    const color = d.createTexture({
      size: [W, H],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const view = color.createView();
    const path = new CompactRelitPath(d, { hdri: sky() });
    path.setCloud(cloud);
    const frame = async () => {
      const enc = d.createCommandEncoder();
      path.render(enc, camera, view, "rgba16float", W, H, {
        clear: { r: 0, g: 0, b: 0, a: 0 },
      });
      path.registry.submit(enc.finish());
      await d.queue.onSubmittedWorkDone();
    };
    await frame();
    expect(await path.drawnCount()).toBe(N);
    const all = await readImage(d, color);
    const evens = new Uint32Array(Math.ceil(N / 2)).map((_, k) => 2 * k);
    path.setDrawList(evens);
    await frame();
    expect(await path.drawnCount()).toBe(evens.length);
    // Every gaussian listed (the frame buffer grows): the same image.
    path.setDrawList(new Uint32Array(N).map((_, k) => k));
    await frame();
    expect(await path.drawnCount()).toBe(N);
    const listed = await readImage(d, color);
    let diff = 0;
    let sum = 0;
    for (let k = 0; k < all.length; k++) {
      diff += (all[k] - listed[k]) ** 2;
      sum += all[k] ** 2;
    }
    process.stderr.write(`DRAWLIST relMSE ${diff / sum}\n`);
    // Equal sort keys land in either order (the atomic append): ~1e-5 to
    // 2e-4 between runs; a lost projection or list would be ~1.
    expect(diff / sum).toBeLessThan(2e-3);
    // Nothing listed: cleared, nothing drawn.
    path.setDrawList(new Uint32Array(0));
    await frame();
    const empty = await readImage(d, color);
    expect(empty.every((v) => v === 0)).toBe(true);
    path.dispose();
    color.destroy();
  });
});

describe.skipIf(!wideDevice)("the compact path's shading options", () => {
  const d = wideDevice as GPUDevice;
  it("bins by cluster and repeats with stableOrder, bit for bit", async () => {
    const s = syntheticCloud();
    const cloud = encodeCompact({
      count: N,
      ...s,
      basis: packJointBasis(syntheticJointBasis(256, 5)),
      clusters: 256,
    });
    const W = 160;
    const H = 120;
    const { camera, place } = placeCamera(
      W,
      H,
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(3, -2, 1),
      40,
      1,
    );
    const color = d.createTexture({
      size: [W, H],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const view = color.createView();
    const imageOf = async (binByCluster: boolean, deg: number) => {
      const path = new CompactRelitPath(d, {
        hdri: sky(),
        binByCluster,
        stableOrder: true,
      });
      path.setCloud(cloud);
      place(deg);
      const enc = d.createCommandEncoder();
      path.render(enc, camera, view, "rgba16float", W, H, {
        clear: { r: 0, g: 0, b: 0, a: 0 },
      });
      path.registry.submit(enc.finish());
      const image = await readImage(d, color);
      path.dispose();
      return image;
    };
    for (const deg of [0, 37]) {
      const plain = await imageOf(false, deg);
      const again = await imageOf(false, deg);
      const binned = await imageOf(true, deg);
      expect(again).toEqual(plain);
      expect(binned).toEqual(plain);
      expect(plain.some((v) => v > 0)).toBe(true);
    }
    color.destroy();
  });

  it("toggles sharp and plain after the env went into the atlas, the same images", async () => {
    // a98d9d2 freed the env buffer once the atlas held it: turning the sharp
    // lobes off (or plain on) and back read a missing buffer (CW).
    const s = syntheticCloud();
    const cloud = encodeCompact({
      count: N,
      ...s,
      basis: packJointBasis(syntheticJointBasis(256, 5)),
      clusters: 256,
    });
    const W = 96;
    const H = 72;
    const { camera } = placeCamera(
      W,
      H,
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(3, -2, 1),
      40,
      1,
    );
    const color = d.createTexture({
      size: [W, H],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const view = color.createView();
    const draw = async (path: CompactRelitPath) => {
      const enc = d.createCommandEncoder();
      path.render(enc, camera, view, "rgba16float", W, H, {
        clear: { r: 0, g: 0, b: 0, a: 0 },
      });
      path.registry.submit(enc.finish());
      return readImage(d, color);
    };
    const fresh = async (o: { sharp?: boolean; plain?: boolean }) => {
      const p = new CompactRelitPath(d, {
        hdri: sky(),
        stableOrder: true,
        ...o,
      });
      p.setCloud(cloud);
      const image = await draw(p);
      p.dispose();
      return image;
    };
    const sharpOn = await fresh({ sharp: true });
    const sharpOff = await fresh({ sharp: false });
    const plainOn = await fresh({ sharp: true, plain: true });
    const path = new CompactRelitPath(d, {
      hdri: sky(),
      stableOrder: true,
      sharp: true,
    });
    path.setCloud(cloud);
    expect(await draw(path)).toEqual(sharpOn);
    // The env buffer is in the atlas now.
    expect(path.sky.envTexels).toBeNull();
    path.options.sharp = false;
    expect(await draw(path)).toEqual(sharpOff);
    path.options.sharp = true;
    expect(await draw(path)).toEqual(sharpOn);
    path.options.plain = true;
    expect(await draw(path)).toEqual(plainOn);
    path.options.plain = false;
    expect(await draw(path)).toEqual(sharpOn);
    expect(sharpOn.some((v) => v > 0)).toBe(true);
    path.dispose();
    color.destroy();
  });

  it("keeps a still frame still when the same draw list comes again", async () => {
    const s = syntheticCloud();
    const cloud = encodeCompact({
      count: N,
      ...s,
      basis: packJointBasis(syntheticJointBasis(256, 5)),
      clusters: 256,
    });
    const W = 64;
    const H = 48;
    const { camera } = placeCamera(
      W,
      H,
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(3, -2, 1),
      40,
      1,
    );
    const color = d.createTexture({
      size: [W, H],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const view = color.createView();
    const path = new CompactRelitPath(d, { hdri: sky() });
    path.setCloud(cloud);
    const frame = () => {
      const enc = d.createCommandEncoder();
      path.render(enc, camera, view, "rgba16float", W, H);
      path.registry.submit(enc.finish());
    };
    const list = new Uint32Array(N >> 1).map((_, k) => 2 * k);
    path.setDrawList(list);
    frame();
    const shaded = path.stats.frames;
    path.setDrawList(list.slice());
    frame();
    // The same cut: only the draw ran again.
    expect(path.stats.frames).toBe(shaded);
    list[0] = 1;
    path.setDrawList(list);
    frame();
    expect(path.stats.frames).toBe(shaded + 1);
    await d.queue.onSubmittedWorkDone();
    path.dispose();
    color.destroy();
  });
});

// 512 clusters (thread CX: a 9-bit cluster id, its 9th bit in the shadow
// index's bit 15, the bins 512): a basis of two copies of a 256-cluster one,
// half the splats pointing at the second copy, draws exactly as the 256.
describe.skipIf(!wideDevice)("512 clusters", () => {
  const d = wideDevice as GPUDevice;
  it("a doubled basis draws as the single one", async () => {
    const s = syntheticCloud();
    const basis = packJointBasis(syntheticJointBasis(256, 5));
    const twice = new Float32Array(2 * basis.length);
    twice.set(basis);
    twice.set(basis, basis.length);
    const q8 = s.q8.slice();
    for (let i = 0; i < N; i += 2) q8[7 * i + 6] += 256;
    const a = encodeCompact({ count: N, ...s, basis, clusters: 256 });
    const b = encodeCompact({
      count: N,
      ...s,
      q8,
      basis: twice,
      clusters: 512,
    });
    for (let i = 0; i < N; i += 101) {
      expect([...decodeCompactSplat(b, i).cells]).toEqual([
        ...s.shadowBits.subarray(8 * i, 8 * i + 8),
      ]);
    }
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
    const run = {
      W,
      H,
      sky: sky(),
      camera,
      place,
      colours: true,
      image: true,
      path: { stableOrder: true },
    };
    const ra = await compactRun(d, a, run);
    const rb = await compactRun(d, b, run);
    expect(rb.drawn).toBe(ra.drawn);
    expect(ra.drawn).toBeGreaterThan(N / 4);
    const ca = ra.colours as Float32Array;
    const cb = rb.colours as Float32Array;
    let worst = 0;
    for (let k = 0; k < ca.length; k++)
      worst = Math.max(worst, Math.abs(ca[k] - cb[k]));
    expect(worst).toBe(0);
    const ia = ra.image as Float32Array;
    const ib = rb.image as Float32Array;
    let worstPx = 0;
    for (let k = 0; k < ia.length; k++)
      worstPx = Math.max(worstPx, Math.abs(ia[k] - ib[k]));
    expect(worstPx).toBe(0);
  });

  // Opt-in (COMPACT_CLUSTERS_TIMING=1, a short GPU hold): the shade's cost at
  // 256 and 512 clusters, the phone's 393 x 852, a cloud of COMPACT_CLUSTERS_N.
  it.skipIf(process.env.COMPACT_CLUSTERS_TIMING !== "1")(
    "the shade's cost at 256 and 512 clusters",
    async () => {
      const n = Number(process.env.COMPACT_CLUSTERS_N ?? 400_000);
      const W = 393;
      const H = 852;
      const { camera, place } = placeCamera(
        W,
        H,
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(0.9, -0.75, 0.45),
        50,
        1,
      );
      const out: Record<string, unknown> = {};
      for (const C of [256, 512, 256, 512]) {
        const s = syntheticCloud(n, C);
        const cloud = encodeCompact({
          count: n,
          ...s,
          basis: packJointBasis(syntheticJointBasis(C, 5)),
          clusters: C,
        });
        const r = await compactRun(d, cloud, {
          W,
          H,
          sky: sky(),
          camera,
          place,
          timing: true,
          frames: 40,
        });
        out[`${C}${out[`${C}`] ? "b" : ""}`] = {
          drawn: r.drawn,
          timing: r.timing,
        };
      }
      console.log(`COMPACT_CLUSTERS ${JSON.stringify(out)}`);
    },
    600_000,
  );
});

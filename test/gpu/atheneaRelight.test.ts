// The athenea relight plugin (atheneaRelightPlugin, slang/athenea_adapter/
// relight.slang) on Dawn against athenea's own relighting: a battery of
// materials -- matte, polished, metal, Schlick metal, lacquer (coat), sheen,
// glass, a thin sheet, emission, a baked body -- with a TX transfer (direct
// 16, indirect 48, reflected field 48) and 16 x 16 open-direction cells,
// under a dome with a sun in it, an analytic sun and a sphere lamp, on an
// object turned, moved and scaled.
//
// The reference (slang/tests/athenea_relight.slang) is splat_project's
// relitToward and relitSplat on athenea's own buffer layout -- a stream a
// buffer, a transfer at i * transferWords halves -- with no kept terms. The
// plugin reads the same streams from the mesh's attribute pool, keeps the
// view-independent terms as halves (athenea's txCache) and hands generate
// the colour; so what differs is the halves, and nothing else.

import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { encodeExtSplat } from "../../src/utils";
import { KernelRegistry } from "../../src/webgpu/KernelRegistry";
import {
  GpuSplatSource,
  WgpuSplatRenderer,
} from "../../src/webgpu/WgpuSplatRenderer";
import type { AtheneaSky, SkyImage } from "../../src/webgpu/athenea/AtheneaSky";
import {
  type AtheneaLightRecord,
  LIGHT_SHADOW,
  packLightRecords,
} from "../../src/webgpu/athenea/lights";
import { atheneaRelightPlugin } from "../../src/webgpu/athenea/relightPlugin";
import refModule from "../../src/webgpu/generated/tests/athenea_relight";
import {
  createReadback,
  createStorage,
  readAndDestroy,
  upload,
} from "../../src/webgpu/gpuBuffers";
import { PluginHost } from "../../src/webgpu/plugins";
import { UniformWriter } from "../../src/webgpu/uniforms";
import { wideDevice } from "./device";

const NONE = 0xffffffff;
const XAXIS = new THREE.Vector3(1, 0, 0);

/** packing.slang packNormal. */
function packNormal(n: number[]): number {
  const l1 = Math.abs(n[0]) + Math.abs(n[1]) + Math.abs(n[2]);
  const v = n.map((c) => c / l1);
  let p = [v[0], v[1]];
  if (v[2] < 0) {
    p = [
      (1 - Math.abs(v[1])) * (v[0] >= 0 ? 1 : -1),
      (1 - Math.abs(v[0])) * (v[1] >= 0 ? 1 : -1),
    ];
  }
  const q = p.map((c) =>
    Math.round(Math.min(Math.max(c * 0.5 + 0.5, 0), 1) * 65535),
  );
  return (q[0] | (q[1] << 16)) >>> 0;
}

const byte = (v: number) => Math.round(Math.min(Math.max(v, 0), 1) * 255);

/** splat_encoding's pbr word. */
const pbrWord = (
  metallic: number,
  roughness: number,
  transmission = 0,
  thin = false,
  schlick = false,
) =>
  (byte(metallic) |
    (byte(roughness) << 8) |
    (byte(transmission) << 16) |
    (thin ? 1 << 24 : 0) |
    (schlick ? 1 << 25 : 0)) >>>
  0;

/** packing.slang packLobes. */
function lobesWords(l: {
  specularWeight?: number;
  specularColour?: number[];
  specularIor?: number;
  coatWeight?: number;
  coatRoughness?: number;
  coatIor?: number;
  coatDarkening?: boolean;
  sheenColour?: number[];
  sheenRoughness?: number;
}): number[] {
  const sc = l.specularColour ?? [1, 1, 1];
  const sh = l.sheenColour ?? [0, 0, 0];
  const ior = (v: number) =>
    Math.min(Math.max(Math.floor((v - 1) * 128 + 0.5), 0), 255);
  const coat =
    Math.min(
      Math.max(Math.floor(((l.coatIor ?? 1.5) - 1) * 64 + 0.5), 0),
      127,
    ) | (l.coatDarkening ? 128 : 0);
  return [
    (byte(sc[0]) |
      (byte(sc[1]) << 8) |
      (byte(sc[2]) << 16) |
      (byte(l.specularWeight ?? 1) << 24)) >>>
      0,
    (byte(l.coatWeight ?? 0) |
      (byte(l.coatRoughness ?? 0) << 8) |
      (coat << 16) |
      (ior(l.specularIor ?? 1.5) << 24)) >>>
      0,
    (byte(sh[0]) |
      (byte(sh[1]) << 8) |
      (byte(sh[2]) << 16) |
      (byte(l.sheenRoughness ?? 0) << 24)) >>>
      0,
  ];
}

/** packing.slang packRgb9e5 for values exact at exponent 15. */
const rgb9e5 = (r: number, g: number, b: number) =>
  (Math.round(r * 512) |
    (Math.round(g * 512) << 9) |
    (Math.round(b * 512) << 18) |
    (15 << 27)) >>>
  0;

interface Material {
  name: string;
  albedo: [number, number, number];
  pbr: number;
  lobes?: number[];
  emission?: number;
  normal?: boolean;
}

const MATERIALS: Material[] = [
  { name: "matte", albedo: [0.6, 0.5, 0.4], pbr: pbrWord(0, 1) },
  { name: "polish", albedo: [0.1, 0.3, 0.6], pbr: pbrWord(0, 0.25) },
  { name: "metal", albedo: [0.9, 0.7, 0.3], pbr: pbrWord(1, 0.15) },
  {
    name: "schlick metal",
    albedo: [0.05, 0.06, 0.05],
    pbr: pbrWord(1, 0.34, 0, false, true),
  },
  {
    name: "lacquer",
    albedo: [0.047, 0.06, 0.047],
    pbr: pbrWord(1, 0.3421),
    lobes: lobesWords({
      coatWeight: 1,
      coatRoughness: 0.02,
      coatIor: 1.45,
      coatDarkening: true,
    }),
  },
  {
    name: "sheen",
    albedo: [0.3, 0.1, 0.1],
    pbr: pbrWord(0, 0.8),
    lobes: lobesWords({ sheenColour: [0.8, 0.8, 0.9], sheenRoughness: 0.4 }),
  },
  {
    name: "tinted specular",
    albedo: [0.4, 0.4, 0.4],
    pbr: pbrWord(0, 0.4),
    lobes: lobesWords({
      specularWeight: 0.6,
      specularColour: [1, 0.8, 0.5],
      specularIor: 1.8,
    }),
  },
  { name: "glass", albedo: [0.9, 0.95, 0.9], pbr: pbrWord(0, 0.05, 1) },
  {
    name: "thin sheet",
    albedo: [0.8, 0.9, 1],
    pbr: pbrWord(0, 0.02, 1, true),
  },
  {
    name: "emitter",
    albedo: [0.2, 0.2, 0.2],
    pbr: pbrWord(0, 0.7),
    emission: rgb9e5(2, 1.5, 0.5),
  },
  {
    name: "stored normal",
    albedo: [0.5, 0.6, 0.5],
    pbr: pbrWord(0, 0.5),
    normal: true,
  },
  { name: "mirror", albedo: [0.7, 0.7, 0.7], pbr: pbrWord(0, 0) },
];

/** A deterministic sequence in [0, 1). */
function random(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

const Y = (k: number, d: THREE.Vector3) => {
  const { x, y, z } = d;
  switch (k) {
    case 0:
      return 0.28209479177387814;
    case 1:
      return -0.4886025119029199 * y;
    case 2:
      return 0.4886025119029199 * z;
    case 3:
      return -0.4886025119029199 * x;
    case 4:
      return 1.0925484305920792 * x * y;
    case 5:
      return -1.0925484305920792 * y * z;
    case 6:
      return 0.31539156525252005 * (2 * z * z - x * x - y * y);
    case 7:
      return -1.0925484305920792 * x * z;
    case 8:
      return 0.5462742152960396 * (x * x - y * y);
    default:
      return 0;
  }
};

/** A sky with a horizon, a ground and a sun, top row first. */
function sky(w: number, h: number): SkyImage {
  const data = new Float32Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    const up = 1 - (y + 0.5) / h; // 1 at the top
    for (let x = 0; x < w; x++) {
      const sun = Math.abs(x - 0.3 * w) <= 1 && Math.abs(y - 0.25 * h) <= 1;
      const v =
        up > 0.5
          ? [0.3 + 0.5 * up, 0.5 + 0.4 * up, 0.9 * up + 0.2]
          : [0.25, 0.2, 0.15].map((c) => c * (0.5 + up));
      const s = sun ? 4000 : 1;
      data.set([v[0] * s, v[1] * s, v[2] * s, 1], (y * w + x) * 4);
    }
  }
  return { width: w, height: h, data };
}

describe.skipIf(!wideDevice)("athenea relight plugin", () => {
  const d = wideDevice as GPUDevice;
  const registry = new KernelRegistry(d);

  // The cloud: one splat a material, a TX transfer each.
  const N = MATERIALS.length;
  const TRANSFER = 112;
  const CELLS = 8;
  const rand = random(7);
  const a = new Uint32Array(N * 4);
  const b = new Uint32Array(N * 4);
  const transfer: number[] = [];
  const cells: number[] = [];
  MATERIALS.forEach((m, i) => {
    const q = new THREE.Quaternion()
      .setFromEuler(
        new THREE.Euler(rand() * 6.28, rand() * 6.28, rand() * 6.28),
      )
      .normalize();
    // The glass shows the eye its near face (a far face passes nothing on).
    if (m.name === "glass")
      q.multiply(new THREE.Quaternion().setFromAxisAngle(XAXIS, Math.PI));
    const x = (i % 4) * 0.6 - 0.9;
    const y = Math.floor(i / 4) * 0.6 - 0.6;
    encodeExtSplat(
      [a, b],
      i,
      x,
      y,
      0,
      0.2,
      0.15,
      0.01,
      q.x,
      q.y,
      q.z,
      q.w,
      0.9,
      ...m.albedo,
    );
    // The direct half: the clamped cosine about the splat's face, a bit
    // occluded; the indirect half a tinted share of it; a field.
    const n = new THREE.Vector3(0, 0, 1).applyQuaternion(q);
    const band = [1, 2 / 3, 2 / 3, 2 / 3, 0.25, 0.25, 0.25, 0.25, 0.25];
    const open = 0.6 + 0.4 * rand();
    for (let k = 0; k < 16; k++)
      transfer.push(k < 9 ? open * band[k] * Y(k, n) : 0.01 * (rand() - 0.5));
    for (let k = 0; k < 16; k++)
      for (const t of [0.3, 0.25, 0.2])
        transfer.push(t * transfer[i * TRANSFER + k] * (0.8 + 0.4 * rand()));
    // The glass's field reads open (a white sky through it): where a lens
    // shows the sky it bends (splat_relight's `open`).
    const fieldDc = m.name === "glass" ? 4 : 0;
    for (let k = 0; k < 48; k++)
      transfer.push(
        k < 3 ? fieldDc + 0.3 + 0.2 * rand() : 0.05 * (rand() - 0.5),
      );
    for (let w = 0; w < CELLS; w++)
      cells.push(((rand() * 4294967296) >>> 0) | 0x0f0f0f0f);
  });

  function source(curvature = 0) {
    const s = GpuSplatSource.fromExt(d, a, b, N);
    if (curvature)
      s.setAttribute(
        "curvature",
        MATERIALS.flatMap(() => [curvature, 0, curvature]),
        "f16",
        3,
      );
    s.setAttribute(
      "pbr",
      MATERIALS.map((m) => (m.pbr === NONE ? 0 : m.pbr)),
      "u32",
      1,
    );
    s.setAttribute(
      "lobes",
      MATERIALS.flatMap((m) => m.lobes ?? lobesWords({})),
      "u32",
      3,
    );
    s.setAttribute(
      "emission",
      MATERIALS.map((m) => m.emission ?? 0),
      "u32",
      1,
    );
    s.setAttribute(
      "normalOct",
      MATERIALS.map((m) => (m.normal ? packNormal([0.3, 0.2, 0.93]) : 0)),
      "u32",
      1,
    );
    s.setAttribute("transfer", transfer, "f16", TRANSFER);
    s.setAttribute("shadowBits", cells, "u32", CELLS);
    return s;
  }

  const object = new THREE.Object3D();
  object.position.set(0.4, -0.3, -0.5);
  object.rotation.set(0.3, 0.7, -0.2);
  object.scale.setScalar(1.3);
  object.updateMatrixWorld();

  const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 100);
  camera.position.set(0.3, 0.8, 4);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();

  const lights: AtheneaLightRecord[] = [
    {
      kind: "sphere",
      flags: LIGHT_SHADOW,
      sizeX: 0.1,
      colour: [40, 30, 20],
      matrix: new THREE.Matrix4().makeTranslation(1, 2, 1),
    },
  ];
  const sun = {
    direction: [0.3, 0.8, 0.5] as [number, number, number],
    intensity: 3,
  };

  async function read(buffer: GPUBuffer) {
    const staging = createReadback(d, buffer.size, "relight readback");
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
    d.queue.submit([enc.finish()]);
    return readAndDestroy(staging);
  }

  /** athenea's colour of every splat (slang/tests/athenea_relight.slang). */
  async function reference(
    sky: AtheneaSky,
    src: GPUBuffer,
    pbrMissing: boolean,
    indirect: boolean,
    curvature = 0,
  ) {
    // The streams one after the other, as athenea keeps them a buffer each.
    const words = new Uint32Array(N * (1 + 3 + 1 + 1 + TRANSFER / 2 + CELLS));
    const pool = source().attribs;
    const at = {
      pbr: 0,
      lobes: N,
      normals: 4 * N,
      emission: 5 * N,
      transfer: 6 * N,
      cells: 6 * N + (N * TRANSFER) / 2,
    };
    const { words: packed, layout } = (pool as NonNullable<typeof pool>).pack();
    const header = packed[2];
    const stride = packed[0];
    const offset = (name: string) => {
      const k = layout.specs.findIndex((s) => s.name === name);
      return packed[4 + 4 * k];
    };
    for (let i = 0; i < N; i++) {
      const base = header + i * stride;
      words[at.pbr + i] = packed[base + offset("pbr")];
      for (let k = 0; k < 3; k++)
        words[at.lobes + 3 * i + k] = packed[base + offset("lobes") + k];
      words[at.normals + i] = packed[base + offset("normalOct")];
      words[at.emission + i] = packed[base + offset("emission")];
      for (let k = 0; k < TRANSFER / 2; k++)
        words[at.transfer + (i * TRANSFER) / 2 + k] =
          packed[base + offset("transfer") + k];
      for (let k = 0; k < CELLS; k++)
        words[at.cells + i * CELLS + k] =
          packed[base + offset("shadowBits") + k];
    }
    const native = upload(d, words, "native streams");
    const records = packLightRecords([
      sky.record(),
      {
        kind: "distant",
        flags: 2 | LIGHT_SHADOW,
        sizeX: (0.53 * Math.PI) / 180,
        colour: [3, 3, 3],
        matrix: (await import("../../src/webgpu/athenea/lights")).distantMatrix(
          new THREE.Vector3(...sun.direction),
        ),
      },
      ...lights,
    ]);
    const lightsBuffer = upload(d, records, "ref lights");
    const envOfLight = upload(
      d,
      new Uint32Array([0, NONE, NONE]),
      "ref envOfLight",
    );
    const out = createStorage(d, N * 16, "ref out");
    const e = object.matrixWorld.elements;
    const row = (k: number) => [e[k], e[k + 4], e[k + 8], e[k + 12]];
    const eye = new THREE.Vector3().setFromMatrixPosition(camera.matrixWorld);
    const eyeCloud = eye
      .clone()
      .applyMatrix4(object.matrixWorld.clone().invert());
    registry.get(refModule, "referenceRelight").run({
      grid: [N],
      buffers: {
        src,
        native,
        lights: lightsBuffer,
        envSh: sky.envSh,
        envSun: sky.envSun,
        envTexels: sky.envTexels as GPUBuffer,
        envOfLight,
        outColour: out,
        iesRecords: createStorage(d, 32, "ies records"),
        iesValues: createStorage(d, 16, "ies values"),
      },
      uniforms: UniformWriter.for(refModule).setAll({
        count: N,
        transferCount: TRANSFER,
        transferWords: TRANSFER / 2,
        shadowBits: CELLS,
        pbrAt: pbrMissing ? NONE : at.pbr,
        lobesAt: at.lobes,
        normalsAt: at.normals,
        emissionAt: at.emission,
        transferAt: at.transfer,
        cellsAt: at.cells,
        lightCount: 3,
        envLights: 1,
        envBaseSide: sky.baseSide,
        transferIndirect: indirect ? 1 : 0,
        litBody: 0,
        linearCloud: 1,
        ior: 1.5,
        curvature,
        w0: row(0),
        w1: row(1),
        w2: row(2),
        eyeCloud: [eyeCloud.x, eyeCloud.y, eyeCloud.z, 1],
        eyeWorld: [eye.x, eye.y, eye.z, 1],
      }).data,
    });
    const got = new Float32Array(await read(out));
    native.destroy();
    lightsBuffer.destroy();
    envOfLight.destroy();
    out.destroy();
    return got;
  }

  const color = d.createTexture({
    size: [32, 32],
    format: "rgba16float",
    usage:
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.TEXTURE_BINDING,
  });
  const target = {
    texture: { colorSpace: THREE.LinearSRGBColorSpace },
    samples: 0,
  } as unknown as THREE.RenderTarget;
  const fakeRenderer = {
    backend: {
      isWebGPUBackend: true,
      device: d,
      context: {} as GPUCanvasContext,
      get: (r: object) =>
        r === target.texture ? { texture: color } : undefined,
    },
  };

  it("relights every material as athenea's relitSplat does", async () => {
    const splats = new WgpuSplatRenderer(fakeRenderer as never, {
      depthTest: false,
      alwaysGenerate: true,
    });
    const src = source();
    const mesh = splats.add(src, object);
    const host = new PluginHost({ capabilities: splats.capabilities, tier: 2 });
    const relight = atheneaRelightPlugin({
      hdri: sky(256, 128),
      rotation: 0.6,
      intensity: 1.5,
      sun,
      lights,
      ior: 1.5,
    });
    relight.setStoredLinear(mesh, true);
    host.register(relight).attach(splats);
    await host.ready();
    splats.render(camera, target);
    await d.queue.onSubmittedWorkDone();
    expect(relight.stats.viewless).toBe(1);
    expect(relight.stats.relit).toBe(1);
    const relit = relight.buffers?.("splat", { frame: null, mesh })
      .atheneaRelit as GPUBuffer;
    const got = new Float32Array(await read(relit));
    const want = await reference(
      relight.sky as AtheneaSky,
      src.src,
      false,
      true,
    );
    let worst = 0;
    MATERIALS.forEach((m, i) => {
      const g = [0, 1, 2].map((c) => got[20 * i + c]);
      const w = [0, 1, 2].map((c) => want[4 * i + c]);
      const err = Math.max(
        ...g.map((v, c) => Math.abs(v - w[c]) / Math.max(Math.abs(w[c]), 0.05)),
      );
      worst = Math.max(worst, err);
      if (process.env.ATHENEA_LOG)
        process.stderr.write(
          `${m.name.padEnd(16)} plugin ${g.map((v) => v.toFixed(4))} athenea ${w.map((v) => v.toFixed(4))} ${err.toExponential(2)}\n`,
        );
      expect(w.every(Number.isFinite)).toBe(true);
      expect(Math.max(...w)).toBeGreaterThan(0);
    });
    // The kept terms are halves (athenea's txCache): 2^-11 relative.
    expect(worst).toBeLessThan(2e-3);
    // The thin sheet's alpha floor; everything else keeps its own.
    const alpha = MATERIALS.map((_, i) => got[20 * i + 3]);
    MATERIALS.forEach((m, i) => {
      if (m.name === "thin sheet")
        expect(alpha[i]).toBeCloseTo(Math.max(0.9 - 1 / 255, 0.1), 3);
      else expect(alpha[i]).toBe(-1);
    });
    host.detach();
    splats.dispose();
  }, 120_000);
  it("reads a solid glass's far face from the cloud's curvature (a lens)", async () => {
    const K = 20; // a ball of 5 cm, exact in f16
    const splats = new WgpuSplatRenderer(fakeRenderer as never, {
      depthTest: false,
      alwaysGenerate: true,
    });
    const src = source(K);
    const mesh = splats.add(src, object);
    const host = new PluginHost({ capabilities: splats.capabilities, tier: 2 });
    const relight = atheneaRelightPlugin({
      hdri: sky(256, 128),
      rotation: 0.6,
      intensity: 1.5,
      sun,
      lights,
      ior: 1.5,
      // The test also reads the per-pixel slope and sharp records.
      pixelDetail: true,
      // athenea shades the first frame's splats at their centres (no
      // footprint prefilter; relightFootprint.test.ts).
      footprint: 0,
    });
    relight.setStoredLinear(mesh, true);
    host.register(relight).attach(splats);
    await host.ready();
    // The first frame: no target drawn yet, so the lens alone (no slope).
    splats.render(camera, target);
    await d.queue.onSubmittedWorkDone();
    const relit = relight.buffers?.("splat", { frame: null, mesh })
      .atheneaRelit as GPUBuffer;
    const got = new Float32Array(await read(relit));
    const sky0 = relight.sky as AtheneaSky;
    const lens = await reference(sky0, src.src, false, true, K);
    const flat = await reference(sky0, src.src, false, true, 0);
    let worst = 0;
    MATERIALS.forEach((m, i) => {
      const g = [0, 1, 2].map((c) => got[20 * i + c]);
      const w = [0, 1, 2].map((c) => lens[4 * i + c]);
      worst = Math.max(
        worst,
        ...g.map((v, c) => Math.abs(v - w[c]) / Math.max(Math.abs(w[c]), 0.05)),
      );
    });
    expect(worst).toBeLessThan(2e-3);
    // The glass is where the curvature shows: through both faces.
    const glass = MATERIALS.findIndex((m) => m.name === "glass");
    const change = [0, 1, 2].map((c) =>
      Math.abs(lens[4 * glass + c] - flat[4 * glass + c]),
    );
    expect(Math.max(...change)).toBeGreaterThan(1e-3);
    // The second frame knows the target: every TX splat with a curvature
    // hands the draw a slope (1) or a sharp lobe (2) about its centre.
    camera.position.x += 1e-3;
    camera.updateMatrixWorld();
    splats.render(camera, target);
    await d.queue.onSubmittedWorkDone();
    const words = new Uint32Array(await read(relit));
    const half = (w: number) => THREE.DataUtils.fromHalfFloat(w >>> 16);
    const marks = MATERIALS.map((_, i) => half(words[20 * i + 4 * 4 + 3]));
    expect(marks.every((v) => v === 1 || v === 2)).toBe(true);
    expect(marks[MATERIALS.findIndex((m) => m.name === "mirror")]).toBe(2);
    expect(marks[MATERIALS.findIndex((m) => m.name === "matte")]).toBe(1);
    // Centres in pixels inside the 32 x 32 target.
    MATERIALS.forEach((_, i) => {
      const f = new Float32Array(words.buffer, (20 * i + 16) * 4, 2);
      expect(f[0]).toBeGreaterThan(-32);
      expect(f[0]).toBeLessThan(64);
    });
    host.detach();
    splats.dispose();
  }, 120_000);

  it("keeps the captured colour where the transfer's page is not resident", async () => {
    const splats = new WgpuSplatRenderer(fakeRenderer as never, {
      depthTest: false,
      alwaysGenerate: true,
    });
    const src = source(20);
    // A paged pool whose relight streams have no page in: the transfer's
    // descriptor points at a page table of one empty slot (ATTRIB_PAGED).
    const pool = src.attribs as NonNullable<typeof src.attribs>;
    const { words, layout } = pool.pack();
    const grown = new Uint32Array(words.length + 1);
    grown.set(words);
    grown[words.length] = NONE;
    for (const name of ["transfer", "shadowBits", "curvature"]) {
      const k = layout.specs.findIndex((spec) => spec.name === name);
      grown[4 + 4 * k] = words.length;
      grown[4 + 4 * k + 3] |= 2;
    }
    pool.gpuBuffer = upload(d, grown, "paged relight pool");
    const mesh = splats.add(src, object);
    const host = new PluginHost({ capabilities: splats.capabilities, tier: 2 });
    const relight = atheneaRelightPlugin({ hdri: sky(64, 32), sun, ior: 1.5 });
    relight.setStoredLinear(mesh, true);
    host.register(relight).attach(splats);
    await host.ready();
    splats.render(camera, target);
    await d.queue.onSubmittedWorkDone();
    const relit = relight.buffers?.("splat", { frame: null, mesh })
      .atheneaRelit as GPUBuffer;
    const got = new Float32Array(await read(relit));
    MATERIALS.forEach((_, i) => {
      expect(got[20 * i + 3]).toBe(-2);
      // Nothing for the draw to shade per pixel.
      expect([...got.slice(20 * i + 16, 20 * i + 20)]).toEqual([0, 0, 0, 0]);
    });
    host.detach();
    splats.dispose();
    pool.gpuBuffer.destroy();
  }, 120_000);

  it("keeps the view-independent terms until the sky moves, and draws the relit light", async () => {
    // One large splat facing the camera: at the centre pixel its alpha is
    // its opacity, so the pixel is its relit colour times that.
    const qa = new Uint32Array(4);
    const qb = new Uint32Array(4);
    encodeExtSplat(
      [qa, qb],
      0,
      0,
      0,
      0,
      3,
      3,
      0.01,
      0,
      0,
      0,
      1,
      0.6,
      0.5,
      0.4,
      0.3,
    );
    const one = GpuSplatSource.fromExt(d, qa, qb, 1);
    one.setAttribute("pbr", [pbrWord(0, 0.4)], "u32", 1);
    one.setAttribute("transfer", transfer.slice(0, TRANSFER), "f16", TRANSFER);
    one.setAttribute("shadowBits", cells.slice(0, CELLS), "u32", CELLS);
    const splats = new WgpuSplatRenderer(fakeRenderer as never, {
      depthTest: false,
      alwaysGenerate: true,
    });
    const mesh = splats.add(one);
    const host = new PluginHost({ capabilities: splats.capabilities, tier: 2 });
    const relight = atheneaRelightPlugin({ hdri: sky(128, 64), sun });
    relight.setStoredLinear(mesh, true);
    host.register(relight).attach(splats);
    await host.ready();
    const view = new THREE.PerspectiveCamera(50, 1, 0.05, 100);
    view.coordinateSystem = THREE.WebGPUCoordinateSystem;
    view.position.set(0, 0, 3);
    view.updateProjectionMatrix();
    view.updateMatrixWorld();
    const frame = async () => {
      const enc = d.createCommandEncoder();
      enc
        .beginRenderPass({
          colorAttachments: [
            {
              view: color.createView(),
              loadOp: "clear",
              storeOp: "store",
              clearValue: [0, 0, 0, 1],
            },
          ],
        })
        .end();
      d.queue.submit([enc.finish()]);
      splats.render(view, target);
      await d.queue.onSubmittedWorkDone();
    };
    await frame();
    await frame();
    // Nothing moved: no pass ran again.
    expect([
      relight.stats.viewless,
      relight.stats.relit,
      relight.stats.skies,
    ]).toEqual([1, 1, 1]);
    const relit = relight.buffers?.("splat", { frame: null, mesh })
      .atheneaRelit as GPUBuffer;
    const c = new Float32Array(await read(relit));
    const bytesPerRow = 256;
    const buf = d.createBuffer({
      size: bytesPerRow * 32,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = d.createCommandEncoder();
    enc.copyTextureToBuffer(
      { texture: color },
      { buffer: buf, bytesPerRow },
      [32, 32],
    );
    d.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const halves = new Uint16Array(buf.getMappedRange());
    const at = (16 * bytesPerRow) / 2 + 16 * 4;
    const px = [0, 1, 2].map((k) =>
      THREE.DataUtils.fromHalfFloat(halves[at + k]),
    );
    buf.unmap();
    buf.destroy();
    const want = [0, 1, 2].map((k) => 0.6 * c[k]);
    expect(
      Math.max(
        ...px.map((v, k) => Math.abs(v - want[k]) / Math.max(want[k], 0.05)),
      ),
    ).toBeLessThan(3e-3);
    // The eye moves: the colours again, the kept terms not.
    view.position.set(0.5, 0.2, 3);
    view.updateMatrixWorld();
    await frame();
    expect([relight.stats.viewless, relight.stats.relit]).toEqual([1, 2]);
    // The sky turns: prepared again, the kept terms again.
    relight.set({ rotation: 1 });
    await frame();
    expect([
      relight.stats.viewless,
      relight.stats.relit,
      relight.stats.skies,
    ]).toEqual([2, 3, 2]);
    host.detach();
    splats.dispose();
  }, 120_000);
  it("culls the opaque splats whose stored face looks away (cullBacks)", async () => {
    const splats = new WgpuSplatRenderer(fakeRenderer as never, {
      depthTest: false,
      alwaysGenerate: true,
    });
    const src = source();
    const mesh = splats.add(src, object);
    const host = new PluginHost({ capabilities: splats.capabilities, tier: 2 });
    const relight = atheneaRelightPlugin({
      hdri: sky(256, 128),
      sun,
      lights,
      ior: 1.5,
      cullBacks: true,
    });
    relight.setStoredLinear(mesh, true);
    host.register(relight).attach(splats);
    await host.ready();
    const relit = () =>
      relight.buffers?.("splat", { frame: null, mesh })
        .atheneaRelit as GPUBuffer;
    const stored = new THREE.Vector3(0.3, 0.2, 0.93).normalize();
    const toObject = object.matrixWorld.clone().invert();
    let culled = 0;
    let kept = 0;
    // From in front and from behind: every splat shows the eye one face.
    for (const side of [1, -1]) {
      const eye = new THREE.PerspectiveCamera(50, 1, 0.05, 100);
      eye.position.set(0.3 * side, 0.8 * side, 4 * side);
      eye.lookAt(0, 0, 0);
      eye.updateMatrixWorld();
      relight.set({ cullBacks: true });
      splats.render(eye, target);
      await d.queue.onSubmittedWorkDone();
      const on = new Float32Array(await read(relit()));
      relight.set({ cullBacks: false });
      splats.render(eye, target);
      await d.queue.onSubmittedWorkDone();
      const off = new Float32Array(await read(relit()));
      const eyeObject = eye.position.clone().applyMatrix4(toObject);
      MATERIALS.forEach((m, i) => {
        const centre = new THREE.Vector3(
          (i % 4) * 0.6 - 0.9,
          Math.floor(i / 4) * 0.6 - 0.6,
          0,
        );
        // The cloud keeps normalOct (word 0, (0, 0, -1), where a material
        // has none): the stored face, not the disc's axis, decides.
        const n = m.normal ? stored : new THREE.Vector3(0, 0, -1);
        const facing = n.dot(eyeObject.clone().sub(centre).normalize());
        if (Math.abs(facing + 0.2) < 0.02) return;
        const opaque = m.name !== "glass" && m.name !== "thin sheet";
        const at = 20 * i;
        if (opaque && facing < -0.2) {
          culled += 1;
          expect([...on.subarray(at, at + 4)]).toEqual([0, 0, 0, 0]);
        } else {
          kept += 1;
          expect([...on.subarray(at, at + 4)]).toEqual([
            ...off.subarray(at, at + 4),
          ]);
        }
      });
    }
    expect(culled).toBeGreaterThanOrEqual(4);
    expect(kept).toBeGreaterThanOrEqual(10);
    host.detach();
    splats.dispose();
  }, 120_000);

  it("draws a shadow catcher as athenea's catcherOpacity", async () => {
    // Ground splats facing +Y, each with its own open cells.
    const M = 6;
    const ga = new Uint32Array(M * 4);
    const gb = new Uint32Array(M * 4);
    const q = new THREE.Quaternion().setFromAxisAngle(
      new THREE.Vector3(1, 0, 0),
      -Math.PI / 2,
    );
    const ground: number[] = [];
    const r2 = random(11);
    for (let i = 0; i < M; i++) {
      encodeExtSplat(
        [ga, gb],
        i,
        i * 0.3,
        -1,
        0,
        0.2,
        0.2,
        0.01,
        q.x,
        q.y,
        q.z,
        q.w,
        0.9,
        0.5,
        0.5,
        0.5,
      );
      for (let w = 0; w < CELLS; w++) {
        ground.push(
          i === 0 ? 0xffffffff : i === 1 ? 0 : (r2() * 4294967296) >>> 0,
        );
      }
    }
    const catcher = GpuSplatSource.fromExt(d, ga, gb, M);
    catcher.setAttribute("transfer", new Array(M * 16).fill(0), "f16", 16);
    catcher.setAttribute("shadowBits", ground, "u32", CELLS);
    const splats = new WgpuSplatRenderer(fakeRenderer as never, {
      depthTest: false,
      alwaysGenerate: true,
    });
    const mesh = splats.add(catcher);
    const host = new PluginHost({ capabilities: splats.capabilities, tier: 2 });
    const relight = atheneaRelightPlugin({
      hdri: sky(256, 128),
      rotation: 0.3,
    });
    relight.setCatcher(mesh, true);
    host.register(relight).attach(splats);
    await host.ready();
    splats.render(camera, target);
    await d.queue.onSubmittedWorkDone();
    const relit = relight.buffers?.("splat", { frame: null, mesh })
      .atheneaRelit as GPUBuffer;
    const got = new Float32Array(await read(relit));
    const { sh, sun: disc } = await (relight.sky as AtheneaSky).read();
    // splat_project's catcherOpacity, in JS.
    const octDecode = (u: number, v: number) => {
      const fx = u * 2 - 1;
      const fy = v * 2 - 1;
      const n = new THREE.Vector3(fx, fy, 1 - Math.abs(fx) - Math.abs(fy));
      const t = Math.min(Math.max(-n.z, 0), 1);
      n.x += n.x >= 0 ? -t : t;
      n.y += n.y >= 0 ? -t : t;
      return n.normalize();
    };
    const octEncode = (n: THREE.Vector3) => {
      const l1 = Math.abs(n.x) + Math.abs(n.y) + Math.abs(n.z);
      let p = [n.x / l1, n.y / l1];
      if (n.z / l1 < 0)
        p = [
          (1 - Math.abs(p[1])) * (p[0] >= 0 ? 1 : -1),
          (1 - Math.abs(p[0])) * (p[1] >= 0 ? 1 : -1),
        ];
      return p.map((c) => c * 0.5 + 0.5);
    };
    const Y16 = (k: number, v: THREE.Vector3) => {
      const { x, y, z } = v;
      const c = [
        -0.5900435899266435 * y * (3 * x * x - y * y),
        2.890611442640554 * x * y * z,
        -0.4570457994644658 * y * (4 * z * z - x * x - y * y),
        0.3731763325901154 * z * (2 * z * z - 3 * x * x - 3 * y * y),
        -0.4570457994644658 * x * (4 * z * z - x * x - y * y),
        1.445305721320277 * z * (x * x - y * y),
        -0.5900435899266435 * x * (x * x - 3 * y * y),
      ];
      return k < 9 ? Y(k, v) : c[k - 9];
    };
    const luma = (r: number, g: number, b: number) =>
      0.2126 * r + 0.7152 * g + 0.0722 * b;
    const skyL = Array.from({ length: 16 }, (_, k) =>
      luma(sh[4 * k], sh[4 * k + 1], sh[4 * k + 2]),
    );
    const n = new THREE.Vector3(0, 0, 1).applyQuaternion(q);
    const side = 16;
    for (let i = 0; i < M; i++) {
      const bit = (cell: number) =>
        (ground[i * CELLS + (cell >> 5)] >>> (cell & 31)) & 1;
      let withO = 0;
      let without = 0;
      for (let cell = 0; cell < side * side; cell++) {
        const dir = octDecode(
          ((cell % side) + 0.5) / side,
          (Math.floor(cell / side) + 0.5) / side,
        );
        const cosine = n.dot(dir);
        if (!(cosine > 0)) continue;
        let radiance = 0;
        for (let k = 0; k < 16; k++) radiance += skyL[k] * Y16(k, dir);
        const l1 = Math.abs(dir.x) + Math.abs(dir.y) + Math.abs(dir.z);
        const worth =
          Math.max(radiance, 0) * cosine * (4 / (side * side)) * l1 ** 3;
        without += worth;
        if (bit(cell)) withO += worth;
      }
      if (disc[3] > 0) {
        const s = new THREE.Vector3(disc[0], disc[1], disc[2]);
        const sunL = luma(disc[4], disc[5], disc[6]) * Math.max(n.dot(s), 0);
        const at = octEncode(s).map((c) => c * side - 0.5);
        const lo = at.map(Math.floor);
        const f = [at[0] - lo[0], at[1] - lo[1]];
        let open = 0;
        let weight = 0;
        for (let k = 0; k < 4; k++) {
          const cx = Math.min(Math.max(lo[0] + (k & 1), 0), side - 1);
          const cy = Math.min(Math.max(lo[1] + (k >> 1), 0), side - 1);
          const wk = (k & 1 ? f[0] : 1 - f[0]) * (k >> 1 ? f[1] : 1 - f[1]);
          if (
            wk <= 0 ||
            octDecode((cx + 0.5) / side, (cy + 0.5) / side).dot(n) <= 0
          )
            continue;
          open += wk * bit(cy * side + cx);
          weight += wk;
        }
        withO += sunL * (weight > 0 ? open / weight : 1);
        without += sunL;
      }
      const ratio = Math.min(Math.max(withO / Math.max(without, 1e-6), 0), 1);
      const want =
        0.9 *
        Math.min(Math.max(-Math.log(Math.max(ratio, 1e-4)) / 6.2831853, 0), 1);
      expect(got.slice(20 * i, 20 * i + 3)).toEqual(new Float32Array(3));
      expect(Math.abs(got[20 * i + 3] - want)).toBeLessThan(2e-3);
    }
    // All open: nothing taken; all closed: as dark as the layer gets.
    expect(got[3]).toBeLessThan(1e-3);
    expect(got[20 + 3]).toBeCloseTo(0.9, 2);
    host.detach();
    splats.dispose();
  }, 120_000);
});

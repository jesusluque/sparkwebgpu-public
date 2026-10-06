// athenea's material layers (packing.slang `SplatLobes`) from usd-athc to the
// relit colour: test/fixtures/athc/lobes_sphere.athc (written by
// rust/spark-lib athc_build's writes_the_lobes_sphere_fixture) is a ball of
// 1200 splats of the Corvette's paint -- a Schlick metal (metallic 1,
// roughness 0.342) under a clear coat (weight 1, roughness 0.047, index
// 1.45, darkening) where x > 0, the same base bare where not -- with a direct
// transfer (16) and every way out open.
//
// The .athc goes through the WASM decoder into an attribute pool; the lobes
// and pbr words must be packLobes' and packPbr's (Schlick bit included), and
// the relight plugin's colour must be athenea's relitSplat on the same
// streams (slang/tests/athenea_relight.slang), where the coat changes it and
// the bare half does not. Everything is compared in linear float, before
// any display transform; the sky is a float HDR image.

import { readFileSync } from "node:fs";
import * as THREE from "three";
import { describe, expect, it, vi } from "vitest";
import { unpackAthcNormal, unpackAthcPbr } from "../../src/athc";
import type { AttribValues } from "../../src/defines";
import { encodeExtSplat, unpackSplat } from "../../src/utils";
import { KernelRegistry } from "../../src/webgpu/KernelRegistry";
import {
  GpuSplatSource,
  WgpuSplatRenderer,
} from "../../src/webgpu/WgpuSplatRenderer";
import type { AtheneaSky, SkyImage } from "../../src/webgpu/athenea/AtheneaSky";
import {
  LIGHT_SHADOW,
  distantMatrix,
  packLightRecords,
} from "../../src/webgpu/athenea/lights";
import { atheneaRelightPlugin } from "../../src/webgpu/athenea/relightPlugin";
import { AttribPool } from "../../src/webgpu/attributes/schema";
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

const wasm = await vi.importActual<typeof import("spark-rs")>("spark-rs");
wasm.initSync({
  module: readFileSync(
    new URL("../../rust/spark-rs/pkg/spark_rs_bg.wasm", import.meta.url),
  ),
});

const BYTES = new Uint8Array(
  readFileSync(new URL("../fixtures/athc/lobes_sphere.athc", import.meta.url)),
);
const NONE = 0xffffffff;

// packing.slang packLobes of the paint's layers, worked out by hand (the
// Rust test packs_lobes_as_athenea_does checks the same words).
const COATED = [0xffffffff, 0x409d0cff, 0x80000000];
const BARE = [0xffffffff, 0x409d0c00, 0x80000000];
// packPbr(1, 0.3421304, 0 + kSchlickMetalMark): bit 25, roughness 87.
const PAINT_PBR = 0x020057ff;

type Decoded = {
  numSplats: number;
  packed: Uint32Array;
  splatEncoding: {
    lodOpacity: boolean;
    rgbMin: number;
    rgbMax: number;
    lnScaleMin: number;
    lnScaleMax: number;
  };
  attribSpecs?: AttribValues["specs"];
  attribColumns?: Float64Array[];
};

function decode(bytes: Uint8Array): Decoded {
  const decoder = wasm.decode_to_packedsplats(
    undefined,
    "cloud.athc",
    undefined,
    undefined,
    undefined,
    undefined,
  );
  decoder.push(bytes);
  return decoder.finish() as Decoded;
}

/** The finest level's splats (after the merged nodes) as ext splats, and
 * their attribute pool. */
function finest(dec: Decoded) {
  const v = new DataView(BYTES.buffer, BYTES.byteOffset, BYTES.byteLength);
  expect(v.getUint32(0, true)).toBe(0x33485441); // "ATH3"
  const count = v.getUint32(12, true);
  const merged = dec.numSplats - count;
  const specs = dec.attribSpecs ?? [];
  const values = (dec.attribColumns ?? []).map((col, k) =>
    col.slice(merged * specs[k].components),
  );
  const pool = AttribPool.fromValues({ count, specs, values });
  const a = new Uint32Array(count * 4);
  const b = new Uint32Array(count * 4);
  const e = dec.splatEncoding;
  for (let i = 0; i < count; i++) {
    const s = unpackSplat(dec.packed, merged + i, e);
    encodeExtSplat(
      [a, b],
      i,
      s.center.x,
      s.center.y,
      s.center.z,
      s.scales.x,
      s.scales.y,
      s.scales.z,
      s.quaternion.x,
      s.quaternion.y,
      s.quaternion.z,
      s.quaternion.w,
      s.opacity,
      s.color.r,
      s.color.g,
      s.color.b,
    );
  }
  return { count, pool, a, b, specs };
}

/** A sky with a horizon, a ground and a sun, top row first (linear float). */
function sky(w: number, h: number): SkyImage {
  const data = new Float32Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    const up = 1 - (y + 0.5) / h;
    for (let x = 0; x < w; x++) {
      const sun = Math.abs(x - 0.3 * w) <= 1 && Math.abs(y - 0.25 * h) <= 1;
      // Bright stripes above the horizon: a coat mirrors them.
      const stripe = up > 0.5 && Math.floor((x / w) * 16) % 2 === 0 ? 3 : 1;
      const v =
        up > 0.5
          ? [0.3 + 0.5 * up, 0.5 + 0.4 * up, 0.9 * up + 0.2].map(
              (c) => c * stripe,
            )
          : [0.25, 0.2, 0.15].map((c) => c * (0.5 + up));
      const s = sun ? 4000 : 1;
      data.set([v[0] * s, v[1] * s, v[2] * s, 1], (y * w + x) * 4);
    }
  }
  return { width: w, height: h, data };
}

describe("lobes_sphere.athc (WASM)", () => {
  it("decodes packLobes' and packPbr's words for every splat", () => {
    const { count, pool, specs } = finest(decode(BYTES));
    expect(count).toBe(1200);
    expect(specs.map((s) => [s.name, s.format, s.components])).toContainEqual([
      "lobes",
      "u32",
      3,
    ]);
    let coated = 0;
    for (let i = 0; i < count; i++) {
      const n = unpackAthcNormal(pool.getAttribute("normalOct", i)[0]);
      expect(pool.getAttribute("pbr", i)[0]).toBe(PAINT_PBR);
      if (Math.abs(n[0]) < 1e-3) continue; // the seam, within a step
      const want = n[0] > 0 ? COATED : BARE;
      expect(pool.getAttribute("lobes", i)).toEqual(want);
      if (n[0] > 0) coated++;
    }
    expect(coated).toBeGreaterThan(500);
    expect(unpackAthcPbr(PAINT_PBR).metallic).toBe(1);
  });
});

describe.skipIf(!wideDevice)("athenea lobes on the GPU", () => {
  const d = wideDevice as GPUDevice;
  const registry = new KernelRegistry(d);
  const { count: N, pool, a, b } = finest(decode(BYTES));
  const TRANSFER = 16;
  const CELLS = 8;

  const object = new THREE.Object3D();
  object.updateMatrixWorld();
  const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 100);
  camera.position.set(0.3, 0.8, 4);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();
  const sun = {
    direction: [0.3, 0.8, 0.5] as [number, number, number],
    intensity: 3,
  };

  async function read(buffer: GPUBuffer) {
    const staging = createReadback(d, buffer.size, "lobes readback");
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
    d.queue.submit([enc.finish()]);
    return readAndDestroy(staging);
  }

  /** athenea's relitSplat on its own stream layout, the lobes as given. */
  async function reference(
    sky: AtheneaSky,
    src: GPUBuffer,
    lobes: (i: number) => number[],
    pbrMask = 0xffffffff,
  ) {
    const at = {
      pbr: 0,
      lobes: N,
      normals: 4 * N,
      emission: 5 * N,
      transfer: 6 * N,
      cells: 6 * N + (N * TRANSFER) / 2,
    };
    const words = new Uint32Array(at.cells + N * CELLS);
    const { words: packed, layout } = pool.pack();
    const header = packed[2];
    const stride = packed[0];
    const offset = (name: string) => {
      const k = layout.specs.findIndex((s) => s.name === name);
      return packed[4 + 4 * k];
    };
    for (let i = 0; i < N; i++) {
      const base = header + i * stride;
      words[at.pbr + i] = (packed[base + offset("pbr")] & pbrMask) >>> 0;
      words.set(lobes(i), at.lobes + 3 * i);
      words[at.normals + i] = packed[base + offset("normalOct")];
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
        matrix: distantMatrix(new THREE.Vector3(...sun.direction)),
      },
    ]);
    const lightsBuffer = upload(d, records, "ref lights");
    const envOfLight = upload(d, new Uint32Array([0, NONE]), "envOfLight");
    const out = createStorage(d, N * 16, "ref out");
    const e = object.matrixWorld.elements;
    const row = (k: number) => [e[k], e[k + 4], e[k + 8], e[k + 12]];
    const eye = new THREE.Vector3().setFromMatrixPosition(camera.matrixWorld);
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
        pbrAt: at.pbr,
        lobesAt: at.lobes,
        normalsAt: at.normals,
        emissionAt: NONE,
        transferAt: at.transfer,
        cellsAt: at.cells,
        lightCount: 2,
        envLights: 1,
        envBaseSide: sky.baseSide,
        transferIndirect: 0,
        litBody: 0,
        linearCloud: 1,
        ior: 1.5,
        curvature: 0,
        w0: row(0),
        w1: row(1),
        w2: row(2),
        eyeCloud: [eye.x, eye.y, eye.z, 1],
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

  it("relights the coat as athenea's relitSplat does, and the coat shows", async () => {
    const splats = new WgpuSplatRenderer(fakeRenderer as never, {
      depthTest: false,
      alwaysGenerate: true,
    });
    const src = GpuSplatSource.fromExt(d, a, b, N);
    src.attribs = pool;
    const mesh = splats.add(src, object);
    const host = new PluginHost({ capabilities: splats.capabilities, tier: 2 });
    const relight = atheneaRelightPlugin({
      hdri: sky(256, 128),
      rotation: 0.6,
      intensity: 1.5,
      sun,
      ior: 1.5,
    });
    relight.setStoredLinear(mesh, true);
    host.register(relight).attach(splats);
    await host.ready();
    splats.render(camera, target);
    await d.queue.onSubmittedWorkDone();
    const relit = relight.buffers?.("splat", { frame: null, mesh })
      .atheneaRelit as GPUBuffer;
    const got = new Float32Array(await read(relit));
    const s = relight.sky as AtheneaSky;
    const asStored = await reference(s, src.src, (i) =>
      pool.getAttribute("lobes", i),
    );
    const bare = await reference(s, src.src, () => BARE);
    // What the published Corvette had: neither the layers nor the Schlick
    // mark (a conductor of the same albedo and roughness).
    const before = await reference(s, src.src, () => BARE, ~(1 << 25));

    const eye = camera.position;
    let worst = 0;
    let coatShift = 0;
    let coatSeen = 0;
    let bareShift = 0;
    let pluginOffBare = 0;
    let beforeShift = 0;
    for (let i = 0; i < N; i++) {
      const g = [0, 1, 2].map((c) => got[20 * i + c]);
      const w = [0, 1, 2].map((c) => asStored[4 * i + c]);
      const o = [0, 1, 2].map((c) => bare[4 * i + c]);
      expect(w.every(Number.isFinite)).toBe(true);
      worst = Math.max(
        worst,
        ...g.map((v, c) => Math.abs(v - w[c]) / Math.max(Math.abs(w[c]), 0.05)),
      );
      const n = unpackAthcNormal(pool.getAttribute("normalOct", i)[0]);
      const p = pool.getAttribute("lobes", i);
      const facing =
        n[0] * eye.x + n[1] * eye.y + n[2] * eye.z > 0.5 * eye.length();
      const rel = (x: number[], y: number[]) =>
        Math.max(
          ...x.map(
            (v, c) => Math.abs(v - y[c]) / Math.max(Math.abs(y[c]), 0.02),
          ),
        );
      if (p[1] === COATED[1]) {
        if (facing) {
          coatShift += rel(w, o);
          pluginOffBare += rel(g, o);
          beforeShift += rel(
            w,
            [0, 1, 2].map((c) => before[4 * i + c]),
          );
          coatSeen++;
        }
      } else {
        bareShift = Math.max(bareShift, rel(w, o));
      }
    }
    if (process.env.ATHENEA_LOG)
      process.stderr.write(
        `lobes: worst ${worst.toExponential(2)}, coat changes athenea's colour by ${(coatShift / coatSeen).toFixed(3)} (plugin ${(pluginOffBare / coatSeen).toFixed(3)}) over ${coatSeen} facing coated splats; against no lobes and no Schlick mark ${(beforeShift / coatSeen).toFixed(3)}\n`,
      );
    // The plugin is athenea's relitSplat on the lobes it read (the kept
    // terms are halves: 2^-11 relative).
    expect(worst).toBeLessThan(2e-3);
    // The bare half is the same with the paint's bare words.
    expect(bareShift).toBe(0);
    // The coat changes the colour, in athenea and in the plugin alike.
    expect(coatSeen).toBeGreaterThan(50);
    expect(coatShift / coatSeen).toBeGreaterThan(0.05);
    expect(Math.abs(pluginOffBare - coatShift) / coatShift).toBeLessThan(0.01);
    expect(beforeShift / coatSeen).toBeGreaterThan(0.05);
    host.detach();
    splats.dispose();
  }, 120_000);
});

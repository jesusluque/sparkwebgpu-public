// The Corvette's clear coat on real data: a piece of the hood of athenea's
// Car_Paint_Main TX bake (usd-athc --drop-backs 0.008 --box, kept as
// test/fixtures/athc/hood_t16.athc), relit by the plugin as the Corvette
// page places it -- athenea's stage is Z-up, the page turns the car to
// three's Y-up -- against athenea's relitSplat in athenea's own stage (no
// turn, the dome turned as the stage's DomeLight is: rotateXYZ (90, 0, 90)).
//
// A transfer, its cells and its field are directions of the stage the bake
// ran in (splat_relight.slang reads them with world directions: only a
// zonal transfer has a frame of its own). Relit in three's world after the
// page's turn, every one of them is read 90 degrees off: the hood's mirror
// (up in the stage) asks the cells about a horizontal direction, which the
// car's own body closes, so the coat's lobe read closed and the paint came
// out matte. The plugin's `frame` relights in the stage's frame instead.
//
// Compared in linear float per splat, under the full-resolution autoshop_01
// HDRI when ATHENEA_HDRI points at it (athenea-renders' bake sky), a float
// sky with a bright ceiling otherwise.
//
// The fixture is 2897 splats of the hood's middle (usd-athc
// Car_Paint_Main_tx_m3.usdc --transfer 16 --drop-backs 0.008 --box
// -0.06,-1.48,0.6,0.06,-1.38,1.3 --gzip). ATHENEA_COAT_ATHC runs the same
// on another cut (a larger box, a full transfer). With autoshop_01 at 4k:
// athenea 0.338 mean luminance (its coat 32%), the plugin turned with the
// car 0.081.

import { existsSync, readFileSync } from "node:fs";
import * as THREE from "three";
import { HDRLoader } from "three/examples/jsm/loaders/HDRLoader.js";
import { describe, expect, it, vi } from "vitest";
import type { AttribValues } from "../../src/defines";
import { encodeExtSplat, unpackSplat } from "../../src/utils";
import { KernelRegistry } from "../../src/webgpu/KernelRegistry";
import {
  GpuSplatSource,
  WgpuSplatRenderer,
} from "../../src/webgpu/WgpuSplatRenderer";
import { AtheneaSky, type SkyImage } from "../../src/webgpu/athenea/AtheneaSky";
import { packLightRecords } from "../../src/webgpu/athenea/lights";
import { atheneaRelightPlugin } from "../../src/webgpu/athenea/relightPlugin";
import { AttribPool, attribWords } from "../../src/webgpu/attributes/schema";
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

const FIXTURE = new URL("../fixtures/athc/hood_t16.athc", import.meta.url);
const ATHC = process.env.ATHENEA_COAT_ATHC ?? FIXTURE.pathname;
const NONE = 0xffffffff;
const log = (s: string) => {
  if (process.env.ATHENEA_LOG) process.stderr.write(`${s}\n`);
};

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
  attribColumns?: AttribValues["values"];
};

/** The finest level's splats as ext splats, and their attribute pool. */
function finest(bytes: Uint8Array) {
  const decoder = wasm.decode_to_packedsplats(
    undefined,
    "cloud.athc",
    undefined,
    undefined,
    undefined,
    undefined,
  );
  decoder.push(bytes);
  const dec = decoder.finish() as Decoded;
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const count = v.getUint32(12, true);
  const merged = dec.numSplats - count;
  const specs = dec.attribSpecs ?? [];
  const values = (dec.attribColumns ?? []).map((col, k) =>
    // A .athc's streams come as their packed words (spec.packed).
    col.slice(
      merged *
        (specs[k].packed
          ? attribWords(specs[k] as never)
          : specs[k].components),
    ),
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
  const components = (name: string) =>
    specs.find((s) => s.name === name)?.components ?? 0;
  return {
    count,
    pool,
    a,
    b,
    transferCount: components("transfer"),
    cellWords: components("shadowBits"),
  };
}

/** A float sky: a bright ceiling of strip lights over a dim room. */
function syntheticSky(w: number, h: number): SkyImage {
  const data = new Float32Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    const up = 1 - (y + 0.5) / h;
    for (let x = 0; x < w; x++) {
      const strip = up > 0.8 && Math.floor((x / w) * 24) % 3 === 0 ? 20 : 1;
      const v = up > 0.5 ? 0.6 * strip : 0.15;
      data.set([v, v * 0.97, v * 0.92, 1], (y * w + x) * 4);
    }
  }
  return { width: w, height: h, data };
}

function loadSky(): { sky: SkyImage; name: string } {
  const path = process.env.ATHENEA_HDRI;
  if (path && existsSync(path)) {
    const buf = readFileSync(path);
    const loader = new HDRLoader().setDataType(THREE.FloatType);
    const t = loader.parse(
      buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
    ) as { width: number; height: number; data: Float32Array };
    return {
      sky: { width: t.width, height: t.height, data: t.data, channels: 4 },
      name: path.split("/").pop() as string,
    };
  }
  return { sky: syntheticSky(512, 256), name: "synthetic" };
}

describe("hood_t16.athc decoded (WASM)", () => {
  it("hands its streams over as their packed words", () => {
    const bytes = new Uint8Array(readFileSync(FIXTURE));
    const decoder = wasm.decode_to_packedsplats(
      undefined,
      "cloud.athc",
      undefined,
      undefined,
      undefined,
      undefined,
    );
    decoder.push(bytes);
    const dec = decoder.finish() as Decoded;
    const specs = dec.attribSpecs ?? [];
    const cols = dec.attribColumns ?? [];
    expect(specs.map((s) => [s.name, s.format, s.components])).toEqual([
      ["normalOct", "u32", 1],
      ["pbr", "u32", 1],
      ["lobes", "u32", 3],
      ["transfer", "f16", 16],
      ["shadowBits", "u32", 8],
      ["curvature", "f16", 3],
      ["athcGroup", "u32", 2],
    ]);
    // Words, not f64 values: a full transfer as f64 (896 bytes a splat) ran
    // the wasm32 heap out on the Corvette's paint ("unreachable").
    specs.forEach((s, k) => {
      expect(s.packed).toBe(true);
      expect(cols[k]).toBeInstanceOf(Uint32Array);
      expect(cols[k].length).toBe(dec.numSplats * attribWords(s as never));
    });
    // The pool reads them as values again: the paint's coat and pbr words.
    const pool = AttribPool.fromValues({
      count: dec.numSplats,
      specs,
      values: cols,
    });
    const last = dec.numSplats - 1;
    expect(pool.getAttribute("lobes", last)[0]).toBe(0xffffffff);
    expect(pool.getAttribute("lobes", last)[1] & 0xff).toBe(0xff);
    expect(pool.getAttribute("pbr", last)[0] >>> 24).toBe(2); // Schlick
    const t = pool.getAttribute("transfer", last);
    expect(t.every(Number.isFinite)).toBe(true);
    expect(t[0]).toBeGreaterThan(0);
  });
});

describe.skipIf(!wideDevice || !existsSync(ATHC))(
  "athenea's coat on the Corvette's hood",
  () => {
    const d = wideDevice as GPUDevice;
    const registry = new KernelRegistry(d);
    const bytes = new Uint8Array(readFileSync(ATHC));
    const { count: N, pool, a, b, transferCount, cellWords } = finest(bytes);
    const { sky: image, name: skyName } = loadSky();
    const domeTurn = Math.PI / 2; // corvette.json domeRotation 90

    // corvette.json's camera (athenea's /World/Camera, turned to Y-up).
    const camera = new THREE.PerspectiveCamera(
      (2 * Math.atan(Math.tan((39.6 * Math.PI) / 360) / (16 / 9)) * 180) /
        Math.PI,
      16 / 9,
      0.05,
      200,
    );
    camera.position.set(4.378, 1.676, 5.854);
    camera.lookAt(0, 0.553, 0.252);
    camera.updateMatrixWorld();

    async function read(buffer: GPUBuffer) {
      const staging = createReadback(d, buffer.size, "coat readback");
      const enc = d.createCommandEncoder();
      enc.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
      d.queue.submit([enc.finish()]);
      return readAndDestroy(staging);
    }

    const car = new THREE.Group();
    car.rotation.x = -Math.PI / 2;
    const object = new THREE.Object3D();
    car.add(object);
    car.updateMatrixWorld(true);
    /** three's world to athenea's stage. */
    const toStage = car.matrixWorld.clone().invert();
    const eyeStage = camera.position.clone().applyMatrix4(toStage);

    /** athenea's sky, in its own stage. */
    const stageSky = new AtheneaSky(d, registry);
    stageSky.set({ image, rotation: domeTurn, frame: toStage });
    // The page's dome is athenea's DomeLight rotateXYZ (90, 0, 90).
    const usd = new THREE.Matrix4()
      .makeRotationZ(Math.PI / 2)
      .multiply(new THREE.Matrix4().makeRotationX(Math.PI / 2));
    expect(
      stageSky.matrix.elements.map((v, k) => v - usd.elements[k]),
    ).toSatisfy((e: number[]) => e.every((x) => Math.abs(x) < 1e-6));

    /** athenea's relitSplat in its stage: no turn, the stage's eye. */
    async function reference(src: GPUBuffer, lobes: (i: number) => number[]) {
      const at = {
        pbr: 0,
        lobes: N,
        normals: 4 * N,
        transfer: 5 * N,
        cells: 5 * N + (N * transferCount) / 2,
      };
      const words = new Uint32Array(at.cells + N * cellWords);
      const { words: packed, layout } = pool.pack();
      const header = packed[2];
      const stride = packed[0];
      const offset = (name: string) => {
        const k = layout.specs.findIndex((s) => s.name === name);
        return packed[4 + 4 * k];
      };
      for (let i = 0; i < N; i++) {
        const base = header + i * stride;
        words[at.pbr + i] = packed[base + offset("pbr")];
        words.set(lobes(i), at.lobes + 3 * i);
        words[at.normals + i] = packed[base + offset("normalOct")];
        for (let k = 0; k < transferCount / 2; k++)
          words[at.transfer + (i * transferCount) / 2 + k] =
            packed[base + offset("transfer") + k];
        for (let k = 0; k < cellWords; k++)
          words[at.cells + i * cellWords + k] =
            packed[base + offset("shadowBits") + k];
      }
      const native = upload(d, words, "native streams");
      const lights = upload(
        d,
        packLightRecords([stageSky.record()]),
        "ref lights",
      );
      const envOfLight = upload(d, new Uint32Array([0]), "envOfLight");
      const out = createStorage(d, N * 16, "ref out");
      const ies = {
        records: createStorage(d, 32, "ies records"),
        values: createStorage(d, 16, "ies values"),
      };
      registry.get(refModule, "referenceRelight").run({
        grid: [N],
        buffers: {
          src,
          native,
          lights,
          envSh: stageSky.envSh,
          envSun: stageSky.envSun,
          envTexels: stageSky.envTexels as GPUBuffer,
          envOfLight,
          outColour: out,
          iesRecords: ies.records,
          iesValues: ies.values,
        },
        uniforms: UniformWriter.for(refModule).setAll({
          count: N,
          transferCount,
          transferWords: transferCount / 2,
          shadowBits: cellWords,
          pbrAt: at.pbr,
          lobesAt: at.lobes,
          normalsAt: at.normals,
          emissionAt: NONE,
          transferAt: at.transfer,
          cellsAt: at.cells,
          lightCount: 1,
          envLights: 1,
          envBaseSide: stageSky.baseSide,
          transferIndirect: 1,
          litBody: 0,
          linearCloud: 0,
          ior: 0,
          curvature: 0,
          w0: [1, 0, 0, 0],
          w1: [0, 1, 0, 0],
          w2: [0, 0, 1, 0],
          eyeCloud: [eyeStage.x, eyeStage.y, eyeStage.z, 1],
          eyeWorld: [eyeStage.x, eyeStage.y, eyeStage.z, 1],
        }).data,
      });
      const got = new Float32Array(await read(out));
      for (const buf of [
        native,
        lights,
        envOfLight,
        out,
        ies.records,
        ies.values,
      ])
        buf.destroy();
      return got;
    }

    const color = d.createTexture({
      size: [64, 36],
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

    /** The plugin as the Corvette page runs it (`frame`: the car, or none). */
    async function plugin(frame: THREE.Object3D | null, pixelDetail = false) {
      const splats = new WgpuSplatRenderer(fakeRenderer as never, {
        depthTest: false,
        alwaysGenerate: true,
      });
      const src = GpuSplatSource.fromExt(d, a, b, N);
      src.attribs = pool;
      const mesh = splats.add(src, object);
      const host = new PluginHost({
        capabilities: splats.capabilities,
        tier: 2,
      });
      const relight = atheneaRelightPlugin({
        hdri: image,
        rotation: domeTurn,
        frame,
        pixelDetail,
      });
      host.register(relight).attach(splats);
      await host.ready();
      splats.render(camera, target);
      // The second frame knows the target's size: the slope and sharp lobes.
      if (pixelDetail) splats.render(camera, target);
      await d.queue.onSubmittedWorkDone();
      const relit = relight.buffers?.("splat", { frame: null, mesh })
        .atheneaRelit as GPUBuffer;
      const got = new Float32Array(await read(relit));
      return {
        got,
        src,
        done: () => {
          host.detach();
          splats.dispose();
        },
      };
    }

    const luma = (v: ArrayLike<number>, at: number) =>
      0.2126 * v[at] + 0.7152 * v[at + 1] + 0.0722 * v[at + 2];

    it("keeps the coat's reflection: relit in athenea's stage", async () => {
      stageSky.prepareNow();
      await d.queue.onSubmittedWorkDone();
      const turned = await plugin(null);
      const staged = await plugin(car);
      const ref = await reference(staged.src.src, (i) =>
        pool.getAttribute("lobes", i),
      );
      const noCoat = await reference(staged.src.src, (i) => {
        const w = pool.getAttribute("lobes", i);
        return [w[0], (w[1] & ~0xff) >>> 0, w[2]];
      });
      let sumRef = 0;
      let sumNoCoat = 0;
      let sumStaged = 0;
      let sumTurned = 0;
      let worst = 0;
      let turnedOff = 0;
      for (let i = 0; i < N; i++) {
        const r = luma(ref, 4 * i);
        sumRef += r;
        sumNoCoat += luma(noCoat, 4 * i);
        sumStaged += luma(staged.got, 20 * i);
        sumTurned += luma(turned.got, 20 * i);
        for (let c = 0; c < 3; c++) {
          const w = ref[4 * i + c];
          worst = Math.max(
            worst,
            Math.abs(staged.got[20 * i + c] - w) / Math.max(Math.abs(w), 0.05),
          );
        }
        turnedOff += Math.abs(luma(turned.got, 20 * i) - r) / Math.max(r, 0.02);
      }
      log(
        `hood ${N} splats (t${transferCount}, ${skyName}): mean luminance athenea ${(sumRef / N).toFixed(4)}, without its coat ${(sumNoCoat / N).toFixed(4)} (coat ${((100 * (sumRef - sumNoCoat)) / sumRef).toFixed(1)}%); plugin in the stage ${(sumStaged / N).toFixed(4)} (worst ${worst.toExponential(2)}), turned with the car ${(sumTurned / N).toFixed(4)} (mean off ${(turnedOff / N).toFixed(3)})`,
      );
      // The plugin relit in athenea's stage is athenea's relitSplat.
      expect(worst).toBeLessThan(2e-3);
      // The coat is a large part of the hood's light ...
      expect(sumRef - sumNoCoat).toBeGreaterThan(0.2 * sumRef);
      // ... which the turned relight lost.
      expect(turnedOff / N).toBeGreaterThan(0.2);
      turned.done();
      staged.done();
    }, 300_000);

    it("hands the sharp coat to the draw with pixelDetail", async () => {
      expect(pool.id("curvature")).toBeGreaterThanOrEqual(0);
      const detail = await plugin(car, true);
      const words = new Uint32Array(detail.got.buffer);
      const half = (w: number) => {
        const e = (w >> 10) & 31;
        const m = w & 1023;
        const v = e === 0 ? m * 2 ** -24 : (1 + m / 1024) * 2 ** (e - 15);
        return w & 0x8000 ? -v : v;
      };
      let sharp = 0;
      let held = 0;
      let colour = 0;
      for (let i = 0; i < N; i++) {
        const centre = 20 * i + 16;
        const mark = half(words[centre + 3] >>> 16);
        if (mark > 1.5) sharp++;
        held +=
          0.2126 * half(words[centre + 2] & 0xffff) +
          0.7152 * half(words[centre + 2] >>> 16) +
          0.0722 * half(words[centre + 3] & 0xffff);
        colour += luma(detail.got, 20 * i);
      }
      log(
        `pixelDetail: ${sharp} of ${N} splats sharp, the centre's sharp reading ${((100 * held) / colour).toFixed(1)}% of their colour`,
      );
      // The hood's coat (rough 0.03-0.07) goes to the blend per pixel.
      expect(sharp).toBeGreaterThan(0.5 * N);
      expect(held).toBeGreaterThan(0.3 * colour);
      detail.done();
    }, 300_000);
  },
);

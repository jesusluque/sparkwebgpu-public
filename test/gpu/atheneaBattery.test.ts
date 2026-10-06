// The relight plugin against athenea's own frames: the paint balls of
// athenea's TX tests (tests/data/tx/paint.usda), converted by athenea
// mesh2splat with a TX transfer (direct 16, indirect 48, reflected field 48,
// 16 x 16 cells), drawn by WgpuSplatRenderer under the skies athenea drew
// them under, and measured in athenea's box (192 x 192 + 96 + 80 of 384 x
// 384, the ball's face) against
//   - athenea's raster (the TX build s95: renders/paint/s95), and
//   - the mesh path traced (the ground truth its gate measures against).
//
// The clouds, skies and frames are athenea's outputs on this machine, read
// only (~/luc/athenea-renders/paint; ATHENEA_RENDERS moves it). The USD
// clouds are read with sdfdump (~/tools/usd-*). It takes a minute or two, so
// it runs only when asked: ATHENEA_BATTERY=1 npm run test:gpu -- atheneaBattery
// (and it skips without the files). Numbers are printed; the assertions
// hold the agreement with athenea's raster loosely (ratio and relMSE).
// ATHENEA_BATTERY_OUT=<dir> writes each frame as a PNG.

import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { zlibSync } from "fflate";
import * as THREE from "three";
import { EXRLoader } from "three/examples/jsm/loaders/EXRLoader.js";
import { HDRLoader } from "three/examples/jsm/loaders/HDRLoader.js";
import { describe, expect, it } from "vitest";
import { encodeExtSplat } from "../../src/utils";
import {
  GpuSplatSource,
  WgpuSplatRenderer,
} from "../../src/webgpu/WgpuSplatRenderer";
import type { SkyImage } from "../../src/webgpu/athenea/AtheneaSky";
import { atheneaRelightPlugin } from "../../src/webgpu/athenea/relightPlugin";
import { PluginHost } from "../../src/webgpu/plugins";
import { wideDevice } from "./device";

const RENDERS =
  process.env.ATHENEA_RENDERS ?? join(homedir(), "luc/athenea-renders/paint");
const SDFDUMP = join(homedir(), "tools/usd-26.08-mx/bin/sdfdump");
const SAN_GIUSEPPE = join(
  homedir(),
  "tools/usd-26.08-mx/resources/Lights/san_giuseppe_bridge.hdr",
);
const OUT = process.env.ATHENEA_BATTERY_OUT;
const SIZE = 384;
const BOX = { x: 96, y: 80, w: 192, h: 192 };

interface Case {
  name: string;
  /** The material, as the paint.usda override says. */
  material: string;
  cloud: string;
  sky: string | null;
  athenea: string;
  gt: string;
}

const CASES: Case[] = [
  {
    name: "green, san giuseppe",
    material: "dark green metal, roughness 0.34",
    cloud: "s81/ball_green.usdc",
    sky: SAN_GIUSEPPE,
    athenea: "s95/green.exr",
    gt: "s94/gt/green_sangiuseppe_tx-s94.exr",
  },
  {
    name: "white, san giuseppe",
    material: "white metal, roughness 0.34",
    cloud: "s81/ball_white.usdc",
    sky: SAN_GIUSEPPE,
    athenea: "s95/white.exr",
    gt: "s94/gt/white_sangiuseppe_tx-s94.exr",
  },
  {
    name: "green, white dome",
    material: "dark green metal, roughness 0.34",
    cloud: "s81/ball_green.usdc",
    sky: null,
    athenea: "s95/green_white.exr",
    gt: "s83/green_white_gt.exr",
  },
  {
    name: "white, white dome",
    material: "white metal, roughness 0.34",
    cloud: "s81/ball_white.usdc",
    sky: null,
    athenea: "s95/white_white.exr",
    gt: "s83/white_white_gt.exr",
  },
  {
    name: "floating, sun only",
    material: "white metal, roughness 0.34, no ground",
    cloud: "s88/float.usdc",
    sky: join(RENDERS, "s88/sun_only.exr"),
    athenea: "s95/float.exr",
    gt: "s94/gt/float_sun_tx-s94.exr",
  },
];

const available =
  Boolean(process.env.ATHENEA_BATTERY) &&
  existsSync(SDFDUMP) &&
  CASES.every(
    (c) =>
      existsSync(join(RENDERS, c.cloud)) &&
      existsSync(join(RENDERS, c.athenea)) &&
      existsSync(join(RENDERS, c.gt)) &&
      (!c.sky || existsSync(c.sky)),
  );

interface UsdCloud {
  count: number;
  arrays: Map<string, number[]>;
}

/** The ParticleField's arrays, by attribute name, out of sdfdump's text. */
function readUsdCloud(file: string): UsdCloud {
  const text = execFileSync(SDFDUMP, ["--fullArrays", "-f", "default", file], {
    maxBuffer: 1 << 30,
    encoding: "utf8",
  });
  const arrays = new Map<string, number[]>();
  let name = "";
  for (const line of text.split("\n")) {
    const head = /^<\/World\/Splats\.(.+)> :/.exec(line);
    if (head) {
      name = head[1];
      continue;
    }
    const body = /^\s+default: VtArray<[^>]+> = \[(.*)\]$/.exec(line);
    if (body && name) {
      const numbers = body[1].replace(/[()]/g, "").split(",");
      arrays.set(name, numbers.map(Number));
    }
  }
  const count = (arrays.get("opacities") ?? []).length;
  return { count, arrays };
}

const PV = "primvars:athenea:splat:";

/** packing.slang packNormal. */
function packNormal(x: number, y: number, z: number): number {
  const l1 = Math.abs(x) + Math.abs(y) + Math.abs(z) || 1;
  const v = [x / l1, y / l1, z / l1];
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

const unit = (v: number) => Math.floor(Math.min(Math.max(v, 0), 1) * 255 + 0.5);

/** The cloud as the relight plugin reads a .athc: ext splats and streams. */
function sourceOf(device: GPUDevice, cloud: UsdCloud): GpuSplatSource {
  const n = cloud.count;
  const get = (k: string) => {
    const v = cloud.arrays.get(k);
    if (!v) throw new Error(`no ${k}`);
    return v;
  };
  const pos = get("positions");
  const rot = get("orientations"); // GfQuatf: real first
  const scale = get("scales");
  const opacity = get("opacities");
  const dc = get("radiance:sphericalHarmonicsCoefficients");
  const a = new Uint32Array(n * 4);
  const b = new Uint32Array(n * 4);
  for (let i = 0; i < n; i++) {
    encodeExtSplat(
      [a, b],
      i,
      pos[3 * i],
      pos[3 * i + 1],
      pos[3 * i + 2],
      scale[3 * i],
      scale[3 * i + 1],
      scale[3 * i + 2],
      rot[4 * i + 1],
      rot[4 * i + 2],
      rot[4 * i + 3],
      rot[4 * i],
      opacity[i],
      0.5 + 0.28209479177387814 * dc[3 * i],
      0.5 + 0.28209479177387814 * dc[3 * i + 1],
      0.5 + 0.28209479177387814 * dc[3 * i + 2],
    );
  }
  const source = GpuSplatSource.fromExt(device, a, b, n);
  const normal = get(`${PV}normal`);
  source.setAttribute(
    "normalOct",
    Array.from({ length: n }, (_, i) =>
      packNormal(normal[3 * i], normal[3 * i + 1], normal[3 * i + 2]),
    ),
    "u32",
    1,
  );
  const metallic = get(`${PV}metallic`);
  const roughness = get(`${PV}roughness`);
  const transmission = get(`${PV}transmission`);
  const schlick = cloud.arrays.get(`${PV}schlickMetal`);
  source.setAttribute(
    "pbr",
    Array.from(
      { length: n },
      (_, i) =>
        (unit(metallic[i]) |
          (unit(roughness[i]) << 8) |
          (unit(transmission[i]) << 16) |
          (schlick?.[i] ? 1 << 25 : 0)) >>>
        0,
    ),
    "u32",
    1,
  );
  const direct = get(`${PV}transferDirect`);
  const indirect = get(`${PV}transferIndirect`);
  const field = get(`${PV}transferReflected`);
  const transfer = new Float32Array(n * 112);
  for (let i = 0; i < n; i++) {
    transfer.set(direct.slice(16 * i, 16 * i + 16), 112 * i);
    transfer.set(indirect.slice(48 * i, 48 * i + 48), 112 * i + 16);
    transfer.set(field.slice(48 * i, 48 * i + 48), 112 * i + 64);
  }
  source.setAttribute("transfer", transfer, "f16", 112);
  const bits = get(`${PV}shadowBits`);
  source.setAttribute(
    "shadowBits",
    bits.map((v) => v >>> 0),
    "u32",
    bits.length / n,
  );
  return source;
}

/** An EXR, top row first (three's EXRLoader hands it bottom row first). */
function loadExr(file: string) {
  const loader = new EXRLoader();
  loader.setDataType(THREE.FloatType);
  const bytes = readFileSync(file);
  const r = loader.parse(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  ) as { width: number; height: number; data: Float32Array };
  const row = r.width * 4;
  const data = new Float32Array(r.data.length);
  for (let y = 0; y < r.height; y++) {
    data.set(
      r.data.subarray((r.height - 1 - y) * row, (r.height - y) * row),
      y * row,
    );
  }
  return { width: r.width, height: r.height, data };
}

function loadSky(file: string): SkyImage {
  const bytes = readFileSync(file);
  const buffer = bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  );
  if (file.endsWith(".exr")) {
    const r = loadExr(file);
    return { width: r.width, height: r.height, data: r.data, channels: 4 };
  }
  const loader = new HDRLoader();
  loader.setDataType(THREE.FloatType);
  const r = loader.parse(buffer) as {
    width: number;
    height: number;
    data: Float32Array;
  };
  return { width: r.width, height: r.height, data: r.data, channels: 4 };
}

/** relMSE over the box, athenea's: (a - b)^2 / (b^2 + 1e-2), channel mean. */
function compare(a: Float32Array, b: Float32Array) {
  let err = 0;
  let sa = 0;
  let sb = 0;
  let n = 0;
  for (let y = BOX.y; y < BOX.y + BOX.h; y++) {
    for (let x = BOX.x; x < BOX.x + BOX.w; x++) {
      const at = (y * SIZE + x) * 4;
      for (let c = 0; c < 3; c++) {
        const d = a[at + c] - b[at + c];
        err += (d * d) / (b[at + c] * b[at + c] + 1e-2);
        sa += a[at + c];
        sb += b[at + c];
        n += 1;
      }
    }
  }
  return { relMse: err / n, ratio: sa / sb };
}

function crc32(bytes: Uint8Array) {
  let c = ~0;
  for (const v of bytes) {
    c ^= v;
    for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
  }
  return ~c >>> 0;
}

/** An sRGB PNG of a linear frame (top row first). */
function writePng(file: string, rgba: Float32Array, w: number, h: number) {
  const raw = new Uint8Array(h * (w * 3 + 1));
  const enc = (v: number) => {
    const x = Math.min(Math.max(v, 0), 1);
    return Math.round(
      255 * (x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055),
    );
  };
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < 3; c++) {
        raw[y * (w * 3 + 1) + 1 + x * 3 + c] = enc(rgba[(y * w + x) * 4 + c]);
      }
    }
  }
  const chunk = (type: string, data: Uint8Array) => {
    const out = new Uint8Array(12 + data.length);
    const view = new DataView(out.buffer);
    view.setUint32(0, data.length);
    out.set(new TextEncoder().encode(type), 4);
    out.set(data, 8);
    view.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
    return out;
  };
  const ihdr = new Uint8Array(13);
  const v = new DataView(ihdr.buffer);
  v.setUint32(0, w);
  v.setUint32(4, h);
  ihdr.set([8, 2, 0, 0, 0], 8);
  const parts = [
    new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk("IHDR", ihdr),
    chunk("IDAT", zlibSync(raw)),
    chunk("IEND", new Uint8Array(0)),
  ];
  writeFileSync(file, Buffer.concat(parts));
}

describe.skipIf(!wideDevice || !available)(
  "athenea relight against athenea's frames",
  () => {
    const d = wideDevice as GPUDevice;
    const clouds = new Map<string, UsdCloud>();
    const skies = new Map<string, SkyImage>();

    const color = d.createTexture({
      size: [SIZE, SIZE],
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
    // tests/data/tx/ground.usda's camera: 35 mm over a 24.576 mm aperture.
    const camera = new THREE.PerspectiveCamera(
      2 * THREE.MathUtils.radToDeg(Math.atan(12.288 / 35)),
      1,
      0.1,
      1000,
    );
    camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
    camera.position.set(0, 0.4, 4.2);
    camera.rotation.set(THREE.MathUtils.degToRad(-8), 0, 0);
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld();

    async function frame(): Promise<Float32Array> {
      const bytesPerRow = SIZE * 8;
      const buf = d.createBuffer({
        size: bytesPerRow * SIZE,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const enc = d.createCommandEncoder();
      enc.copyTextureToBuffer(
        { texture: color },
        { buffer: buf, bytesPerRow },
        [SIZE, SIZE],
      );
      d.queue.submit([enc.finish()]);
      await buf.mapAsync(GPUMapMode.READ);
      const halves = new Uint16Array(buf.getMappedRange());
      const out = new Float32Array(halves.length);
      for (let k = 0; k < halves.length; k++) {
        out[k] = THREE.DataUtils.fromHalfFloat(halves[k]);
      }
      buf.unmap();
      buf.destroy();
      return out;
    }

    const rows: string[] = [];

    for (const c of CASES) {
      it(`${c.name} (${c.material})`, async () => {
        let cloud = clouds.get(c.cloud);
        if (!cloud) {
          cloud = readUsdCloud(join(RENDERS, c.cloud));
          clouds.set(c.cloud, cloud);
        }
        let sky: SkyImage | null = null;
        if (c.sky) {
          sky = skies.get(c.sky) ?? loadSky(c.sky);
          skies.set(c.sky, sky);
        }
        const splats = new WgpuSplatRenderer(fakeRenderer as never, {
          depthTest: false,
          alwaysGenerate: true,
        });
        const mesh = splats.add(sourceOf(d, cloud));
        const host = new PluginHost({
          capabilities: splats.capabilities,
          tier: 2,
        });
        const relight = atheneaRelightPlugin({ hdri: sky });
        relight.setStoredLinear(mesh, true);
        host.register(relight).attach(splats);
        await host.ready();
        const enc = d.createCommandEncoder();
        enc
          .beginRenderPass({
            colorAttachments: [
              {
                view: color.createView(),
                loadOp: "clear",
                storeOp: "store",
                clearValue: [0, 0, 0, 0],
              },
            ],
          })
          .end();
        d.queue.submit([enc.finish()]);
        splats.render(camera, target);
        await d.queue.onSubmittedWorkDone();
        const ours = await frame();
        const athenea = loadExr(join(RENDERS, c.athenea)).data;
        const gt = loadExr(join(RENDERS, c.gt)).data;
        const vsAthenea = compare(ours, athenea);
        const vsGt = compare(ours, gt);
        const atheneaVsGt = compare(athenea, gt);
        const row = `${c.name.padEnd(22)} ours/athenea ratio ${vsAthenea.ratio.toFixed(3)} relMSE ${vsAthenea.relMse.toFixed(4)} | ours/GT ratio ${vsGt.ratio.toFixed(3)} relMSE ${vsGt.relMse.toFixed(4)} | athenea/GT ratio ${atheneaVsGt.ratio.toFixed(3)} relMSE ${atheneaVsGt.relMse.toFixed(4)}`;
        rows.push(row);
        console.log(row);
        if (OUT) {
          mkdirSync(OUT, { recursive: true });
          const slug = c.name.replace(/[^a-z]+/g, "_");
          writePng(join(OUT, `${slug}_ours.png`), ours, SIZE, SIZE);
          writePng(join(OUT, `${slug}_athenea.png`), athenea, SIZE, SIZE);
          writePng(join(OUT, `${slug}_gt.png`), gt, SIZE, SIZE);
        }
        // Measured (2026-10-06): ratio 0.96-1.03, relMSE 0.003-0.006.
        expect(vsAthenea.ratio).toBeGreaterThan(0.93);
        expect(vsAthenea.ratio).toBeLessThan(1.07);
        expect(vsAthenea.relMse).toBeLessThan(0.02);
        host.detach();
        splats.dispose();
      }, 300_000);
    }
  },
);

// The sphere battery: athenea's TX balls through OUR whole pipeline --
// athenea's USD cloud converted by usd-athc (full transfer 112, lobes,
// curvature), decoded by the WASM decoder as a page decodes it
// (decode_to_extsplats), relit by atheneaRelightPlugin, drawn by
// WgpuSplatRenderer into a float target -- against
//   - athenea's relitSplat on athenea's own buffer layout, built from the
//     USD arrays themselves (slang/tests/athenea_relight.slang): per splat,
//     in float, which proves the conversion and the plugin carry every
//     input athenea's raster reads;
//   - athenea's raster frames (s95) and the path-traced ground truth (s94)
//     of the paint balls, per pixel, in float (EXR, linear, no display
//     transform), measured as athenea measures them.
//
// The data is athenea's, read only, and the .athc files are converted
// beforehand (docs/docs/athenea-spheres.md says how):
//
//   ATHENEA_SPHERES=<dir with green.athc, white.athc, float.athc,
//                    tx_{paint,chrome,rubber,glass}.athc>
//   npm run test:gpu -- atheneaSpheres
//
// ATHENEA_SPHERES_WEB (default ~/luc/athenea-renders/spheres-web) and
// ATHENEA_RENDERS (default ~/luc/athenea-renders/paint) move athenea's
// files; ATHENEA_SPHERES_OUT=<dir> writes each frame as a PFM (float).
// One sphere is drawn at a time, at 384 x 384.

import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import * as THREE from "three";
import { afterAll, describe, expect, it, vi } from "vitest";
import type { AttribValues } from "../../src/defines";
import { encodeExtSplat } from "../../src/utils";
import { KernelRegistry } from "../../src/webgpu/KernelRegistry";
import {
  GpuSplatSource,
  WgpuSplatRenderer,
} from "../../src/webgpu/WgpuSplatRenderer";
import type { SkyImage } from "../../src/webgpu/athenea/AtheneaSky";
import {
  type AtheneaLightRecord,
  LIGHT_SHADOW,
  distantMatrix,
  packLightRecords,
} from "../../src/webgpu/athenea/lights";
import {
  type AtheneaRelightPlugin,
  atheneaRelightPlugin,
} from "../../src/webgpu/athenea/relightPlugin";
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
import { loadExr, loadSky } from "./atheneaFrames";
import { wideDevice } from "./device";

const ATHC_DIR = process.env.ATHENEA_SPHERES ?? "";
const WEB =
  process.env.ATHENEA_SPHERES_WEB ??
  join(homedir(), "luc/athenea-renders/spheres-web");
const RENDERS =
  process.env.ATHENEA_RENDERS ?? join(homedir(), "luc/athenea-renders/paint");
const OUT = process.env.ATHENEA_SPHERES_OUT;
const SDFDUMP = join(homedir(), "tools/usd-26.08-mx/bin/sdfdump");
const SAN_GIUSEPPE = join(
  homedir(),
  "tools/usd-26.08-mx/resources/Lights/san_giuseppe_bridge.hdr",
);
const SIZE = 384;
/** athenea's measure of the balls: 192 x 192 + 96 + 80 of 384 x 384. */
const BOX = { x: 96, y: 80, w: 192, h: 192 };
const NONE = 0xffffffff;

const available =
  Boolean(ATHC_DIR) && existsSync(ATHC_DIR) && existsSync(SDFDUMP);

// ---------------------------------------------------------------- the skies

/** sRGB 8-bit to linear (HioImage reads an 8-bit PNG as sRGB). */
const srgbToLinear = (v: number) =>
  v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;

/**
 * athenea's tests/usd/test_usd.cpp "[tx_conversion]" window sky: a 64 x 32
 * lat-long, blue above, dark brown below, a white window (rows 8-13,
 * columns 20-29), as the PNG it writes is read: sRGB.
 */
function windowSky(): SkyImage {
  const w = 64;
  const h = 32;
  const data = new Float32Array(w * h * 4);
  const blue = [0x50, 0x80, 0xe0].map((c) => srgbToLinear(c / 255));
  const brown = [0x4a, 0x30, 0x20].map((c) => srgbToLinear(c / 255));
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let c = y < h / 2 ? blue : brown;
      if (y >= 8 && y < 14 && x >= 20 && x < 30) c = [1, 1, 1];
      data.set([c[0], c[1], c[2], 1], (y * w + x) * 4);
    }
  }
  return { width: w, height: h, data, channels: 4 };
}

/** A UsdLux xformOp:rotateXYZ (degrees), applied x first. */
function rotateXYZ(x: number, y: number, z: number): THREE.Matrix4 {
  const r = (a: number) => THREE.MathUtils.degToRad(a);
  return new THREE.Matrix4()
    .makeRotationZ(r(z))
    .multiply(new THREE.Matrix4().makeRotationY(r(y)))
    .multiply(new THREE.Matrix4().makeRotationX(r(x)));
}

interface Sky {
  name: string;
  hdri: SkyImage | null;
  colour: [number, number, number];
  intensity: number;
  noDome: boolean;
  /** The stage's lights besides the dome, as athenea's LightTable writes them. */
  lights: AtheneaLightRecord[];
}

/**
 * tests/data/lobes/sphere.usda: a DomeLight of colour (0.6, 0.75, 1) at 0.6
 * and a DistantLight of intensity 3, angle 4, rotateXYZ (-40, 30, 0), its
 * `normalize` off (UsdLux's default: the intensity is the disc's radiance).
 */
const pale: Sky = {
  name: "pale",
  hdri: null,
  colour: [0.6, 0.75, 1],
  intensity: 0.6,
  noDome: false,
  lights: [
    {
      kind: "distant",
      flags: LIGHT_SHADOW,
      sizeX: THREE.MathUtils.degToRad(4),
      colour: [3, 3, 3],
      matrix: distantMatrix(
        new THREE.Vector3(0, 0, 1).applyMatrix4(rotateXYZ(-40, 30, 0)),
      ),
    },
  ],
};

const windowed: Sky = {
  name: "window",
  hdri: windowSky(),
  colour: [1, 1, 1],
  intensity: 1.5,
  noDome: false,
  lights: [],
};

/** No dome: a SphereLight of radius 0.3 at 40, at (2.2, 2.5, 1.5). */
const lamp: Sky = {
  name: "lamp",
  hdri: null,
  colour: [1, 1, 1],
  intensity: 1,
  noDome: true,
  lights: [
    {
      kind: "sphere",
      flags: LIGHT_SHADOW,
      sizeX: 0.3,
      colour: [40, 40, 40],
      matrix: new THREE.Matrix4().makeTranslation(2.2, 2.5, 1.5),
    },
  ],
};

let sanGiuseppeImage: SkyImage | null = null;
const sanGiuseppe = (): Sky => {
  sanGiuseppeImage ??= loadSky(SAN_GIUSEPPE);
  return {
    name: "san giuseppe",
    hdri: sanGiuseppeImage,
    colour: [1, 1, 1],
    intensity: 1,
    noDome: false,
    lights: [],
  };
};

const whiteDome: Sky = {
  name: "white dome",
  hdri: null,
  colour: [1, 1, 1],
  intensity: 1,
  noDome: false,
  lights: [],
};

let sunOnlyImage: SkyImage | null = null;
const sunOnly = (): Sky => {
  sunOnlyImage ??= loadSky(join(RENDERS, "s88/sun_only.exr"));
  return {
    name: "sun only",
    hdri: sunOnlyImage,
    colour: [1, 1, 1],
    intensity: 1,
    noDome: false,
    lights: [],
  };
};

// ---------------------------------------------------------- the clouds

const wasm = await vi.importActual<typeof import("spark-rs")>("spark-rs");
wasm.initSync({
  module: readFileSync(
    new URL("../../rust/spark-rs/pkg/spark_rs_bg.wasm", import.meta.url),
  ),
});

interface Ours {
  count: number;
  a: Uint32Array;
  b: Uint32Array;
  pool: AttribPool;
  /** .athc flag bit 1: athenea's colours are linear light. */
  linearFile: boolean;
  /**
   * Whether the decoded colours are still linear: a whole-file decode
   * encodes a linear cloud's to sRGB for Spark's blend (athc.rs
   * DecodeOptions; only the pager keeps them linear, PagedSplats
   * athcStoredLinear), which is what the plugin reads by default.
   */
  storedLinear: boolean;
}

/** A .athc as a page decodes it (ext), its finest level. */
function decodeAthc(file: string): Ours {
  const bytes = new Uint8Array(readFileSync(file));
  const decoder = wasm.decode_to_extsplats(
    undefined,
    "cloud.athc",
    undefined,
    undefined,
    undefined,
  );
  decoder.push(bytes);
  const dec = decoder.finish() as {
    numSplats: number;
    ext0: Uint32Array;
    ext1: Uint32Array;
    attribSpecs?: AttribValues["specs"];
    attribColumns?: AttribValues["values"];
  };
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const flags = v.getUint32(8, true);
  const count = v.getUint32(12, true);
  const merged = dec.numSplats - count;
  const specs = dec.attribSpecs ?? [];
  const values = (dec.attribColumns ?? []).map((col, k) =>
    col.slice(
      merged *
        (specs[k].packed
          ? attribWords(specs[k] as never)
          : specs[k].components),
    ),
  );
  return {
    count,
    a: dec.ext0.slice(4 * merged),
    b: dec.ext1.slice(4 * merged),
    pool: AttribPool.fromValues({ count, specs, values }),
    linearFile: (flags & 2) !== 0,
    storedLinear: false,
  };
}

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
      arrays.set(
        name,
        body[1]
          .replace(/[()]/g, "")
          .split(",")
          .map((s) => {
            const t = s.trim();
            return t === "true" ? 1 : t === "false" ? 0 : Number(t);
          }),
      );
    }
  }
  return { count: (arrays.get("opacities") ?? []).length, arrays };
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

const byte = (v: number) => Math.floor(Math.min(Math.max(v, 0), 1) * 255 + 0.5);

/** splat_encoding.slang packPbr (transmission carries +2 thin, +4 Schlick). */
function packPbr(m: number, r: number, t: number): number {
  const schlick = t >= 3.5;
  const rest = schlick ? t - 4 : t;
  const thin = rest >= 1.5;
  return (
    (byte(m) |
      (byte(r) << 8) |
      (byte(thin ? rest - 2 : rest) << 16) |
      (thin ? 1 << 24 : 0) |
      (schlick ? 1 << 25 : 0)) >>>
    0
  );
}

/** packing.slang packLobes. */
function packLobes(l: {
  sw: number;
  sc: number[];
  sior: number;
  cw: number;
  cr: number;
  cior: number;
  cd: number;
  hc: number[];
  hr: number;
}): number[] {
  const iorByte = (ior: number) =>
    Math.floor(Math.min(Math.max((ior - 1) * 128 + 0.5, 0), 255));
  const coatByte = (ior: number, d: number) =>
    Math.floor(Math.min(Math.max((ior - 1) * 64 + 0.5, 0), 127)) |
    (d >= 0.5 ? 128 : 0);
  return [
    (byte(l.sc[0]) |
      (byte(l.sc[1]) << 8) |
      (byte(l.sc[2]) << 16) |
      (byte(l.sw) << 24)) >>>
      0,
    (byte(l.cw) |
      (byte(l.cr) << 8) |
      (coatByte(l.cior, l.cd) << 16) |
      (iorByte(l.sior) << 24)) >>>
      0,
    (byte(l.hc[0]) |
      (byte(l.hc[1]) << 8) |
      (byte(l.hc[2]) << 16) |
      (byte(l.hr) << 24)) >>>
      0,
  ];
}

/** Which USD splat each of ours is (by its centre, which both keep in f32). */
function matchOrder(ours: Ours, usd: UsdCloud): Int32Array {
  const pos = usd.arrays.get("positions") as number[];
  const key = (x: number, y: number, z: number) =>
    `${Math.fround(x)},${Math.fround(y)},${Math.fround(z)}`;
  const op = usd.arrays.get("opacities") as number[];
  const nrm = usd.arrays.get(`${PV}normal`) as number[];
  // A few splats share a centre: among them, the nearest opacity, once.
  const at = new Map<string, number[]>();
  for (let j = 0; j < usd.count; j++) {
    const k = key(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]);
    const list = at.get(k);
    if (list) list.push(j);
    else at.set(k, [j]);
  }
  const f = new Float32Array(ours.a.buffer, ours.a.byteOffset, ours.a.length);
  const order = new Int32Array(ours.count);
  for (let i = 0; i < ours.count; i++) {
    const list = at.get(key(f[4 * i], f[4 * i + 1], f[4 * i + 2]));
    if (!list?.length) {
      order[i] = -1;
      continue;
    }
    // Two faces of a shell can share a centre: told apart by the normal,
    // then the opacity.
    const alpha = THREE.DataUtils.fromHalfFloat(ours.a[4 * i + 3] & 0xffff);
    const word = ours.pool.getAttribute("normalOct", i)[0] >>> 0;
    const ox = (word & 0xffff) / 65535;
    const oy = (word >>> 16) / 65535;
    const cost = (j: number) => {
      const u = packNormal(nrm[3 * j], nrm[3 * j + 1], nrm[3 * j + 2]);
      const dx = (u & 0xffff) / 65535 - ox;
      const dy = (u >>> 16) / 65535 - oy;
      return dx * dx + dy * dy + 1e-3 * Math.abs(op[j] - alpha);
    };
    let best = 0;
    for (let k = 1; k < list.length; k++) {
      if (cost(list[k]) < cost(list[best])) best = k;
    }
    order[i] = list[best];
    list.splice(best, 1);
  }
  return order;
}

/**
 * athenea's buffers for our splat order, from the USD arrays alone: the
 * ext splats (`src`) and one `native` buffer of pbr, lobes, normals, the
 * transfer (halves, as athenea uploads it), the cells and the curvature
 * (f32, as athenea keeps it).
 */
function athenea(
  device: GPUDevice,
  usd: UsdCloud,
  order: Int32Array,
  geometry: Ours | null = null,
) {
  const n = order.length;
  const get = (k: string) => usd.arrays.get(k) as number[];
  const opt = (k: string) => usd.arrays.get(k);
  const pos = get("positions");
  const rot = get("orientations"); // GfQuatf: real first
  const scale = get("scales");
  const opacity = get("opacities");
  const dc = get("radiance:sphericalHarmonicsCoefficients");
  const a = new Uint32Array(n * 4);
  const b = new Uint32Array(n * 4);
  for (let i = 0; i < n; i++) {
    const j = order[i];
    encodeExtSplat(
      [a, b],
      i,
      pos[3 * j],
      pos[3 * j + 1],
      pos[3 * j + 2],
      scale[3 * j],
      scale[3 * j + 1],
      scale[3 * j + 2],
      rot[4 * j + 1],
      rot[4 * j + 2],
      rot[4 * j + 3],
      rot[4 * j],
      opacity[j],
      0.5 + 0.28209479177387814 * dc[3 * j],
      0.5 + 0.28209479177387814 * dc[3 * j + 1],
      0.5 + 0.28209479177387814 * dc[3 * j + 2],
    );
  }
  // `geometry`: our own splats (the .athc's quantised rotation and sRGB
  // colours) where only the arithmetic is compared.
  const src = geometry
    ? GpuSplatSource.fromExt(device, geometry.a, geometry.b, n)
    : GpuSplatSource.fromExt(device, a, b, n);
  const cellWords = get(`${PV}shadowBits`).length / usd.count;
  const at = {
    pbr: 0,
    lobes: n,
    normals: 4 * n,
    transfer: 5 * n,
    cells: 5 * n + 56 * n,
    curvature: 5 * n + 56 * n + cellWords * n,
  };
  const words = new Uint32Array(at.curvature + 3 * n);
  const metallic = get(`${PV}metallic`);
  const roughness = get(`${PV}roughness`);
  const transmission = get(`${PV}transmission`);
  const thin = opt(`${PV}thinWalled`);
  const schlick = opt(`${PV}schlickMetal`);
  const hasLobes = Boolean(opt(`${PV}coatWeight`));
  const lobe = (k: string, d: number, j: number, c = 1, o = 0) =>
    opt(`${PV}${k}`)?.[c * j + o] ?? d;
  const normal = get(`${PV}normal`);
  const direct = get(`${PV}transferDirect`);
  const indirect = get(`${PV}transferIndirect`);
  const field = get(`${PV}transferReflected`);
  const bits = get(`${PV}shadowBits`);
  const curvature = get(`${PV}curvature`);
  // Rounded to nearest even, as athenea's upload does (three's
  // toHalfFloat truncates the subnormals a transfer is full of).
  const halves = new Float16Array(words.buffer, at.transfer * 4, 112 * n);
  const floats = new Float32Array(words.buffer);
  for (let i = 0; i < n; i++) {
    const j = order[i];
    words[at.pbr + i] = packPbr(
      metallic[j],
      roughness[j],
      transmission[j] + (thin?.[j] ? 2 : 0) + (schlick?.[j] ? 4 : 0),
    );
    if (hasLobes) {
      words.set(
        packLobes({
          sw: lobe("specularWeight", 1, j),
          sc: [0, 1, 2].map((c) => lobe("specularColor", 1, j, 3, c)),
          sior: lobe("specularIor", 1.5, j),
          cw: lobe("coatWeight", 0, j),
          cr: lobe("coatRoughness", 0, j),
          cior: lobe("coatIor", 1.5, j),
          cd: lobe("coatDarkening", 0, j),
          hc: [0, 1, 2].map((c) => lobe("sheenColor", 0, j, 3, c)),
          hr: lobe("sheenRoughness", 0, j),
        }),
        at.lobes + 3 * i,
      );
    }
    words[at.normals + i] = packNormal(
      normal[3 * j],
      normal[3 * j + 1],
      normal[3 * j + 2],
    );
    for (let k = 0; k < 16; k++) halves[112 * i + k] = direct[16 * j + k];
    for (let k = 0; k < 48; k++) {
      halves[112 * i + 16 + k] = indirect[48 * j + k];
      halves[112 * i + 64 + k] = field[48 * j + k];
    }
    for (let k = 0; k < cellWords; k++)
      words[at.cells + cellWords * i + k] = bits[cellWords * j + k] >>> 0;
    for (let k = 0; k < 3; k++)
      floats[at.curvature + 3 * i + k] = curvature[3 * j + k];
  }
  return {
    src,
    native: upload(device, words, "athenea native streams"),
    words,
    b,
    at,
    hasLobes,
    cellWords,
  };
}

/**
 * Every stream the relight reads, ours (the .athc decoded) against
 * athenea's (the USD arrays packed as athenea packs them): the words alike,
 * the transfer's halves within one step, the curvature within a half's.
 */
function streamsAgree(ours: Ours, ref: ReturnType<typeof athenea>, n: number) {
  const pool = ours.pool;
  const w = ref.words;
  const f = new Float32Array(w.buffer);
  const h = new Uint16Array(w.buffer);
  const off: Record<string, number> = {};
  const first: Record<string, string> = {};
  const miss = (k: string, i: number, what: string) => {
    off[k] = (off[k] ?? 0) + 1;
    first[k] ??= `splat ${i}: ${what}`;
  };
  const toLinear = (v: number) =>
    v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  for (let i = 0; i < n; i++) {
    // The colour: ours sRGB-encoded halves (a whole-file decode), athenea's
    // the base 0.5 + SH0 dc in linear light, kept as halves.
    const half = (w: number) => THREE.DataUtils.fromHalfFloat(w & 0xffff);
    const mine = [
      half(ours.b[4 * i]),
      half(ours.b[4 * i] >>> 16),
      half(ours.b[4 * i + 1]),
    ].map((v) => (ours.storedLinear ? v : toLinear(v)));
    const theirs = [
      half(ref.b[4 * i]),
      half(ref.b[4 * i] >>> 16),
      half(ref.b[4 * i + 1]),
    ];
    for (let c = 0; c < 3; c++) {
      if (Math.abs(mine[c] - theirs[c]) > 2e-3 * Math.max(theirs[c], 0.05)) {
        miss("colour", i, `${c}: ${mine[c]} vs ${theirs[c]}`);
        break;
      }
    }
    const pbr = pool.getAttribute("pbr", i)[0] >>> 0;
    if (pbr !== w[ref.at.pbr + i])
      miss(
        "pbr",
        i,
        `${pbr.toString(16)} vs ${w[ref.at.pbr + i].toString(16)}`,
      );
    if (ref.hasLobes) {
      const l = pool.getAttribute("lobes", i);
      for (let k = 0; k < 3; k++)
        if (l[k] >>> 0 !== w[ref.at.lobes + 3 * i + k])
          miss(
            "lobes",
            i,
            `${k}: ${(l[k] >>> 0).toString(16)} vs ${w[ref.at.lobes + 3 * i + k].toString(16)}`,
          );
    } else if (pool.id("lobes") >= 0) {
      miss("lobes", i, "ours has lobes, the USD none");
    }
    const nrm = pool.getAttribute("normalOct", i)[0] >>> 0;
    const wn = w[ref.at.normals + i];
    // A step of 1/65535 either way: f32 rounding of the octahedral fold.
    if (
      Math.abs((nrm & 0xffff) - (wn & 0xffff)) > 1 ||
      Math.abs((nrm >>> 16) - (wn >>> 16)) > 1
    )
      miss(
        "normalOct",
        i,
        `${nrm.toString(16)} vs ${w[ref.at.normals + i].toString(16)}`,
      );
    const t = pool.getAttribute("transfer", i);
    for (let k = 0; k < 112; k++) {
      const want = THREE.DataUtils.fromHalfFloat(
        h[2 * ref.at.transfer + 112 * i + k],
      );
      if (Math.abs(t[k] - want) > 1e-3 * Math.max(Math.abs(want), 1e-2)) {
        miss("transfer", i, `${k}: ${t[k]} vs ${want}`);
        break;
      }
    }
    const c = pool.getAttribute("shadowBits", i);
    for (let k = 0; k < ref.cellWords; k++)
      if (c[k] >>> 0 !== w[ref.at.cells + ref.cellWords * i + k]) {
        miss("shadowBits", i, `${k}`);
        break;
      }
    const k3 = pool.getAttribute("curvature", i);
    for (let k = 0; k < 3; k++) {
      const want = f[ref.at.curvature + 3 * i + k];
      if (Math.abs(k3[k] - want) > 2e-3 * Math.max(Math.abs(want), 1)) {
        miss("curvature", i, `${k}: ${k3[k]} vs ${want}`);
        break;
      }
    }
  }
  report(
    `  streams: ${
      Object.keys(off).length
        ? Object.entries(off)
            .map(([k, v]) => `${k} ${v} apart (${first[k]})`)
            .join("; ")
        : "all alike"
    }`,
  );
  expect(off).toEqual({});
}

// ---------------------------------------------------------- measuring

const luma = (v: ArrayLike<number>, k: number) =>
  0.2126 * v[k] + 0.7152 * v[k + 1] + 0.0722 * v[k + 2];

/** Pixels of the ball itself: its rays meet the sphere of radius `r`. */
function ballMask(camera: THREE.PerspectiveCamera, r: number): Uint8Array {
  const mask = new Uint8Array(SIZE * SIZE);
  const o = camera.position;
  const v = new THREE.Vector3();
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      v.set(((x + 0.5) / SIZE) * 2 - 1, 1 - ((y + 0.5) / SIZE) * 2, 0.5)
        .unproject(camera)
        .sub(o)
        .normalize();
      const b = o.dot(v);
      const c = o.lengthSq() - r * r;
      mask[y * SIZE + x] = b * b - c > 0 && b < 0 ? 1 : 0;
    }
  }
  return mask;
}

/** athenea's relMSE ((a - b)^2 / (b^2 + 1e-2), channel mean) and mean ratio. */
function compare(
  a: Float32Array,
  b: Float32Array,
  inside: (p: number) => boolean,
) {
  let err = 0;
  let sa = 0;
  let sb = 0;
  let n = 0;
  for (let p = 0; p < SIZE * SIZE; p++) {
    if (!inside(p)) continue;
    for (let c = 0; c < 3; c++) {
      const d = a[4 * p + c] - b[4 * p + c];
      err += (d * d) / (b[4 * p + c] * b[4 * p + c] + 1e-2);
      sa += a[4 * p + c];
      sb += b[4 * p + c];
      n += 1;
    }
  }
  return { relMse: err / n, ratio: sa / sb, mean: sa / n };
}

const inBox = (p: number) => {
  const x = p % SIZE;
  const y = Math.floor(p / SIZE);
  return x >= BOX.x && x < BOX.x + BOX.w && y >= BOX.y && y < BOX.y + BOX.h;
};

/** A float frame as a PFM (linear, bottom row first). */
function writePfm(file: string, rgba: Float32Array) {
  const head = Buffer.from(`PF\n${SIZE} ${SIZE}\n-1.0\n`, "ascii");
  const body = new Float32Array(SIZE * SIZE * 3);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const from = ((SIZE - 1 - y) * SIZE + x) * 4;
      body.set(rgba.subarray(from, from + 3), (y * SIZE + x) * 3);
    }
  }
  writeFileSync(file, Buffer.concat([head, Buffer.from(body.buffer)]));
}

const rows: string[] = [];
const report = (row: string) => {
  rows.push(row);
  process.stderr.write(`${row}\n`);
};

describe.skipIf(!wideDevice || !available)(
  "athenea's spheres through the whole pipeline",
  () => {
    const d = wideDevice as GPUDevice;
    const registry = new KernelRegistry(d);

    afterAll(() => {
      if (OUT) {
        mkdirSync(OUT, { recursive: true });
        writeFileSync(join(OUT, "spheres.txt"), `${rows.join("\n")}\n`);
      }
    });

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
    // tests/data/tx/ground.usda's /Camera (cloud_<c>.usda's): 35 mm over a
    // 24.576 mm aperture, at (0, 0.4, 4.2), rotateXYZ (-8, 0, 0).
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
    const ball = ballMask(camera, 0.97);

    async function read(buffer: GPUBuffer) {
      const staging = createReadback(d, buffer.size, "spheres readback");
      const enc = d.createCommandEncoder();
      enc.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
      d.queue.submit([enc.finish()]);
      return readAndDestroy(staging);
    }

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

    function clear() {
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
    }

    /** Our pipeline: the decoded .athc, relit by the plugin, drawn. */
    async function ours(
      cloud: Ours,
      sky: Sky,
      opts: {
        ior?: number;
        pixelDetail?: boolean;
        /**
         * athenea's stage placed in three's world: the cloud a child of it,
         * the eye and the lights carried along, relit with `frame`.
         */
        stage?: THREE.Object3D;
      } = {},
    ) {
      const splats = new WgpuSplatRenderer(fakeRenderer as never, {
        depthTest: false,
        alwaysGenerate: true,
      });
      const src = GpuSplatSource.fromExt(d, cloud.a, cloud.b, cloud.count);
      src.attribs = cloud.pool;
      let eye: THREE.Camera = camera;
      let lights = sky.lights;
      let mesh: ReturnType<typeof splats.add>;
      if (opts.stage) {
        const object = new THREE.Object3D();
        opts.stage.add(object);
        opts.stage.updateMatrixWorld(true);
        mesh = splats.add(src, object);
        const placed = camera.clone();
        placed.matrixAutoUpdate = false;
        placed.matrix.multiplyMatrices(
          opts.stage.matrixWorld,
          camera.matrixWorld,
        );
        placed.matrixWorld.copy(placed.matrix);
        placed.matrixWorldInverse.copy(placed.matrixWorld).invert();
        placed.updateMatrixWorld = () => {};
        placed.matrix.decompose(
          placed.position,
          placed.quaternion,
          placed.scale,
        );
        eye = placed;
        lights = sky.lights.map((l) => ({
          ...l,
          matrix: opts.stage?.matrixWorld
            .clone()
            .multiply(l.matrix ?? new THREE.Matrix4()),
        }));
      } else {
        mesh = splats.add(src);
      }
      const host = new PluginHost({
        capabilities: splats.capabilities,
        tier: 2,
      });
      const relight: AtheneaRelightPlugin = atheneaRelightPlugin({
        hdri: sky.hdri,
        colour: sky.colour,
        intensity: sky.intensity,
        noDome: sky.noDome,
        lights,
        pixelDetail: opts.pixelDetail ?? false,
        frame: opts.stage ?? null,
      });
      relight.setStoredLinear(mesh, cloud.storedLinear);
      if (opts.ior) relight.setIor(mesh, opts.ior);
      host.register(relight).attach(splats);
      await host.ready();
      clear();
      splats.render(eye, target);
      // The second frame knows the target's size: the slope and sharp lobes.
      if (opts.pixelDetail) {
        clear();
        splats.render(eye, target);
      }
      await d.queue.onSubmittedWorkDone();
      const image = await frame();
      const relitBuffer = relight.buffers?.("splat", {
        frame: null,
        mesh,
      } as never).atheneaRelit as GPUBuffer;
      const relit = new Float32Array(await read(relitBuffer));
      return {
        image,
        relit,
        relight,
        done: () => {
          host.detach();
          splats.dispose();
          src.destroy();
        },
      };
    }

    /** athenea's relitSplat on its own layout, under the plugin's sky. */
    async function reference(
      ref: ReturnType<typeof athenea>,
      n: number,
      relight: AtheneaRelightPlugin,
      sky: Sky,
      linear: boolean,
      ior: number,
      pixel = false,
    ) {
      const prepared = relight.sky;
      if (!prepared) throw new Error("no sky");
      const list: AtheneaLightRecord[] = [
        ...(sky.noDome ? [] : [prepared.record()]),
        ...sky.lights,
      ];
      const lights = upload(d, packLightRecords(list), "ref lights");
      const of = new Uint32Array(Math.max(list.length, 1)).fill(NONE);
      if (!sky.noDome) of[0] = 0;
      const envOfLight = upload(d, of, "ref envOfLight");
      const out = createStorage(d, n * 16 * (pixel ? 4 : 1), "ref out");
      const ies = {
        records: createStorage(d, 32, "ies records"),
        values: createStorage(d, 16, "ies values"),
      };
      const eye = camera.position;
      // The plugin's projection (relightPlugin.ts relightMesh), at SIZE.
      const p = camera.projectionMatrix.elements;
      const e = camera.matrixWorldInverse.elements;
      const viewRow = (k: number, sign: number) => [
        sign * e[k],
        sign * e[k + 4],
        sign * e[k + 8],
        sign * e[k + 12],
      ];
      registry
        .get(refModule, pixel ? "referenceRelightPixel" : "referenceRelight")
        .run({
          grid: [n],
          buffers: {
            src: ref.src.src,
            native: ref.native,
            lights,
            envSh: prepared.envSh,
            envSun: prepared.envSun,
            envTexels: prepared.envTexels as GPUBuffer,
            envOfLight,
            outColour: out,
            iesRecords: ies.records,
            iesValues: ies.values,
          },
          uniforms: UniformWriter.for(refModule).setAll({
            count: n,
            transferCount: 112,
            transferWords: 56,
            shadowBits: ref.cellWords,
            pbrAt: ref.at.pbr,
            lobesAt: ref.hasLobes ? ref.at.lobes : NONE,
            normalsAt: ref.at.normals,
            emissionAt: NONE,
            transferAt: ref.at.transfer,
            cellsAt: ref.at.cells,
            lightCount: list.length,
            envLights: sky.noDome ? 0 : 1,
            envBaseSide: prepared.baseSide,
            transferIndirect: 1,
            litBody: 0,
            linearCloud: linear ? 1 : 0,
            ior,
            curvature: 0,
            perSplatCurvature: 1,
            curvatureAt: ref.at.curvature,
            w0: [1, 0, 0, 0],
            w1: [0, 1, 0, 0],
            w2: [0, 0, 1, 0],
            eyeCloud: [eye.x, eye.y, eye.z, 1],
            eyeWorld: [eye.x, eye.y, eye.z, 1],
            view0: viewRow(0, 1),
            view1: viewRow(1, 1),
            view2: viewRow(2, -1),
            focal: [
              (SIZE / 2) * p[0],
              -(SIZE / 2) * p[5],
              (SIZE / 2) * (1 - p[8]),
              (SIZE / 2) * (1 + p[9]),
            ],
            viewport: [SIZE, SIZE, 0, 0],
          }).data,
        });
      const got = new Float32Array(await read(out));
      for (const buf of [lights, envOfLight, out, ies.records, ies.values])
        buf.destroy();
      return got;
    }

    /**
     * The records the draw reads per pixel (pixelDetail), ours against
     * splat_project's: which splats are sharp or sloped, the colour without
     * the centre's sharp reading, the mirror and its turn, each lobe's weight
     * and roughness, the sun through them, and the slope across the
     * footprint.
     */
    async function pixelParity(
      cloud: Ours,
      ref: ReturnType<typeof athenea>,
      sky: Sky,
      ior: number,
      facing: (i: number) => number,
      refLinear: boolean,
    ) {
      const n = cloud.count;
      const run = await ours(cloud, sky, { ior, pixelDetail: true });
      const want = await reference(
        ref,
        n,
        run.relight,
        sky,
        refLinear,
        ior,
        true,
      );
      const mine = new Uint32Array(run.relit.buffer);
      const theirs = new Uint32Array(want.buffer);
      const half = (w: number) => THREE.DataUtils.fromHalfFloat(w & 0xffff);
      const halves = (w: Uint32Array, at: number, count: number) =>
        Array.from({ length: count }, (_, k) =>
          half(k % 2 ? w[at + (k >> 1)] >>> 16 : w[at + (k >> 1)]),
        );
      let marks = 0;
      let markOff = 0;
      let sharp = 0;
      let sloped = 0;
      let worstColour = 0;
      let worstRecord = 0;
      let worstSlope = 0;
      let worstAt = -1;
      let worstRecordAt = "";
      let worstSlopeAt = "";
      let recordsOff = 0;
      const slopeSum = [0, 0];
      const rel = (a: number, b: number, floor: number) =>
        Math.abs(a - b) / Math.max(Math.abs(b), floor);
      for (let i = 0; i < n; i++) {
        const mark = half(mine[20 * i + 19] >>> 16);
        const wantMark = want[16 * i + 3];
        if (Math.abs(facing(i)) < 0.02) continue;
        marks++;
        if (Math.round(mark) !== Math.round(wantMark)) {
          markOff++;
          continue;
        }
        if (mark > 1.5) {
          sharp++;
          // The colour Spark carries keeps the centre's sharp reading
          // (`held`); athenea's has it out.
          const held = halves(mine, 20 * i + 18, 3);
          for (let c = 0; c < 3; c++) {
            const e = rel(
              run.relit[20 * i + c] - held[c],
              want[16 * i + c],
              0.05,
            );
            if (e > worstColour) {
              worstColour = e;
              worstAt = i;
            }
          }
          const a = halves(mine, 20 * i + 8, 16);
          const b = halves(theirs, 16 * i + 4, 16);
          let off = false;
          for (let k = 0; k < 16; k++) {
            // The mirror's octahedral steps are small: an absolute floor.
            const e = rel(a[k], b[k], k < 6 ? 0.05 : 0.02);
            if (e > 0.02) off = true;
            if (e > worstRecord) {
              worstRecord = e;
              worstRecordAt = `splat ${i} k ${k}: ${a.map((v) => v.toPrecision(3))} vs ${b.map((v) => v.toPrecision(3))}`;
            }
          }
          if (off) recordsOff++;
        } else if (mark > 0.5) {
          sloped++;
          const a = halves(mine, 20 * i + 8, 6);
          const b = halves(theirs, 16 * i + 4, 6);
          const scale = Math.max(...b.map(Math.abs), 1e-3);
          for (let k = 0; k < 6; k++) {
            slopeSum[0] += Math.abs(a[k]);
            slopeSum[1] += Math.abs(b[k]);
          }
          let off = false;
          for (let k = 0; k < 6; k++) {
            const e = Math.abs(a[k] - b[k]) / scale;
            if (e > 0.02) off = true;
            if (e > worstSlope) {
              worstSlope = e;
              worstSlopeAt = `splat ${i} k ${k}: ${a.map((v) => v.toPrecision(3))} vs ${b.map((v) => v.toPrecision(3))}`;
            }
          }
          if (off) recordsOff++;
          for (let c = 0; c < 3; c++)
            worstColour = Math.max(
              worstColour,
              rel(run.relit[20 * i + c], want[16 * i + c], 0.05),
            );
        }
      }
      report(
        `${"".padEnd(14)} ${sky.name.padEnd(13)} per pixel records: ${sharp} sharp, ${sloped} sloped of ${marks} (${markOff} marked apart, ${recordsOff} records over 2% apart); worst colour ${worstColour.toExponential(2)}${worstAt >= 0 ? ` (splat ${worstAt})` : ""}, sharp record ${worstRecord.toExponential(2)}, slope ${worstSlope.toExponential(2)}`,
      );
      if (process.env.ATHENEA_LOG) {
        process.stderr.write(
          `    sharp record: ${worstRecordAt}\n    slope: ${worstSlopeAt}\n    mean |slope| ours ${(slopeSum[0] / Math.max(sloped, 1)).toExponential(3)} athenea ${(slopeSum[1] / Math.max(sloped, 1)).toExponential(3)}\n`,
        );
      }
      run.done();
      return {
        markOff,
        recordsOff,
        marks,
        worstColour,
        worstRecord,
        worstSlope,
        sharp,
        sloped,
      };
    }

    /** Ours against athenea's relitSplat, per splat, in float. */
    async function parity(
      file: string,
      usdFile: string,
      skies: Sky[],
      ior = 0,
      curvatureScale = 1,
      sameGeometry = false,
    ) {
      const cloud = decodeAthc(join(ATHC_DIR, file));
      const usd = readUsdCloud(usdFile);
      const order = matchOrder(cloud, usd);
      const unmatched = order.filter((j) => j < 0).length;
      expect(unmatched).toBe(0);
      // Every splat usd-athc dropped is one under 1/255 of opacity.
      const op = usd.arrays.get("opacities") as number[];
      const kept = new Set(order);
      let droppedVisible = 0;
      for (let j = 0; j < usd.count; j++)
        if (!kept.has(j) && op[j] >= 1 / 255) droppedVisible++;
      expect(droppedVisible).toBe(0);
      const ref = athenea(d, usd, order, sameGeometry ? cloud : null);
      const refLinear = sameGeometry ? cloud.storedLinear : cloud.linearFile;
      const n = cloud.count;
      streamsAgree(cloud, ref, n);
      if (curvatureScale !== 1) {
        // A crease: the same shape operator, sharper, in both clouds.
        const k = new Float32Array(3 * n);
        for (let i = 0; i < n; i++)
          k.set(
            cloud.pool
              .getAttribute("curvature", i)
              .map((v) => v * curvatureScale),
            3 * i,
          );
        cloud.pool.setAttribute("curvature", k, "f16", 3);
        const f = new Float32Array(ref.words.buffer);
        for (let i = 0; i < 3 * n; i++) f[ref.at.curvature + i] = k[i];
        d.queue.writeBuffer(ref.native, 0, ref.words);
      }
      const results = [];
      for (const sky of skies) {
        const run = await ours(cloud, sky, { ior });
        const want = await reference(ref, n, run.relight, sky, refLinear, ior);
        // The eye's side of each splat: a splat seen edge on (|n.v| under
        // 0.02, its silhouette) turns its face over for a quantisation step
        // of its rotation, ours (the .athc's) or athenea's (its own GPU
        // packing), and its colour with it -- a few tens of splats a ball,
        // which draw as a line. Counted, and kept out of the worst.
        const fa = new Float32Array(cloud.a.buffer, cloud.a.byteOffset);
        const facing = (i: number) => {
          const word = cloud.pool.getAttribute("normalOct", i)[0] >>> 0;
          const u = ((word & 0xffff) / 65535) * 2 - 1;
          const v = ((word >>> 16) / 65535) * 2 - 1;
          const nz = 1 - Math.abs(u) - Math.abs(v);
          const t = Math.max(-nz, 0);
          const nn = new THREE.Vector3(
            u + (u >= 0 ? -t : t),
            v + (v >= 0 ? -t : t),
            nz,
          ).normalize();
          return nn.dot(
            camera.position
              .clone()
              .sub(new THREE.Vector3(fa[4 * i], fa[4 * i + 1], fa[4 * i + 2]))
              .normalize(),
          );
        };
        let worst = 0;
        let worstAt = 0;
        let over = 0;
        let edgeOn = 0;
        let sumOurs = 0;
        let sumRef = 0;
        let sumErr = 0;
        for (let i = 0; i < n; i++) {
          sumOurs += luma(run.relit, 20 * i);
          sumRef += luma(want, 4 * i);
          let most = 0;
          for (let c = 0; c < 3; c++) {
            const w = want[4 * i + c];
            const e =
              Math.abs(run.relit[20 * i + c] - w) / Math.max(Math.abs(w), 0.05);
            most = Math.max(most, e);
            sumErr += e;
          }
          if (most <= 1e-2) continue;
          over++;
          if (Math.abs(facing(i)) < 0.02) {
            edgeOn++;
            continue;
          }
          if (most > worst) {
            worst = most;
            worstAt = i;
          }
        }
        if (process.env.ATHENEA_LOG && worst > 0) {
          const i = worstAt;
          const o = Array.from(run.relit.subarray(20 * i, 20 * i + 4));
          const w = Array.from(want.subarray(4 * i, 4 * i + 4));
          process.stderr.write(
            `    worst splat ${i} (n.v ${facing(i).toFixed(4)}): ours ${o.map((v) => v.toPrecision(4))} athenea ${w.map((v) => v.toPrecision(4))}\n`,
          );
        }
        const row = `${(sameGeometry ? `${file}*` : file).padEnd(14)} ${sky.name.padEnd(13)} per splat (${n}): mean luminance ours ${(sumOurs / n).toFixed(5)} athenea ${(sumRef / n).toFixed(5)} ratio ${(sumOurs / sumRef).toFixed(5)}, relative error mean ${(sumErr / (3 * n)).toExponential(2)} ${over} over 1% (${edgeOn} edge on), worst of the others ${worst.toExponential(2)}`;
        report(row);
        run.done();
        const px = await pixelParity(cloud, ref, sky, ior, facing, refLinear);
        results.push({
          sky: sky.name,
          worst,
          over,
          edgeOn,
          mean: sumErr / (3 * n),
          px,
        });
      }
      ref.native.destroy();
      ref.src.destroy();
      return results;
    }

    const TX = (m: string) => join(WEB, `tx-balls/${m}_tx.usdc`);
    const PAINT = (c: string) => join(WEB, `paint-balls/ball_${c}.usdc`);

    for (const m of ["paint", "chrome", "rubber", "glass"]) {
      it.skipIf(
        !existsSync(join(ATHC_DIR, `tx_${m}.athc`)) || !existsSync(TX(m)),
      )(
        `tx ball ${m}: the plugin is athenea's relitSplat, pale / window / lamp`,
        async () => {
          const ior = m === "glass" ? 1.45 : 0; // the cloud's constant `ior`
          const res = await parity(
            `tx_${m}.athc`,
            TX(m),
            [pale, windowed, lamp],
            ior,
          );
          for (const r of res) {
            // The kept terms are halves (athenea's txCache too).
            expect(r.mean).toBeLessThan(2e-3);
            expect(r.worst).toBeLessThan(2e-2);
            // Per pixel: the same splats sharp or sloped. Their records
            // follow the rotation's quantisation (the .athc's ten-bit
            // smallest three, then Spark's ext encoding, against the USD's
            // floats): a few percent of them turn their step a pixel away
            // by more than 2%.
            expect(r.px.markOff).toBe(0);
            expect(r.px.recordsOff).toBeLessThan(0.05 * r.px.marks);
            expect(r.px.worstColour).toBeLessThan(2e-2);
          }
          // On our own splats the records are athenea's arithmetic exactly.
          const same = await parity(
            `tx_${m}.athc`,
            TX(m),
            [pale, windowed],
            ior,
            1,
            true,
          );
          for (const r of same) {
            expect(r.px.markOff).toBe(0);
            expect(r.px.recordsOff).toBeLessThan(5);
          }
          if (m === "glass") {
            // The cloud's `ior` (a constant primvar, 1.45) is not in the
            // .athc: usd-athc reports it (--json constants.ior) and the page
            // hands it over with setIor. Without it nothing bends (athenea
            // bends only where the cloud has an index; the splat's own
            // specularIor is used once it does).
            const cloud = decodeAthc(join(ATHC_DIR, "tx_glass.athc"));
            const bent = await ours(cloud, windowed, { ior: 1.45 });
            const flat = await ours(cloud, windowed);
            let off = 0;
            let sum = 0;
            for (let i = 0; i < cloud.count; i++) {
              off += Math.abs(
                luma(flat.relit, 20 * i) - luma(bent.relit, 20 * i),
              );
              sum += luma(bent.relit, 20 * i);
            }
            report(
              `tx_glass.athc  window        without setIor: mean |change| of the luminance ${(off / sum).toFixed(3)} of its mean with 1.45 (${(sum / cloud.count).toFixed(5)})`,
            );
            expect(off / sum).toBeGreaterThan(0.05);
            bent.done();
            flat.done();
          }
        },
        600_000,
      );
    }

    it.skipIf(
      !existsSync(join(ATHC_DIR, "tx_rubber.athc")) ||
        !existsSync(TX("rubber")),
    )(
      "a crease (the rubber ball's curvature x 40): the slope is held as athenea holds it",
      async () => {
        // splat_project holds a slope's fall over three sigma of the
        // footprint to the colour, measured on the covariance after its
        // third of a pixel of dilation: a chrome strip turns its normal so
        // fast that the ramp would go below black. On our own splats (the
        // .athc's rotation): what is compared is the arithmetic, where the
        // rotation's quantisation (athenea's from USD) turns a sharply
        // curved splat's step a pixel down by as much as the fix.
        const res = await parity(
          "tx_rubber.athc",
          TX("rubber"),
          [windowed],
          0,
          40,
          true,
        );
        expect(res[0].px.sloped).toBeGreaterThan(1000);
        expect(res[0].px.markOff).toBe(0);
        expect(res[0].px.recordsOff).toBeLessThan(0.001 * res[0].px.marks);
      },
      600_000,
    );

    it.skipIf(!existsSync(join(ATHC_DIR, "tx_paint.athc")))(
      "frame: the coated paint ball in a turned and moved stage relights as in its own",
      async () => {
        // A Z-up stage turned to three's Y-up (the Corvette's), then turned
        // and moved some more; the eye and the sun carried along. Under a
        // dome of one colour the sky has no turn to tell, so everything the
        // relight reads in the stage must come out as it did unturned:
        // the colours and the per-pixel records (the projection included).
        const cloud = decodeAthc(join(ATHC_DIR, "tx_paint.athc"));
        const plain = await ours(cloud, pale, { pixelDetail: true });
        const stage = new THREE.Group();
        stage.rotation.set(-Math.PI / 2 + 0.3, 0.7, -0.2);
        stage.position.set(1.5, -0.4, 2);
        const turned = await ours(cloud, pale, { pixelDetail: true, stage });
        let worst = 0;
        let records = 0;
        const a = new Uint32Array(plain.relit.buffer);
        const b = new Uint32Array(turned.relit.buffer);
        const half = (w: number) => THREE.DataUtils.fromHalfFloat(w & 0xffff);
        for (let i = 0; i < cloud.count; i++) {
          for (let c = 0; c < 3; c++) {
            const w = plain.relit[20 * i + c];
            worst = Math.max(
              worst,
              Math.abs(turned.relit[20 * i + c] - w) /
                Math.max(Math.abs(w), 0.05),
            );
          }
          for (let k = 8; k < 20; k++) {
            for (const sh of [0, 16]) {
              const x = half(a[20 * i + k] >>> sh);
              const y = half(b[20 * i + k] >>> sh);
              // The centre in pixels (k 16, 17) is a float.
              const u = k === 16 || k === 17 ? plain.relit[20 * i + k] : x;
              const v = k === 16 || k === 17 ? turned.relit[20 * i + k] : y;
              if (Math.abs(u - v) > 2e-2 * Math.max(Math.abs(u), 0.05)) {
                records++;
                k = 20;
                break;
              }
            }
          }
        }
        report(
          `frame (turned, moved stage): worst colour ${worst.toExponential(2)}, ${records} of ${cloud.count} per-pixel records apart`,
        );
        expect(worst).toBeLessThan(1e-2);
        expect(records).toBeLessThan(0.001 * cloud.count);
        plain.done();
        turned.done();
      },
      600_000,
    );

    it.skipIf(
      !existsSync(join(ATHC_DIR, "green.athc")) || !existsSync(PAINT("green")),
    )(
      "paint ball green: the plugin is athenea's relitSplat under san giuseppe",
      async () => {
        const res = await parity("green.athc", PAINT("green"), [sanGiuseppe()]);
        expect(res[0].mean).toBeLessThan(2e-3);
        expect(res[0].px.markOff).toBe(0);
        expect(res[0].px.recordsOff).toBeLessThan(0.05 * res[0].px.marks);
      },
      600_000,
    );

    interface Frame {
      name: string;
      athc: string;
      sky: () => Sky;
      athenea: string;
      gt: string;
    }
    const FRAMES: Frame[] = [
      {
        name: "green, san giuseppe",
        athc: "green.athc",
        sky: sanGiuseppe,
        athenea: "s95/green.exr",
        gt: join(WEB, "paint-balls/green_sangiuseppe_tx-s94.exr"),
      },
      {
        name: "white, san giuseppe",
        athc: "white.athc",
        sky: sanGiuseppe,
        athenea: "s95/white.exr",
        gt: join(WEB, "paint-balls/white_sangiuseppe_tx-s94.exr"),
      },
      {
        name: "green, white dome",
        athc: "green.athc",
        sky: () => whiteDome,
        athenea: "s95/green_white.exr",
        gt: join(RENDERS, "s83/green_white_gt.exr"),
      },
      {
        name: "white, white dome",
        athc: "white.athc",
        sky: () => whiteDome,
        athenea: "s95/white_white.exr",
        gt: join(RENDERS, "s83/white_white_gt.exr"),
      },
      {
        name: "floating, sun only",
        athc: "float.athc",
        sky: sunOnly,
        athenea: "s95/float.exr",
        gt: join(WEB, "paint-balls/float_sun_tx-s94.exr"),
      },
    ];

    for (const f of FRAMES) {
      const files = [join(ATHC_DIR, f.athc), join(RENDERS, f.athenea), f.gt];
      it.skipIf(!files.every(existsSync))(
        `${f.name}: ours against athenea's raster and the path traced`,
        async () => {
          const cloud = decodeAthc(join(ATHC_DIR, f.athc));
          const sky = f.sky();
          const raster = loadExr(join(RENDERS, f.athenea)).data;
          const gt = loadExr(f.gt).data;
          const inBall = (p: number) => ball[p] === 1;
          const out: Record<string, Float32Array> = {};
          for (const pixelDetail of [true, false]) {
            const run = await ours(cloud, sky, { pixelDetail });
            // Over athenea's own frame where our splats leave it uncovered:
            // the ground (a mesh athenea draws, not the cloud) in the box's
            // corners and behind the ball's rim. Premultiplied, as drawn.
            const img = run.image.slice();
            for (let q = 0; q < SIZE * SIZE; q++) {
              const left = 1 - Math.min(Math.max(img[4 * q + 3], 0), 1);
              for (let c = 0; c < 3; c++)
                img[4 * q + c] += left * raster[4 * q + c];
            }
            out[pixelDetail ? "ours" : "ours_centre"] = img;
            const label = pixelDetail ? "per pixel" : "centre  ";
            for (const [where, inside] of [
              ["box ", inBox],
              ["ball", inBall],
            ] as const) {
              const vsRaster = compare(img, raster, inside);
              const vsGt = compare(img, gt, inside);
              const rasterVsGt = compare(raster, gt, inside);
              report(
                `${f.name.padEnd(20)} ${label} ${where}: ours/athenea ${vsRaster.ratio.toFixed(3)} relMSE ${vsRaster.relMse.toFixed(4)} | ours/GT ${vsGt.ratio.toFixed(3)} relMSE ${vsGt.relMse.toFixed(4)} | athenea/GT ${rasterVsGt.ratio.toFixed(3)} relMSE ${rasterVsGt.relMse.toFixed(4)} | means ours ${vsRaster.mean.toFixed(4)} athenea ${compare(raster, gt, inside).mean.toFixed(4)}`,
              );
              if (pixelDetail && where === "box ") {
                // athenea's raster draws the lobes per pixel: so do we.
                expect(vsRaster.ratio).toBeGreaterThan(0.95);
                expect(vsRaster.ratio).toBeLessThan(1.05);
                expect(vsRaster.relMse).toBeLessThan(0.02);
              }
            }
            run.done();
          }
          if (OUT) {
            mkdirSync(OUT, { recursive: true });
            const slug = f.name.replace(/[^a-z]+/g, "_");
            for (const [k, v] of Object.entries(out))
              writePfm(join(OUT, `${slug}_${k}.pfm`), v);
            writePfm(join(OUT, `${slug}_athenea.pfm`), raster);
            writePfm(join(OUT, `${slug}_gt.pfm`), gt);
          }
        },
        600_000,
      );
    }
  },
);

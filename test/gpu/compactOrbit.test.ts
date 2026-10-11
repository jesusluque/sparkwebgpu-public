// The compact path (thread CB) on a cloud dumped from the stress page
// (thread CF): the real Corvette geometry, materials, curvature and cells of
// a step, its LoD cut, the page's sky and frame, and the cameras of its
// orbit. Times each pass over the orbit and compares the images of the
// shading modes frame by frame (float).
//
// Opt-in (the data is outside git), either
//   COMPACT_DUMP=<dir with meta.json, data.bin, list.bin, hdri.bin>
// (cf. the stress page: scratch dump.mjs), or a real TX cut as thread CB's
// iphoneCompact test builds it (athenea's joint basis, 4-bit weights, the
// shadow dictionary), every leaf drawn, the camera turning 1 degree a frame
// about athenea's target at 2.2x:
//   ORBIT_ATHC=cut.athc:cut-vq.athc CPCA8_DIR=<athenea q97 data> [ORBIT_HDR=x.hdr]
//   [ORBIT_FLAGS=0,128,...] (debug flag bits ORed into the path's flags)
//   [ORBIT_MODES=full,split,...] [ORBIT_STEP=1] (cameras a frame)
//   [ORBIT_FRAMES=120] [ORBIT_IMAGES=0]
//   npx vitest run --dir test/gpu test/gpu/compactOrbit.test.ts
// Prints COMPACT_ORBIT_* {json} lines.

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import * as THREE from "three";
import { HDRLoader } from "three/examples/jsm/loaders/HDRLoader.js";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { GpuProfiler } from "../../src/webgpu/GpuProfiler";
import {
  type CompactCloud,
  type CompactPathOptions,
  CompactRelitPath,
} from "../../src/webgpu/athenea/compact";
import { encodeCompact } from "../../src/webgpu/athenea/compact";
import {
  Q8_RECORD_WORDS,
  encodeJointCpca,
  halfToFloat,
  packJointBasis,
  parseJointCpca,
} from "../../src/webgpu/athenea/cpcaJoint";
import { compare, placeCamera, readImage } from "./compactHarness";
import { wideDevice } from "./device";

const DIR = process.env.COMPACT_DUMP ?? "";
const ATHC = process.env.ORBIT_ATHC ?? "";
const ok =
  ((!!DIR && existsSync(`${DIR}/meta.json`)) || !!ATHC) && !!wideDevice;
const FLAGS = (process.env.ORBIT_FLAGS ?? "")
  .split(",")
  .filter(Boolean)
  .map(Number);
const MODES = (process.env.ORBIT_MODES ?? "full").split(",").filter(Boolean);
const STEP = Number(process.env.ORBIT_STEP ?? 1);
const FRAMES = Number(process.env.ORBIT_FRAMES ?? 120);
const IMAGES = process.env.ORBIT_IMAGES === "1";

function words(path: string) {
  const b = readFileSync(path);
  return new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4).slice();
}

function median(v: number[]) {
  const s = [...v].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)] ?? 0;
}

interface Dump {
  cloud: CompactCloud;
  list: Uint32Array | null;
  sky: { width: number; height: number; data: Float32Array; channels: number };
  frame: THREE.Matrix4 | null;
  matrixWorld: THREE.Matrix4;
  options: Record<string, unknown>;
  cams: {
    world: number[];
    proj: number[];
    fov: number;
    aspect: number;
    near: number;
    far: number;
    w: number;
    h: number;
  }[];
  capacity: number;
}

function load(): Dump {
  const meta = JSON.parse(readFileSync(`${DIR}/meta.json`, "utf8"));
  const hdri = readFileSync(`${DIR}/hdri.bin`);
  return {
    cloud: { data: words(`${DIR}/data.bin`), layout: meta.layout },
    list: words(`${DIR}/list.bin`),
    sky: {
      ...meta.hdri,
      data: new Float32Array(
        hdri.buffer,
        hdri.byteOffset,
        hdri.byteLength / 4,
      ).slice(),
    },
    frame: meta.frame ? new THREE.Matrix4().fromArray(meta.frame) : null,
    matrixWorld: new THREE.Matrix4().fromArray(meta.matrixWorld),
    options: meta.options,
    cams: meta.cams,
    capacity: meta.capacity,
  };
}

/** A real TX cut (CB's iphoneCompact recipe), cameras 1 degree apart. */
async function loadAthc(): Promise<Dump> {
  const [file, vqFile] = ATHC.split(":");
  const wasm = await vi.importActual<typeof import("spark-rs")>("spark-rs");
  wasm.initSync({
    module: readFileSync(
      new URL("../../rust/spark-rs/pkg/spark_rs_bg.wasm", import.meta.url),
    ),
  });
  const decode = (f: string) => {
    const bytes = new Uint8Array(readFileSync(f));
    const decoder = wasm.decode_to_extsplats(undefined, "cloud.athc");
    decoder.push(bytes);
    const d = decoder.finish() as {
      numSplats: number;
      ext0: Uint32Array;
      ext1: Uint32Array;
      attribSpecs: { name: string }[];
      attribColumns: Uint32Array[];
    };
    const n = (
      wasm.athc_layout(bytes, bytes.length) as { header: { count: number } }
    ).header.count;
    const first = d.numSplats - n;
    const col = (name: string, w: number) => {
      const k = d.attribSpecs.findIndex((s) => s.name === name);
      return k < 0
        ? null
        : d.attribColumns[k].slice(first * w, d.numSplats * w);
    };
    return {
      n,
      ext0: d.ext0.slice(first * 4, d.numSplats * 4),
      ext1: d.ext1.slice(first * 4, d.numSplats * 4),
      normalOct: col("normalOct", 1) as Uint32Array,
      pbr: col("pbr", 1) as Uint32Array,
      lobes: col("lobes", 3) as Uint32Array,
      curvature: col("curvature", 2),
      shadowBits: col("shadowBits", 8) as Uint32Array,
      transfer: col("transfer", 56) as Uint32Array,
    };
  };
  const raw = decode(file);
  const n = raw.n;
  const dir = process.env.CPCA8_DIR ?? "";
  const basis = parseJointCpca(
    JSON.parse(readFileSync(`${dir}/cpca_joint_C256_K8.json`, "utf8")),
    new Uint8Array(readFileSync(`${dir}/cpca_joint_C256_K8.bin`)),
  );
  // 4-bit weights (athenea's 7 levels), cached beside the cut as CB's.
  const cache = `${file}.q8-7`;
  let q8: Uint32Array;
  if (existsSync(cache)) {
    const b = readFileSync(cache);
    q8 = new Uint32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length));
  } else {
    const t16 = new Uint16Array(
      raw.transfer.buffer,
      raw.transfer.byteOffset,
      n * 112,
    );
    const lut = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) lut[h] = halfToFloat(h);
    q8 = encodeJointCpca(
      basis,
      n,
      (i, out) => {
        for (let v = 0; v < 112; v++) out[v] = lut[t16[i * 112 + v]];
      },
      7,
    ).records;
    writeFileSync(cache, new Uint8Array(q8.buffer));
  }
  expect(q8.length).toBe(n * Q8_RECORD_WORDS);
  const shadowBits = vqFile ? decode(vqFile).shadowBits : raw.shadowBits;
  const cloud = encodeCompact({
    count: n,
    ...raw,
    shadowBits,
    q8,
    weightBits: 4,
    basis: packJointBasis(basis),
    clusters: basis.clusters,
    normalBits: 12,
  });
  const hdrPath =
    process.env.ORBIT_HDR ??
    new URL(
      "../../../publish-r2/sparkwebgpu/corvette-lights-v8-light/hdri/golden_gate_hills_1k.hdr",
      import.meta.url,
    ).pathname;
  const hdr = new HDRLoader()
    .setDataType(THREE.FloatType)
    .parse(readFileSync(hdrPath).buffer as ArrayBuffer) as {
    width: number;
    height: number;
    data: Float32Array;
  };
  const [W, H] = (process.env.ORBIT_SIZE ?? "393x852").split("x").map(Number);
  const { camera, place } = placeCamera(
    W,
    H,
    new THREE.Vector3(0, -0.252, 0.553),
    new THREE.Vector3(4.378, -5.854, 1.676),
    39.6,
    2.2,
  );
  const cams: Dump["cams"] = [];
  for (let k = 0; k < 360; k++) {
    place(k);
    cams.push({
      world: camera.matrixWorld.toArray(),
      proj: camera.projectionMatrix.toArray(),
      fov: camera.fov,
      aspect: camera.aspect,
      near: camera.near,
      far: camera.far,
      w: W,
      h: H,
    });
  }
  return {
    cloud,
    list: null,
    sky: { width: hdr.width, height: hdr.height, data: hdr.data, channels: 4 },
    frame: null,
    matrixWorld: new THREE.Matrix4(),
    options: {},
    cams,
    capacity: n,
  };
}

/** A camera placed as the page's was in frame k. */
function cameraOf(dump: Dump) {
  const c0 = dump.cams[0];
  const camera = new THREE.PerspectiveCamera(
    c0.fov,
    c0.aspect,
    c0.near,
    c0.far,
  );
  const place = (k: number) => {
    const c = dump.cams[k % dump.cams.length];
    new THREE.Matrix4()
      .fromArray(c.world)
      .decompose(camera.position, camera.quaternion, camera.scale);
    camera.projectionMatrix.fromArray(c.proj);
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    camera.updateMatrixWorld(true);
  };
  place(0);
  return { camera, place };
}

export interface OrbitRun {
  timing: Record<string, number>;
  images: Float32Array[];
  stats: CompactRelitPath["stats"];
  memory: Record<string, number>;
}

/**
 * The orbit through one path: `frames` cameras (every `step`-th of the
 * dump), timed back to back; with `images`, each frame's float image too
 * (then the frames are not timed).
 */
export async function orbit(
  d: GPUDevice,
  dump: Dump,
  o: {
    path?: CompactPathOptions;
    flags?: number;
    mode?: string;
    frames: number;
    step: number;
    images?: boolean;
  },
): Promise<OrbitRun> {
  const W = dump.cams[0].w;
  const H = dump.cams[0].h;
  const color = d.createTexture({
    size: [W, H],
    format: "rgba16float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const view = color.createView();
  const { frame: stage, ...rest } = dump.options as CompactPathOptions;
  const path = new CompactRelitPath(d, {
    ...(rest as CompactPathOptions),
    hdri: dump.sky,
    frame: dump.frame,
    ...o.path,
    ...(o.images ? { stableOrder: true } : {}),
  });
  path.matrixWorld.copy(dump.matrixWorld);
  path.setCloud(dump.cloud);
  path.capacity = o.images
    ? Math.max(dump.capacity, dump.list?.length ?? 0)
    : dump.capacity;
  if (dump.list) path.setDrawList(dump.list);
  if (o.flags) {
    const p = path as unknown as { flags(): number };
    const base = p.flags.bind(path);
    p.flags = () => base() | (o.flags as number);
  }
  const { camera, place } = cameraOf(dump);
  const profiler = GpuProfiler.create(d, 0, 64);
  // Staleness checks: the reflected field read for an earlier frame's eye.
  const [modeName, modeArg] = (o.mode ?? "").split("+")[0].split(":");
  const eyeAt = (k: number) => {
    place(Math.max(0, k) * o.step);
    return camera.position.clone();
  };
  const frameAt = (k: number, prof: GpuProfiler | null) => {
    if (modeName === "stale") {
      // The whole field refreshed every N frames.
      const n = Number(modeArg ?? 2);
      path.options.fieldEye = eyeAt(k - (k % n));
    } else if (modeName === "inter") {
      // Half the gaussians (by index parity, alternating) each frame.
      path.options.fieldEye = eyeAt(k - 1);
      path.options.fieldFreshParity = (k & 1) as 0 | 1;
    }
    place(k * o.step);
    const enc = d.createCommandEncoder();
    path.render(enc, camera, view, "rgba16float", W, H, {
      clear: { r: 0, g: 0, b: 0, a: 0 },
      profiler: prof,
    });
    const after = prof?.resolve(enc);
    path.registry.submit(enc.finish());
    after?.();
  };
  // Warm up (pipelines, the sky, the caches).
  for (let k = 0; k < 3; k++) {
    frameAt(0, null);
    await d.queue.onSubmittedWorkDone();
  }
  const images: Float32Array[] = [];
  const got: Record<string, number>[] = [];
  if (o.images) {
    for (let k = 0; k < o.frames; k++) {
      frameAt(k, null);
      images.push(await readImage(d, color));
    }
  } else {
    // Back to back in batches, so the GPU stays busy (its clock up).
    if (profiler) profiler.onFrame = (f) => got.push({ ...f });
    for (let round = 0; round < 2; round++) {
      got.length = 0;
      const t0 = performance.now();
      for (let k = 0; k < o.frames; k++) {
        frameAt(k, profiler);
        if (k % 16 === 15) await d.queue.onSubmittedWorkDone();
      }
      await d.queue.onSubmittedWorkDone();
      for (let k = 0; k < 4; k++) await new Promise((r) => setTimeout(r, 10));
      got.push({ wallPerFrame: (performance.now() - t0) / o.frames });
    }
    if (profiler) profiler.onFrame = null;
  }
  const timing: Record<string, number> = {};
  const labels = new Set(got.flatMap((f) => Object.keys(f)));
  for (const l of labels)
    timing[l] = +median(got.filter((f) => l in f).map((f) => f[l])).toFixed(3);
  const memory = path.memory?.() ?? {};
  const stats = { ...path.stats };
  profiler?.destroy();
  path.dispose();
  color.destroy();
  return { timing, images, stats, memory };
}

describe.skipIf(!ok)("compact path on the stress page's dump", () => {
  const d = wideDevice as GPUDevice;
  let dump = null as unknown as Dump;
  beforeAll(async () => {
    if (ok) dump = ATHC ? await loadAthc() : load();
  }, 600_000);

  it("times the passes over the orbit", { timeout: 600_000 }, async () => {
    const variants = FLAGS.length ? FLAGS : [0];
    for (const mode of MODES) {
      for (const flags of variants) {
        const r = await orbit(d, dump, {
          path: modeOptions(mode),
          mode,
          flags: flags | modeFlags(mode),
          frames: FRAMES,
          step: STEP,
        });
        process.stderr.write(
          `COMPACT_ORBIT_TIMING ${JSON.stringify({ mode, flags, step: STEP, ...r.timing, stats: r.stats, memory: r.memory })}\n`,
        );
        expect(r.timing["compact.draw"]).toBeGreaterThan(0);
      }
    }
  });

  it.skipIf(!process.env.COLOURS_OUT)(
    "writes every gaussian's relit colour at the first view",
    { timeout: 300_000 },
    async () => {
      const { frame: _, ...rest } = dump.options as CompactPathOptions;
      const path = new CompactRelitPath(d, {
        ...(rest as CompactPathOptions),
        hdri: dump.sky,
        frame: dump.frame,
      });
      path.matrixWorld.copy(dump.matrixWorld);
      path.setCloud(dump.cloud);
      const { camera } = cameraOf(dump);
      const c = dump.cams[0];
      const { colours, shapes } = await path.colours(camera, c.w, c.h);
      const out = new Float32Array(colours.length * 2);
      out.set(colours);
      out.set(shapes, colours.length);
      writeFileSync(process.env.COLOURS_OUT as string, Buffer.from(out.buffer));
      path.dispose();
    },
  );

  it.skipIf(!IMAGES)(
    "compares the modes' images frame by frame",
    { timeout: 900_000 },
    async () => {
      const frames = Math.min(FRAMES, 60);
      const refMode = process.env.ORBIT_REF ?? "full";
      const ref = await orbit(d, dump, {
        path: modeOptions(refMode),
        flags: modeFlags(refMode),
        frames,
        step: STEP,
        images: true,
      });
      const out: Record<string, unknown>[] = [];
      for (const mode of MODES.filter((m) => m !== refMode)) {
        const got = await orbit(d, dump, {
          path: modeOptions(mode),
          flags: modeFlags(mode),
          mode,
          frames,
          step: STEP,
          images: true,
        });
        out.push(errorSeries(mode, got.images, ref.images));
      }
      for (const line of out)
        process.stderr.write(`COMPACT_ORBIT_ERROR ${JSON.stringify(line)}\n`);
      if (process.env.ORBIT_DUMP_IMAGES)
        writeFileSync(
          `${process.env.ORBIT_DUMP_IMAGES}/ref0.f32`,
          Buffer.from(ref.images[0].buffer),
        );
    },
  );
});

/** Debug flag bits a mode's name carries ("name+flags"). */
export function modeFlags(mode: string): number {
  return Number(mode.split("+")[1] ?? 0);
}

/** The path's options for a shading mode's name. */
export function modeOptions(mode: string): CompactPathOptions {
  const [name, arg] = mode.split("+")[0].split(":");
  switch (name) {
    case "full":
      return {};
    case "nobin":
      return { binByCluster: false };
    case "bin":
      return { binByCluster: true };
    case "stale":
    case "inter":
      return {};
    case "split":
      return { shading: "split" } as CompactPathOptions;
    case "half":
      return {
        shading: "split",
        reflectedEvery: Number(arg ?? 2),
      } as CompactPathOptions;
    case "angle":
      return {
        shading: "split",
        reshadeAngle: Number(arg ?? 1),
      } as CompactPathOptions;
    default:
      throw new Error(`mode ${mode}`);
  }
}

/**
 * A mode's images against the reference's, frame by frame: the image error
 * (relMSE, PSNR over the car: pixels either covers), and the temporal error
 * (the frame-to-frame change against the reference's own change).
 */
function errorSeries(mode: string, got: Float32Array[], ref: Float32Array[]) {
  const per = got.map((g, k) => carCompare(g, ref[k]));
  const temporal: number[] = [];
  // The largest per-pixel jump the mode adds to the reference's own change
  // between frames (a pop), and its 99.9th percentile.
  let popMax = 0;
  let codeMax = 0;
  let codesOver1 = 0;
  const pops: number[] = [];
  for (let k = 1; k < got.length; k++) {
    const dg = diff(got[k], got[k - 1]);
    const dr = diff(ref[k], ref[k - 1]);
    temporal.push(carCompare(dg, dr, ref[k]).psnr);
    const px: number[] = [];
    for (let q = 0; q < dg.length; q += 4) {
      const e = Math.max(
        Math.abs(dg[q] - dr[q]),
        Math.abs(dg[q + 1] - dr[q + 1]),
        Math.abs(dg[q + 2] - dr[q + 2]),
      );
      if (e > 0) px.push(e);
      if (e > popMax) popMax = e;
    }
    // In display codes (clamp, sRGB, 8 bits): the frame-to-frame jump a
    // mode adds on screen.
    for (let q = 0; q < dg.length; q += 4) {
      for (let c = 0; c < 3; c++) {
        const e = Math.abs(
          srgb8(got[k][q + c]) -
            srgb8(got[k - 1][q + c]) -
            (srgb8(ref[k][q + c]) - srgb8(ref[k - 1][q + c])),
        );
        if (e > codeMax) codeMax = e;
        if (e > 1) codesOver1++;
      }
    }
    px.sort((a, b) => a - b);
    pops.push(px.length ? px[Math.floor(0.999 * (px.length - 1))] : 0);
  }
  return {
    mode,
    frames: got.length,
    psnrMin: Math.min(...per.map((p) => p.psnr)),
    psnrMedian: median(per.map((p) => p.psnr)),
    relMSEMax: Math.max(...per.map((p) => p.relMSE)),
    relMSEMedian: median(per.map((p) => p.relMSE)),
    ratio: median(per.map((p) => p.ratio)),
    temporalPsnrMin: temporal.length ? Math.min(...temporal) : null,
    temporalPsnrMedian: temporal.length ? median(temporal) : null,
    popMax: +popMax.toFixed(4),
    pop999Max: +Math.max(0, ...pops).toFixed(4),
    displayJumpMaxCodes: +codeMax.toFixed(2),
    displayJumpsOver1Code: codesOver1,
  };
}

/** A linear value on an 8-bit sRGB display (clamped), unrounded. */
function srgb8(v: number) {
  const x = Math.min(Math.max(v, 0), 1);
  return 255 * (x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055);
}

function diff(a: Float32Array, b: Float32Array) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
  return out;
}

/** compare() over the pixels the car covers (alpha > 1e-3 in `mask` or `ref`). */
function carCompare(
  got: Float32Array,
  ref: Float32Array,
  mask: Float32Array = ref,
) {
  const g: number[] = [];
  const r: number[] = [];
  for (let k = 0; k < ref.length; k += 4) {
    if (!(mask[k + 3] > 1e-3 || got[k + 3] > 1e-3 || ref[k + 3] > 1e-3))
      continue;
    for (let c = 0; c < 4; c++) {
      g.push(got[k + c]);
      r.push(ref[k + c]);
    }
  }
  return compare(Float32Array.from(g), Float32Array.from(r));
}

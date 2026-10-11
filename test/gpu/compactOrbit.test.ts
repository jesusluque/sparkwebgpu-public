// The compact path (thread CB) on a cloud dumped from the stress page
// (thread CF): the real Corvette geometry, materials, curvature and cells of
// a step, its LoD cut, the page's sky and frame, and the cameras of its
// orbit. Times each pass over the orbit and compares the images of the
// shading modes frame by frame (float).
//
// Opt-in (the dump is outside git):
//   COMPACT_DUMP=<dir with meta.json, data.bin, list.bin, hdri.bin>
//   [ORBIT_FLAGS=0,128,...] (debug flag bits ORed into the path's flags)
//   [ORBIT_MODES=full,split,...] [ORBIT_STEP=1] (cameras a frame)
//   [ORBIT_FRAMES=120] [ORBIT_IMAGES=0]
//   npx vitest run --dir test/gpu test/gpu/compactOrbit.test.ts
// Prints COMPACT_ORBIT_* {json} lines.

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { GpuProfiler } from "../../src/webgpu/GpuProfiler";
import {
  type CompactCloud,
  type CompactPathOptions,
  CompactRelitPath,
} from "../../src/webgpu/athenea/compact";
import { compare, readImage } from "./compactHarness";
import { wideDevice } from "./device";

const DIR = process.env.COMPACT_DUMP ?? "";
const ok = !!DIR && existsSync(`${DIR}/meta.json`) && !!wideDevice;
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
  list: Uint32Array;
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
    ? Math.max(dump.capacity, dump.list.length)
    : dump.capacity;
  path.setDrawList(dump.list);
  if (o.flags) {
    const p = path as unknown as { flags(): number };
    const base = p.flags.bind(path);
    p.flags = () => base() | (o.flags as number);
  }
  const { camera, place } = cameraOf(dump);
  const profiler = GpuProfiler.create(d, 0, 64);
  const frameAt = (k: number, prof: GpuProfiler | null) => {
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
  const dump = ok ? load() : (null as unknown as Dump);

  it("times the passes over the orbit", { timeout: 600_000 }, async () => {
    const variants = FLAGS.length ? FLAGS : [0];
    for (const mode of MODES) {
      for (const flags of variants) {
        const r = await orbit(d, dump, {
          path: modeOptions(mode),
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
  for (let k = 1; k < got.length; k++) {
    const dg = diff(got[k], got[k - 1]);
    const dr = diff(ref[k], ref[k - 1]);
    temporal.push(carCompare(dg, dr, ref[k]).psnr);
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
  };
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

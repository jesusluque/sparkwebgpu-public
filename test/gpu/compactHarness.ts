// Shared by compact.test.ts and iphoneCompact.test.ts (thread CB): today's
// relit path (WgpuSplatRenderer + atheneaRelightPlugin, cpca8) and the
// compact path (src/webgpu/athenea/compact.ts) on the same cloud, dome and
// views: per-splat relit colours, float images, GPU ms by pass and GPU
// memory by buffer.

import * as THREE from "three";
import { GpuProfiler } from "../../src/webgpu/GpuProfiler";
import { GpuSplatSource } from "../../src/webgpu/GpuSplatSource";
import { WgpuSplatRenderer } from "../../src/webgpu/WgpuSplatRenderer";
import type { SkyImage } from "../../src/webgpu/athenea/AtheneaSky";
import {
  type CompactCloud,
  type CompactPathOptions,
  CompactRelitPath,
} from "../../src/webgpu/athenea/compact";
import {
  type JointCpcaBasis,
  Q8_RECORD_WORDS,
  packJointBasis,
  q8Column,
} from "../../src/webgpu/athenea/cpcaJoint";
import { atheneaRelightPlugin } from "../../src/webgpu/athenea/relightPlugin";
import {
  AttribPool,
  type AttributeSpec,
} from "../../src/webgpu/attributes/schema";
import { createReadback, readAndDestroy } from "../../src/webgpu/gpuBuffers";
import { PluginHost } from "../../src/webgpu/plugins";

/** The .athc streams the relight reads, as pool columns. */
export interface Streams {
  ext0: Uint32Array;
  ext1: Uint32Array;
  normalOct: Uint32Array;
  pbr: Uint32Array;
  lobes: Uint32Array;
  curvature: Uint32Array | null;
  shadowBits: Uint32Array;
  q8: Uint32Array;
}

const SPECS: Record<string, AttributeSpec> = {
  normalOct: {
    name: "normalOct",
    format: "u32",
    components: 1,
    lodMerge: "first",
  },
  pbr: { name: "pbr", format: "u32", components: 1, lodMerge: "first" },
  lobes: { name: "lobes", format: "u32", components: 3, lodMerge: "first" },
  curvature: {
    name: "curvature",
    format: "f16",
    components: 4,
    lodMerge: "weightedMean",
    packed: true,
  } as AttributeSpec,
  shadowBits: {
    name: "shadowBits",
    format: "u32",
    components: 8,
    lodMerge: "first",
  },
};

/** The relight's pool for `s` (the joint CPCA resident, transferResident "cpca8"). */
export function poolOf(
  s: Streams,
  n: number,
  basis: JointCpcaBasis,
): AttribPool {
  const pool = new AttribPool(n);
  pool.setColumn({ spec: SPECS.normalOct, words: s.normalOct });
  pool.setColumn({ spec: SPECS.pbr, words: s.pbr });
  pool.setColumn({ spec: SPECS.lobes, words: s.lobes });
  if (s.curvature)
    pool.setColumn({ spec: SPECS.curvature, words: s.curvature });
  pool.setColumn({ spec: SPECS.shadowBits, words: s.shadowBits });
  pool.setColumn(q8Column(s.q8.subarray(0, n * Q8_RECORD_WORDS)));
  pool.cpcaJoint = { basis, table: packJointBasis(basis), values: 112 };
  return pool;
}

/** GPU buffers and textures made while alive (BY's iphoneCost tracker). */
export function tracker(d: GPUDevice) {
  const live = new Map<object, { label: string; bytes: number }>();
  const createBuffer = d.createBuffer.bind(d);
  const createTexture = d.createTexture.bind(d);
  d.createBuffer = (desc: GPUBufferDescriptor) => {
    const b = createBuffer(desc);
    live.set(b, { label: desc.label ?? "", bytes: desc.size });
    const destroy = b.destroy.bind(b);
    b.destroy = () => {
      live.delete(b);
      destroy();
    };
    return b;
  };
  d.createTexture = (desc: GPUTextureDescriptor) => {
    const t = createTexture(desc);
    const size = desc.size as number[];
    const px = (size[0] ?? 1) * (size[1] ?? 1) * (size[2] ?? 1);
    const fmt = String(desc.format);
    const bpp = /32float/.test(fmt)
      ? 4 * (fmt.startsWith("rgba") ? 4 : fmt.startsWith("rg") ? 2 : 1)
      : /16float/.test(fmt)
        ? 2 * (fmt.startsWith("rgba") ? 4 : fmt.startsWith("rg") ? 2 : 1)
        : 4;
    // With mips: about a third more.
    const mips = (desc.mipLevelCount ?? 1) > 1 ? 4 / 3 : 1;
    live.set(t, {
      label: `texture ${desc.label ?? fmt}`,
      bytes: Math.round(px * bpp * mips * (desc.sampleCount ?? 1)),
    });
    const destroy = t.destroy.bind(t);
    t.destroy = () => {
      live.delete(t);
      destroy();
    };
    return t;
  };
  return {
    snapshot() {
      const by = new Map<string, number>();
      for (const { label, bytes } of live.values())
        by.set(label, (by.get(label) ?? 0) + bytes);
      return Object.fromEntries(
        [...by].sort((a, b) => b[1] - a[1]).map(([k, v]) => [k, v]),
      );
    },
    reset() {
      live.clear();
    },
    restore() {
      d.createBuffer = createBuffer;
      d.createTexture = createTexture;
    },
  };
}

/** A camera at `distance` x (camera - target) from the target, turned `deg` about +Z. */
export function placeCamera(
  W: number,
  H: number,
  target: THREE.Vector3,
  from: THREE.Vector3,
  hfov: number,
  distance: number,
) {
  const aspect = W / H;
  const vfov =
    (2 * Math.atan(Math.tan((hfov * Math.PI) / 360) / aspect) * 180) / Math.PI;
  const camera = new THREE.PerspectiveCamera(vfov, aspect, 0.05, 100);
  camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
  camera.up.set(0, 0, 1);
  const away = from.clone().sub(target).multiplyScalar(distance);
  const place = (deg: number) => {
    camera.position
      .copy(away)
      .applyAxisAngle(new THREE.Vector3(0, 0, 1), (deg * Math.PI) / 180)
      .add(target);
    camera.lookAt(target);
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld();
  };
  place(0);
  return { camera, place };
}

/** An rgba16float target's pixels as floats. */
export async function readImage(
  d: GPUDevice,
  tex: GPUTexture,
): Promise<Float32Array> {
  const W = tex.width;
  const H = tex.height;
  const row = Math.ceil((W * 8) / 256) * 256;
  const staging = createReadback(d, row * H, "image readback");
  const enc = d.createCommandEncoder();
  enc.copyTextureToBuffer(
    { texture: tex },
    { buffer: staging, bytesPerRow: row },
    [W, H],
  );
  d.queue.submit([enc.finish()]);
  const bytes = new Uint16Array(await readAndDestroy(staging));
  const out = new Float32Array(W * H * 4);
  const h2f = (h: number) => {
    const s = h >> 15;
    const e = (h >> 10) & 31;
    const m = h & 1023;
    const v =
      e === 0
        ? m * 2 ** -24
        : e === 31
          ? Number.NaN
          : (1 + m / 1024) * 2 ** (e - 15);
    return s ? -v : v;
  };
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W * 4; x++)
      out[y * W * 4 + x] = h2f(bytes[(y * row) / 2 + x]);
  return out;
}

/** relMSE, sum ratio and PSNR (peak 1) of `got` against `ref` over finite values; rgb of rgba when `stride` 4. */
export function compare(
  got: Float32Array,
  ref: Float32Array,
  stride = 4,
  channels = 3,
) {
  let e = 0;
  let r = 0;
  let gs = 0;
  let rs = 0;
  let n = 0;
  for (let k = 0; k < ref.length; k += stride) {
    for (let c = 0; c < channels; c++) {
      const a = got[k + c];
      const b = ref[k + c];
      if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
      e += (a - b) ** 2;
      r += b * b;
      gs += a;
      rs += b;
      n += 1;
    }
  }
  return {
    relMSE: +(e / Math.max(r, 1e-30)).toExponential(3),
    ratio: +(gs / Math.max(rs, 1e-30)).toFixed(4),
    psnr: +(10 * Math.log10(1 / Math.max(e / Math.max(n, 1), 1e-30))).toFixed(
      2,
    ),
    psnrSignal: +(
      10 * Math.log10(Math.max(r, 1e-30) / Math.max(e, 1e-30))
    ).toFixed(2),
  };
}

export interface RunOptions {
  W: number;
  H: number;
  sky: SkyImage;
  frames?: number;
  /** The view a frame: k (0 first) -> the camera placed. */
  place: (deg: number) => void;
  camera: THREE.PerspectiveCamera;
  timing?: boolean;
  image?: boolean;
  colours?: boolean;
  /** Reference only: the draw's footprints (projectOnce's records) by splat. */
  shapes?: boolean;
  /** Reference only: more WgpuSplatRenderer options. */
  renderer?: Record<string, unknown>;
}

function median(v: number[]) {
  const s = [...v].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)] ?? 0;
}

/** GPU ms by label (median) over `frames` frames from `frame(k)`, and the wall ms. */
async function timed(
  d: GPUDevice,
  profiler: GpuProfiler | null,
  frames: number,
  frame: (k: number) => void,
) {
  const got: Record<string, number>[] = [];
  if (profiler) profiler.onFrame = (f) => got.push({ ...f });
  const wall: number[] = [];
  for (let k = 0; k < frames; k++) {
    const s = performance.now();
    frame(k);
    await d.queue.onSubmittedWorkDone();
    wall.push(performance.now() - s);
  }
  for (let k = 0; k < 4; k++) {
    await d.queue.onSubmittedWorkDone();
    await new Promise((r) => setTimeout(r, 5));
  }
  if (profiler) profiler.onFrame = null;
  const labels = new Set(got.flatMap((f) => Object.keys(f)));
  const out: Record<string, number> = {};
  for (const l of labels)
    out[l] = +median(got.map((f) => f[l] ?? 0)).toFixed(3);
  out.wall = +median(wall).toFixed(2);
  out.frames = got.length;
  return out;
}

function target(d: GPUDevice, W: number, H: number) {
  return d.createTexture({
    size: [W, H],
    format: "rgba16float",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    label: "target",
  });
}

/** Today's path: WgpuSplatRenderer + atheneaRelightPlugin({ transferResident: "cpca8" }). */
export async function referenceRun(
  d: GPUDevice,
  s: Streams,
  n: number,
  basis: JointCpcaBasis,
  o: RunOptions,
  track?: ReturnType<typeof tracker>,
) {
  track?.reset();
  const color = target(d, o.W, o.H);
  const tgt = {
    texture: { colorSpace: THREE.LinearSRGBColorSpace },
    samples: 0,
    width: o.W,
    height: o.H,
  } as unknown as THREE.RenderTarget;
  const fakeRenderer = {
    backend: {
      isWebGPUBackend: true,
      device: d,
      context: {} as GPUCanvasContext,
      get: (r: object) => (r === tgt.texture ? { texture: color } : undefined),
    },
  };
  const source = GpuSplatSource.fromExt(d, s.ext0, s.ext1, n);
  source.attribs = poolOf(s, n, basis);
  const splats = new WgpuSplatRenderer(fakeRenderer as never, {
    depthTest: false,
    alwaysGenerate: true,
    ...(o.shapes ? { projectOnce: true } : {}),
    ...o.renderer,
  });
  const host = new PluginHost({ capabilities: splats.capabilities, tier: 2 });
  const relight = atheneaRelightPlugin({
    hdri: o.sky,
    transferResident: "cpca8",
  });
  host.register(relight).attach(splats);
  const mesh = splats.add(source);
  await host.ready();
  o.place(0);
  for (let k = 0; k < 4; k++) {
    splats.render(o.camera, tgt);
    await d.queue.onSubmittedWorkDone();
  }
  let colours: Float32Array | null = null;
  if (o.colours) {
    const relit = relight.relitBufferOf(mesh) as GPUBuffer;
    const staging = createReadback(d, relit.size, "relit readback");
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(relit, 0, staging, 0, relit.size);
    d.queue.submit([enc.finish()]);
    const all = new Float32Array(await readAndDestroy(staging));
    const stride = 4 * relight.relitStrideOf(mesh);
    colours = new Float32Array(n * 4);
    for (let i = 0; i < n; i++)
      for (let c = 0; c < 4; c++) colours[4 * i + c] = all[i * stride + c];
  }
  let image: Float32Array | null = null;
  if (o.image) {
    // The renderer draws over what the target holds (three clears it): the
    // warm-up frames stacked, so the target is cleared for one more frame.
    const enc = d.createCommandEncoder();
    enc
      .beginRenderPass({
        colorAttachments: [
          {
            view: color.createView(),
            loadOp: "clear",
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
            storeOp: "store",
          },
        ],
      })
      .end();
    d.queue.submit([enc.finish()]);
    splats.render(o.camera, tgt);
    await d.queue.onSubmittedWorkDone();
    image = await readImage(d, color);
  }
  let shapes: Float32Array | null = null;
  let places: Float32Array | null = null;
  if (o.shapes) {
    const internals = splats as unknown as {
      projected: GPUBuffer;
      sorter: { drawArgs: GPUBuffer };
    };
    const p = internals.projected;
    const staging = createReadback(d, p.size, "projected readback");
    const args = createReadback(d, 16, "args readback");
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(p, 0, staging, 0, p.size);
    enc.copyBufferToBuffer(internals.sorter.drawArgs, 0, args, 0, 16);
    d.queue.submit([enc.finish()]);
    const rec = await readAndDestroy(staging);
    const f = new Float32Array(rec);
    const u = new Uint32Array(rec);
    const count = new Uint32Array(await readAndDestroy(args))[1];
    shapes = new Float32Array(n * 4);
    places = new Float32Array(n * 4);
    for (let slot = 0; slot < count; slot++) {
      const b = 16 * slot;
      if (u[b + 12] === 0) continue;
      const i = u[b + 13];
      if (i >= n) continue;
      shapes[4 * i] = Math.hypot(f[b + 8], f[b + 9]);
      shapes[4 * i + 1] = Math.hypot(f[b + 10], f[b + 11]);
      shapes[4 * i + 2] = f[b + 7];
      shapes[4 * i + 3] = f[b + 12];
      places[4 * i] = ((f[b] / f[b + 3] + 1) / 2) * o.W;
      places[4 * i + 1] = ((1 - f[b + 1] / f[b + 3]) / 2) * o.H;
      let angle = Math.atan2(-f[b + 9], f[b + 8]);
      if (angle < 0) angle += Math.PI;
      places[4 * i + 2] = angle;
    }
  }
  let timing: Record<string, Record<string, number>> | null = null;
  if (o.timing) {
    const frames = o.frames ?? 30;
    splats.options.profile = true;
    splats.render(o.camera, tgt);
    await d.queue.onSubmittedWorkDone();
    const profiler = (splats as unknown as { profiler: GpuProfiler | null })
      .profiler;
    let tick = 0;
    timing = {
      moving: await timed(d, profiler, frames, (k) => {
        o.place(0.2 * (k + 1));
        splats.render(o.camera, tgt);
      }),
      still: await timed(d, profiler, frames, () =>
        splats.render(o.camera, tgt),
      ),
      lights: await timed(d, profiler, frames, (k) => {
        o.place(0.2 * (k + 1));
        relight.set({ emission: 1 + 1e-6 * ++tick });
        splats.render(o.camera, tgt);
      }),
    };
    splats.options.profile = false;
  }
  const memory = track?.snapshot() ?? null;
  host.detach();
  splats.dispose();
  source.dispose?.();
  color.destroy();
  o.place(0);
  return { colours, shapes, places, image, timing, memory };
}

/** The compact path on `cloud`. */
export async function compactRun(
  d: GPUDevice,
  cloud: CompactCloud,
  o: RunOptions & { path?: CompactPathOptions; fit?: boolean },
  track?: ReturnType<typeof tracker>,
) {
  track?.reset();
  const color = target(d, o.W, o.H);
  const view = color.createView();
  const path = new CompactRelitPath(d, { hdri: o.sky, ...o.path });
  path.setCloud(cloud);
  const frame = (profiler: GpuProfiler | null = null) => {
    const enc = d.createCommandEncoder();
    path.render(enc, o.camera, view, "rgba16float", o.W, o.H, {
      clear: { r: 0, g: 0, b: 0, a: 0 },
      profiler,
    });
    const after = profiler?.resolve(enc);
    path.registry.submit(enc.finish());
    after?.();
  };
  o.place(0);
  for (let k = 0; k < 3; k++) {
    frame();
    await d.queue.onSubmittedWorkDone();
  }
  const drawn = await path.drawnCount();
  if (o.fit) {
    path.fitCapacity(drawn);
    frame();
    await d.queue.onSubmittedWorkDone();
  }
  const both = o.colours ? await path.colours(o.camera, o.W, o.H) : null;
  const colours = both?.colours ?? null;
  const shapes = both?.shapes ?? null;
  const places = both?.places ?? null;
  const image = o.image ? await readImage(d, color) : null;
  let timing: Record<string, Record<string, number>> | null = null;
  if (o.timing) {
    const frames = o.frames ?? 30;
    const profiler = GpuProfiler.create(d, 0, 8);
    timing = {
      moving: await timed(d, profiler, frames, (k) => {
        o.place(0.2 * (k + 1));
        frame(profiler);
      }),
      still: await timed(d, profiler, frames, () => frame(profiler)),
      lights: await timed(d, profiler, frames, (k) => {
        o.place(0.2 * (k + 1));
        path.touchLight();
        frame(profiler);
      }),
    };
    profiler?.destroy();
  }
  const memory = track?.snapshot() ?? null;
  const capacity = path.capacity;
  path.dispose();
  color.destroy();
  o.place(0);
  return { colours, shapes, places, image, timing, memory, drawn, capacity };
}

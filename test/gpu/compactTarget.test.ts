// The compact path's draw target (thread CI): the float target the stress
// page draws into (rgba16float, then its display pass: the backdrop under
// the coverage, a clamp and the sRGB curve, into the 8-bit canvas) against
// drawing straight into an 8-bit sRGB target (an rgba8unorm texture seen
// through an rgba8unorm-srgb view: the hardware blends in linear light and
// re-encodes after every splat), cleared to the same backdrop.
//
// In exact arithmetic the two are the same picture: blending "over" from a
// backdrop B gives sum + B * prod(1 - a), which is what the display adds to
// the float target's premultiplied sum. On the GPU the 8-bit target rounds
// and clamps (HDR light past 1) after every blend, the float target once
// at the end. This test measures what that changes, in display codes.
//
// Measured (thread CI, M5 Pro, Dawn, stress-page dumps of 0.4M and 1M, 8
// views): the 8-bit target is NOT the same picture: mean 0.05-0.21 codes,
// but 0.3-3 % of channels off by more than one code and up to 85-138
// codes, almost all in pixels whose final light is within [0, 1] (HDR light
// clamped under splats drawn over it, and 8-bit rounding stalling faint
// layers). And it buys nothing: the draw pass costs the same in both
// targets (Apple GPUs blend in tile memory; the format only changes the
// one store), and the display pass is ~0 (its timestamps overlap the draw;
// the frame's span does not move). So the page keeps the float target.
//
// Always: a synthetic TX cloud (compact.test.ts's), small target.
// Opt-in, on the stress page's dump (scratch dump.mjs: meta.json, data.bin,
// list.bin, hdri.bin):
//   COMPACT_DUMP=<dir> [TARGET_SIZES=393x852,1920x1080] [TARGET_FRAMES=96]
//   npx vitest run --dir test/gpu test/gpu/compactTarget.test.ts
// prints COMPACT_TARGET_IMAGE {json} (code differences over 8 views) and
// COMPACT_TARGET_TIMING {json} (the draw and display passes by target,
// interleaved blocks over the orbit).

import { existsSync, readFileSync } from "node:fs";
import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { encodeExtSplat } from "../../src/utils";
import { GpuProfiler } from "../../src/webgpu/GpuProfiler";
import {
  COMPACT_PIECE,
  type CompactCloud,
  type CompactPathOptions,
  CompactRelitPath,
  encodeCompact,
  packNormal16,
} from "../../src/webgpu/athenea/compact";
import {
  packJointBasis,
  syntheticJointBasis,
  syntheticJointRecords,
} from "../../src/webgpu/athenea/cpcaJoint";
import { createReadback, readAndDestroy } from "../../src/webgpu/gpuBuffers";
import { placeCamera, readImage } from "./compactHarness";
import { wideDevice } from "./device";

// A stress page's display pass (DISPLAY_WGSL), verbatim but for the backdrop's name.
const BACKDROP = [0.06, 0.065, 0.08] as const;
const DISPLAY_WGSL = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) at: vec4f) -> @location(0) vec4f {
  let c = textureLoad(src, vec2i(at.xy), 0);
  let lin = c.rgb + vec3f(${BACKDROP.join(", ")}) * (1.0 - c.a);
  let v = clamp(lin, vec3f(0.0), vec3f(1.0));
  let s = select(1.055 * pow(v, vec3f(1.0 / 2.4)) - 0.055, v * 12.92, v <= vec3f(0.0031308));
  return vec4f(s, 1.0);
}`;

/** The two targets for one size; frame(enc, path, camera, kind) draws one. */
function targets(d: GPUDevice, W: number, H: number) {
  const usage =
    GPUTextureUsage.RENDER_ATTACHMENT |
    GPUTextureUsage.COPY_SRC |
    GPUTextureUsage.TEXTURE_BINDING;
  const float = d.createTexture({
    size: [W, H],
    format: "rgba16float",
    usage,
  });
  // The canvas stand-in: 8-bit, written as display codes.
  const canvas = d.createTexture({
    size: [W, H],
    format: "rgba8unorm",
    usage,
    viewFormats: ["rgba8unorm-srgb"],
  });
  const module = d.createShaderModule({ code: DISPLAY_WGSL });
  const display = d.createRenderPipeline({
    layout: "auto",
    vertex: { module, entryPoint: "vs" },
    fragment: { module, entryPoint: "fs", targets: [{ format: "rgba8unorm" }] },
    primitive: { topology: "triangle-list" },
  });
  const group = d.createBindGroup({
    layout: display.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: float.createView() }],
  });
  const floatView = float.createView();
  const canvasView = canvas.createView();
  const srgbView = canvas.createView({ format: "rgba8unorm-srgb" });
  type Kind = "float" | "srgb8";
  const frame = (
    enc: GPUCommandEncoder,
    path: CompactRelitPath,
    camera: THREE.Camera,
    kind: Kind,
    profiler: GpuProfiler | null,
  ) => {
    if (kind === "float") {
      path.render(enc, camera, floatView, "rgba16float", W, H, {
        clear: { r: 0, g: 0, b: 0, a: 0 },
        profiler,
      });
      const pass = enc.beginRenderPass({
        label: "display",
        colorAttachments: [
          {
            view: canvasView,
            loadOp: "clear",
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            storeOp: "store",
          },
        ],
        timestampWrites: profiler?.timestampWrites("display"),
      });
      pass.setPipeline(display);
      pass.setBindGroup(0, group);
      pass.draw(3);
      pass.end();
    } else {
      // Linear backdrop, encoded by the view on store.
      path.render(enc, camera, srgbView, "rgba8unorm-srgb", W, H, {
        clear: { r: BACKDROP[0], g: BACKDROP[1], b: BACKDROP[2], a: 1 },
        profiler,
      });
    }
  };
  const codes = async () => {
    const row = Math.ceil((W * 4) / 256) * 256;
    const staging = createReadback(d, row * H, "codes readback");
    const enc = d.createCommandEncoder();
    enc.copyTextureToBuffer(
      { texture: canvas },
      { buffer: staging, bytesPerRow: row },
      [W, H],
    );
    d.queue.submit([enc.finish()]);
    const bytes = new Uint8Array(await readAndDestroy(staging));
    const out = new Uint8Array(W * H * 3);
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++)
        for (let c = 0; c < 3; c++)
          out[3 * (y * W + x) + c] = bytes[y * row + 4 * x + c];
    return out;
  };
  const destroy = () => {
    float.destroy();
    canvas.destroy();
  };
  return { frame, codes, destroy, float };
}

/** Display-code differences of `b` against `a` (rgb, 0-255). */
function codeDiff(a: Uint8Array, b: Uint8Array) {
  let max = 0;
  let sum = 0;
  let over1 = 0;
  let over2 = 0;
  const hist = new Array(9).fill(0);
  for (let k = 0; k < a.length; k++) {
    const e = Math.abs(a[k] - b[k]);
    max = Math.max(max, e);
    sum += e;
    if (e > 1) over1 += 1;
    if (e > 2) over2 += 1;
    hist[Math.min(e, 8)] += 1;
  }
  const n = a.length;
  return {
    max,
    mean: +(sum / n).toFixed(4),
    fracOver1: +(over1 / n).toExponential(2),
    fracOver2: +(over2 / n).toExponential(2),
    hist,
  };
}

const N = COMPACT_PIECE + 3000;

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

// compact.test.ts's synthetic sphere (copied: that file exports nothing).
/** A synthetic TX cloud: a sphere shell of splats facing out, a few materials. */
function syntheticStreams() {
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
    const a0 = i % 7 === 0 ? 0.05 + 0.3 * r() : i % 3 === 0 ? 1 + r() : 0.9;
    const a = a0;
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

function syntheticCloud(): CompactCloud {
  return encodeCompact({
    count: N,
    ...syntheticStreams(),
    basis: packJointBasis(syntheticJointBasis(256, 5)),
    clusters: 256,
  });
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

describe.skipIf(!wideDevice)(
  "the compact path into an 8-bit sRGB target",
  () => {
    const d = wideDevice as GPUDevice;
    it("draws the float path's picture, up to the 8-bit target's rounding", async () => {
      const W = 200;
      const H = 150;
      const { camera, place } = placeCamera(
        W,
        H,
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(3, -2, 1),
        40,
        1,
      );
      const t = targets(d, W, H);
      const path = new CompactRelitPath(d, { hdri: sky(), stableOrder: true });
      path.setCloud(syntheticCloud());
      const out: ReturnType<typeof codeDiff>[] = [];
      for (const deg of [0, 120]) {
        place(deg);
        const got: Record<string, Uint8Array> = {};
        for (const kind of ["float", "srgb8"] as const) {
          const enc = d.createCommandEncoder();
          t.frame(enc, path, camera, kind, null);
          path.registry.submit(enc.finish());
          got[kind] = await t.codes();
        }
        // Not empty: the sphere covers part of the view.
        let lit = 0;
        for (let k = 0; k < got.float.length; k += 3)
          if (got.float[k] > 40) lit += 1;
        expect(lit).toBeGreaterThan(0.05 * W * H);
        const e = codeDiff(got.float, got.srgb8);
        out.push(e);
        // The same picture to a few codes (per-blend rounding and clamping);
        // a wrong backdrop, colour space or blend would be tens.
        expect(e.mean).toBeLessThan(2);
      }
      process.stderr.write(`COMPACT_TARGET_SYNTHETIC ${JSON.stringify(out)}\n`);
      path.dispose();
      t.destroy();
    });
  },
);

// ------------------------------------------------- the stress page's dump

const DIR = process.env.COMPACT_DUMP ?? "";
const dumpOk = !!DIR && existsSync(`${DIR}/meta.json`) && !!wideDevice;

function words(path: string) {
  const b = readFileSync(path);
  return new Uint32Array(b.buffer, b.byteOffset, b.byteLength / 4).slice();
}

function median(v: number[]) {
  const s = [...v].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)] ?? 0;
}

describe.skipIf(!dumpOk)(
  "the compact path's target on the stress page's dump",
  () => {
    const d = wideDevice as GPUDevice;
    it(
      "compares and times the float and 8-bit targets",
      { timeout: 1_200_000 },
      async () => {
        const meta = JSON.parse(readFileSync(`${DIR}/meta.json`, "utf8"));
        const hdri = readFileSync(`${DIR}/hdri.bin`);
        const cloud = { data: words(`${DIR}/data.bin`), layout: meta.layout };
        const list = words(`${DIR}/list.bin`);
        const sizes = (process.env.TARGET_SIZES ?? "393x852,1920x1080")
          .split(",")
          .map((s) => s.split("x").map(Number));
        const FRAMES = Number(process.env.TARGET_FRAMES ?? 96);
        const { frame: _f, ...rest } = meta.options as CompactPathOptions;
        for (const [W, H] of sizes) {
          const c0 = meta.cams[0];
          // The page's cameras, the vertical fov kept, at this aspect.
          const camera = new THREE.PerspectiveCamera(
            c0.fov,
            W / H,
            c0.near,
            c0.far,
          );
          camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
          const place = (k: number) => {
            const c = meta.cams[k % meta.cams.length];
            new THREE.Matrix4()
              .fromArray(c.world)
              .decompose(camera.position, camera.quaternion, camera.scale);
            camera.updateProjectionMatrix();
            camera.updateMatrixWorld(true);
          };
          const make = (stableOrder: boolean) => {
            const path = new CompactRelitPath(d, {
              ...(rest as CompactPathOptions),
              hdri: {
                ...meta.hdri,
                data: new Float32Array(
                  hdri.buffer,
                  hdri.byteOffset,
                  hdri.byteLength / 4,
                ),
              },
              frame: meta.frame
                ? new THREE.Matrix4().fromArray(meta.frame)
                : null,
              stableOrder,
            });
            path.matrixWorld.fromArray(meta.matrixWorld);
            path.setCloud(cloud);
            path.capacity = Math.max(meta.capacity, list.length);
            path.setDrawList(list);
            return path;
          };
          const t = targets(d, W, H);
          // Images: 8 views around the orbit, the same order in both targets.
          {
            const path = make(true);
            const per: Record<string, unknown>[] = [];
            const all = {
              max: 0,
              sum: 0,
              n: 0,
              over1: 0,
              over2: 0,
              hist: new Array(9).fill(0),
            };
            // Warm-up (the sky's prefilter, the pipelines).
            for (let k = 0; k < 2; k++) {
              place(0);
              const enc = d.createCommandEncoder();
              t.frame(enc, path, camera, "float", null);
              path.registry.submit(enc.finish());
              await d.queue.onSubmittedWorkDone();
            }
            for (let v = 0; v < 8; v++) {
              place(v * 30);
              const got: Record<string, Uint8Array> = {};
              let linear: Float32Array | null = null;
              for (const kind of ["float", "srgb8", "again"] as const) {
                const enc = d.createCommandEncoder();
                t.frame(
                  enc,
                  path,
                  camera,
                  kind === "again" ? "float" : kind,
                  null,
                );
                path.registry.submit(enc.finish());
                got[kind] = await t.codes();
                if (kind === "float") linear = await readImage(d, t.float);
              }
              const e = codeDiff(got.float, got.srgb8);
              // The pixels whose composited light stays within [0, 1] (no
              // clamp in the display): what is left there is the per-blend
              // rounding, and HDR light clamped under splats in front.
              const L = linear as Float32Array;
              const inA: number[] = [];
              const inB: number[] = [];
              let hdrPixels = 0;
              for (let p = 0; p < W * H; p++) {
                const a = L[4 * p + 3];
                let hdr = false;
                for (let c = 0; c < 3; c++)
                  if (L[4 * p + c] + BACKDROP[c] * (1 - a) > 1) hdr = true;
                if (hdr) {
                  hdrPixels += 1;
                  continue;
                }
                for (let c = 0; c < 3; c++) {
                  inA.push(got.float[3 * p + c]);
                  inB.push(got.srgb8[3 * p + c]);
                }
              }
              const inRange = codeDiff(
                Uint8Array.from(inA),
                Uint8Array.from(inB),
              );
              per.push({
                view: v * 30,
                ...e,
                repeat: codeDiff(got.float, got.again).max,
                hdrPixels: +(hdrPixels / (W * H)).toExponential(2),
                inRange: {
                  max: inRange.max,
                  mean: inRange.mean,
                  fracOver1: inRange.fracOver1,
                },
              });
              const n = got.float.length;
              all.max = Math.max(all.max, e.max);
              all.sum += e.mean * n;
              all.n += n;
              all.over1 += e.fracOver1 * n;
              all.over2 += e.fracOver2 * n;
              e.hist.forEach((h, k) => {
                all.hist[k] += h;
              });
            }
            process.stderr.write(
              `COMPACT_TARGET_IMAGE ${JSON.stringify({ size: `${W}x${H}`, splats: list.length, all: { max: all.max, mean: +(all.sum / all.n).toFixed(4), fracOver1: +(all.over1 / all.n).toExponential(2), fracOver2: +(all.over2 / all.n).toExponential(2), hist: all.hist }, per })}\n`,
            );
            path.dispose();
          }
          // Timing: moving frames, blocks of 16 alternating between targets.
          {
            const path = make(false);
            const profiler = GpuProfiler.create(d, 0, 64) as GpuProfiler;
            const got: Record<string, Record<string, number>[]> = {
              float: [],
              srgb8: [],
            };
            let kind: "float" | "srgb8" = "float";
            // Frames read back in submission order: their targets queued.
            const queued: ("float" | "srgb8")[] = [];
            profiler.onFrame = (f) => {
              const k = queued.shift();
              if (k) got[k].push({ ...f });
            };
            for (let k = 0; k < 6; k++) {
              const enc = d.createCommandEncoder();
              place(k);
              t.frame(enc, path, camera, k % 2 ? "float" : "srgb8", null);
              path.registry.submit(enc.finish());
            }
            await d.queue.onSubmittedWorkDone();
            for (let b = 0; b < (2 * FRAMES) / 16; b++) {
              kind = b % 2 ? "srgb8" : "float";
              for (let k = 0; k < 16; k++) {
                place(b * 16 + k);
                const enc = d.createCommandEncoder();
                t.frame(enc, path, camera, kind, profiler);
                const after = profiler.resolve(enc);
                path.registry.submit(enc.finish());
                if (after) queued.push(kind);
                after?.();
              }
              await d.queue.onSubmittedWorkDone();
              for (let k = 0; k < 4; k++)
                await new Promise((r) => setTimeout(r, 5));
            }
            profiler.onFrame = null;
            const timing: Record<string, Record<string, number>> = {};
            for (const [name, frames] of Object.entries(got)) {
              const labels = new Set(frames.flatMap((f) => Object.keys(f)));
              timing[name] = { frames: frames.length };
              for (const l of labels)
                timing[name][l] = +median(
                  frames.filter((f) => l in f).map((f) => f[l]),
                ).toFixed(3);
            }
            process.stderr.write(
              `COMPACT_TARGET_TIMING ${JSON.stringify({ size: `${W}x${H}`, splats: list.length, ...timing })}\n`,
            );
            expect(timing.float["compact.draw"]).toBeGreaterThan(0);
            profiler.destroy();
            path.dispose();
          }
          t.destroy();
        }
      },
    );
  },
);

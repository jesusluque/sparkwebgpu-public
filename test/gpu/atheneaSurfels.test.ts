// athenea's surfel delivery (a): the car-paint shader ball, unmerged (600k
// thin 3DGS) and merged 10x (52k surfels + 3k thin 3DGS, coverage), through
// OUR whole pipeline -- usd-athc, the WASM decoder, atheneaRelightPlugin,
// WgpuSplatRenderer into a half-float target -- against athenea's raster of
// each cloud and the mesh path-traced, in float (linear Rec.709, no display
// transform). athenea's stage: ~/luc/athenea-matx/bench/matx/scene/base.usda
// (Y-up, metres; one DomeLight autoshop_01_4k.hdr at intensity 1, no other
// light; the camera of camera.json, 512 x 512).
//
//   usd-athc carpaint_full.usdc $DIR/cfull.athc
//   usd-athc carpaint_10x_mixed.usdc $DIR/c10x.athc
//   ATHENEA_SURFELS=$DIR [ATHENEA_SURFELS_OUT=<dir>] npm run test:gpu -- atheneaSurfels
//
// ATHENEA_SURFELS_WEB (default ~/luc/athenea-renders/surfels-web/a) moves
// athenea's files; ATHENEA_SURFELS_HDRI the dome. research/surfels-a.md has
// the numbers.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import * as THREE from "three";
import { afterAll, describe, expect, it, vi } from "vitest";
import type { AttribValues } from "../../src/defines";
import {
  GpuSplatSource,
  WgpuSplatRenderer,
} from "../../src/webgpu/WgpuSplatRenderer";
import { atheneaRelightPlugin } from "../../src/webgpu/athenea/relightPlugin";
import { AttribPool, attribWords } from "../../src/webgpu/attributes/schema";
import { PluginHost } from "../../src/webgpu/plugins";
import { loadExr, loadSky, writePng } from "./atheneaFrames";
import { wideDevice } from "./device";

const DIR = process.env.ATHENEA_SURFELS ?? "";
const WEB =
  process.env.ATHENEA_SURFELS_WEB ??
  join(homedir(), "luc/athenea-renders/surfels-web/a");
const HDRI =
  process.env.ATHENEA_SURFELS_HDRI ??
  join(homedir(), "tools/assets/hdri/autoshop_01_4k.hdr");
const OUT = process.env.ATHENEA_SURFELS_OUT;
/**
 * The chrome ball's merge tree (chrome_tree.usdc) as cut by usd-athc --tree
 * (`ath<N>/chrome.athc`) and by our own error cut of its leaves (athc-convert
 * --keep-splats N --error, `bc<N>/` and `bco<N>/`), against the leaves
 * (`leaves/chrome.athc`), relit.
 */
const TREE = process.env.ATHENEA_SURFELS_TREE ?? "";
const SIZE = 512;

const available =
  (Boolean(DIR) || Boolean(TREE)) &&
  existsSync(join(WEB, "camera.json")) &&
  existsSync(HDRI);

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
}

/** A .athc as a page decodes it (ext), its finest level. */
function decodeAthc(file: string, drop: string[] = []): Ours {
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
  const keep = specs.map((sp) => !drop.includes(sp.name));
  return {
    count,
    a: dec.ext0.slice(4 * merged),
    b: dec.ext1.slice(4 * merged),
    pool: AttribPool.fromValues({
      count,
      specs: specs.filter((_, k) => keep[k]),
      values: values.filter((_, k) => keep[k]),
    }),
  };
}

const half = (w: number) => THREE.DataUtils.fromHalfFloat(w & 0xffff);

/** The three ln-scales of splat i (ext). */
const lnScales = (o: Ours, i: number) => [
  half(o.b[4 * i + 1] >>> 16),
  half(o.b[4 * i + 2]),
  half(o.b[4 * i + 2] >>> 16),
];
/** athc SURFEL_LN: a surfel's flat axis, -65504. */
const isFlat = (l: number) => l <= -65000;

function census(o: Ours) {
  let surfels = 0;
  let twoFlat = 0;
  let over1 = 0;
  let maxOpacity = 0;
  for (let i = 0; i < o.count; i++) {
    const flat = lnScales(o, i).filter(isFlat).length;
    if (flat === 1) surfels++;
    if (flat > 1) twoFlat++;
    const a = half(o.a[4 * i + 3]);
    if (a > 1) over1++;
    maxOpacity = Math.max(maxOpacity, a);
  }
  return { count: o.count, surfels, twoFlat, over1, maxOpacity };
}

/** A copy of `o` with every opacity changed by `f(alpha, i)`. */
function withOpacity(o: Ours, f: (a: number, i: number) => number): Ours {
  const a = o.a.slice();
  for (let i = 0; i < o.count; i++) {
    const v = f(half(a[4 * i + 3]), i);
    a[4 * i + 3] = (a[4 * i + 3] & 0xffff0000) | THREE.DataUtils.toHalfFloat(v);
  }
  return { ...o, a };
}

/** A copy of `o` whose surfels are thin 3DGS (flat axis 1e-4 x widest). */
function thickened(o: Ours, ratio = 1e-4): Ours {
  const b = o.b.slice();
  for (let i = 0; i < o.count; i++) {
    const l = lnScales(o, i);
    if (!l.some(isFlat)) continue;
    const wide = Math.max(...l.filter((v) => !isFlat(v)));
    const t = l.map((v) => (isFlat(v) ? wide + Math.log(ratio) : v));
    const h = t.map((v) => THREE.DataUtils.toHalfFloat(v));
    b[4 * i + 1] = (b[4 * i + 1] & 0xffff) | (h[0] << 16);
    b[4 * i + 2] = h[1] | (h[2] << 16);
  }
  return { ...o, b };
}

// ---------------------------------------------------------- measuring

/**
 * athenea's relMSE ((a - b)^2 / (b^2 + 1e-2), channel mean), the mean ratio
 * and the PSNR in linear light against a peak of 1.
 */
function compare(
  a: Float32Array,
  b: Float32Array,
  inside: (p: number) => boolean,
) {
  let err = 0;
  let se = 0;
  let sa = 0;
  let sb = 0;
  let n = 0;
  for (let p = 0; p < SIZE * SIZE; p++) {
    if (!inside(p)) continue;
    for (let c = 0; c < 3; c++) {
      const d = a[4 * p + c] - b[4 * p + c];
      err += (d * d) / (b[4 * p + c] * b[4 * p + c] + 1e-2);
      se += d * d;
      sa += a[4 * p + c];
      sb += b[4 * p + c];
      n += 1;
    }
  }
  return {
    relMse: err / n,
    ratio: sa / sb,
    psnr: 10 * Math.log10(1 / (se / n)),
    mean: sa / n,
  };
}

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
  "athenea's surfel delivery (a): the car-paint ball, full and 10x",
  () => {
    const d = wideDevice as GPUDevice;
    const out: Record<string, Float32Array> = {};

    afterAll(() => {
      if (!OUT) return;
      mkdirSync(OUT, { recursive: true });
      writeFileSync(join(OUT, "surfels.txt"), `${rows.join("\n")}\n`);
      for (const [k, v] of Object.entries(out)) {
        writePfm(join(OUT, `${k}.pfm`), v);
        writePng(join(OUT, `${k}.png`), v, SIZE, SIZE);
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

    // camera.json: /World/Camera, USD row vectors (rows are the axes and
    // the eye), so its rows read as three's columns.
    const cam = available
      ? JSON.parse(readFileSync(join(WEB, "camera.json"), "utf8"))
      : null;
    const camera = new THREE.PerspectiveCamera(
      cam
        ? 2 *
            THREE.MathUtils.radToDeg(
              Math.atan(cam.verticalAperture / 2 / cam.focalLength),
            )
        : 30,
      1,
      0.1,
      1000,
    );
    camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
    if (cam) {
      camera.matrix.fromArray(cam.cameraToWorld_rowMajor.flat());
      camera.matrix.decompose(camera.position, camera.quaternion, camera.scale);
    }
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld(true);

    const sky = available ? loadSky(HDRI) : null;

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
      const img = new Float32Array(halves.length);
      for (let k = 0; k < halves.length; k++) img[k] = half(halves[k]);
      buf.unmap();
      buf.destroy();
      return img;
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

    /** Our pipeline under the autoshop dome; the frame premultiplied. */
    async function ours(
      cloud: Ours,
      opts: {
        pixelDetail?: boolean;
        surfels?: "auto" | "ewa";
        faintSplats?: "keep" | "drop";
      } = {},
    ) {
      const splats = new WgpuSplatRenderer(fakeRenderer as never, {
        depthTest: false,
        alwaysGenerate: true,
        surfels: opts.surfels ?? "auto",
        faintSplats: opts.faintSplats ?? "keep",
      });
      const src = GpuSplatSource.fromExt(d, cloud.a, cloud.b, cloud.count);
      src.attribs = cloud.pool;
      const mesh = splats.add(src);
      const host = new PluginHost({
        capabilities: splats.capabilities,
        tier: 2,
      });
      const relight = atheneaRelightPlugin({
        hdri: sky,
        colour: [1, 1, 1],
        intensity: 1,
        noDome: false,
        lights: [],
        pixelDetail: opts.pixelDetail ?? true,
        frame: null,
        // athenea's centre shading: this validates against athenea itself.
        footprint: 0,
      });
      relight.setStoredLinear(mesh, false);
      host.register(relight).attach(splats);
      await host.ready();
      clear();
      splats.render(camera, target);
      if (opts.pixelDetail ?? true) {
        clear();
        splats.render(camera, target);
      }
      await d.queue.onSubmittedWorkDone();
      const image = await frame();
      host.detach();
      splats.dispose();
      src.destroy();
      return image;
    }

    /** `img` over `under` where our splats leave it uncovered. */
    const over = (img: Float32Array, under: Float32Array) => {
      const o = img.slice();
      for (let q = 0; q < SIZE * SIZE; q++) {
        const left = 1 - Math.min(Math.max(img[4 * q + 3], 0), 1);
        for (let c = 0; c < 3; c++) o[4 * q + c] += left * under[4 * q + c];
        o[4 * q + 3] = 1;
      }
      return o;
    };

    const carpaint =
      existsSync(join(DIR, "cfull.athc")) && existsSync(join(DIR, "c10x.athc"));

    it.skipIf(!carpaint)(
      "the clouds decode with their surfels and coverage",
      () => {
        const full = census(decodeAthc(join(DIR, "cfull.athc")));
        const tenx = census(decodeAthc(join(DIR, "c10x.athc")));
        report(`census full: ${JSON.stringify(full)}`);
        report(`census 10x:  ${JSON.stringify(tenx)}`);
        expect(full.count).toBe(599_863);
        expect(full.surfels).toBe(0);
        expect(tenx.count).toBe(55_477);
        expect(tenx.surfels).toBe(52_485);
        expect(tenx.twoFlat).toBe(0);
      },
    );

    it.skipIf(!carpaint)(
      "full and 10x against athenea's raster and the path traced",
      async () => {
        const rFull = loadExr(join(WEB, "carpaint_full_raster.exr")).data;
        const r10 = loadExr(join(WEB, "carpaint_10x_raster.exr")).data;
        const gt = loadExr(join(WEB, "carpaint_mesh_pt_256spp.exr")).data;
        const full = decodeAthc(join(DIR, "cfull.athc"));
        const tenx = decodeAthc(join(DIR, "c10x.athc"));

        const fullImg = await ours(full);
        // The ball: where the full cloud covers the pixel. Same mask for all.
        const mask = new Uint8Array(SIZE * SIZE);
        let inBall = 0;
        for (let p = 0; p < SIZE * SIZE; p++) {
          mask[p] = fullImg[4 * p + 3] > 0.99 ? 1 : 0;
          inBall += mask[p];
        }
        const ball = (p: number) => mask[p] === 1;
        const all = () => true;
        report(`ball mask: ${inBall} px of ${SIZE * SIZE}`);

        const line = (name: string, a: Float32Array, b: Float32Array) => {
          const m = compare(a, b, ball);
          const f = compare(a, b, all);
          report(
            `${name.padEnd(34)} ball: ratio ${m.ratio.toFixed(3)} relMSE ${m.relMse.toFixed(4)} PSNR ${m.psnr.toFixed(2)} | frame: ratio ${f.ratio.toFixed(3)} relMSE ${f.relMse.toFixed(4)} PSNR ${f.psnr.toFixed(2)}`,
          );
          return m;
        };

        out.athenea_full = rFull;
        out.athenea_10x = r10;
        out.gt = gt;
        line("athenea 10x / athenea full", r10, rFull);
        line("athenea full / GT", rFull, gt);
        line("athenea 10x / GT", r10, gt);

        const of = over(fullImg, rFull);
        out.ours_full = of;
        const fullVsAthenea = line("ours full / athenea full", of, rFull);
        line("ours full / GT", of, gt);

        const runs: [string, Ours, Parameters<typeof ours>[1]][] = [
          ["10x", tenx, {}],
          ["10x_centre", tenx, { pixelDetail: false }],
          ["10x_ewa", tenx, { surfels: "ewa" }],
          ["10x_drop", tenx, { faintSplats: "drop" }],
          ["10x_cap099", withOpacity(tenx, (a) => Math.min(a, 0.99)), {}],
          ["10x_thick", thickened(tenx), {}],
          ["10x_cap1", withOpacity(tenx, (a) => Math.min(a, 1)), {}],
          [
            "10x_cap099_ewa",
            withOpacity(tenx, (a) => Math.min(a, 0.99)),
            { surfels: "ewa" },
          ],
          [
            "10x_cap099_centre",
            withOpacity(tenx, (a) => Math.min(a, 0.99)),
            { pixelDetail: false },
          ],
        ];
        const measured: Record<string, ReturnType<typeof compare>> = {};
        for (const [name, cloud, opts] of runs) {
          const img = over(await ours(cloud, opts), r10);
          out[`ours_${name}`] = img;
          measured[`${name}/athenea`] = line(
            `ours ${name} / athenea 10x`,
            img,
            r10,
          );
          measured[`${name}/full`] = line(`ours ${name} / ours full`, img, of);
          line(`ours ${name} / GT`, img, gt);
        }
        const centre = over(await ours(full, { pixelDetail: false }), rFull);
        out.ours_full_centre = centre;
        line("ours full_centre / athenea full", centre, rFull);

        // Ours draws athenea's full cloud as athenea does.
        expect(fullVsAthenea.ratio).toBeGreaterThan(0.98);
        expect(fullVsAthenea.ratio).toBeLessThan(1.02);
        expect(fullVsAthenea.relMse).toBeLessThan(2e-3);
        // Ours at 10x keeps the full cloud's light (athenea's raster loses
        // 14% to its 0.99 cap on the coverage) ...
        expect(measured["10x/full"].ratio).toBeGreaterThan(0.98);
        expect(measured["10x/full"].ratio).toBeLessThan(1.02);
        expect(measured["10x/full"].relMse).toBeLessThan(3e-3);
        // ... and with athenea's cap and without the per-pixel curvature
        // (athenea reads it in the moments' frame), draws athenea's 10x.
        expect(measured["10x_cap099_centre/athenea"].ratio).toBeLessThan(1.03);
        expect(measured["10x_cap099_centre/athenea"].relMse).toBeLessThan(6e-3);
      },
      900_000,
    );

    it.skipIf(!TREE || !existsSync(join(TREE, "leaves/chrome.athc")))(
      "the chrome tree: athenea's cuts and ours against the leaves, relit",
      async () => {
        const grey = new Float32Array(SIZE * SIZE * 4).fill(0.18);
        const leaves = over(
          await ours(decodeAthc(join(TREE, "leaves/chrome.athc"))),
          grey,
        );
        out.chrome_leaves = leaves;
        const mask = new Uint8Array(SIZE * SIZE);
        for (let p = 0; p < SIZE * SIZE; p++) {
          // Where the leaves are (they cover it, or differ from the grey).
          let d = 0;
          for (let c = 0; c < 3; c++) d += Math.abs(leaves[4 * p + c] - 0.18);
          mask[p] = d > 1e-4 ? 1 : 0;
        }
        const inside = (p: number) => mask[p] === 1;
        const { readdirSync } = await import("node:fs");
        const cuts = readdirSync(TREE)
          .filter((n) => /^(ath|bc|bco)\d+$/.test(n))
          .sort(
            (a, b) =>
              Number(b.replace(/\D/g, "")) - Number(a.replace(/\D/g, "")) ||
              a.localeCompare(b),
          );
        for (const name of cuts) {
          const cloud = decodeAthc(join(TREE, name, "chrome.athc"));
          let maxLn = Number.NEGATIVE_INFINITY;
          let over1 = 0;
          for (let i = 0; i < cloud.count; i++) {
            maxLn = Math.max(maxLn, ...lnScales(cloud, i));
            if (half(cloud.a[4 * i + 3]) > 1) over1++;
          }
          const img = over(await ours(cloud), grey);
          out[`chrome_${name}`] = img;
          const m = compare(img, leaves, inside);
          if (name === "ath300000" || name === "bc300000") {
            const variants: [string, Ours, Parameters<typeof ours>[1]][] = [
              [
                `${name}_nocurv`,
                decodeAthc(join(TREE, name, "chrome.athc"), ["curvature"]),
                {},
              ],
              [`${name}_centre`, cloud, { pixelDetail: false }],
              [`${name}_thick`, thickened(cloud), {}],
              [`${name}_ewa`, cloud, { surfels: "ewa" }],
            ];
            for (const [label, c, o] of variants) {
              const v = over(await ours(c, o), grey);
              out[`chrome_${label}`] = v;
              const e = compare(v, leaves, inside);
              report(
                `chrome ${label.padEnd(22)}: ratio ${e.ratio.toFixed(3)} relMSE ${e.relMse.toFixed(4)} PSNR ${e.psnr.toFixed(2)}`,
              );
            }
            const lc = over(
              await ours(
                decodeAthc(join(TREE, "leaves/chrome.athc"), ["curvature"]),
              ),
              grey,
            );
            out.chrome_leaves_nocurv = lc;
            const e = compare(lc, leaves, inside);
            report(
              `chrome leaves_nocurv: ratio ${e.ratio.toFixed(3)} relMSE ${e.relMse.toFixed(4)} PSNR ${e.psnr.toFixed(2)}`,
            );
            const nc = compare(out[`chrome_${name}_nocurv`], lc, inside);
            report(
              `chrome ${name}_nocurv / leaves_nocurv: ratio ${nc.ratio.toFixed(3)} relMSE ${nc.relMse.toFixed(4)} PSNR ${nc.psnr.toFixed(2)}`,
            );
          }
          report(
            `chrome ${name.padEnd(10)} ${String(cloud.count).padStart(7)} splats (${census(cloud).surfels} surfels, ${over1} LoD opacity, widest ${Math.exp(maxLn).toFixed(3)} m): ratio ${m.ratio.toFixed(3)} relMSE ${m.relMse.toFixed(4)} PSNR ${m.psnr.toFixed(2)}`,
          );
          // Every cut keeps the leaves' light.
          expect(m.ratio).toBeGreaterThan(0.95);
          expect(m.ratio).toBeLessThan(1.05);
        }
      },
      900_000,
    );
  },
);

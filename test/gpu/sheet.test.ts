// A thin sheet of glass (athenea's relight: generate's GEN_MARK_SHEET, from
// a colour plugin's negative alpha; splat_shape.slang's SHEET_ALPHA): the
// draw floors its alpha at 0.1 AFTER the anti-alias compensation, with its
// colour scaled to keep alpha x colour, as athenea's splat_project does
// (max(paid x own, 0.1)). A plain splat of the same alpha is taken down to
// paid x 0.1 by the compensation. A sheet seen at a slant (thin on screen,
// paid well under 1) so covers a tenth of what is behind it where the
// plain one covers paid of a tenth -- the headlight cover that let 0.27 of
// the room through where athenea lets 0.18 -- and reflects the same light.
//
// The source alpha's sign stands in for the relight's sheet: the default
// colour plugin hands it on, generate marks the splat.

import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { encodeExtSplat } from "../../src/utils";
import { GpuSplatSource } from "../../src/webgpu/GpuSplatSource";
import {
  WgpuSplatRenderer,
  type WgpuSplatRendererOptions,
} from "../../src/webgpu/WgpuSplatRenderer";
import { device } from "./device";

const W = 64;
const H = 48;
const Z = 4;
// Screen sigmas (px) before the blur: wide along x, thin along y.
const SX = 2;
const SY = 0.08;
const BLUR = 0.3;
// The colour, sRGB-encoded with a power of 2.2 (the draw's transfer).
const C = 0.5;

describe.skipIf(!device)("thin sheets", () => {
  const d = device as GPUDevice;
  const camera = new THREE.PerspectiveCamera(60, W / H, 0.05, 100);
  camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();
  const focal = H / 2 / Math.tan(Math.PI / 6);

  // One splat on a pixel centre, facing the eye; the frame's summed alpha
  // and red (linear), and its largest alpha.
  async function draw(
    alpha: number,
    options: WgpuSplatRendererOptions,
  ): Promise<{ alpha: number; red: number; peak: number }> {
    const a = new Uint32Array(4);
    const b = new Uint32Array(4);
    const px = Z / focal;
    encodeExtSplat(
      [a, b],
      0,
      0.5 * px,
      0.5 * px,
      -Z,
      SX * px,
      SY * px,
      SY * px,
      0,
      0,
      0,
      1,
      alpha,
      C,
      C,
      C,
    );
    const color = d.createTexture({
      size: [W, H],
      format: "rgba16float",
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.STORAGE_BINDING,
    });
    const fakeTarget = {
      texture: { colorSpace: THREE.LinearSRGBColorSpace },
      samples: 0,
    } as unknown as THREE.RenderTarget;
    const renderer = {
      backend: {
        isWebGPUBackend: true,
        device: d,
        context: {
          getCurrentTexture: () => color,
        } as unknown as GPUCanvasContext,
        get: (r: object) =>
          r === fakeTarget.texture ? { texture: color } : undefined,
      },
    };
    const splats = new WgpuSplatRenderer(renderer as never, {
      alwaysGenerate: true,
      accumulator: "ext",
      blurAmount: BLUR,
      ...options,
    });
    splats.add(GpuSplatSource.fromExt(d, a, b, 1));
    for (let frame = 0; frame < 2; frame++) {
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
      splats.render(camera, fakeTarget);
      await d.queue.onSubmittedWorkDone();
    }
    const bytesPerRow = W * 8;
    const buf = d.createBuffer({
      size: bytesPerRow * H,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = d.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: color }, { buffer: buf, bytesPerRow }, [
      W,
      H,
    ]);
    d.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const p = new Uint16Array(buf.getMappedRange());
    const half = THREE.DataUtils.fromHalfFloat;
    let sa = 0;
    let sr = 0;
    let peak = 0;
    for (let i = 0; i < p.length; i += 4) {
      sr += half(p[i]);
      const al = half(p[i + 3]);
      sa += al;
      peak = Math.max(peak, al);
    }
    buf.unmap();
    buf.destroy();
    splats.dispose();
    color.destroy();
    return { alpha: sa, red: sr, peak };
  }

  // The blurred Gaussian's area (2 pi sigma1 sigma2), the compensation and
  // a Gaussian's mass inside the squared radius r2.
  const v1 = SX * SX + BLUR;
  const v2 = SY * SY + BLUR;
  const area = 2 * Math.PI * Math.sqrt(v1 * v2);
  const paid = Math.sqrt((SX * SX * SY * SY) / (v1 * v2));
  const inside = (r2: number) => 1 - Math.exp(-r2 / 2);
  const linear = C ** 2.2;

  for (const [name, options] of [
    ["quads", {}],
    ["the projector", { projectOnce: true }],
    ["tiles", { rasterizer: "tiles" }],
  ] as [string, WgpuSplatRendererOptions][]) {
    it(`floors a sheet's alpha after the compensation (${name})`, async () => {
      const sheet = await draw(-0.1, options);
      const plain = await draw(0.1, options);
      // The sheet: 0.1 at its peak, cut where it falls to 1/255
      // (athenea's cutoffPower), its colour scaled by paid.
      const sheetMass = 0.1 * area * inside(2 * Math.log(25.5));
      // The plain splat: paid x 0.1, cut at minAlpha (0.5 / 255).
      const plainMass =
        0.1 * paid * area * inside(2 * Math.log((0.1 * paid * 255) / 0.5));
      if (process.env.SHEET_LOG)
        process.stderr.write(
          `${name} ${JSON.stringify({ paid, sheet, sheetMass, plain, plainMass })}\n`,
        );
      expect(paid).toBeLessThan(0.15);
      expect(sheet.peak).toBeCloseTo(0.1, 2);
      expect(plain.peak).toBeCloseTo(0.1 * paid, 2);
      expect(sheet.alpha / sheetMass).toBeCloseTo(1, 1);
      expect(plain.alpha / plainMass).toBeCloseTo(1, 1);
      // The reflection: alpha x colour as the compensation leaves it, the
      // sheet's over its whole footprint (paid x 0.1 x the colour).
      expect(sheet.red / (linear * paid * sheetMass)).toBeCloseTo(1, 1);
      expect(plain.red / (linear * plainMass)).toBeCloseTo(1, 1);
    });
  }
});

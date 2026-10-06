// The relight plugin on athenea's TX pawn (OpenChessSet Pawn, White): the
// two clouds usd-athc converted (publish-r2/sparkwebgpu/pawn: the body, a
// glossy white material, and the glass top, transmission 1 at index 1.5),
// decoded by the WASM .athc v3 reader, relit under autoshop_01 -- the sky
// athenea baked and validated them under -- from /World/Cam, and measured per
// material as athenea's validate does (relMSE and mean over the material's
// pixels, ~/luc/athenea-renders/pawn-tx/s95/tx/validate.json):
//   - against athenea's raster of that material's cloud in the frame
//     (M_Pawn_*_W_gs.exr), and
//   - against the mesh path traced (gt.exr).
// The splats are drawn over the dome's own image (each pixel's ray read off
// the lat-long), as athenea's frame shows the dome behind.
//
// Read only, and on this machine only: ATHENEA_BATTERY=1 runs it (it decodes
// two 300 MB clouds). PAWN_DIR and ATHENEA_PAWN move the inputs;
// ATHENEA_BATTERY_OUT=<dir> writes the frames as PNGs.

import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import * as THREE from "three";
import { describe, expect, it, vi } from "vitest";
import {
  GpuSplatSource,
  WgpuSplatRenderer,
} from "../../src/webgpu/WgpuSplatRenderer";
import type { SkyImage } from "../../src/webgpu/athenea/AtheneaSky";
import { atheneaRelightPlugin } from "../../src/webgpu/athenea/relightPlugin";
import { AttribPool } from "../../src/webgpu/attributes/schema";
import { PluginHost } from "../../src/webgpu/plugins";
import { loadExr, loadSky, writePng } from "./atheneaFrames";
import { wideDevice } from "./device";

const PAWN =
  process.env.PAWN_DIR ??
  join(homedir(), "luc/sparkwebGPU/publish-r2/sparkwebgpu/pawn");
const RENDERS =
  process.env.ATHENEA_PAWN ??
  join(homedir(), "luc/athenea-renders/pawn-tx/s95/tx");
const SKY = [
  join(homedir(), "tools/assets/hdri/autoshop_01_4k.hdr"),
  join(PAWN, "hdri/autoshop_01_2k.hdr"),
].find((f) => existsSync(f));
const OUT = process.env.ATHENEA_BATTERY_OUT;
const SIZE = 512;

const FILES = {
  body: join(PAWN, "body-full.athc"),
  top: join(PAWN, "top-full.athc"),
};
const available =
  Boolean(process.env.ATHENEA_BATTERY) &&
  Boolean(SKY) &&
  [
    FILES.body,
    FILES.top,
    join(RENDERS, "gt.exr"),
    join(RENDERS, "M_Pawn_Body_W_gs.exr"),
    join(RENDERS, "M_Pawn_Top_W_gs.exr"),
  ].every(existsSync);

type Decoded = {
  numSplats: number;
  ext0: Uint32Array;
  ext1: Uint32Array;
  attribSpecs: { name: string; format: string; components: number }[];
  attribColumns: Float64Array[];
};

/** The cloud's own splats (the file's leaves, after its merged LoD nodes). */
async function loadCloud(device: GPUDevice, file: string) {
  const wasm = await vi.importActual<typeof import("spark-rs")>("spark-rs");
  wasm.initSync({
    module: readFileSync(
      new URL("../../rust/spark-rs/pkg/spark_rs_bg.wasm", import.meta.url),
    ),
  });
  const bytes = new Uint8Array(readFileSync(file));
  const count = new DataView(bytes.buffer, bytes.byteOffset).getUint32(
    12,
    true,
  );
  const decoder = wasm.decode_to_extsplats(undefined, "pawn.athc");
  decoder.push(bytes);
  const d = decoder.finish() as Decoded;
  const first = d.numSplats - count;
  const source = GpuSplatSource.fromExt(
    device,
    d.ext0.subarray(first * 4),
    d.ext1.subarray(first * 4),
    count,
  );
  source.attribs = AttribPool.fromValues({
    count,
    specs: d.attribSpecs as never,
    values: d.attribSpecs.map((s, k) =>
      d.attribColumns[k].subarray(first * s.components),
    ),
  });
  return source;
}

/** The dome along each pixel's ray (domeUv, identity frame), top row first. */
function background(sky: SkyImage, camera: THREE.PerspectiveCamera) {
  const out = new Float32Array(SIZE * SIZE * 4);
  const n = sky.channels ?? 4;
  const inverse = camera.projectionMatrixInverse;
  const dir = new THREE.Vector3();
  const at = (x: number, y: number, c: number) =>
    sky.data[(y * sky.width + x) * n + c];
  for (let py = 0; py < SIZE; py++) {
    for (let px = 0; px < SIZE; px++) {
      dir
        .set(((px + 0.5) / SIZE) * 2 - 1, 1 - ((py + 0.5) / SIZE) * 2, 0.5)
        .applyMatrix4(inverse)
        .transformDirection(camera.matrixWorld);
      const u =
        ((((Math.atan2(dir.z, dir.x) + 0.5 * Math.PI) / (2 * Math.PI)) % 1) +
          1) %
        1;
      // v = 1 - acos(y) / pi counts up from the bottom; rows go down.
      const row =
        (Math.acos(Math.min(Math.max(dir.y, -1), 1)) / Math.PI) * sky.height -
        0.5;
      const col = u * sky.width - 0.5;
      const x0 = Math.floor(col);
      const y0 = Math.floor(row);
      const fx = col - x0;
      const fy = row - y0;
      const xs = [
        ((x0 % sky.width) + sky.width) % sky.width,
        (((x0 + 1) % sky.width) + sky.width) % sky.width,
      ];
      const ys = [
        Math.min(Math.max(y0, 0), sky.height - 1),
        Math.min(Math.max(y0 + 1, 0), sky.height - 1),
      ];
      for (let c = 0; c < 3; c++) {
        const top = at(xs[0], ys[0], c) * (1 - fx) + at(xs[1], ys[0], c) * fx;
        const bottom =
          at(xs[0], ys[1], c) * (1 - fx) + at(xs[1], ys[1], c) * fx;
        out[(py * SIZE + px) * 4 + c] = top * (1 - fy) + bottom * fy;
      }
      out[(py * SIZE + px) * 4 + 3] = 1;
    }
  }
  return out;
}

/** validate's measures over `mask`: relMSE and the mean, rgb. */
function measure(a: Float32Array, b: Float32Array, mask: Uint8Array) {
  let err = 0;
  let n = 0;
  const ma = [0, 0, 0];
  const mb = [0, 0, 0];
  for (let p = 0; p < mask.length; p++) {
    if (!mask[p]) continue;
    for (let c = 0; c < 3; c++) {
      const x = a[p * 4 + c];
      const y = b[p * 4 + c];
      err += ((x - y) * (x - y)) / (y * y + 1e-2);
      ma[c] += x;
      mb[c] += y;
    }
    n += 1;
  }
  return {
    relMse: err / (3 * n),
    ratio: (ma[0] + ma[1] + ma[2]) / (mb[0] + mb[1] + mb[2]),
    pixels: n,
  };
}

describe.skipIf(!wideDevice || !available)(
  "athenea relight on the TX pawn",
  () => {
    const d = wideDevice as GPUDevice;
    const color = d.createTexture({
      size: [SIZE, SIZE],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
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
    // /World/Cam: 50 mm over 36 mm, at (0, 0.048, 0.18) looking down -Z.
    const camera = new THREE.PerspectiveCamera(
      2 * THREE.MathUtils.radToDeg(Math.atan(18 / 50)),
      1,
      0.01,
      10,
    );
    camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
    camera.position.set(0, 0.048, 0.18);
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld();

    async function draw(splats: WgpuSplatRenderer): Promise<Float32Array> {
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
      const bytesPerRow = SIZE * 8;
      const buf = d.createBuffer({
        size: bytesPerRow * SIZE,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const copy = d.createCommandEncoder();
      copy.copyTextureToBuffer(
        { texture: color },
        { buffer: buf, bytesPerRow },
        [SIZE, SIZE],
      );
      d.queue.submit([copy.finish()]);
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

    it("relights the body and the glass top as athenea's raster does", async () => {
      const sky = loadSky(SKY as string);
      const body = await loadCloud(d, FILES.body);
      const top = await loadCloud(d, FILES.top);
      const splats = new WgpuSplatRenderer(fakeRenderer as never, {
        depthTest: false,
        alwaysGenerate: true,
      });
      const host = new PluginHost({
        capabilities: splats.capabilities,
        tier: 2,
      });
      const relight = atheneaRelightPlugin({ hdri: sky });
      host.register(relight).attach(splats);
      // One cloud at a time for its coverage, then both.
      const bodyMesh = splats.add(body);
      await host.ready();
      const bodyAlone = await draw(splats);
      splats.remove(bodyMesh);
      const topMesh = splats.add(top);
      relight.setIor(topMesh, 1.5);
      const topAlone = await draw(splats);
      splats.add(body);
      const both = await draw(splats);
      const bg = background(sky, camera);
      const ours = new Float32Array(both.length);
      for (let p = 0; p < SIZE * SIZE; p++) {
        const a = both[p * 4 + 3];
        for (let c = 0; c < 3; c++)
          ours[p * 4 + c] = both[p * 4 + c] + (1 - a) * bg[p * 4 + c];
        ours[p * 4 + 3] = 1;
      }
      const gt = loadExr(join(RENDERS, "gt.exr")).data;
      const report: string[] = [];
      for (const [name, alone, other, frame] of [
        ["M_Pawn_Body_W", bodyAlone, topAlone, "M_Pawn_Body_W_gs.exr"],
        ["M_Pawn_Top_W", topAlone, bodyAlone, "M_Pawn_Top_W_gs.exr"],
      ] as const) {
        // The glass is drawn at what it covers (its opacity), well under one.
        const cover = name === "M_Pawn_Top_W" ? 0.3 : 0.99;
        const mask = new Uint8Array(SIZE * SIZE);
        for (let p = 0; p < mask.length; p++) {
          mask[p] = alone[p * 4 + 3] > cover && other[p * 4 + 3] < 0.01 ? 1 : 0;
        }
        const athenea = loadExr(join(RENDERS, frame)).data;
        const vsAthenea = measure(ours, athenea, mask);
        const vsGt = measure(ours, gt, mask);
        const atheneaVsGt = measure(athenea, gt, mask);
        const row = `${name.padEnd(14)} ${vsGt.pixels} px | ours/athenea ratio ${vsAthenea.ratio.toFixed(3)} relMSE ${vsAthenea.relMse.toFixed(4)} | ours/GT ratio ${vsGt.ratio.toFixed(3)} relMSE ${vsGt.relMse.toFixed(4)} | athenea/GT ratio ${atheneaVsGt.ratio.toFixed(3)} relMSE ${atheneaVsGt.relMse.toFixed(4)}`;
        report.push(row);
        process.stderr.write(`${row}\n`);
        if (OUT) {
          mkdirSync(OUT, { recursive: true });
          writePng(join(OUT, `pawn_${name}_athenea.png`), athenea, SIZE, SIZE);
        }
        expect(vsGt.pixels).toBeGreaterThan(500);
      }
      if (OUT) {
        writePng(join(OUT, "pawn_ours.png"), ours, SIZE, SIZE);
        writePng(join(OUT, "pawn_gt.png"), gt, SIZE, SIZE);
      }
      host.detach();
      splats.dispose();
    }, 600_000);
  },
);

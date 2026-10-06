// The skin plugin on athenea's skinned, relit sparrow against athenea's own
// frames: SparrowBird.usda (the Eurasian tree sparrow, 609 joints, a 61-frame
// flight at 30 fps) converted by athenea mesh2splat --skinned --transfer
// --time 1 (athenea-tx 4c7dc40: four joints, weight gradients and a zonal
// transfer a gaussian), then by usd-athc (sparrow160tx.athc), posed at a
// time code by atheneaSkinPlugin and relit under a white dome; athenea draws
// the same .usdc through its Hydra raster at the same time code from the same
// camera (sparrow_ref.usda: athenea stage --technique raster). Measured as
// athenea's validate measures (relMSE and the mean ratio over the bird's
// pixels), against the rest pose and the neighbouring time codes, so a
// wrong pose, a wrong time or a transposed matrix shows.
//
// Read only, and on this machine only: ATHENEA_SKIN_FRAMES=1 runs it;
// ATHENEA_SKIN_DIR moves the inputs (default ~/luc/sparkwebGPU/
// athenea-skin-data), ATHENEA_SKIN_OUT=<dir> writes the frames as PNGs.

import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import * as THREE from "three";
import { describe, expect, it, vi } from "vitest";
import type { AthcSkeleton } from "../../src/athc";
import {
  GpuSplatSource,
  WgpuSplatRenderer,
} from "../../src/webgpu/WgpuSplatRenderer";
import { atheneaRelightPlugin } from "../../src/webgpu/athenea/relightPlugin";
import { atheneaSkinPlugin } from "../../src/webgpu/athenea/skinPlugin";
import { AttribPool, attribWords } from "../../src/webgpu/attributes/schema";
import { PluginHost } from "../../src/webgpu/plugins";
import { loadExr, writePng } from "./atheneaFrames";
import { wideDevice } from "./device";

const DIR =
  process.env.ATHENEA_SKIN_DIR ??
  join(homedir(), "luc/sparkwebGPU/athenea-skin-data");
const OUT = process.env.ATHENEA_SKIN_OUT;
const W = 800;
const H = 600;
const TIMES = [1, 20, 40];
const CLOUD = join(DIR, "sparrow160tx.athc");
const available =
  Boolean(process.env.ATHENEA_SKIN_FRAMES) &&
  [CLOUD, ...TIMES.map((t) => join(DIR, `athenea_t${t}.exr`))].every(
    existsSync,
  );

type Decoded = {
  numSplats: number;
  ext0: Uint32Array;
  ext1: Uint32Array;
  attribSpecs: {
    name: string;
    format: string;
    components: number;
    packed?: boolean;
  }[];
  attribColumns: (Float64Array | Uint32Array)[];
};

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
  const decoder = wasm.decode_to_extsplats(undefined, "sparrow.athc");
  decoder.push(bytes);
  const d = decoder.finish() as Decoded;
  // The cloud's own splats (the file's leaves, after its merged LoD nodes).
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
      d.attribColumns[k].subarray(
        first * (s.packed ? attribWords(s as never) : s.components),
      ),
    ),
  });
  return { source, skeleton: wasm.athc_skeleton(bytes) as AthcSkeleton };
}

/** relMSE and the mean ratio over `mask` (athenea's validate). */
function measure(a: Float32Array, b: Float32Array, mask: Uint8Array) {
  let err = 0;
  let n = 0;
  let sa = 0;
  let sb = 0;
  for (let p = 0; p < mask.length; p++) {
    if (!mask[p]) continue;
    for (let c = 0; c < 3; c++) {
      const x = a[p * 4 + c];
      const y = b[p * 4 + c];
      err += ((x - y) * (x - y)) / (y * y + 1e-2);
      sa += x;
      sb += y;
    }
    n += 1;
  }
  return { relMse: err / (3 * n), ratio: sa / sb, pixels: n };
}

describe.skipIf(!wideDevice || !available)(
  "athenea skinning: the sparrow against athenea's frames",
  () => {
    const d = wideDevice as GPUDevice;
    const color = d.createTexture({
      size: [W, H],
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
    // sparrow_ref.usda's /Cam in the stage's own (Z-up) space: 35 mm over a
    // 24 x 18 mm aperture, translate then rotateXYZ (78, 0, 8).
    const camera = new THREE.PerspectiveCamera(
      2 * THREE.MathUtils.radToDeg(Math.atan(9 / 35)),
      W / H,
      0.01,
      100,
    );
    camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
    camera.position.set(0.05, -0.42, 0.12);
    camera.rotation.set(
      THREE.MathUtils.degToRad(78),
      0,
      THREE.MathUtils.degToRad(8),
      "ZYX",
    );
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
      const bytesPerRow = Math.ceil((W * 8) / 256) * 256;
      const buf = d.createBuffer({
        size: bytesPerRow * H,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const copy = d.createCommandEncoder();
      copy.copyTextureToBuffer(
        { texture: color },
        { buffer: buf, bytesPerRow },
        [W, H],
      );
      d.queue.submit([copy.finish()]);
      await buf.mapAsync(GPUMapMode.READ);
      const halves = new Uint16Array(buf.getMappedRange());
      const out = new Float32Array(W * H * 4);
      for (let y = 0; y < H; y++) {
        for (let k = 0; k < W * 4; k++) {
          out[y * W * 4 + k] = THREE.DataUtils.fromHalfFloat(
            halves[(y * bytesPerRow) / 2 + k],
          );
        }
      }
      buf.unmap();
      buf.destroy();
      // Over the white dome athenea's frame shows behind.
      for (let p = 0; p < W * H; p++) {
        const a = out[p * 4 + 3];
        for (let c = 0; c < 3; c++) out[p * 4 + c] += 1 - a;
      }
      return out;
    }

    it("poses the sparrow as athenea does at its time codes", async () => {
      const { source, skeleton } = await loadCloud(d, CLOUD);
      const splats = new WgpuSplatRenderer(fakeRenderer as never, {
        depthTest: false,
        alwaysGenerate: true,
      });
      const host = new PluginHost({
        capabilities: splats.capabilities,
        tier: 2,
      });
      const white = {
        width: 64,
        height: 32,
        data: new Float32Array(64 * 32 * 4).fill(1),
        channels: 4,
      };
      const relight = atheneaRelightPlugin({ hdri: white });
      const skin = atheneaSkinPlugin({
        skeleton,
        playing: false,
        time: TIMES[0],
      });
      host.register(skin).register(relight).attach(splats);
      splats.add(source);
      await host.ready();
      const ours = new Map<number, Float32Array>();
      for (const t of [...TIMES, 18, 22]) {
        skin.set({ time: t });
        ours.set(t, await draw(splats));
      }
      // The blend of the joints' linear parts alone (athenea before 7dff879).
      const blendOnly = new Map<number, Float32Array>();
      skin.set({ jacobian: false });
      for (const t of TIMES) {
        skin.set({ time: t });
        blendOnly.set(t, await draw(splats));
      }
      skin.set({ jacobian: true });
      // The bind pose, as converted.
      skin.set({ bindPose: true });
      const rest = await draw(splats);
      const rows: string[] = [];
      for (const t of TIMES) {
        const athenea = loadExr(join(DIR, `athenea_t${t}.exr`)).data;
        const mine = ours.get(t) as Float32Array;
        // The bird in either frame (darker than the dome somewhere).
        const mask = new Uint8Array(W * H);
        for (let p = 0; p < mask.length; p++) {
          const dark = (f: Float32Array) =>
            f[p * 4] + f[p * 4 + 1] + f[p * 4 + 2] < 2.9;
          mask[p] = dark(mine) || dark(athenea) ? 1 : 0;
        }
        const same = measure(mine, athenea, mask);
        const atRest = measure(rest, athenea, mask);
        const blend = measure(blendOnly.get(t) as Float32Array, athenea, mask);
        const near = [t - 2, t + 2]
          .filter((u) => ours.has(u))
          .map((u) => measure(ours.get(u) as Float32Array, athenea, mask));
        const row = `t ${t}: ${same.pixels} px | posed relMSE ${same.relMse.toFixed(4)} ratio ${same.ratio.toFixed(3)} | blend only relMSE ${blend.relMse.toFixed(4)} | rest relMSE ${atRest.relMse.toFixed(4)}${near.map((m) => ` | t±2 relMSE ${m.relMse.toFixed(4)}`).join("")}`;
        rows.push(row);
        process.stderr.write(`${row}\n`);
        if (OUT) {
          mkdirSync(OUT, { recursive: true });
          writePng(join(OUT, `sparrow_t${t}_ours.png`), mine, W, H);
          writePng(join(OUT, `sparrow_t${t}_athenea.png`), athenea, W, H);
        }
        expect(same.pixels).toBeGreaterThan(2000);
        expect(same.relMse).toBeLessThan(atRest.relMse);
        // athenea draws it by the whole Jacobian (its weights' gradients).
        expect(same.relMse).toBeLessThan(blend.relMse);
        for (const m of near) expect(same.relMse).toBeLessThan(m.relMse);
      }
      if (OUT) writePng(join(OUT, "sparrow_rest_ours.png"), rest, W, H);
      host.detach();
      splats.dispose();
    }, 600_000);
  },
);

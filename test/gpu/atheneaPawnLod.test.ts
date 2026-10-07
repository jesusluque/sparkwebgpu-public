// The pawn's glass head drawn from its merged LoD levels, as the page draws
// it (a SplatMesh's LoD cut picks merged nodes wherever the 0.05 mm splats
// are under a pixel): each level of top-full-gz.athc drawn alone, close up,
// against the head drawn from its splats. A merged gaussian's axes were its
// moments' eigenvectors, in any order and sign, and the relight reads the
// curvature on its third axis' side: half of every level's cells read the
// head as a concave lens (a checker of dark cells, no lens image).
// athc::orient_merged turns them to the stored normal.
//
// Also: the head drawn from a cut after the eye moved is the head a fresh
// renderer draws there (the per-eye relight follows the eye).
//
// Read only, and on this machine only: ATHENEA_BATTERY=1 runs it (it decodes
// the 180 MB top). PAWN_DIR moves the input; ATHENEA_BATTERY_OUT=<dir> writes
// the frames as PNGs.

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
import { loadSky, writePng } from "./atheneaFrames";
import { wideDevice } from "./device";

const PAWN =
  process.env.PAWN_DIR ??
  join(homedir(), "luc/sparkwebGPU/publish-r2/sparkwebgpu/pawn");
const SKY = [
  join(homedir(), "tools/assets/hdri/autoshop_01_4k.hdr"),
  join(PAWN, "hdri/autoshop_01_2k.hdr"),
].find((f) => existsSync(f));
const TOP = join(PAWN, "top-full-gz.athc");
const OUT = process.env.ATHENEA_BATTERY_OUT;
const SIZE = 512;
// top-full-gz.athc: its splats and its levels' groups (1 .. 8), coarsest
// first, after the root the decoder adds.
const LEAVES = 673711;
const LEVELS = [8, 56, 272, 1142, 4555, 17701, 66852, 234275];

const available =
  Boolean(process.env.ATHENEA_BATTERY) && Boolean(SKY) && existsSync(TOP);

type Decoded = {
  numSplats: number;
  ext0: Uint32Array;
  ext1: Uint32Array;
  attribSpecs: { name: string; format: string; components: number }[];
  attribColumns: (Float64Array | Uint32Array)[];
};

/** The whole decoded cloud: its merged LoD nodes, then its splats. */
async function loadWhole(device: GPUDevice, file: string) {
  const wasm = await vi.importActual<typeof import("spark-rs")>("spark-rs");
  wasm.initSync({
    module: readFileSync(
      new URL("../../rust/spark-rs/pkg/spark_rs_bg.wasm", import.meta.url),
    ),
  });
  const decoder = wasm.decode_to_extsplats(undefined, "top.athc");
  decoder.push(new Uint8Array(readFileSync(file)));
  const d = decoder.finish() as Decoded;
  const source = GpuSplatSource.fromExt(device, d.ext0, d.ext1, d.numSplats);
  source.attribs = AttribPool.fromValues({
    count: d.numSplats,
    specs: d.attribSpecs as never,
    values: d.attribColumns as never,
  });
  return { source, count: d.numSplats };
}

/** The dome along each pixel's ray (domeUv, identity frame), nearest texel. */
function background(sky: SkyImage, camera: THREE.PerspectiveCamera) {
  const out = new Float32Array(SIZE * SIZE * 4);
  const n = sky.channels ?? 4;
  const dir = new THREE.Vector3();
  for (let py = 0; py < SIZE; py++) {
    for (let px = 0; px < SIZE; px++) {
      dir
        .set(((px + 0.5) / SIZE) * 2 - 1, 1 - ((py + 0.5) / SIZE) * 2, 0.5)
        .applyMatrix4(camera.projectionMatrixInverse)
        .transformDirection(camera.matrixWorld);
      const u =
        ((((Math.atan2(dir.z, dir.x) + 0.5 * Math.PI) / (2 * Math.PI)) % 1) +
          1) %
        1;
      const v = Math.acos(Math.min(Math.max(dir.y, -1), 1)) / Math.PI;
      const x = Math.min(Math.floor(u * sky.width), sky.width - 1);
      const y = Math.min(Math.floor(v * sky.height), sky.height - 1);
      for (let c = 0; c < 3; c++)
        out[(py * SIZE + px) * 4 + c] = sky.data[(y * sky.width + x) * n + c];
      out[(py * SIZE + px) * 4 + 3] = 1;
    }
  }
  return out;
}

const luminance = (f: Float32Array, p: number) =>
  0.2126 * f[p * 4] + 0.7152 * f[p * 4 + 1] + 0.0722 * f[p * 4 + 2];

describe.skipIf(!wideDevice || !available)(
  "athenea relight on the pawn's merged glass",
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
    // A 260 mm lens from /World/Cam's place (or `x` to its side), on the head.
    const camera = (x = 0) => {
      const c = new THREE.PerspectiveCamera(
        2 * THREE.MathUtils.radToDeg(Math.atan(18 / 260)),
        1,
        0.01,
        10,
      );
      c.coordinateSystem = THREE.WebGPUCoordinateSystem;
      c.position.set(x, 0.048, 0.18);
      c.lookAt(0, 0.07, 0);
      c.updateProjectionMatrix();
      c.updateMatrixWorld();
      return c;
    };

    async function draw(splats: WgpuSplatRenderer, cam: THREE.Camera) {
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
      splats.render(cam, target);
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

    function relit(source: GpuSplatSource, sky: SkyImage) {
      const splats = new WgpuSplatRenderer(fakeRenderer as never, {
        depthTest: false,
      });
      const host = new PluginHost({
        capabilities: splats.capabilities,
        tier: 2,
      });
      const relight = atheneaRelightPlugin({ hdri: sky });
      host.register(relight).attach(splats);
      const mesh = splats.add(source);
      relight.setIor(mesh, 1.5);
      return { splats, host, mesh };
    }

    it("draws the head's merged levels as its splats, and follows the eye", async () => {
      const sky = loadSky(SKY as string);
      const { source, count } = await loadWhole(d, TOP);
      const merged = LEVELS.reduce((a, b) => a + b, 0);
      const root = count - LEAVES - merged;
      expect(root === 0 || root === 1).toBe(true);
      const range = (a: number, b: number) =>
        Uint32Array.from({ length: b - a }, (_, k) => a + k);
      const cuts: [string, Uint32Array][] = [
        ["splats", range(count - LEAVES, count)],
      ];
      let at = root;
      LEVELS.forEach((n, l) => {
        if (l >= 5) cuts.push([`level ${l + 1}`, range(at, at + n)]);
        at += n;
      });
      const { splats, host, mesh } = relit(source, sky);
      await host.ready();
      const front = camera();
      const bg = background(sky, front);
      let reference: Float32Array | null = null;
      const shown = (f: Float32Array) => {
        const out = new Float32Array(f.length);
        for (let p = 0; p < SIZE * SIZE; p++) {
          for (let c = 0; c < 3; c++)
            out[p * 4 + c] = f[p * 4 + c] + (1 - f[p * 4 + 3]) * bg[p * 4 + c];
          out[p * 4 + 3] = f[p * 4 + 3];
        }
        return out;
      };
      const rows: { name: string; err: number; dark: number }[] = [];
      for (const [name, list] of cuts) {
        splats.setLodIndices(mesh, list);
        const f = shown(await draw(splats, front));
        if (OUT) {
          mkdirSync(OUT, { recursive: true });
          writePng(
            join(OUT, `pawn_lod_${name.replace(" ", "")}.png`),
            f,
            SIZE,
            SIZE,
          );
        }
        if (!reference) {
          reference = f;
          continue;
        }
        // Over the head as shown (over the dome), against its splats: relMSE
        // of the luminance, and the pixels at less than half of it (the
        // dark cells).
        let err = 0;
        let dark = 0;
        let n = 0;
        for (let p = 0; p < SIZE * SIZE; p++) {
          if (reference[p * 4 + 3] < 0.3) continue;
          const a = luminance(f, p);
          const b = luminance(reference, p);
          err += ((a - b) * (a - b)) / (b * b + 1e-2);
          if (a < 0.5 * b) dark += 1;
          n += 1;
        }
        process.stderr.write(
          `${name}: ${list.length} nodes, relMSE ${(err / n).toFixed(4)} against the splats, ${((100 * dark) / n).toFixed(2)}% of ${n} px under half\n`,
        );
        expect(n).toBeGreaterThan(20000);
        rows.push({ name, err: err / n, dark: dark / n });
      }
      for (const r of rows) {
        expect(r.dark, r.name).toBeLessThan(0.01);
        expect(r.err, r.name).toBeLessThan(0.1);
      }
      // The eye moves with a cut in place: what is drawn is what a fresh
      // renderer draws from there.
      const cut = cuts[cuts.length - 1][1];
      splats.setLodIndices(mesh, cut);
      const side = camera(0.06);
      const moved = await draw(splats, side);
      host.detach();
      splats.dispose();
      const fresh = relit(source, sky);
      await fresh.host.ready();
      fresh.splats.setLodIndices(fresh.mesh, cut);
      const there = await draw(fresh.splats, side);
      let diff = 0;
      let sum = 0;
      for (let k = 0; k < there.length; k++) {
        diff += Math.abs(moved[k] - there[k]);
        sum += Math.abs(there[k]);
      }
      process.stderr.write(
        `moved eye: |drawn - fresh| / |fresh| ${(diff / sum).toExponential(2)}\n`,
      );
      expect(diff / sum).toBeLessThan(1e-4);
      fresh.host.detach();
      fresh.splats.dispose();
    }, 900_000);
  },
);

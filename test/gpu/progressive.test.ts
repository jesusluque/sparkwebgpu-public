// ProgressiveAccumulator (src/webgpu/fx): jittered frames of a still view
// average to the pixel's box, and any change starts over.
//
// The "renderer" here is the CPU: a hard vertical edge, each pixel taking
// the value at its centre moved by the jitter the camera's projection
// carries this frame, written into an rgba32float texture. The mean of the
// Halton samples must approach the edge pixel's coverage, equal the plain
// average of what was drawn, and give way to the plain frame on a change.

import * as THREE from "three";
import { describe, expect, it } from "vitest";
import {
  FxChain,
  ProgressiveAccumulator,
  centredJitter,
  halton,
  jitterOffset,
} from "../../src/webgpu/fx";
import { createReadback, readAndDestroy } from "../../src/webgpu/gpuBuffers";
import { device } from "./device";

const W = 16; // 16 x 16 B = 256 B rows: no copy padding
const H = 8;
const EDGE = 9.3; // the edge's x in pixels: pixel 9 is 30% left of it

function makeTexture(d: GPUDevice) {
  return d.createTexture({
    size: { width: W, height: H },
    format: "rgba32float",
    usage:
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.TEXTURE_BINDING,
  });
}

// The jitter in pixels that `camera`'s projection carries over `base`.
function jitterOf(camera: THREE.PerspectiveCamera, base: THREE.Matrix4) {
  const e = camera.projectionMatrix.elements;
  const b = base.elements;
  // Rows 0/1 of column 2 gain n * (w row = -1).
  return [((b[8] - e[8]) * W) / 2, ((b[9] - e[9]) * H) / 2];
}

// Draws the edge as seen through the camera's jitter; returns the pixels.
function draw(
  d: GPUDevice,
  texture: GPUTexture,
  camera: THREE.PerspectiveCamera,
  base: THREE.Matrix4,
  level = 1,
) {
  const [dx] = jitterOf(camera, base);
  const data = new Float32Array(W * H * 4);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const v = x + 0.5 + dx < EDGE ? level : 0;
      data.set([v, v * 0.5, v * 0.25, 1], (y * W + x) * 4);
    }
  d.queue.writeTexture({ texture }, data, { bytesPerRow: W * 16 }, [W, H]);
  return data;
}

async function read(d: GPUDevice, texture: GPUTexture) {
  const staging = createReadback(d, W * H * 16, "test read");
  const enc = d.createCommandEncoder();
  enc.copyTextureToBuffer(
    { texture },
    { buffer: staging, bytesPerRow: W * 16 },
    [W, H],
  );
  d.queue.submit([enc.finish()]);
  return new Float32Array(await readAndDestroy(staging));
}

describe.skipIf(!device)("ProgressiveAccumulator", () => {
  it("halton (2, 3) offsets are the radical inverses, centred", () => {
    expect(halton(1, 2)).toBe(0.5);
    expect(halton(3, 2)).toBe(0.75);
    expect(halton(2, 3)).toBeCloseTo(2 / 3, 12);
    expect(jitterOffset(0)).toEqual([0, 1 / 3 - 0.5]);
    // Centred: the finished mean's offsets average to the pixel's centre.
    for (const n of [1, 4, 16]) {
      let sx = 0;
      let sy = 0;
      for (let k = 0; k < n; k++) {
        const [x, y] = centredJitter(k, n);
        sx += x;
        sy += y;
      }
      expect(Math.abs(sx / n) + Math.abs(sy / n)).toBeLessThan(1e-12);
    }
    for (let k = 0; k < 64; k++) {
      const [x, y] = jitterOffset(k);
      expect(Math.abs(x)).toBeLessThan(0.5 + 1e-12);
      expect(Math.abs(y)).toBeLessThan(0.5 + 1e-12);
    }
  });

  it("averages jittered frames to the pixel's box and resets on change", async () => {
    const d = device as GPUDevice;
    const texture = makeTexture(d);
    const camera = new THREE.PerspectiveCamera(50, W / H, 0.1, 100);
    camera.position.set(0, 0, 5);
    camera.updateMatrixWorld();
    const base = camera.projectionMatrix.clone();
    const spark = {
      blurAmount: 0.3,
      webgpu: { splats: { contentVersion: 0 } },
    };
    const accum = new ProgressiveAccumulator(d, { frames: 16, blur: 0.1 });
    const frame = () => ({ camera, width: W, height: H, spark });
    const row = (px: Float32Array, x: number) => px[(2 * W + x) * 4];

    // First frame: nothing to compare with -> plain, the frame as drawn.
    expect(accum.begin(frame())).toBe("plain");
    expect(spark.blurAmount).toBe(0.3);
    draw(d, texture, camera, base);
    accum.end(texture);
    expect(row(await read(d, texture), 9)).toBe(0); // centre 9.5 > 9.3

    // Still: 16 jittered samples, the blur sharpened only while drawing.
    const drawn: number[] = [];
    for (let k = 0; k < 16; k++) {
      expect(accum.begin(frame())).toBe("sample");
      expect(spark.blurAmount).toBe(0.1);
      const [dx, dy] = jitterOf(camera, base);
      expect(dx).toBeCloseTo(centredJitter(k, 16)[0], 5);
      expect(dy).toBeCloseTo(centredJitter(k, 16)[1], 5);
      // A point in view moves by exactly the jitter on screen.
      const p = new THREE.Vector3(0.3, -0.2, 0).project(camera);
      const q = new THREE.Vector3(0.3, -0.2, 0).applyMatrix4(
        new THREE.Matrix4().multiplyMatrices(base, camera.matrixWorldInverse),
      );
      expect(((p.x - q.x) * W) / 2).toBeCloseTo(dx, 4);
      const px = draw(d, texture, camera, base);
      drawn.push(row(px, 9));
      accum.end(texture);
      expect(spark.blurAmount).toBe(0.3);
      expect(camera.projectionMatrix.equals(base)).toBe(true);
      expect(camera.userData.unjitteredProjectionMatrix).toBeUndefined();
      const shown = await read(d, texture);
      const mean = drawn.reduce((a, b) => a + b, 0) / drawn.length;
      expect(row(shown, 9)).toBeCloseTo(mean, 5);
      expect(shown[(2 * W + 9) * 4 + 1]).toBeCloseTo(mean * 0.5, 5);
    }
    expect(accum.converged).toBe(true);
    // The box: 30% of pixel 9 lies left of the edge.
    const converged = row(await read(d, texture), 9);
    expect(Math.abs(converged - 0.3)).toBeLessThan(1 / 16 + 1e-6);
    expect(row(await read(d, texture), 8)).toBe(1);
    expect(row(await read(d, texture), 10)).toBe(0);

    // Converged: the ordinary frame is drawn, the mean shown over it.
    expect(accum.begin(frame())).toBe("hold");
    expect(spark.blurAmount).toBe(0.3);
    draw(d, texture, camera, base, 7);
    accum.end(texture);
    expect(row(await read(d, texture), 9)).toBeCloseTo(converged, 6);

    // The camera moves: the plain frame, as drawn, then a new mean.
    camera.position.x += 0.01;
    camera.updateMatrixWorld();
    expect(accum.begin(frame())).toBe("plain");
    draw(d, texture, camera, base, 5);
    accum.end(texture);
    expect(row(await read(d, texture), 8)).toBe(5);
    expect(accum.samples).toBe(0);
    expect(accum.begin(frame())).toBe("sample");
    draw(d, texture, camera, base, 2);
    accum.end(texture);
    expect(accum.samples).toBe(1);
    expect(row(await read(d, texture), 8)).toBe(2);

    // The splats change (contentVersion), the page's version, reset().
    spark.webgpu.splats.contentVersion += 1;
    expect(accum.begin(frame())).toBe("plain");
    accum.end(texture);
    expect(accum.begin(frame())).toBe("sample");
    accum.end(texture);
    expect(accum.begin({ ...frame(), version: 1 })).toBe("plain");
    accum.end(texture);
    accum.reset();
    expect(accum.begin({ ...frame(), version: 1 })).toBe("plain");
    accum.end(texture);
    // A change while a sample draws: that frame stands, the next is plain.
    expect(accum.begin({ ...frame(), version: 1 })).toBe("sample");
    draw(d, texture, camera, base, 3);
    spark.webgpu.splats.contentVersion += 1;
    accum.end(texture);
    expect(accum.samples).toBe(0);
    expect(accum.mode).toBe("plain");
    expect(row(await read(d, texture), 8)).toBe(3);
    expect(accum.begin({ ...frame(), version: 1 })).toBe("plain");
    accum.end(texture);

    // frames 0: off, nothing touched.
    accum.frames = 0;
    expect(accum.begin(frame())).toBe("off");
    accum.end(texture);
    accum.dispose();
    texture.destroy();
  });

  it("runs the chain over the mean", async () => {
    const d = device as GPUDevice;
    const texture = makeTexture(d);
    const camera = new THREE.PerspectiveCamera(50, W / H, 0.1, 100);
    camera.updateMatrixWorld();
    const base = camera.projectionMatrix.clone();
    const chain = new FxChain(d);
    const { Invert } = await import("../../src/webgpu/fx");
    chain.add(new Invert());
    const accum = new ProgressiveAccumulator(chain, { frames: 4 });
    const frame = { camera, width: W, height: H };
    accum.begin(frame);
    draw(d, texture, camera, base);
    accum.end(texture); // plain: the chain on the frame
    expect((await read(d, texture))[(2 * W + 8) * 4]).toBeCloseTo(0, 6);
    const drawn: number[] = [];
    for (let k = 0; k < 4; k++) {
      expect(accum.begin(frame)).toBe("sample");
      drawn.push(draw(d, texture, camera, base)[(2 * W + 9) * 4]);
      accum.end(texture);
    }
    const mean = drawn.reduce((a, b) => a + b, 0) / 4;
    // Invert after the mean: 1 - mean.
    expect((await read(d, texture))[(2 * W + 9) * 4]).toBeCloseTo(1 - mean, 5);
    accum.dispose();
    texture.destroy();
  });
});

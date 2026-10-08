// WgpuSplatRenderer's rendering features on Dawn (depth of field, covariance
// splats...): each rendered offscreen and checked against an equivalent
// rendering without the feature.

import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { RgbaArray } from "../../src/RgbaArray";
import { CovSplatTransformer } from "../../src/SplatGenerator";
import { maybeInjectSplatRgba } from "../../src/SplatMesh";
import { SplatSkinning, SplatSkinningMode } from "../../src/SplatSkinning";
import { dynoBlock } from "../../src/dyno/base";
import { CovSplat, Gsplat, splitGsplat } from "../../src/dyno/splats";
import { dynoConst } from "../../src/dyno/value";
import { encodeExtSplat } from "../../src/utils";
import {
  GpuSplatSource,
  type WgpuDyno,
  WgpuSplatRenderer,
  type WgpuSplatRendererOptions,
} from "../../src/webgpu/WgpuSplatRenderer";
import { device } from "./device";

const W = 96;
const H = 64;

interface TestSplat {
  center: number[];
  scales: number[];
  quat?: number[];
  rgba: number[];
}

// A grid of small axis-aligned splats in the z = 0 plane.
function grid(scales = [0.04, 0.02, 0.03]): TestSplat[] {
  const out: TestSplat[] = [];
  for (let y = -4; y <= 4; y++) {
    for (let x = -6; x <= 6; x++) {
      out.push({
        center: [x * 0.12, y * 0.12, 0],
        scales,
        rgba: [0.5 + x / 12, 0.5 + y / 8, 0.6, 0.8],
      });
    }
  }
  return out;
}

describe.skipIf(!device)("WgpuSplatRenderer features", () => {
  const d = device as GPUDevice;
  const canvas = d.createTexture({
    size: [W, H],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const fakeRenderer = {
    backend: {
      isWebGPUBackend: true,
      device: d,
      context: {
        getCurrentTexture: () => canvas,
      } as unknown as GPUCanvasContext,
      get: () => undefined,
    },
  };

  async function readCanvas(): Promise<Uint8Array> {
    const bytesPerRow = 256 * Math.ceil((W * 4) / 256);
    const buf = d.createBuffer({
      size: bytesPerRow * H,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = d.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: canvas }, { buffer: buf, bytesPerRow }, [
      W,
      H,
    ]);
    d.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buf.getMappedRange());
    const out = new Uint8Array(W * H * 4);
    for (let y = 0; y < H; y++) {
      out.set(
        src.subarray(y * bytesPerRow, y * bytesPerRow + W * 4),
        y * W * 4,
      );
    }
    buf.unmap();
    buf.destroy();
    return out;
  }

  function clearCanvas() {
    const enc = d.createCommandEncoder();
    enc
      .beginRenderPass({
        colorAttachments: [
          {
            view: canvas.createView(),
            loadOp: "clear",
            storeOp: "store",
            clearValue: [0, 0, 0, 1],
          },
        ],
      })
      .end();
    d.queue.submit([enc.finish()]);
  }

  async function render(
    list: TestSplat[],
    options: WgpuSplatRendererOptions,
    object = new THREE.Object3D(),
    dyno?: WgpuDyno,
  ): Promise<Uint8Array> {
    const a = new Uint32Array(list.length * 4);
    const b = new Uint32Array(list.length * 4);
    list.forEach((s, i) => {
      const q = s.quat ?? [0, 0, 0, 1];
      encodeExtSplat(
        [a, b],
        i,
        s.center[0],
        s.center[1],
        s.center[2],
        s.scales[0],
        s.scales[1],
        s.scales[2],
        q[0],
        q[1],
        q[2],
        q[3],
        s.rgba[3],
        s.rgba[0],
        s.rgba[1],
        s.rgba[2],
      );
    });
    const splats = new WgpuSplatRenderer(fakeRenderer as never, {
      depthTest: false,
      ...options,
    });
    splats.add(GpuSplatSource.fromExt(d, a, b, list.length), object, dyno);
    const camera = new THREE.PerspectiveCamera(60, W / H, 0.05, 100);
    camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
    camera.updateProjectionMatrix();
    camera.position.set(0.1, 0.2, 1.6);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    clearCanvas();
    splats.render(camera);
    await d.queue.onSubmittedWorkDone();
    const px = await readCanvas();
    splats.dispose();
    return px;
  }

  function meanDiff(a: Uint8Array, b: Uint8Array) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
    return sum / a.length;
  }

  function lit(px: Uint8Array, threshold = 30) {
    let n = 0;
    for (let i = 0; i < W * H; i++) {
      if (px[i * 4] + px[i * 4 + 1] > threshold) n++;
    }
    return n;
  }

  it("covariance splats match Gsplats under a similarity transform", async () => {
    const object = new THREE.Object3D();
    object.rotation.set(0.3, -0.4, 0.2);
    object.scale.setScalar(1.3);
    const list = grid().map((s, i) => {
      const q = new THREE.Quaternion()
        .setFromEuler(new THREE.Euler(i * 0.7, i * 0.3, i * 1.1))
        .toArray();
      return { ...s, quat: q };
    });
    const plain = await render(list, {}, object);
    const cov = await render(list, { covSplats: true }, object);
    expect(lit(plain)).toBeGreaterThan(W * H * 0.1);
    expect(meanDiff(plain, cov)).toBeLessThan(0.5);
  });

  it("covariance splats take a non-uniform object scale", async () => {
    const s = [1.8, 0.6, 1.2];
    const object = new THREE.Object3D();
    object.scale.set(s[0], s[1], s[2]);
    const cov = await render(grid(), { covSplats: true }, object);
    // Axis-aligned splats: the same scaling baked into centers and scales.
    const baked = grid().map((g) => ({
      ...g,
      center: g.center.map((c, k) => c * s[k]),
      scales: g.scales.map((c, k) => c * s[k]),
    }));
    const plain = await render(baked, {});
    expect(lit(plain)).toBeGreaterThan(W * H * 0.1);
    expect(meanDiff(plain, cov)).toBeLessThan(0.5);
    // ...and differs from the similarity transform's best effort.
    const uniform = await render(grid(), {}, object);
    expect(meanDiff(plain, uniform)).toBeGreaterThan(2);
  });

  // Axis-aligned splats scaled by s: the same baked into centers and scales.
  function bakedScale(s: number[], offset = [0, 0, 0]) {
    return grid().map((g) => ({
      ...g,
      center: g.center.map((c, k) => c * s[k] + offset[k]),
      scales: g.scales.map((c, k) => c * s[k]),
    }));
  }

  it("runs CovSplat modifiers (CovSplatTransformer)", async () => {
    const s = [0.7, 1.6, 1.1];
    const transformer = new CovSplatTransformer();
    transformer.updateFromMatrix(
      new THREE.Matrix4().makeScale(s[0], s[1], s[2]),
    );
    const modifier = dynoBlock(
      { covsplat: CovSplat },
      { covsplat: CovSplat },
      ({ covsplat }) => ({
        covsplat: transformer.applyCovSplat(covsplat as never),
      }),
    );
    const cov = await render(grid(), { covSplats: true }, undefined, {
      covObjectModifiers: [modifier],
    });
    const plain = await render(bakedScale(s), {});
    expect(meanDiff(plain, cov)).toBeLessThan(0.5);
  });

  it("runs linear-blend skinning on CovSplats", async () => {
    const list = grid();
    const skinning = new SplatSkinning({
      mesh: { numSplats: list.length } as never,
      numBones: 2,
      mode: SplatSkinningMode.LINEAR_BLEND,
    });
    // Every splat on bone 1, scaled non-uniformly and moved.
    const s = [1.5, 0.5, 1];
    const offset = [0.1, -0.05, 0];
    for (let i = 0; i < list.length; i++) {
      skinning.setSplatBones(
        i,
        new THREE.Vector4(1, 0, 0, 0),
        new THREE.Vector4(1, 0, 0, 0),
      );
    }
    skinning.setRestMatrix(1, new THREE.Matrix4());
    skinning.setBoneMatrix(
      1,
      new THREE.Matrix4()
        .makeScale(s[0], s[1], s[2])
        .setPosition(offset[0], offset[1], offset[2]),
    );
    skinning.skinTexture.needsUpdate = true;
    skinning.boneTexture.needsUpdate = true;
    const modifier = dynoBlock(
      { covsplat: CovSplat },
      { covsplat: CovSplat },
      ({ covsplat }) => ({ covsplat: skinning.modifyCov(covsplat as never) }),
    );
    const skinned = await render(list, { covSplats: true }, undefined, {
      covObjectModifiers: [modifier],
    });
    const plain = await render(bakedScale(s, offset), {});
    expect(lit(plain)).toBeGreaterThan(W * H * 0.1);
    expect(meanDiff(plain, skinned)).toBeLessThan(0.5);
  });

  it("follows the whole Jacobian of a linear blend with weightGradients", async () => {
    // Bone 1 moves by T; its weight is a ramp in x, w = a x + b. The blend
    // p + w(p) T is then the affine map A = [I + T (a, 0, 0)^T | b T], so
    // with the weights' gradients the skinned cloud is the cloud under A
    // (a shear here); without them its splats only move.
    const list = grid();
    const T = [0, 0.5, 0];
    const a = 0.6;
    const b = 0.5;
    const skinned = async (gradients: boolean) => {
      const skinning = new SplatSkinning({
        mesh: { numSplats: list.length } as never,
        numBones: 2,
        mode: SplatSkinningMode.LINEAR_BLEND,
        weightGradients: gradients,
      });
      list.forEach((g, i) => {
        const w = a * g.center[0] + b;
        skinning.setSplatBones(
          i,
          new THREE.Vector4(0, 1, 0, 0),
          new THREE.Vector4(1 - w, w, 0, 0),
        );
        if (gradients) {
          skinning.setSplatWeightGradients(
            i,
            new THREE.Vector3(-a, 0, 0),
            new THREE.Vector3(a, 0, 0),
            new THREE.Vector3(),
          );
        }
      });
      skinning.setRestMatrix(0, new THREE.Matrix4());
      skinning.setRestMatrix(1, new THREE.Matrix4());
      skinning.setBoneMatrix(
        1,
        new THREE.Matrix4().makeTranslation(T[0], T[1], T[2]),
      );
      skinning.skinTexture.needsUpdate = true;
      skinning.boneTexture.needsUpdate = true;
      skinning.gradientTexture.needsUpdate = true;
      const modifier = dynoBlock(
        { covsplat: CovSplat },
        { covsplat: CovSplat },
        ({ covsplat }) => ({ covsplat: skinning.modifyCov(covsplat as never) }),
      );
      return render(list, { covSplats: true }, undefined, {
        covObjectModifiers: [modifier],
      });
    };
    // The cloud under A, by hand: centres moved, each covariance sheared
    // (J = I + T (a, 0, 0)^T mixes x into y: its 2x2 in xy diagonalised).
    const sheared = list.map((g) => {
      const [sx, sy, sz] = g.scales;
      const k = a * T[1];
      const c00 = sx * sx;
      const c01 = k * sx * sx;
      const c11 = k * k * sx * sx + sy * sy;
      const angle = 0.5 * Math.atan2(2 * c01, c00 - c11);
      const mean = 0.5 * (c00 + c11);
      const radius = Math.hypot(0.5 * (c00 - c11), c01);
      const q = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(0, 0, 1),
        angle,
      );
      const y = g.center[1] + (a * g.center[0] + b) * T[1];
      return {
        ...g,
        center: [g.center[0], y, g.center[2]],
        scales: [Math.sqrt(mean + radius), Math.sqrt(mean - radius), sz],
        quat: q.toArray(),
      };
    });
    const affine = await render(sheared, {});
    const whole = await skinned(true);
    const blend = await skinned(false);
    expect(lit(affine)).toBeGreaterThan(W * H * 0.1);
    // 8-bit weights move a splat by up to |T| / 510.
    expect(meanDiff(affine, whole)).toBeLessThan(0.3);
    expect(meanDiff(affine, blend)).toBeGreaterThan(
      5 * meanDiff(affine, whole),
    );
  });

  it("draws zero-scale splats as surfels, and enable2DGS forces it", async () => {
    // Flat in z, seen nearly face-on: the surfel covers what the projected
    // 3D one does (surfels: "ewa"), with Huang's filter for the blur.
    const flat = grid([0.04, 0.03, 0]);
    const projected = await render(flat, { surfels: "ewa" });
    const surfels = await render(flat, {});
    const forced = await render(flat, { surfels: "ewa", enable2DGS: true });
    expect(lit(surfels)).toBeGreaterThan(W * H * 0.1);
    expect(meanDiff(surfels, forced)).toBe(0);
    const diff = meanDiff(projected, surfels);
    expect(diff).toBeGreaterThan(0);
    expect(diff).toBeLessThan(4);
  });

  it("replaces colours from an RgbaArray (SplatMesh.splatRgba)", async () => {
    const list = grid();
    // Pure blue for every splat but the last row, past the array's count.
    const count = list.length - 13;
    const array = new Uint8Array(count * 4);
    for (let i = 0; i < count; i++) array.set([0, 0, 255, 204], 4 * i);
    const rgba = new RgbaArray({ array, count });
    const modifier = dynoBlock(
      { gsplat: Gsplat },
      { gsplat: Gsplat },
      ({ gsplat }) => ({
        gsplat: maybeInjectSplatRgba(
          gsplat as never,
          rgba.dyno,
          splitGsplat(gsplat as never).outputs.index,
          dynoConst("bool", false),
        ),
      }),
    );
    const px = await render(list, {}, undefined, {
      objectModifiers: [modifier],
    });
    let blue = 0;
    let other = 0;
    for (let i = 0; i < W * H; i++) {
      const [r, g, b] = [px[4 * i], px[4 * i + 1], px[4 * i + 2]];
      if (b > 40 && r < 5 && g < 5) blue++;
      else if (r + g > 40) other++;
    }
    expect(blue).toBeGreaterThan(W * H * 0.1);
    expect(other).toBeGreaterThan(0); // the last row keeps its colours
    expect(other).toBeLessThan(blue / 4);
  });

  it("depth of field blurs splats off the focal plane", async () => {
    const sharp = await render(grid([0.01, 0.01, 0.01]), {});
    const focused = await render(grid([0.01, 0.01, 0.01]), {
      focalDistance: 1.6,
      apertureAngle: 0.05,
    });
    const blurred = await render(grid([0.01, 0.01, 0.01]), {
      focalDistance: 0.5,
      apertureAngle: 0.05,
    });
    expect(meanDiff(sharp, focused)).toBeLessThan(1);
    // Blur keeps each splat's energy: fainter, over a larger area.
    expect(lit(blurred, 2)).toBeGreaterThan(lit(sharp, 2) * 1.5);
  });
});

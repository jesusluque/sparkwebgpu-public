// SparkRenderer with backend "webgpu" on Dawn: a stand-in for three's
// WebGPURenderer opens the render pass, as three does before drawing its
// transparent objects, and SparkRenderer.onBeforeRender draws into it.

import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { ExtSplats } from "../../src/ExtSplats";
import { PackedSplats } from "../../src/PackedSplats";
import { RgbaArray } from "../../src/RgbaArray";
import { SparkRenderer } from "../../src/SparkRenderer";
import { SplatMesh, type SplatSource } from "../../src/SplatMesh";
import { loadWebGPU } from "../../src/webgpuLoader";
import { device } from "./device";

// SparkRenderer constructs its WebGPU backend at once when it has loaded.
await loadWebGPU();

const W = 64;
const H = 64;

describe.skipIf(!device)("SparkRenderer on WebGPU", () => {
  const d = device as GPUDevice;
  const canvas = d.createTexture({
    size: [W, H],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  const rc = {
    textures: null,
    depthTexture: null,
    renderTarget: null,
    width: W,
    height: H,
    viewport: false,
    viewportValue: new THREE.Vector4(0, 0, W, H),
  };
  const rcData: Record<string, unknown> = {};
  const fakeRenderer = {
    isWebGPURenderer: true,
    info: { frame: 0 },
    outputColorSpace: THREE.SRGBColorSpace,
    getOutputRenderTarget: () => null,
    _currentRenderContext: rc,
    _frameBufferTarget: null,
    backend: {
      isWebGPUBackend: true,
      device: d,
      context: {
        getCurrentTexture: () => canvas,
        canvas: { width: W, height: H },
      },
      get: (resource: object) => (resource === rc ? rcData : {}),
      utils: {
        getCurrentColorFormat: () => canvas.format,
        getCurrentDepthStencilFormat: () => undefined,
        getSampleCountRenderContext: () => 1,
        getCurrentColorSpace: () => THREE.SRGBColorSpace,
      },
      updateViewport: () => {},
      pipelineUtils: { _activePipelines: new WeakMap() },
    },
  };

  // One three frame: a pass cleared to black, with the splats drawn into it.
  async function render(
    spark: SparkRenderer,
    scene: THREE.Scene,
    camera: THREE.Camera,
  ) {
    fakeRenderer.info.frame += 1;
    const encoder = d.createCommandEncoder();
    const descriptor: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: canvas.createView(),
          loadOp: "clear",
          storeOp: "store",
          clearValue: [0, 0, 0, 1],
        },
      ],
    };
    rcData.descriptor = descriptor;
    rcData.currentPass = encoder.beginRenderPass(descriptor);
    spark.onBeforeRender(fakeRenderer as never, scene, camera);
    (rcData.currentPass as GPURenderPassEncoder).end();
    d.queue.submit([encoder.finish()]);
    await d.queue.onSubmittedWorkDone();
  }

  async function litPixels(): Promise<number> {
    const bytesPerRow = 256;
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
    const px = new Uint8Array(buf.getMappedRange());
    let lit = 0;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) if (px[y * bytesPerRow + x * 4] > 40) lit++;
    }
    buf.unmap();
    buf.destroy();
    return lit;
  }

  // The lit pixels' bounding box, [width, height], and the lit count in the
  // left and right halves.
  async function litBox() {
    const bytesPerRow = 256;
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
    const px = new Uint8Array(buf.getMappedRange());
    let [x0, y0, x1, y1] = [W, H, -1, -1];
    const halves = [0, 0];
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        if (px[y * bytesPerRow + x * 4] <= 40) continue;
        x0 = Math.min(x0, x);
        x1 = Math.max(x1, x);
        y0 = Math.min(y0, y);
        y1 = Math.max(y1, y);
        halves[x < W / 2 ? 0 : 1] += 1;
      }
    }
    buf.unmap();
    buf.destroy();
    return { size: [x1 - x0 + 1, y1 - y0 + 1], halves };
  }

  // A red ball of splats; `flat`: zero z scales (2DGS).
  function ball(count = 2000, flat = false) {
    const packed = new PackedSplats();
    const center = new THREE.Vector3();
    const scales = new THREE.Vector3(0.08, 0.08, flat ? 0 : 0.08);
    const quaternion = new THREE.Quaternion();
    const color = new THREE.Color(1, 0.1, 0.1);
    for (let i = 0; i < count; i++) {
      const u = 1 - (2 * (i + 0.5)) / count;
      const t = i * Math.PI * (3 - Math.sqrt(5));
      const r = Math.sqrt(1 - u * u);
      center.set(r * Math.cos(t), u, r * Math.sin(t)).multiplyScalar(0.5);
      packed.pushSplat(center, scales, quaternion, 1, color);
    }
    return packed;
  }

  function setup() {
    const spark = new SparkRenderer({ renderer: fakeRenderer as never });
    const scene = new THREE.Scene();
    scene.add(spark);
    const camera = new THREE.PerspectiveCamera(60, W / H, 0.05, 100);
    camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
    camera.updateProjectionMatrix();
    camera.position.set(0, 0, 3);
    return { spark, scene, camera };
  }

  it("draws the scene's SplatMeshes into three's pass", async () => {
    const { spark, scene, camera } = setup();
    expect(spark.webgpu).toBeDefined();
    const mesh = new SplatMesh({ packedSplats: ball() });
    scene.add(mesh);
    await render(spark, scene, camera);
    expect(await litPixels()).toBeGreaterThan(W * H * 0.1);
    const splats = spark.webgpu?.splats;
    expect(splats?.meshes.length).toBe(1);
    expect(splats?.meshes[0].object).toBe(mesh);

    // Options and colours follow SparkRenderer and the mesh.
    spark.maxStdDev = 2;
    mesh.opacity = 0.5;
    await render(spark, scene, camera);
    expect(splats?.options.maxStdDev).toBe(2);
    expect(splats?.meshes[0].recolor.w).toBe(0.5);
    spark.material.depthTest = false;
    await render(spark, scene, camera);
    expect(splats?.options.depthTest).toBe(false);
    spark.dispose();
  });

  // Tone mapping the composite can't invert (a custom node here) puts the
  // splats on the canvas after three's output pass, with MSAA too (its
  // depth resolved to one sample), not into the tone-mapped frame.
  it("draws after the output pass under custom tone mapping with MSAA", async () => {
    const { spark, scene, camera } = setup();
    scene.add(new SplatMesh({ packedSplats: ball() }));
    const r = fakeRenderer as Record<string, unknown>;
    const utils = fakeRenderer.backend.utils;
    const frameBufferTarget = {};
    r.toneMapping = THREE.CustomToneMapping;
    r._frameBufferTarget = frameBufferTarget;
    Object.assign(rc, { renderTarget: frameBufferTarget, textures: [{}] });
    utils.getSampleCountRenderContext = () => 4;
    try {
      await render(spark, scene, camera);
      expect(Object.hasOwn(scene, "onAfterRender")).toBe(true);
      scene.onAfterRender(
        fakeRenderer as never,
        scene,
        camera,
        null as never,
        null as never,
        null as never,
      );
      await d.queue.onSubmittedWorkDone();
      expect(await litPixels()).toBeGreaterThan(W * H * 0.1);
    } finally {
      r.toneMapping = undefined;
      r._frameBufferTarget = null;
      Object.assign(rc, { renderTarget: null, textures: null });
      utils.getSampleCountRenderContext = () => 1;
      spark.dispose();
    }
  });

  // rawColor on the canvas: three's output pass would encode the values
  // the splats store in its linear frame, so they go on after it, unconverted,
  // as WebGL Spark writes them to its canvas.
  it("draws rawColor splats on the canvas after the output pass", async () => {
    const spark = new SparkRenderer({
      renderer: fakeRenderer as never,
      rawColor: true,
    });
    const { spark: other, scene, camera } = setup();
    scene.remove(other);
    other.dispose();
    scene.add(spark);
    scene.add(new SplatMesh({ packedSplats: ball() }));
    const r = fakeRenderer as Record<string, unknown>;
    const frameBufferTarget = {};
    r._frameBufferTarget = frameBufferTarget;
    Object.assign(rc, { renderTarget: frameBufferTarget, textures: [{}] });
    try {
      await render(spark, scene, camera);
      expect(Object.hasOwn(scene, "onAfterRender")).toBe(true);
    } finally {
      r._frameBufferTarget = null;
      Object.assign(rc, { renderTarget: null, textures: null });
      spark.dispose();
    }
  });

  it("follows visibility, shared splats and removal", async () => {
    const { spark, scene, camera } = setup();
    const packed = ball();
    const a = new SplatMesh({ packedSplats: packed });
    const b = new SplatMesh({ packedSplats: packed });
    b.position.x = 0.5;
    scene.add(a, b);
    await render(spark, scene, camera);
    const splats = spark.webgpu?.splats;
    expect(splats?.meshes.length).toBe(2);
    // One upload for both meshes.
    expect(splats?.meshes[0].source).toBe(splats?.meshes[1].source);

    a.visible = false;
    b.visible = false;
    await render(spark, scene, camera);
    expect(splats?.meshes.length).toBe(0);
    expect(await litPixels()).toBe(0);

    b.visible = true;
    await render(spark, scene, camera);
    expect(splats?.meshes.map((m) => m.object)).toEqual([b]);
    expect(await litPixels()).toBeGreaterThan(0);

    scene.remove(b);
    await render(spark, scene, camera);
    expect(splats?.meshes.length).toBe(0);
    spark.dispose();
  });

  it("re-uploads edited splats", async () => {
    const { spark, scene, camera } = setup();
    const packed = ball(100);
    scene.add(new SplatMesh({ packedSplats: packed }));
    await render(spark, scene, camera);
    const splats = spark.webgpu?.splats;
    expect(splats?.meshes[0].source.count).toBe(100);
    packed.pushSplat(
      new THREE.Vector3(),
      new THREE.Vector3().setScalar(0.1),
      new THREE.Quaternion(),
      1,
      new THREE.Color(1, 1, 1),
    );
    await render(spark, scene, camera);
    expect(splats?.meshes[0].source.count).toBe(101);
    spark.dispose();
  });

  it("clips by the portal disk uniforms, as SparkPortals' shader", async () => {
    const spark = new SparkRenderer({
      renderer: fakeRenderer as never,
      extraUniforms: {
        diskCenter: { value: new THREE.Vector3(0, 0, -1) },
        diskNormal: { value: new THREE.Vector3(0, 0, 1) },
        diskRadius: { value: 0 },
        diskTwoSided: { value: true },
      },
    });
    const scene = new THREE.Scene();
    scene.add(spark);
    const camera = new THREE.PerspectiveCamera(60, W / H, 0.05, 100);
    camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
    camera.updateProjectionMatrix();
    camera.position.set(0, 0, 3);
    scene.add(new SplatMesh({ packedSplats: ball() }));
    const radius = (
      spark.uniforms as unknown as Record<string, { value: number }>
    ).diskRadius;
    await render(spark, scene, camera);
    const all = await litPixels();
    expect(all).toBeGreaterThan(W * H * 0.1);
    // A big disk in front: the front pass drops what is behind it...
    radius.value = -10;
    await render(spark, scene, camera);
    expect(await litPixels()).toBe(0);
    // ...and the behind pass draws only that.
    radius.value = 10;
    await render(spark, scene, camera);
    expect(await litPixels()).toBe(all);
    spark.dispose();
  });

  it("bakes a mesh's generated RGBA with getRgba", async () => {
    const { spark, scene, camera } = setup();
    const packed = ball(300);
    const mesh = new SplatMesh({ packedSplats: packed });
    mesh.recolor.setRGB(0.5, 1, 1);
    scene.add(mesh);
    await render(spark, scene, camera);
    const rgba = spark.getRgba({ generator: mesh });
    expect(rgba.count).toBe(300);
    const bytes = await rgba.read();
    // red 1, green 0.1 (26/255) from the packed bytes, times the recolor.
    for (let i = 0; i < 300; i += 37) {
      expect(Math.abs(bytes[i * 4] - 128)).toBeLessThanOrEqual(1);
      expect(bytes[i * 4 + 1]).toBe(26);
      expect(bytes[i * 4 + 3]).toBe(255);
    }
    // fromPackedSplats decodes on the CPU on WebGPU.
    const original = new RgbaArray().fromPackedSplats({
      packedSplats: packed,
      base: 0,
      count: 300,
      renderer: fakeRenderer as never,
    });
    const orig = await original.getArray();
    expect(Array.from(orig.subarray(0, 4))).toEqual([255, 26, 26, 255]);
    rgba.dispose();
    spark.dispose();
  });

  // The same ball as ExtSplats (covSplats meshes need them).
  function extBall(count = 2000) {
    const ext = new ExtSplats();
    ball(count).forEachSplat((_, center, scales, quaternion, opacity, color) =>
      ext.pushSplat(center, scales, quaternion, opacity, color),
    );
    return ext;
  }

  it("accumulates covariance splats with covSplats", async () => {
    // A ball squashed by a non-uniform scale: an ellipse 4x as wide as tall
    // with covariance splats; the similarity transform can't do that.
    const box = async (covSplats: boolean, meshCov: boolean) => {
      const spark = new SparkRenderer({
        renderer: fakeRenderer as never,
        covSplats,
      });
      const { spark: other, scene, camera } = setup();
      scene.remove(other);
      other.dispose();
      scene.add(spark);
      const mesh = new SplatMesh({
        extSplats: extBall(),
        covSplats: meshCov,
      });
      mesh.scale.set(1.6, 0.4, 1);
      scene.add(mesh);
      await render(spark, scene, camera);
      const options = spark.webgpu?.splats?.options;
      const result = { ...(await litBox()), cov: options?.covSplats };
      spark.dispose();
      return result;
    };
    const cov = await box(true, false);
    expect(cov.cov).toBe(true);
    expect(cov.size[0] / cov.size[1]).toBeGreaterThan(2.5);
    // A covSplats mesh turns them on by itself (WebGL throws instead).
    const meshCov = await box(false, true);
    expect(meshCov.cov).toBe(true);
    expect(meshCov.size).toEqual(cov.size);
    const plain = await box(false, false);
    expect(plain.cov).toBe(false);
    expect(plain.size[0] / plain.size[1]).toBeLessThan(1.5);
  });

  it("follows accumExtSplats and enable2DGS", async () => {
    const spark = new SparkRenderer({
      renderer: fakeRenderer as never,
      accumExtSplats: true,
      enable2DGS: true,
    });
    const { spark: other, scene, camera } = setup();
    scene.remove(other);
    other.dispose();
    scene.add(spark);
    const mesh = new SplatMesh({ packedSplats: ball(2000, true) });
    scene.add(mesh);
    await render(spark, scene, camera);
    const options = spark.webgpu?.splats?.options;
    expect(options?.accumulator).toBe("ext");
    expect(options?.enable2DGS).toBe(true);
    // Flat splats facing the camera draw as 2D Gaussians.
    const flat = await litPixels();
    expect(flat).toBeGreaterThan(W * H * 0.1);
    spark.accumExtSplats = false;
    spark.enable2DGS = false;
    await render(spark, scene, camera);
    expect(options?.accumulator).toBe("auto");
    expect(options?.enable2DGS).toBe(false);
    spark.dispose();
  });

  // A SplatSource of its own: its fetchSplat graph is read in generate.
  it("draws SplatMeshes with a custom SplatSource", async () => {
    const packed = ball();
    class Wrapped implements SplatSource {
      prepareFetchSplat() {}
      dispose() {}
      getNumSplats() {
        return packed.getNumSplats();
      }
      hasRgbDir() {
        return false;
      }
      getNumSh() {
        return 0;
      }
      setMaxSh() {}
      fetchSplat(args: Parameters<SplatSource["fetchSplat"]>[0]) {
        return packed.fetchSplat(args);
      }
      forEachSplat(callback: Parameters<SplatSource["forEachSplat"]>[0]) {
        packed.forEachSplat(callback);
      }
    }
    const { spark, scene, camera } = setup();
    const own = new SplatMesh({ packedSplats: packed });
    scene.add(own);
    await render(spark, scene, camera);
    const expected = await litPixels();
    scene.remove(own);
    const mesh = new SplatMesh({ splats: new Wrapped() });
    scene.add(mesh);
    await render(spark, scene, camera);
    const splats = spark.webgpu?.splats;
    expect(splats?.meshes.map((m) => m.object)).toEqual([mesh]);
    expect(await litPixels()).toBe(expected);
    // Hidden and shown again.
    mesh.visible = false;
    await render(spark, scene, camera);
    expect(await litPixels()).toBe(0);
    mesh.visible = true;
    await render(spark, scene, camera);
    expect(await litPixels()).toBe(expected);
    spark.dispose();
  });

  // An ArrayCamera: each sub-camera in its viewport, drawn once three's
  // frame is submitted (scene.onAfterRender).
  it("draws an ArrayCamera's views into their viewports", async () => {
    const { spark, scene } = setup();
    scene.add(new SplatMesh({ packedSplats: ball() }));
    const view = (x: number, lookX: number) => {
      const c = new THREE.PerspectiveCamera(60, W / 2 / H, 0.05, 100);
      c.coordinateSystem = THREE.WebGPUCoordinateSystem;
      c.updateProjectionMatrix();
      c.position.set(0, 0, 3);
      c.lookAt(lookX, 0, 0);
      c.updateMatrixWorld();
      (c as THREE.PerspectiveCamera & { viewport: THREE.Vector4 }).viewport =
        new THREE.Vector4(x, 0, W / 2, H);
      return c;
    };
    const r = fakeRenderer as Record<string, unknown>;
    const frameBufferTarget = {};
    r._frameBufferTarget = frameBufferTarget;
    r.getPixelRatio = () => 1;
    Object.assign(rc, { renderTarget: frameBufferTarget, textures: [{}] });
    const draw = async (a: THREE.Camera, b: THREE.Camera) => {
      const camera = new THREE.ArrayCamera([a, b]);
      await render(spark, scene, camera);
      expect(Object.hasOwn(scene, "onAfterRender")).toBe(true);
      scene.onAfterRender(
        fakeRenderer as never,
        scene,
        camera,
        null as never,
        null as never,
        null as never,
      );
      await d.queue.onSubmittedWorkDone();
      return (await litBox()).halves;
    };
    try {
      const both = await draw(view(0, 0), view(W / 2, 0));
      expect(both[0]).toBeGreaterThan(W * H * 0.05);
      expect(both[1]).toBe(both[0]);
      // The second view looks away: its half stays dark.
      const one = await draw(view(0, 0), view(W / 2, 50));
      expect(one[0]).toBe(both[0]);
      expect(one[1]).toBe(0);
    } finally {
      r._frameBufferTarget = null;
      r.getPixelRatio = undefined;
      Object.assign(rc, { renderTarget: null, textures: null });
      spark.dispose();
    }
  });

  it("refuses what WebGPU doesn't do yet", () => {
    const { spark } = setup();
    expect(
      () =>
        new SparkRenderer({
          renderer: fakeRenderer as never,
          fragmentShader: "void main() {}",
        }),
    ).toThrow(/not supported/);
    expect(
      () =>
        new SparkRenderer({
          renderer: new THREE.Object3D() as never,
          backend: "webgpu",
        }),
    ).toThrow(/WebGPURenderer/);
    spark.dispose();
  });
});

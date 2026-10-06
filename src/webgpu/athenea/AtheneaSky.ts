// A dome prepared on the GPU as athenea's technique::Environment prepares it
// (modules/technique/src/Environment.cpp at txf 89a04d9), by the passes of
// slang/athenea_adapter/env.slang: the lat-long's area mips, the sun taken
// out (envSun), sixteen harmonics of what is left (envSh), the sky without
// its sun and its mips, then eight GGX levels of an octahedral map
// (envTexels). Those three buffers are what athenea's relighting reads
// (light/environment.slang), and what the relight plugin binds.
//
//   const sky = new AtheneaSky(device);
//   sky.set({ image: { width, height, data }, rotation: 0.5, intensity: 1 });
//   sky.prepare(encoder);   // when sky.dirty; bumps sky.version
//
// One dome (slice 0). The dome's image, its intensity and its rotation are
// all baked into the buffers (athenea multiplies by the light's emission and
// looks the image up in the light's frame), so changing any of them prepares
// the sky again; the relight plugin's kept terms follow `version`.

import * as THREE from "three";
import { KernelRegistry } from "../KernelRegistry";
import envModule from "../generated/athenea_adapter/env";
import { createReadback, createStorage, readAndDestroy } from "../gpuBuffers";
import { UniformWriter } from "../uniforms";
import {
  type AtheneaLightRecord,
  LIGHT_NONE,
  packLightRecords,
} from "./lights";

/** athenea/light/environment.slang's constants (vendored files export none). */
export const ENV = {
  kEnvLevels: 8,
  kEnvCoarseSide: 256,
  kEnvFloorSide: 16,
  kEnvCoefficients: 16,
  kEnvIrradianceCoefficients: 9,
  kEnvNone: 0xffffffff,
} as const;
/** Environment.h kEnvironmentWidestSide. */
const WIDEST_SIDE = 2048;
/** Threads a dispatch row: WebGPU's 65535 workgroups of 64. */
const ROW_THREADS = 64 * 32768;

type Rgb = [number, number, number];

/** A lat-long (equirectangular) image of linear radiance, top row first. */
export interface SkyImage {
  width: number;
  height: number;
  /** width * height * channels floats. */
  data: Float32Array;
  /** 3 (rgb) or 4 (rgba); default 4. */
  channels?: 3 | 4;
}

export interface AtheneaSkyOptions {
  /** The dome's image; null: the dome is its colour alone. */
  image?: SkyImage | null;
  /** Multiplies the image (UsdLux color). Default white. */
  colour?: Rgb;
  /** UsdLux intensity. Default 1. */
  intensity?: number;
  /** Stops. Default 0. */
  exposure?: number;
  /** Turns the dome about +Y, radians. Default 0. */
  rotation?: number;
}

/** Environment::baseSideFor: a map texel as wide as a lat-long texel. */
export function envBaseSideFor(width: number, height: number): number {
  if (width === 0 || height === 0) return ENV.kEnvCoarseSide;
  const want = Math.sqrt((2 * width * height) / Math.PI);
  let side = ENV.kEnvCoarseSide;
  while (side < WIDEST_SIDE && side * 2 <= want) side *= 2;
  return side;
}

/** environment.slang envSideOf. */
export function envSideOf(level: number, baseSide: number): number {
  const base = Math.max(baseSide, ENV.kEnvCoarseSide);
  return Math.max(
    base >> Math.min(level, ENV.kEnvLevels - 1),
    ENV.kEnvFloorSide,
  );
}

/** environment.slang envTexelsPerDome. */
export function envTexelsPerDome(baseSide: number): number {
  let texels = 0;
  for (let k = 0; k < ENV.kEnvLevels; k++)
    texels += envSideOf(k, baseSide) ** 2;
  return texels;
}

/** Environment.cpp samplesOf: directions a level's texel takes. */
export function envSamplesOf(level: number): number {
  return level === 0 ? 1 : Math.min(64 << (level - 1), 1024);
}

/** The source level env_project sums over: no wider than 256. */
export function envProjectLevel(width: number): number {
  let level = 0;
  let widest = width;
  while (widest > 256 && level < 12) {
    widest >>= 1;
    level += 1;
  }
  return level;
}

const mipCount = (w: number, h: number) =>
  Math.floor(Math.log2(Math.max(w, h, 1))) + 1;

export class AtheneaSky {
  readonly registry: KernelRegistry;
  /** Bumped by every prepare. */
  version = 0;
  /** Set by set(); cleared by prepare(). */
  dirty = true;
  options: Required<Omit<AtheneaSkyOptions, "image">> & {
    image: SkyImage | null;
  } = {
    image: null,
    colour: [1, 1, 1],
    intensity: 1,
    exposure: 0,
    rotation: 0,
  };

  /** kEnvCoefficients float4: the sky's harmonics (rgb), its sun taken out. */
  envSh: GPUBuffer;
  /** Two float4: (direction, half angle; 0: no sun), (irradiance, threshold). */
  envSun: GPUBuffer;
  /** The GGX levels, two words a texel (environment.slang's layout). */
  envTexels: GPUBuffer | null = null;
  /** The dome's LightRecord (what env.slang's envDome holds). */
  domeBuffer: GPUBuffer;
  baseSide: number = ENV.kEnvCoarseSide;

  private source: GPUTexture | null = null;
  private sourceImage: SkyImage | null = null;
  private residual: GPUTexture | null = null;

  constructor(
    readonly device: GPUDevice,
    registry?: KernelRegistry,
  ) {
    this.registry = registry ?? new KernelRegistry(device);
    this.envSh = createStorage(device, ENV.kEnvCoefficients * 16, "env sh");
    this.envSun = createStorage(device, 2 * 16, "env sun");
    this.domeBuffer = createStorage(device, 160, "env dome");
  }

  set(options: AtheneaSkyOptions) {
    const o = this.options;
    if (options.image !== undefined) o.image = options.image;
    if (options.colour) o.colour = [...options.colour];
    if (options.intensity !== undefined) o.intensity = options.intensity;
    if (options.exposure !== undefined) o.exposure = options.exposure;
    if (options.rotation !== undefined) o.rotation = options.rotation;
    this.dirty = true;
  }

  /** Light to world: the dome turned about +Y. */
  get matrix(): THREE.Matrix4 {
    return new THREE.Matrix4().makeRotationY(this.options.rotation);
  }

  /** The dome as a LightRecord (for a lights buffer beside it). */
  record(): AtheneaLightRecord {
    const o = this.options;
    return {
      kind: "dome",
      colour: o.colour.map((c) => c * o.intensity) as Rgb,
      exposure: o.exposure,
      texture: o.image ? 0 : LIGHT_NONE,
      matrix: this.matrix,
    };
  }

  private uploadSource(image: SkyImage | null): GPUTexture {
    if (this.source && this.sourceImage === image) return this.source;
    this.source?.destroy();
    const w = image?.width ?? 1;
    const h = image?.height ?? 1;
    const texture = this.device.createTexture({
      label: "env source",
      size: [w, h],
      format: "rgba32float",
      mipLevelCount: mipCount(w, h),
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.COPY_DST,
    });
    const rgba = new Float32Array(w * h * 4);
    if (image) {
      const n = image.channels ?? 4;
      // athenea's texture rows count up from the bottom (domeUv: v = 1 is
      // +Y, and texel row 0 is v = 0): the file's top row goes last.
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const from = n * ((h - 1 - y) * w + x);
          const to = 4 * (y * w + x);
          rgba[to] = image.data[from];
          rgba[to + 1] = image.data[from + 1];
          rgba[to + 2] = image.data[from + 2];
          rgba[to + 3] = 1;
        }
      }
    } else {
      rgba.fill(1);
    }
    this.device.queue.writeTexture(
      { texture },
      rgba,
      { bytesPerRow: w * 16, rowsPerImage: h },
      [w, h],
    );
    this.source = texture;
    this.sourceImage = image;
    this.sourceMips = false;
    return texture;
  }

  private sourceMips = false;

  // athenea/algo/mips.slang over a lat-long: level k + 1 from level k.
  private encodeMips(pass: GPUComputePassEncoder, texture: GPUTexture) {
    const kernel = this.registry.get(envModule, "atheneaEnvMip");
    for (let level = 0; level + 1 < texture.mipLevelCount; level++) {
      const sw = Math.max(texture.width >> level, 1);
      const sh = Math.max(texture.height >> level, 1);
      const w = Math.max(texture.width >> (level + 1), 1);
      const h = Math.max(texture.height >> (level + 1), 1);
      const params = UniformWriter.for(envModule, "envParams").setAll({
        width: w,
        height: h,
        sourceMipWidth: sw,
        sourceMipHeight: sh,
      });
      kernel.dispatch(pass, {
        grid: [w, h],
        uniforms: params.data,
        bindings: {
          skySource: texture.createView({
            baseMipLevel: level,
            mipLevelCount: 1,
          }),
          mipTarget: texture.createView({
            baseMipLevel: level + 1,
            mipLevelCount: 1,
          }),
        },
      });
    }
  }

  /** Encodes the preparation (when dirty) into `encoder`. */
  prepare(encoder: GPUCommandEncoder) {
    if (!this.dirty) return;
    this.dirty = false;
    const { device } = this;
    const image = this.options.image;
    const source = this.uploadSource(image);
    const w = image?.width ?? 1;
    const h = image?.height ?? 1;
    this.device.queue.writeBuffer(
      this.domeBuffer,
      0,
      packLightRecords([this.record()]),
    );
    const baseSide = image ? envBaseSideFor(w, h) : ENV.kEnvCoarseSide;
    const words = envTexelsPerDome(baseSide) * 2;
    if (!this.envTexels || this.envTexels.size !== words * 4) {
      this.envTexels?.destroy();
      this.envTexels = createStorage(device, words * 4, "env texels");
    }
    this.baseSide = baseSide;
    const projectLevel = image ? envProjectLevel(w) : 0;
    const sunLevel = projectLevel >= 2 ? projectLevel - 2 : 0;
    const rw = Math.max(w >> sunLevel, 1);
    const rh = Math.max(h >> sunLevel, 1);
    if (
      !this.residual ||
      this.residual.width !== rw ||
      this.residual.height !== rh
    ) {
      this.residual?.destroy();
      this.residual = device.createTexture({
        label: "env residual",
        size: [rw, rh],
        format: "rgba32float",
        mipLevelCount: mipCount(rw, rh),
        usage:
          GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      });
    }
    const residual = this.residual;
    const common = {
      domes: 1,
      projectLevel,
      sourceLevel: projectLevel,
      baseSide,
      residualLevel: sunLevel,
      sourceWidth: w,
      sourceHeight: h,
      residualLevels: residual.mipLevelCount,
      rowThreads: ROW_THREADS,
    };
    const all = source.createView();
    const pass = encoder.beginComputePass({ label: "athenea sky" });
    if (!this.sourceMips) {
      this.encodeMips(pass, source);
      this.sourceMips = true;
    }
    const r = this.registry;
    r.get(envModule, "atheneaEnvSun").dispatch(pass, {
      grid: [1],
      buffers: { envDome: this.domeBuffer, sunOut: this.envSun },
      uniforms: UniformWriter.for(envModule, "envParams").setAll({
        ...common,
        level: sunLevel,
      }).data,
      bindings: { skySource: all },
    });
    r.get(envModule, "atheneaEnvProject").dispatch(pass, {
      grid: [ENV.kEnvCoefficients],
      buffers: {
        envDome: this.domeBuffer,
        sun: this.envSun,
        shOut: this.envSh,
      },
      uniforms: UniformWriter.for(envModule, "envParams").setAll(common).data,
      bindings: { skySource: all },
    });
    r.get(envModule, "atheneaEnvResidual").dispatch(pass, {
      grid: [rw, rh],
      buffers: { envDome: this.domeBuffer, sun: this.envSun },
      uniforms: UniformWriter.for(envModule, "envParams").setAll({
        ...common,
        width: rw,
        height: rh,
      }).data,
      bindings: {
        skySource: all,
        residualOut: residual.createView({ baseMipLevel: 0, mipLevelCount: 1 }),
      },
    });
    this.encodeMips(pass, residual);
    const residualView = residual.createView();
    for (let level = 0; level < ENV.kEnvLevels; level++) {
      const side = envSideOf(level, baseSide);
      const threads = side * side;
      const step = level / (ENV.kEnvLevels - 1);
      r.get(envModule, "atheneaEnvPrefilter").dispatch(pass, {
        grid: [
          Math.min(threads, ROW_THREADS),
          Math.ceil(threads / ROW_THREADS),
        ],
        buffers: { envDome: this.domeBuffer, texelsOut: this.envTexels },
        uniforms: UniformWriter.for(envModule, "envParams").setAll({
          ...common,
          level,
          samples: envSamplesOf(level),
          roughness: step * step,
        }).data,
        bindings: { residual: residualView },
      });
    }
    pass.end();
    this.version += 1;
  }

  /** Prepares now, on a submit of its own. */
  prepareNow() {
    const encoder = this.device.createCommandEncoder({ label: "athenea sky" });
    this.dirty = true;
    this.prepare(encoder);
    this.registry.submit(encoder.finish());
  }

  /** The prepared harmonics and sun, read back (for tests and logs). */
  async read(): Promise<{
    sh: Float32Array;
    sun: Float32Array;
    texels?: Uint32Array;
  }> {
    const copy = async (b: GPUBuffer) => {
      const staging = createReadback(this.device, b.size, "env readback");
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(b, 0, staging, 0, b.size);
      this.device.queue.submit([encoder.finish()]);
      return readAndDestroy(staging);
    };
    return {
      sh: new Float32Array(await copy(this.envSh)),
      sun: new Float32Array(await copy(this.envSun)),
      texels: this.envTexels
        ? new Uint32Array(await copy(this.envTexels))
        : undefined,
    };
  }

  destroy() {
    this.source?.destroy();
    this.residual?.destroy();
    this.envSh.destroy();
    this.envSun.destroy();
    this.envTexels?.destroy();
    this.domeBuffer.destroy();
  }
}

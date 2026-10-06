// The one type Spark's WebGPU modules take a three.js WebGPURenderer as:
// its internals (threeInternals.ts, checked against the three revision)
// plus the public API they call. Tests and narrow helpers Pick from it.

import type * as THREE from "three";
import type { ThreeWebGPURenderer } from "./threeInternals";

export interface WebGPURendererLike extends ThreeWebGPURenderer {
  isWebGPURenderer?: boolean;
  info: { frame: number };
  xr?: { enabled?: boolean; isPresenting?: boolean };
  coordinateSystem: THREE.CoordinateSystem;
  outputColorSpace: string;
  toneMapping?: THREE.ToneMapping;
  toneMappingExposure?: number;
  getOutputRenderTarget(): THREE.RenderTarget | null;
  getPixelRatio?(): number;
  getRenderTarget(): THREE.RenderTarget | null;
  getActiveCubeFace(): number;
  getActiveMipmapLevel(): number;
  setRenderTarget(target: THREE.RenderTarget | null, face?: number): void;
  render(scene: THREE.Object3D, camera: THREE.Camera): void;
  initTexture(texture: THREE.Texture): void;
  readRenderTargetPixelsAsync(
    target: THREE.RenderTarget,
    x: number,
    y: number,
    width: number,
    height: number,
    textureIndex?: number,
    faceIndex?: number,
  ): Promise<ArrayBufferView>;
}

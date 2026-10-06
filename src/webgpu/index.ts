// "@sparkjsdev/spark/webgpu": Spark's WebGPU backend, a chunk of its own so
// WebGL apps don't download it. SparkRenderer loads it on demand
// (src/webgpuLoader.ts); importing it statically registers it up front.

import { registerWebGPU } from "../webgpuLoader";
import { SparkWebGPU } from "./SparkWebGPU";
import { readCubeFaces } from "./WgpuCubeMap";
import { readTargetPixels, rowStride } from "./WgpuReadTarget";

/** @internal What SparkRenderer uses from the backend. */
const backend = { SparkWebGPU, readCubeFaces, readTargetPixels, rowStride };
export type WebGPUBackend = typeof backend;
registerWebGPU(backend);

export { SparkWebGPU, type WebGPURendererLike } from "./SparkWebGPU";
export {
  WgpuSplatRenderer,
  GpuSplatSource,
  type WgpuSplatMesh,
  type WgpuSplatRendererOptions,
} from "./WgpuSplatRenderer";
export { KernelRegistry } from "./KernelRegistry";
export {
  WgpuLod,
  type WgpuLodMesh,
  type WgpuLodMeshOptions,
  type WgpuLodOptions,
} from "./WgpuLod";
export { WgpuSplatPager } from "./WgpuSplatPager";
export {
  WgpuReadTarget,
  type WgpuReadTargetOptions,
} from "./WgpuReadTarget";
export {
  WgpuCubeMap,
  type WgpuCubeMapOptions,
} from "./WgpuCubeMap";
export { UniformWriter } from "./uniforms";
export * as fx from "./fx";
export * as plugins from "./plugins";
export {
  capabilitiesOf,
  splatRequiredLimits,
  type GpuCapabilities,
} from "./capabilities";
export {
  splatGeneratorDyno,
  splatMeshDyno,
} from "./dyno/adapters";
export type { WgpuDyno, WgpuDynoFrame } from "./dyno/DynoKernels";
export {
  SplatAttributes,
  type SplatAttributesOptions,
  type AttribColorMode,
  type PickResult,
} from "./attributes/SplatAttributes";
export {
  AttribPool,
  type AttribFormat,
  type AttributeSpec,
  type LodMerge,
} from "./attributes/schema";
export {
  ATHENEA_VIEWABLE,
  type AtheneaCamera,
  type AtheneaScene,
  type AtheneaSceneFile,
  type AtheneaTransform,
  applyAtheneaCamera,
  applyAtheneaTransform,
  atheneaSceneMesh,
  loadAtheneaScene,
  parseAtheneaScene,
  pickAtheneaFile,
} from "./athenea/sceneJson";
export {
  type AthcLayout,
  type AthcPaging,
  readAthcLayout,
  unpackAthcLobes,
  unpackAthcNormal,
  unpackAthcPbr,
  unpackRgb9e5,
} from "../athc";

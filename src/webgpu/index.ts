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
  ATHENEA_DISPLAYS,
  ATHENEA_VIEWS,
  AtheneaDisplay,
  type AtheneaDisplayEncoding,
  type AtheneaView,
  prepareAces2,
} from "./athenea/AtheneaDisplay";
export {
  ATHC_GROUP_ATTRIBUTE,
  ATHENEA_OUTPUT_ID,
  ATHENEA_RASTER_ID,
  MAX_ATHENEA_OVERRIDES,
  type AtheneaGroupOverride,
  type AtheneaOutputOptions,
  type AtheneaOutputPlugin,
  type AtheneaRasterOptions,
  type AtheneaRasterPlugin,
  atheneaOutputPlugin,
  atheneaRasterPlugin,
} from "./athenea/rasterPlugin";
export {
  ATHENEA_RELIGHT_ID,
  FOOTPRINT_GAIN,
  type AtheneaRelightDebug,
  type AtheneaRelightOptions,
  type AtheneaRelightPlugin,
  type AtheneaSun,
  atheneaRelightPlugin,
} from "./athenea/relightPlugin";
export {
  ATHENEA_SKIN_ID,
  type AtheneaSkinOptions,
  type AtheneaSkinPlugin,
  SKIN_GRADIENTS_ATTRIBUTE,
  SKIN_INFLUENCES_ATTRIBUTE,
  atheneaSkinPlugin,
  atheneaSkinPose,
  skinClipIndex,
} from "./athenea/skinPlugin";
export {
  AtheneaSky,
  type AtheneaSkyOptions,
  type SkyImage,
} from "./athenea/AtheneaSky";
export {
  type AtheneaLightKind,
  type AtheneaLightRecord,
  LIGHT_NORMALIZE,
  LIGHT_SHADOW,
  distantMatrix,
} from "./athenea/lights";
export {
  ATHENEA_LIGHTS_ID,
  type AtheneaLightTerms,
  type AtheneaLightsAssetOptions,
  type AtheneaLightsOptions,
  type AtheneaLightsPlugin,
  type LightGroupState,
  atheneaLightsPlugin,
  athlPagerChunks,
  lightGroupSwatch,
} from "./athenea/lightsPlugin";
export {
  type AthlData,
  type AthlHeader,
  type AthlLayer,
  AthlPager,
  AthlStore,
  athcCloudHash,
  athlChunkRange,
  decodeAthl,
  decodeAthlChunk,
  openAthl,
  readAthlHeader,
  urlRange,
} from "./athenea/athl";
export {
  type LightGroupSpec,
  type LightSidecar,
  LightRig,
  blackbody,
  parseLightSidecar,
} from "./athenea/lightSidecar";
export { type UsdLayer, type UsdPrim, parseUsda } from "./athenea/usda";
export {
  type AtheneaPick,
  atheneaGroupOf,
  pickAtheneaGroup,
} from "./athenea/pick";
export {
  type AthcLayout,
  type AthcPaging,
  type AthcSkeleton,
  type AthcSkinClip,
  decodeAthcClip,
  fetchAthcClip,
  readAthcLayout,
  readAthcSkeleton,
  unpackAthcLobes,
  unpackAthcNormal,
  unpackAthcPbr,
  unpackRgb9e5,
} from "../athc";

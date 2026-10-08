// aopenfx's effect model on WebGPU: effects written against the aofx SDK's
// contract, their Slang kernels unchanged from aopenfx/examples, and a host
// (Gpu, FxGraph, FxChain) that runs them on Spark's KernelRegistry.

export { Effect, RenderRequest } from "./Effect";
export type { InputPlane, OutputPlane, RegionQuery } from "./Effect";
export { Gpu, declaredBuffers } from "./Gpu";
export type { BoundBuffer, RunBuffers, RunUniforms } from "./Gpu";
export { FxGraph, isSource } from "./FxGraph";
export type { FxContext, FxInput, FxNode, FxSource, Rendered } from "./FxGraph";
export { FxChain } from "./FxChain";
export type { FxStep } from "./FxChain";
export { TextureBridge, bridgeFormats } from "./TextureBridge";
export type { BridgeOptions } from "./TextureBridge";
export * from "./types";
export { Invert } from "./effects/Invert";
export { Crop } from "./effects/Crop";
export { Blur } from "./effects/Blur";
export { Grade } from "./effects/Grade";
export { Merge, MERGE_OPERATIONS } from "./effects/Merge";
export { Transform, TRANSFORM_FILTERS } from "./effects/Transform";
export { CornerPin, CornerPinSS } from "./effects/CornerPin";
export { Bloom, bloomLevels, BLOOM_MAX_LEVELS } from "./effects/Bloom";
export {
  CheckerBoard,
  ColorBars,
  ColorWheel,
  Constant,
  ImageStatistics,
  NoOp,
} from "./effects/Generate";

import { Bloom } from "./effects/Bloom";
import { Blur } from "./effects/Blur";
import { CornerPin, CornerPinSS } from "./effects/CornerPin";
import { Crop } from "./effects/Crop";
import {
  CheckerBoard,
  ColorBars,
  ColorWheel,
  Constant,
  ImageStatistics,
  NoOp,
} from "./effects/Generate";
import { Grade } from "./effects/Grade";
import { Invert } from "./effects/Invert";
import { Merge } from "./effects/Merge";
import { Transform } from "./effects/Transform";

/** One instance of every example effect, by identifier. */
export function exampleEffects() {
  const all = [
    new Invert(),
    new Crop(),
    new Blur(),
    new Bloom(),
    new Grade(),
    new Merge(),
    new Transform(),
    new CornerPin(),
    new CornerPinSS(),
    new Constant(),
    new CheckerBoard(),
    new ColorBars(),
    new ColorWheel(),
    new NoOp(),
    new ImageStatistics(),
  ];
  return new Map(all.map((e) => [e.describe().identifier, e]));
}

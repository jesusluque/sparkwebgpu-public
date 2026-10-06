// Plugins of the WebGPU splat renderer: manifest, host, tiers and the
// built-in plugins. See PluginHost.ts.

export {
  PluginHost,
  orderPlugins,
  pickVariant,
  type GenerateKernel,
  type PixelKernels,
  type PluginHostOptions,
} from "./PluginHost";
export { TIERS, tierOf, tierSpec, hasFeature, type TierSpec } from "./tiers";
export {
  tintPlugin,
  fadePlugin,
  attributesPlugin,
  dynoPlugin,
  fxPlugin,
  type TintPlugin,
  type FadePlugin,
  type FadeOptions,
  type AttributesPlugin,
} from "./builtin";
export type * from "./types";
export { presets, pluginSlang } from "../generated/presets";

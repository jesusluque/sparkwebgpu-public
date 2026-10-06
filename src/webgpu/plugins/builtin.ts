// Plugins for what the renderer already does, so a PluginHost can switch it
// per asset and order it with the rest, plus two small example plugins that
// exercise the Slang extension points. The classes they wrap keep working on
// their own: these are additions, not replacements.

import {
  SplatAttributes,
  type SplatAttributesOptions,
} from "../attributes/SplatAttributes";
import type { AttributeSpec } from "../attributes/schema";
import type { WgpuDyno } from "../dyno/DynoKernels";
import type { EffectDesc, SplatPlugin } from "./types";

type Vec4 = [number, number, number, number];

export interface TintPlugin extends SplatPlugin {
  /** The tint for assets without their own. */
  tint: Vec4;
  /** Per-asset tints (WgpuSplatMesh keys). */
  readonly tints: WeakMap<object, Vec4>;
  setTint(tint: Vec4, asset?: object): void;
}

/**
 * Example ISplatColour plugin (slang/plugins/examples/tint.slang): each
 * splat's RGBA times a tint, per asset. Generate runs its preset variant on
 * the meshes it is active for and the default kernel on the rest.
 */
export function tintPlugin(tint: Vec4 = [1, 1, 1, 1]): TintPlugin {
  const plugin: TintPlugin = {
    id: "example.tint",
    slang: { module: "plugins.examples.tint", colour: "TintColour" },
    tint,
    tints: new WeakMap(),
    uniforms: [
      {
        block: "tintParams",
        stage: "splat",
        write(w, ctx) {
          const own = ctx.asset && plugin.tints.get(ctx.asset);
          w.set("tint", own ?? plugin.tint);
        },
      },
    ],
    setTint(value, asset) {
      if (asset) plugin.tints.set(asset, value);
      else plugin.tint = value;
      dirty = true;
    },
    onFrame() {
      const was = dirty;
      dirty = false;
      return was ? "dirty" : "clean";
    },
  };
  let dirty = false;
  return plugin;
}

export interface FadeOptions {
  /** Fog colour, rgb. */
  color?: [number, number, number];
  /** Fog at NDC depth 1 (0: none). */
  fog?: number;
  /** fog * depth^power. */
  power?: number;
  /** Alpha multiplier. */
  opacity?: number;
}

export interface FadePlugin extends SplatPlugin {
  options: Required<FadeOptions>;
}

/**
 * Example IBlendTerm plugin (slang/plugins/examples/fade.slang): per pixel,
 * fog towards a colour by depth and an opacity scale, in the quad draw and
 * the tile rasterizer. A pixel-stage plugin: options take effect on the next
 * draw without regenerating.
 */
export function fadePlugin(options: FadeOptions = {}): FadePlugin {
  const plugin: FadePlugin = {
    id: "example.fade",
    slang: { module: "plugins.examples.fade", blend: "FadeBlend" },
    options: { color: [1, 1, 1], fog: 0, power: 8, opacity: 1, ...options },
    uniforms: [
      {
        block: "fadeParams",
        stage: "pixel",
        write(w) {
          const o = plugin.options;
          w.setAll({
            color: [...o.color, o.fog],
            opacity: o.opacity,
            power: o.power,
          });
        },
      },
    ],
    ui: [
      {
        id: "fog",
        type: "number",
        min: 0,
        max: 1,
        get: () => plugin.options.fog,
        set: (v) => {
          plugin.options.fog = Number(v);
        },
      },
      {
        id: "opacity",
        type: "number",
        min: 0,
        max: 1,
        get: () => plugin.options.opacity,
        set: (v) => {
          plugin.options.opacity = Number(v);
        },
      },
    ],
  };
  return plugin;
}

export interface AttributesPlugin extends SplatPlugin {
  /** The stage while attached to a renderer. */
  stage: SplatAttributes | null;
  options: SplatAttributesOptions;
}

/**
 * Per-Gaussian attributes (SplatAttributes: attrib_gather after generate and
 * the attribute draw) as a plugin: attached with the host, it adds the stage
 * to the renderer; `requires` lists the attributes it reads, so the host
 * reports assets without them.
 */
export function attributesPlugin(
  options: SplatAttributesOptions = {},
  extra: { id?: string; attributes?: AttributeSpec[] } = {},
): AttributesPlugin {
  const slots = (options.slots ?? []).map((s) =>
    typeof s === "string" ? s : s.name,
  );
  const plugin: AttributesPlugin = {
    id: extra.id ?? "spark.attributes",
    attributes: extra.attributes,
    requires: { attributes: [...new Set(slots)] },
    options,
    stage: null,
    attach(renderer) {
      plugin.stage ??= new SplatAttributes(renderer, plugin.options);
    },
    detach() {
      plugin.stage?.dispose();
      plugin.stage = null;
    },
  };
  return plugin;
}

/**
 * Dyno graphs (generators, object/world/covariance modifiers) as a plugin:
 * the host chains the graphs of every dyno plugin active on a mesh, in stage
 * order, into the mesh's WgpuDyno. Meshes added with a dyno of their own
 * keep it untouched.
 */
export function dynoPlugin(id: string, dyno: WgpuDyno): SplatPlugin {
  return { id, dyno };
}

/** aofx post effects as a plugin: PluginHost.applyFx(chain) appends them. */
export function fxPlugin(
  id: string,
  fx: EffectDesc[],
  options: Pick<SplatPlugin, "after" | "before" | "order" | "minTier"> = {},
): SplatPlugin {
  return { id, fx, ...options };
}

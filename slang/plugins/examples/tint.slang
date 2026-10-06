// Example colour plugin (ISplatColour): multiplies each splat's colour by a
// per-asset tint, set by src/webgpu/plugins/builtin.ts tintPlugin. Small on
// purpose: it shows a plugin module with its own uniform block, which the
// PluginHost binds by name per generate dispatch.

module tint;

import core.splat_types;
import plugins.splat_colour;

public struct TintParams {
    public float4 tint; // rgba multiplier
};

public ConstantBuffer<TintParams> tintParams;

public struct TintColour : ISplatColour {
    public static float4 colour(float4 rgba, SplatColourInput i) {
        return rgba * tintParams.tint;
    }
}

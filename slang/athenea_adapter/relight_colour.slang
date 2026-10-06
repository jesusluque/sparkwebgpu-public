// The relit colour into generate: an ISplatColour that takes each splat's
// colour from what the relight pass (relight.slang, run per mesh before
// generate by src/webgpu/athenea/relightPlugin.ts) wrote for its source
// index -- linear light, athenea's relitSplat -- and hands it on encoded
// with Spark's transfer, as the raster plugin does (raster.slang), so a
// draw into a linear target blends exactly this light. The splat's own
// alpha is kept unless the pass set one (a thin sheet held to its floor).
module relight_colour;

import core.splat_types;
import plugins.splat_colour;

/// Two uint4 a source splat (relight.slang's kRelightStride): the first is
/// the colour, asuint of (linear rgb, alpha or -1).
public StructuredBuffer<uint4> atheneaRelit;

public struct AtheneaRelightColour : ISplatColour {
    public static float4 colour(float4 rgba, SplatColourInput i) {
        const float4 c = asfloat(atheneaRelit[2u * i.index]);
        return float4(pow(max(c.rgb, float3(0.0)), float3(1.0 / 2.2)), c.a >= 0.0 ? c.a : rgba.a);
    }
}

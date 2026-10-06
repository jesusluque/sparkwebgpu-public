// ISplatColour: the plugin extension point of kernels/generate.slang, where a
// splat's view-dependent colour is made (SH today; TX relighting, lobes and
// emission for the athenea plugins). Per splat, in object space, after the
// source read (or the dyno generator) and before the object modifiers.
//
// A preset (tools/slang-build, slang/plugins/presets.json) compiles generate
// with SPLAT_COLOUR set to a chain of plugin colours; the default preset
// uses DefaultSplatColour alone, which is today's `rgb += sh`, so its kernel
// is the one Spark always had.
//
// Implementations are stateless structs with a static method; a plugin's
// data comes from globals its own module declares (a uniform block, an
// attribute buffer), bound by name by src/webgpu/plugins/PluginHost.ts.

module splat_colour;

import core.splat_types;

public struct SplatColourInput {
    /** The splat as read, in object space (rgba before any colour plugin). */
    public Gsplat splat;
    /** Source splat index (through the LOD list). */
    public uint index;
    /** The spherical-harmonics colour for this view (0 without SH). */
    public float3 sh;
    /** Camera position in object space. */
    public float3 viewObject;
};

public interface ISplatColour {
    /** The colour so far (`rgba`) to the colour after this term. */
    static float4 colour(float4 rgba, SplatColourInput i);
}

/** Spark's colour: the base colour plus the SH term. */
public struct DefaultSplatColour : ISplatColour {
    [ForceInline]
    public static float4 colour(float4 rgba, SplatColourInput i) {
        return float4(rgba.rgb + i.sh, rgba.a);
    }
}

/** A then B. Presets nest these for longer chains. */
public struct ColourChain<A : ISplatColour, B : ISplatColour> : ISplatColour {
    [ForceInline]
    public static float4 colour(float4 rgba, SplatColourInput i) {
        return B.colour(A.colour(rgba, i), i);
    }
}

[ForceInline]
public float4 applySplatColour<C : ISplatColour>(float4 rgba, SplatColourInput i) {
    return C.colour(rgba, i);
}

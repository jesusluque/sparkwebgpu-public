// athenea's base raster, as an ISplatColour plugin of generate
// (src/webgpu/athenea/rasterPlugin.ts): a splat's colour as athenea's
// splat_project makes it, in linear light, for a draw that blends in linear
// light into a float target and an output stage (AtheneaDisplay) after it.
//
//   colour   base + SH as stored (the colour so far), to linear light the way
//            athenea's `cloudLight` takes a cloud: sRGB-decoded unless the
//            colours are stored linear (a .athc flagged linear, decoded with
//            athcKeepLinear); negative harmonics are no light either way
//   override per finest-group range (the .athc's athcGroup attribute, what
//            a pick returns): tint or replace the colour, as athenea's
//            splatOverrideColour does, and scale the opacity (0 hides)
//   emission + scale * the emission stream (RGB9E5, athenea's unpackRgb9e5)
//
// The result goes back out encoded with Spark's own transfer (core.math's
// gamma 2.2), which the draw's DRAW_ENCODE_LINEAR undoes into a linear
// target: the blend then sums exactly this linear light, as splat_blend
// does. (Into an sRGB target it blends encoded, which is athenea's
// displayBlend with Spark's curve.)

module raster;

import core.attrib;
import core.splat_types;
import plugins.splat_colour;
import athenea.common.color;
import athenea.common.packing;
import athenea.common.splat_override;

public static const uint kAtheneaOverrideRows = 16;

public struct AtheneaRasterParams {
    /// 1: the stored colour is linear light already; 0: sRGB-encoded.
    public uint  storedLinear;
    /// Attribute ids in `atheneaPool` (ATTRIB_NONE: the cloud has none).
    public uint  emissionId;
    public uint  groupId;
    /// Rows of `overrides` in use.
    public uint  overrideCount;
    /// Emission added to the colour, times this (0: none).
    public float emissionScale;
    public float pad0;
    public float pad1;
    public float pad2;
    /// Two float4 a row: (asfloat(lo), asfloat(hi), replace, opacity),
    /// (tint rgb, 0). A row applies to an element whose finest groups
    /// [glo, ghi) lie inside [lo, hi): a splat, or a merged LoD node whose
    /// whole subtree is in the range. The first row that applies is taken.
    public float4 overrides[2 * kAtheneaOverrideRows];
};

public ConstantBuffer<AtheneaRasterParams> atheneaRaster;
/// The mesh's attribute pool (core/attrib.slang), or an empty one.
public StructuredBuffer<uint> atheneaPool;

/// The override row for an element covering finest groups [glo, ghi), as
/// athenea's SplatOverride (metallic, roughness and transmission left the
/// gaussian's: the base raster has no material), and its opacity scale.
public SplatOverride atheneaOverrideFor(uint2 groups, out float opacity) {
    SplatOverride o = splatOverrideNone();
    opacity = 1.0;
    for (uint k = 0; k < atheneaRaster.overrideCount; ++k) {
        const float4 head = atheneaRaster.overrides[2 * k];
        const uint lo = asuint(head.x);
        const uint hi = asuint(head.y);
        if (groups.x < lo || groups.y > hi) {
            continue;
        }
        o.tint = atheneaRaster.overrides[2 * k + 1].xyz;
        o.replace = head.z > 0.5;
        o.found = true;
        opacity = head.w;
        return o;
    }
    return o;
}

/// The finest groups element `index` covers; [~0, ~0) where the pool has
/// none (no override applies).
public uint2 atheneaGroups(uint index) {
    const uint id = atheneaRaster.groupId;
    if (id == ATTRIB_NONE || !attribResident(atheneaPool, id, index)) {
        return uint2(0xFFFFFFFFu);
    }
    return uint2(readAttribU(atheneaPool, id, index, 0u), readAttribU(atheneaPool, id, index, 1u));
}

/// Light to what Spark's draw decodes (core.math srgbToLinear, gamma 2.2).
public float3 atheneaEncodeForDraw(float3 light) {
    return pow(max(light, float3(0.0)), float3(1.0 / 2.2));
}

public struct AtheneaRasterColour : ISplatColour {
    public static float4 colour(float4 rgba, SplatColourInput i) {
        float3 light = cloudLight(rgba.rgb, atheneaRaster.storedLinear != 0);
        float alpha = rgba.a;
        if (atheneaRaster.overrideCount != 0) {
            float opacity;
            const SplatOverride said = atheneaOverrideFor(atheneaGroups(i.index), opacity);
            if (said.found) {
                light = splatOverrideColour(light, said);
                alpha *= opacity;
            }
        }
        if (atheneaRaster.emissionId != ATTRIB_NONE && atheneaRaster.emissionScale != 0.0) {
            light += atheneaRaster.emissionScale *
                     unpackRgb9e5(readAttribU(atheneaPool, atheneaRaster.emissionId, i.index, 0u));
        }
        return float4(atheneaEncodeForDraw(light), alpha);
    }
}

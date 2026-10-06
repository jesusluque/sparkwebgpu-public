// The relit colour into generate, and what athenea's blend reads per pixel
// into the draw.
//
// AtheneaRelightColour (ISplatColour, generate): each splat's colour from
// what the relight pass (relight.slang, run per mesh before generate by
// src/webgpu/athenea/relightPlugin.ts) wrote for its source index -- linear
// light, athenea's relitSplat -- handed on encoded with Spark's transfer, as
// the raster plugin does (raster.slang), so a draw into a linear target
// blends exactly this light. The splat's own alpha is kept unless the pass
// set one (a thin sheet held to its floor), and its whole rgba where the
// pass kept it (its transfer page is not resident). It also copies the
// splat's per-pixel record (relight.slang's kRelightPixel: athenea's
// `slopes` and `sharpPolish` records, the centre in pixels) to its slot in
// the accumulator, for the draw.
//
// AtheneaRelightBlend (IBlendTerm, draw and tiles): splat_blend's per-pixel
// shading of a relit splat, read from that record -- the colour plus its
// slope over the step from the centre (kSlopeMark), or the sky read again
// along the mirror this pixel turns to, at the sharp coat's and the
// polish's roughness, with the sun through each lobe (kSharpMark) -- in
// linear light, and encoded again.
module relight_colour;

import core.splat_types;
import plugins.splat_colour;
import plugins.blend_term;
import athenea.common.packing;
import athenea.common.octahedral;
import athenea.light.environment;
import athenea.splat.splat_relight;

/// Five uint4 a source splat (relight.slang's kRelightStride): the colour,
/// asuint of (linear rgb, alpha, -1 or -2); the kept terms; then the three
/// the draw reads.
public StructuredBuffer<uint4> atheneaRelit;
/// Three uint4 an accumulator slot: the draw's copy of those three
/// (generate writes it as atheneaPixelOut, the draw reads it).
public RWStructuredBuffer<uint4> atheneaPixelOut;
public StructuredBuffer<uint4> atheneaPixel;
/// The first dome's prefiltered sky (AtheneaSky's envTexels).
public StructuredBuffer<uint> atheneaEnvTexels;

public struct AtheneaPixelParams {
    public float4 sunDisc;      ///< the first dome's sun: direction, half angle (0: none)
    public float4 sunIrradiance;
    public uint   envBaseSide;
    public uint   pad0;
    public uint   pad1;
    public uint   pad2;
};
public ConstantBuffer<AtheneaPixelParams> atheneaPixelParams;

static const uint kStride = 5u;
static const uint kPixel = 2u;

public struct AtheneaRelightColour : ISplatColour {
    public static float4 colour(float4 rgba, SplatColourInput i) {
        const uint at = kStride * i.index;
        const float4 c = asfloat(atheneaRelit[at]);
        const uint o = 3u * i.outIndex;
        const bool keepAll = c.a < -1.5;
        atheneaPixelOut[o] = keepAll ? uint4(0u) : atheneaRelit[at + kPixel];
        atheneaPixelOut[o + 1u] = keepAll ? uint4(0u) : atheneaRelit[at + kPixel + 1u];
        atheneaPixelOut[o + 2u] = keepAll ? uint4(0u) : atheneaRelit[at + kPixel + 2u];
        if (keepAll) {
            return rgba;
        }
        return float4(pow(max(c.rgb, float3(0.0)), float3(1.0 / 2.2)), c.a >= 0.0 ? c.a : rgba.a);
    }
}

public struct AtheneaRelightBlend : IBlendTerm {
    public static float4 blend(float4 rgba, BlendInput i) {
        const uint o = 3u * i.splatIndex;
        const uint4 head = atheneaPixel[o + 2u];
        const float mark = highHalf(head.w);
        if (!(mark > 0.5)) {
            return rgba;
        }
        const float2 d = i.fragCoord.xy - float2(asfloat(head.x), asfloat(head.y));
        const float3 held = float3(lowHalf(head.z), highHalf(head.z), lowHalf(head.w));
        float3 shade = pow(max(rgba.rgb, float3(0.0)), float3(2.2)) - held;
        const uint4 w = atheneaPixel[o];
        if (mark < 1.5) {
            // splat_blend's kSlopeMark: the colour's slope over the step.
            const float3 slopeX = float3(lowHalf(w.x), highHalf(w.x), lowHalf(w.y));
            const float3 slopeY = float3(highHalf(w.y), lowHalf(w.z), highHalf(w.z));
            shade = shade + slopeX * d.x + slopeY * d.y;
        } else {
            // splat_blend's kSharpMark: the sky along the mirror this pixel
            // turns to, at the lobe's own roughness, and the sun through it.
            const float2 at = float2(lowHalf(w.x), highHalf(w.x)) + float2(lowHalf(w.y), highHalf(w.y)) * d.x +
                              float2(lowHalf(w.z), highHalf(w.z)) * d.y;
            const float3 mirror = octDecode(saturate(at));
            const uint side = atheneaPixelParams.envBaseSide;
            float3 sky = float3(0.0);
            if (lowHalf(w.w) > 0.0) {
                sky += lowHalf(w.w) * envRadiance(atheneaEnvTexels, 0u, mirror, highHalf(w.w), side);
            }
            const uint4 p = atheneaPixel[o + 1u];
            const float3 polish = float3(lowHalf(p.x), highHalf(p.x), lowHalf(p.y));
            if (any(polish > float3(0.0))) {
                sky += polish * envRadiance(atheneaEnvTexels, 0u, mirror, highHalf(p.y), side);
            }
            const float4 disc = atheneaPixelParams.sunDisc;
            if (disc.w > 0.0) {
                const float3 polishSun = float3(lowHalf(p.z), highHalf(p.z), lowHalf(p.w));
                sky += atheneaPixelParams.sunIrradiance.rgb *
                       (polishSun * sunReflectPdf(mirror, disc.xyz, highHalf(p.y), disc.w) +
                        highHalf(p.w) * sunReflectPdf(mirror, disc.xyz, highHalf(w.w), disc.w));
            }
            shade = shade + sky;
        }
        return float4(pow(max(shade, float3(0.0)), float3(1.0 / 2.2)), rgba.a);
    }
}

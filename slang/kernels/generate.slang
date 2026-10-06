// One SplatMesh's splats into the frame's accumulator: the compute pass that
// replaces SplatAccumulator.generate's fragment-shader "compute".
//
// Per output splat: remap the index through the LOD list, read the source
// splat (packed or ext), add its SH colour for the view direction, transform
// it from object to world space, recolour, and write it in the accumulator's
// format together with its sort metric. Mirrors SplatMesh.constructGenerator
// (src/SplatMesh.ts) and the accumulator outputs (src/SplatAccumulator.ts).
//
// Buffer layout, interleaved per splat so a stage needs only five storage
// buffers (WebGPU guarantees eight):
//   src  packed: 1 x uint4      ext: 2 x uint4 (a, b)
//   sh   packed: 3 x uint4 (sh1.xy, sh2, sh3)   ext: 4 x uint4 (sh1, sh2, sh3a, sh3b)
//   out  as src, in the accumulator's format

import core.math;
import core.splat_types;
import core.packed;
import core.ext;
import core.sh;
import core.dyno_hooks;
import core.cov;
import plugins.splat_colour;

// The colour plugins of this build (slang/plugins/splat_colour.slang): a
// preset defines SPLAT_COLOUR before including this file.
#ifndef SPLAT_COLOUR
#define SPLAT_COLOUR DefaultSplatColour
#endif

static const uint GEN_SRC_EXT = 1u;      // source is ext (else packed)
static const uint GEN_OUT_EXT = 2u;      // accumulator is ext (else packed)
static const uint GEN_USE_LOD = 4u;      // remap through lodIndices
static const uint GEN_LOD_OPACITY = 8u;  // packed source stores alpha / 2
static const uint GEN_SORT_RADIAL = 16u; // radial metric (else along viewDir)
static const uint GEN_DYNO_SOURCE = 256u; // dynoSource generates the splats
static const uint GEN_OUT_COV = 512u;    // accumulator holds covariance splats
static const uint GEN_COV_TRANSFORM = 1024u; // covBasis transforms the CovSplat
                                         // (else the similarity, before it)
static const uint GEN_CULL = 2048u;      // drop splats the draw would skip

struct GenerateParams {
    uint numSplats;   // output splats for this mesh (the LOD count when LOD is on)
    uint outBase;     // first accumulator slot
    uint flags;
    int numSh;        // SH degrees to evaluate, 0-3

    uint srcCount;    // splats in the source
    uint shStride;    // uint4s per splat in sh; 0: all degrees (3 packed, 4 ext)
    uint pad1;
    uint pad2;

    float4 rotate;          // object to world quaternion
    float4 translateScale;  // xyz translation, w uniform scale
    float4 recolor;         // rgba multiplier
    float4 encoding;        // packed source (rgbMin, rgbMax, lnScaleMin, lnScaleMax)
    float4 shMax;           // packed SH ranges, xyz
    float4 viewObject;      // camera position in object space (SH direction)
    float4 viewCenter;      // camera position in world space (sort metric)
    float4 viewDir;         // camera forward in world space (z sort)
    float4 outOrigin;       // subtracted from centers before writing (packed)
    float4 covBasis0;       // object to world linear part, columns
    float4 covBasis1;       // (GEN_COV_TRANSFORM)
    float4 covBasis2;
    float4 cullProj0;       // world to clip, columns (GEN_CULL)
    float4 cullProj1;
    float4 cullProj2;
    float4 cullProj3;
    float4 cullViewZ;       // world to view z, a row
    float4 cullParams;      // x clipXY, y minAlpha
};

// The draw's per-splat early outs (splat_draw.slang computeSplatVertex) on
// the world center and alpha: splats it would drop get no sort slot and no
// draw instance. Same tests, so the image is unchanged.
bool culled(float3 center, float alpha) {
    if (alpha == 0.0 || alpha < params.cullParams.y) {
        return true;
    }
    if (dot(params.cullViewZ.xyz, center) + params.cullViewZ.w >= 0.0) {
        return true;
    }
    float4 clip = params.cullProj0 * center.x + params.cullProj1 * center.y + params.cullProj2 * center.z + params.cullProj3;
    float limit = params.cullParams.x * clip.w;
    return clip.z < 0.0 || clip.z >= clip.w || abs(clip.x) > limit || abs(clip.y) > limit;
}

StructuredBuffer<uint4> src;
StructuredBuffer<uint4> sh;
StructuredBuffer<uint> lodIndices;
RWStructuredBuffer<uint4> outSplats;
RWStructuredBuffer<float> sortMetric;
ConstantBuffer<GenerateParams> params;

bool hasFlag(uint f) { return (params.flags & f) != 0u; }

Gsplat readSource(uint index) {
    if (hasFlag(GEN_SRC_EXT)) {
        return readExtSplat(src[2u * index], src[2u * index + 1u], int(index));
    }
    Gsplat s = readPackedSplat(src[index], int(index), params.encoding);
    if (hasFlag(GEN_LOD_OPACITY)) {
        s.rgba.a *= 2.0;
    }
    return s;
}

float3 shColor(uint index, float3 center) {
    if (params.numSh <= 0) {
        return float3(0.0);
    }
    float3 dir = normalize(center - params.viewObject.xyz);
    // Load only the degrees used: a pool sized for fewer has no room for more.
    int n = params.numSh;
    uint4 zero = uint4(0u);
    if (hasFlag(GEN_SRC_EXT)) {
        uint b = (params.shStride != 0u ? params.shStride : 4u) * index;
        return evaluateExtSH(n, dir, sh[b], n >= 2 ? sh[b + 1u] : zero,
                             n >= 3 ? sh[b + 2u] : zero, n >= 3 ? sh[b + 3u] : zero);
    }
    uint b = (params.shStride != 0u ? params.shStride : 3u) * index;
    return evaluatePackedSH(n, dir, params.shMax.xyz, sh[b].xy, n >= 2 ? sh[b + 1u] : zero,
                            n >= 3 ? sh[b + 2u] : zero);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void generate(uint3 tid: SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= params.numSplats) {
        return;
    }
    uint index = hasFlag(GEN_USE_LOD) ? lodIndices[i] : i;

    Gsplat s;
    s.flags = 0u;
    if (index < params.srcCount) {
        s = hasFlag(GEN_DYNO_SOURCE) ? dynoSource(int(index)) : readSource(index);
    }

    float metric = INFINITY;
    bool cov = hasFlag(GEN_OUT_COV);
    CovSplat c;
    c.flags = 0u;
    if (isActive(s.flags)) {
        SplatColourInput colourIn = { s, index, shColor(index, s.center), params.viewObject.xyz };
        s.rgba = applySplatColour<SPLAT_COLOUR>(s.rgba, colourIn);
        s = dynoObjectModifier(s);

        if (cov && hasFlag(GEN_COV_TRANSFORM)) {
            // The full linear transform, as SplatMesh.constructCovGenerator.
            c = dynoCovObjectModifier(gsplatToCovSplat(s));
            float3x3 basis = transpose(float3x3(params.covBasis0.xyz, params.covBasis1.xyz, params.covBasis2.xyz));
            c = transformCovSplat(c, basis, params.translateScale.xyz);
            c.rgba *= params.recolor;
            c = dynoCovWorldModifier(c);
            s.center = c.center;
            s.flags = c.flags;
        } else {
            float scale = params.translateScale.w;
            s.center = quatVec(params.rotate, s.center * scale) + params.translateScale.xyz;
            s.scales *= scale;
            s.quaternion = quatQuat(params.rotate, s.quaternion);
            s.rgba *= params.recolor;
            s = dynoWorldModifier(s);
            if (cov && isActive(s.flags)) {
                // A Gsplat pipeline into a covariance accumulator.
                c = dynoCovWorldModifier(dynoCovObjectModifier(gsplatToCovSplat(s)));
                s.center = c.center;
                s.flags = c.flags;
            }
        }

        float3 rel = s.center - params.viewCenter.xyz;
        metric = hasFlag(GEN_SORT_RADIAL) ? length(rel) : dot(rel, params.viewDir.xyz) + 100.0;
        if (!isActive(s.flags)) metric = INFINITY; // a modifier dropped it
        if (hasFlag(GEN_CULL) && culled(s.center, cov ? c.rgba.a : s.rgba.a)) metric = INFINITY;
    }

    uint o = params.outBase + i;
    sortMetric[o] = metric;
    float3 center = s.center - params.outOrigin.xyz;
    if (hasFlag(GEN_OUT_EXT)) {
        uint4 a = uint4(0u);
        uint4 b = uint4(0u);
        if (isActive(s.flags)) {
            if (cov) {
                packSplatExtCov(a, b, center, c.rgba, c.xxyyzz, c.xyxzyz);
            } else {
                packSplatExt(a, b, center, s.scales, s.quaternion, s.rgba);
            }
        }
        outSplats[2u * o] = a;
        outSplats[2u * o + 1u] = b;
    } else {
        // The packed accumulator keeps alpha / 2 so LOD opacity can reach 2.
        uint4 p = uint4(0u);
        if (isActive(s.flags)) {
            if (cov) {
                p = packSplatCov(center, float4(c.rgba.rgb, 0.5 * c.rgba.a), c.xxyyzz, c.xyxzyz);
            } else {
                p = packSplat(center, s.scales, s.quaternion, float4(s.rgba.rgb, 0.5 * s.rgba.a));
            }
        }
        outSplats[o] = p;
    }
}

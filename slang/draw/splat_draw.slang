// Draws the accumulator's splats as instanced quads, back to front in the
// sorted order. Ported from src/shaders/splatVertex.glsl and
// splatFragment.glsl; the footprint and alpha maths are in
// draw/splat_shape.slang, shared with the tile rasterizer.
//
// One instance per ordering slot, four vertices as a triangle strip.

import core.math;
__exported import draw.splat_shape;
__exported import plugins.blend_term;

// The blend plugins of this build (slang/plugins/blend_term.slang): a preset
// defines SPLAT_BLEND before including this file (or splat_attrib_draw).
#ifndef SPLAT_BLEND
#define SPLAT_BLEND DefaultBlendTerm
#endif

StructuredBuffer<uint> ordering;

// The projector (projectSplats, a compute pass after the sort): each sorted
// slot's footprint, computed once instead of in each of the quad's four
// vertices, in draw order, PROJECTED_WORDS uint4 a slot:
//   [0] clip centre (f32 x4)   [1] rgba (f32 x4)
//   [2] axis1.xy, axis2.xy (f32, scaled pixels)
//   [3] adjustedStdDev (f32), splat index, 1, 0; all 0 when not drawn
// A surfel's (projectSplat's surfel) holds its SurfelRows instead:
//   [0] x.xyz, y.x   [1] y.yz, w.xy   [2] w.z, ndc depth, r, g
//   [3] -adjustedStdDev (f32, the mark), splat index, b, a
static const uint PROJECTED_WORDS = 4u;
static const uint PROJECT_WG = 256u;
static const uint PROJECT_ROUNDS = 4u; // slots a thread: PROJECT_WG x this = the sort's TILE
StructuredBuffer<uint> sortCount;          // the sort's drawArgs: [1] the active count
RWStructuredBuffer<uint4> projectedOut;
StructuredBuffer<uint4> projected;

struct VertexOut {
    float4 position : SV_Position;
    nointerpolation float4 rgba : COLOR0; // the same at the four corners
    float2 splatUv : TEXCOORD0;
    nointerpolation float adjustedStdDev : TEXCOORD1;
    nointerpolation uint splatIndex : TEXCOORD2;
    // View ray through the fragment (z = -1) and the splat's axial depth.
    float3 viewRay : TEXCOORD3;
    nointerpolation float viewDepth : TEXCOORD4;
    // A surfel's SurfelRows (adjustedStdDev < 0; splatUv is then the offset
    // from its centre in surfel pixels).
    nointerpolation float3 surfelX : TEXCOORD5;
    nointerpolation float3 surfelY : TEXCOORD6;
    nointerpolation float3 surfelW : TEXCOORD7;
};

// VertexOut without semantics, for the stage functions below: slangc gives
// every field of a semantic struct location 0 when it is not an entry's own
// input or output.
struct SplatVertex {
    float4 position;
    float4 rgba;
    float2 splatUv;
    float adjustedStdDev; // negative for a surfel
    uint splatIndex;
    float3 surfelX;
    float3 surfelY;
    float3 surfelW;
};

SplatVertex emptySplatVertex() {
    SplatVertex o;
    // Outside the clip volume, so the quad is dropped if we return early.
    o.position = float4(0.0, 0.0, 2.0, 1.0);
    o.rgba = float4(0.0);
    o.splatUv = float2(0.0);
    o.adjustedStdDev = 0.0;
    o.splatIndex = 0u;
    o.surfelX = float3(0.0);
    o.surfelY = float3(0.0);
    o.surfelW = float3(0.0);
    return o;
}

// A surfel's quad corner: surfelBox around its centre, at its centre's
// depth (as the 3D quads), the offset from the centre as splatUv.
void surfelCorner(inout SplatVertex o, float2 corner, SurfelRows rows, float adjustedStdDev, float ndcDepth) {
    float2 lo, hi;
    surfelQuad(rows, adjustedStdDev, lo, hi);
    const float2 c = float2(rows.x.z, rows.y.z);
    const float2 d = surfelOffset(corner > float2(0.0) ? hi : lo, c);
    const float w = rows.w.z;
    const float2 ndc = (c + d) / surfelPixelScale();
    o.position = float4(ndc * w, ndcDepth * w, w);
    o.splatUv = d;
    o.adjustedStdDev = -adjustedStdDev;
    o.surfelX = rows.x;
    o.surfelY = rows.y;
    o.surfelW = rows.w;
}

static const float2 CORNERS[4] = { float2(-1.0, -1.0), float2(1.0, -1.0), float2(-1.0, 1.0), float2(1.0, 1.0) };

// Whether this pass leaves the splat out (DRAW_ADDITIVE_ONLY /
// DRAW_NO_ADDITIVE; alpha's sign is the additive mark, markedRgba).
bool layerDrops(float markedAlpha) {
    const bool additive = markedAlpha < 0.0;
    return (hasFlag(DRAW_ADDITIVE_ONLY) && !additive) || (hasFlag(DRAW_NO_ADDITIVE) && additive);
}

// The stages as plain functions, for draw variants to call (slangc hangs
// calling an entry point from another).
SplatVertex computeSplatVertex(uint vertexId, uint instanceId) {
    SplatVertex o = emptySplatVertex();
    uint splatIndex = ordering[instanceId];
    if (splatIndex == 0xffffffffu) {
        return o;
    }
    float2 corner = CORNERS[vertexId & 3u];
    SplatShape s;
    if (!projectSplat(splatIndex, s)) {
        return o;
    }
    o.rgba = markedRgba(s);
    if (layerDrops(o.rgba.a)) {
        o.rgba = float4(0.0);
        return o;
    }
    o.splatIndex = splatIndex;
    if (s.surfel) {
        surfelCorner(o, corner, s.rows, s.adjustedStdDev, s.clipCenter.z / s.clipCenter.w);
        return o;
    }
    o.splatUv = corner * s.adjustedStdDev;
    o.adjustedStdDev = s.adjustedStdDev;

    o.position = quadCorner(corner, s.clipCenter, s.axis1, s.axis2);
    return o;
}

// A quad corner's clip position from the footprint.
float4 quadCorner(float2 corner, float4 clipCenter, float2 axis1, float2 axis2) {
    float2 scaledRenderSize = float2(params.renderWidth, params.renderHeight) * params.focalAdjustment;
    float2 pixelOffset = corner.x * axis1 + corner.y * axis2;
    float2 ndcOffset = (2.0 / scaledRenderSize) * pixelOffset;
    float2 ndc = clipCenter.xy / clipCenter.w + ndcOffset;
    return float4(ndc * clipCenter.w, clipCenter.zw);
}

// The projector: projectSplat once per sorted slot, dispatched with the sort's radix dispatch args
// (a workgroup per TILE active slots).
[shader("compute")]
[numthreads(256, 1, 1)]
void projectSplats(uint3 lid: SV_GroupThreadID, uint3 gid: SV_GroupID) {
    uint count = sortCount[1];
    for (uint r = 0u; r < PROJECT_ROUNDS; r++) {
        uint slot = (gid.x * PROJECT_ROUNDS + r) * PROJECT_WG + lid.x;
        if (slot >= count) {
            return;
        }
        uint splatIndex = ordering[slot];
        SplatShape s;
        bool drawn = splatIndex != 0xffffffffu && projectSplat(splatIndex, s);
        uint o = PROJECTED_WORDS * slot;
        if (!drawn) {
            projectedOut[o + 3u] = uint4(0u);
            continue;
        }
        if (s.surfel) {
            const SurfelRows r = s.rows;
            const float4 rgba = markedRgba(s);
            projectedOut[o] = asuint(float4(r.x, r.y.x));
            projectedOut[o + 1u] = asuint(float4(r.y.yz, r.w.xy));
            projectedOut[o + 2u] = asuint(float4(r.w.z, s.clipCenter.z / s.clipCenter.w, rgba.rg));
            projectedOut[o + 3u] = uint4(asuint(-s.adjustedStdDev), splatIndex, asuint(rgba.b), asuint(rgba.a));
            continue;
        }
        projectedOut[o] = asuint(s.clipCenter);
        projectedOut[o + 1u] = asuint(markedRgba(s));
        projectedOut[o + 2u] = asuint(float4(s.axis1, s.axis2));
        projectedOut[o + 3u] = uint4(asuint(s.adjustedStdDev), splatIndex, 1u, 0u);
    }
}

// computeSplatVertex from the projector's record of slot `instanceId`.
SplatVertex projectedSplatVertex(uint vertexId, uint instanceId) {
    SplatVertex o = emptySplatVertex();
    uint b = PROJECTED_WORDS * instanceId;
    uint4 e = projected[b + 3u];
    if (e.x == 0u) {
        return o;
    }
    float2 corner = CORNERS[vertexId & 3u];
    if (asfloat(e.x) < 0.0) {
        const float4 r0 = asfloat(projected[b]);
        const float4 r1 = asfloat(projected[b + 1u]);
        const float4 r2 = asfloat(projected[b + 2u]);
        o.rgba = float4(r2.zw, asfloat(e.zw));
        if (layerDrops(o.rgba.a)) {
            o.rgba = float4(0.0);
            return o;
        }
        o.splatIndex = e.y;
        const SurfelRows rows = { r0.xyz, float3(r0.w, r1.xy), float3(r1.zw, r2.x) };
        surfelCorner(o, corner, rows, -asfloat(e.x), r2.y);
        return o;
    }
    float4 axes = asfloat(projected[b + 2u]);
    float adjustedStdDev = asfloat(e.x);
    o.rgba = asfloat(projected[b + 1u]);
    if (layerDrops(o.rgba.a)) {
        o.rgba = float4(0.0);
        return o;
    }
    o.splatUv = corner * adjustedStdDev;
    o.adjustedStdDev = adjustedStdDev;
    o.splatIndex = e.y;
    o.position = quadCorner(corner, asfloat(projected[b]), axes.xy, axes.zw);
    return o;
}

// `v.position` is the fragment's SV_Position. B: the blend plugins.
float4 computeSplatFragment<B : IBlendTerm>(SplatVertex v) {
    float4 rgba = v.rgba;
    // An additive splat (alpha's sign, markedRgba): its light is added,
    // premultiplied, and its alpha written 0 so nothing behind is covered.
    const bool additive = rgba.a < 0.0;
    rgba.a = abs(rgba.a);
    float2 uv = v.splatUv;
    const float stdDev = abs(v.adjustedStdDev);
    const float z2 = v.adjustedStdDev < 0.0 ? surfelRadius2(v.splatUv, v.surfelX, v.surfelY, v.surfelW, uv) : dot(uv, uv);
    if (z2 > stdDev * stdDev) {
        discard;
    }
    rgba.a = splatAlpha(rgba.a, z2);
    if (rgba.a < params.minAlpha) {
        discard;
    }
    BlendInput blendIn = { v.splatIndex, uv, v.position };
    rgba = applyBlendTerm<B>(rgba, blendIn);
    if (hasFlag(DRAW_ENCODE_LINEAR)) {
        rgba.rgb = srgbToLinear(rgba.rgb);
    }
    // In its own layer (DRAW_ADDITIVE_ONLY) an additive splat covers the
    // additive splats behind it: the layer is their composite.
    const float cover = additive && !hasFlag(DRAW_ADDITIVE_ONLY) ? 0.0 : rgba.a;
    return hasFlag(DRAW_PREMULTIPLIED) ? float4(rgba.rgb * rgba.a, cover) : float4(rgba.rgb, cover);
}

[shader("vertex")]
VertexOut splatVertex(uint vertexId: SV_VertexID, uint instanceId: SV_InstanceID) {
    SplatVertex s = computeSplatVertex(vertexId, instanceId);
    float2 ndc = s.position.xy / s.position.w;
    float3 viewRay = float3(ndc.x / params.proj0.x, ndc.y / params.proj1.y, -1.0);
    return { s.position, s.rgba, s.splatUv, s.adjustedStdDev, s.splatIndex, viewRay, s.position.w, s.surfelX, s.surfelY, s.surfelW };
}

[shader("vertex")]
VertexOut splatVertexProjected(uint vertexId: SV_VertexID, uint instanceId: SV_InstanceID) {
    SplatVertex s = projectedSplatVertex(vertexId, instanceId);
    float2 ndc = s.position.xy / s.position.w;
    float3 viewRay = float3(ndc.x / params.proj0.x, ndc.y / params.proj1.y, -1.0);
    return { s.position, s.rgba, s.splatUv, s.adjustedStdDev, s.splatIndex, viewRay, s.position.w, s.surfelX, s.surfelY, s.surfelW };
}

[shader("fragment")]
float4 splatFragment(VertexOut v) : SV_Target {
    if (hasFlag(DRAW_DISK_CLIP) && params.diskCenter.w != 0.0 && diskClipped(v.viewRay, v.viewDepth)) {
        discard;
    }
    return computeSplatFragment<SPLAT_BLEND>({ v.position, v.rgba, v.splatUv, v.adjustedStdDev, v.splatIndex, v.surfelX, v.surfelY, v.surfelW });
}

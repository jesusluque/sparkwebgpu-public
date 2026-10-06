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

struct VertexOut {
    float4 position : SV_Position;
    float4 rgba : COLOR0;
    float2 splatUv : TEXCOORD0;
    nointerpolation float adjustedStdDev : TEXCOORD1;
    nointerpolation uint splatIndex : TEXCOORD2;
    // View ray through the fragment (z = -1) and the splat's axial depth.
    float3 viewRay : TEXCOORD3;
    nointerpolation float viewDepth : TEXCOORD4;
};

// VertexOut without semantics, for the stage functions below: slangc gives
// every field of a semantic struct location 0 when it is not an entry's own
// input or output.
struct SplatVertex {
    float4 position;
    float4 rgba;
    float2 splatUv;
    float adjustedStdDev;
    uint splatIndex;
};

static const float2 CORNERS[4] = { float2(-1.0, -1.0), float2(1.0, -1.0), float2(-1.0, 1.0), float2(1.0, 1.0) };

// The stages as plain functions, for draw variants to call (slangc hangs
// calling an entry point from another).
SplatVertex computeSplatVertex(uint vertexId, uint instanceId) {
    SplatVertex o;
    // Outside the clip volume, so the quad is dropped if we return early.
    o.position = float4(0.0, 0.0, 2.0, 1.0);
    o.rgba = float4(0.0);
    o.splatUv = float2(0.0);
    o.adjustedStdDev = 0.0;
    o.splatIndex = 0u;

    uint splatIndex = ordering[instanceId];
    if (splatIndex == 0xffffffffu) {
        return o;
    }
    float2 corner = CORNERS[vertexId & 3u];
    SplatShape s;
    if (!projectSplat(splatIndex, s)) {
        return o;
    }
    o.rgba = s.rgba;
    o.splatUv = corner * s.adjustedStdDev;
    o.adjustedStdDev = s.adjustedStdDev;
    o.splatIndex = splatIndex;

    if (s.flat) {
        float3 offset;
        if (s.zeroScales.z) {
            offset = float3(o.splatUv * s.scales.xy, 0.0);
        } else if (s.zeroScales.y) {
            offset = float3(o.splatUv.x * s.scales.x, 0.0, o.splatUv.y * s.scales.z);
        } else {
            offset = float3(0.0, o.splatUv * s.scales.yz);
        }
        o.position = project(s.viewCenter + quatVec(s.viewQuaternion, offset));
        return o;
    }

    float2 scaledRenderSize = float2(params.renderWidth, params.renderHeight) * params.focalAdjustment;
    float2 pixelOffset = corner.x * s.axis1 + corner.y * s.axis2;
    float2 ndcOffset = (2.0 / scaledRenderSize) * pixelOffset;
    float2 ndc = s.clipCenter.xy / s.clipCenter.w + ndcOffset;
    o.position = float4(ndc * s.clipCenter.w, s.clipCenter.zw);
    return o;
}

// `v.position` is the fragment's SV_Position. B: the blend plugins.
float4 computeSplatFragment<B : IBlendTerm>(SplatVertex v) {
    float4 rgba = v.rgba;
    float z2 = dot(v.splatUv, v.splatUv);
    if (z2 > v.adjustedStdDev * v.adjustedStdDev) {
        discard;
    }
    rgba.a = splatAlpha(rgba.a, z2);
    if (rgba.a < params.minAlpha) {
        discard;
    }
    BlendInput blendIn = { v.splatIndex, v.splatUv, v.position };
    rgba = applyBlendTerm<B>(rgba, blendIn);
    if (hasFlag(DRAW_ENCODE_LINEAR)) {
        rgba.rgb = srgbToLinear(rgba.rgb);
    }
    return hasFlag(DRAW_PREMULTIPLIED) ? float4(rgba.rgb * rgba.a, rgba.a) : rgba;
}

[shader("vertex")]
VertexOut splatVertex(uint vertexId: SV_VertexID, uint instanceId: SV_InstanceID) {
    SplatVertex s = computeSplatVertex(vertexId, instanceId);
    float2 ndc = s.position.xy / s.position.w;
    float3 viewRay = float3(ndc.x / params.proj0.x, ndc.y / params.proj1.y, -1.0);
    return { s.position, s.rgba, s.splatUv, s.adjustedStdDev, s.splatIndex, viewRay, s.position.w };
}

[shader("fragment")]
float4 splatFragment(VertexOut v) : SV_Target {
    if (hasFlag(DRAW_DISK_CLIP) && params.diskCenter.w != 0.0 && diskClipped(v.viewRay, v.viewDepth)) {
        discard;
    }
    return computeSplatFragment<SPLAT_BLEND>({ v.position, v.rgba, v.splatUv, v.adjustedStdDev, v.splatIndex });
}

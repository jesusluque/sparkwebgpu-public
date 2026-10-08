// The splat draw with per-Gaussian attributes: draw/splat_draw.slang's
// vertex and fragment stages, plus attributes read from the draw pool
// (kernels/attrib_gather.slang) into flat varyings, colour modes driven by
// them, and extra render targets:
//   target 1  id:    the front-most splat with alpha >= pickAlpha, as
//                    accumulator index + 1 in rgb bytes (rgba8unorm), for
//                    picking; alpha 1 selects it through the blend state
//   target 2, 3:     varying slot 0 / 1, alpha-composited like the colour
//                    (rgba16float, xyz premultiplied, w coverage)
// Targets the pipeline leaves out are simply not written.

import core.math;
import core.packed;
import core.ext;
import core.attrib;
import draw.splat_draw;

#ifndef SPLAT_BLEND
#define SPLAT_BLEND DefaultBlendTerm
#endif

static const uint ATTRIB_SLOTS = 4u;

// Colour modes.
static const uint COLOR_SPLAT = 0u;   // the splat's own colour
static const uint COLOR_LABEL = 1u;   // palette by integer value of colorSlot.x
static const uint COLOR_RELIGHT = 2u; // Lambert with colorSlot.xyz as normal
static const uint COLOR_PROJECT = 3u; // rgb = offset + project * (slot, slot + 1)
static const uint COLOR_VALUE = 4u;   // colorSlot.xyz as colour

struct AttribDrawParams {
    // Each varying slot resolved on the host against the draw layout (a
    // comp4 group of one attribute), so the vertex stage reads its words
    // directly instead of the pool's descriptor table: 5-20% off the draw
    // when vertex-bound (1-2M small splats on Dawn).
    uint4 slotFormat;   // ATTRIB_NONE if unused
    uint4 slotWord;     // first word of the group within a splat's record
    uint4 slotWords;    // its words (1 to 4)
    uint4 slotCount;    // its valid components (1 to 4)
    uint strideWords;
    uint headerWords;
    uint pad0;
    uint pad1;

    uint colorMode;
    uint colorSlot;
    float pickAlpha;
    uint highlight;     // label + 1 to highlight in COLOR_LABEL, 0 for none

    float4 light;       // world direction towards the light xyz, ambient w
    float4 projectR0;   // COLOR_PROJECT rows over 8 components
    float4 projectR1;
    float4 projectG0;
    float4 projectG1;
    float4 projectB0;
    float4 projectB1;
    float4 projectOffset;
};

StructuredBuffer<uint> attribs;
ConstantBuffer<AttribDrawParams> attribParams;

// SplatVertex as varyings, plus the slots. (Nesting VertexOut instead
// crashes slangc 2026.14 when both stages compile together.)
struct AttribVertexOut {
    float4 position : SV_Position;
    nointerpolation float4 rgba : COLOR0; // the same at the four corners
    float2 splatUv : TEXCOORD0;
    nointerpolation float adjustedStdDev : TEXCOORD1;
    nointerpolation uint splatIndex : TEXCOORD2;
    nointerpolation float4 slot0 : TEXCOORD3;
    nointerpolation float4 slot1 : TEXCOORD4;
    nointerpolation float4 slot2 : TEXCOORD5;
    nointerpolation float4 slot3 : TEXCOORD6;
    // For DRAW_DISK_CLIP, as splat_draw.slang's VertexOut.
    float3 viewRay : TEXCOORD7;
    nointerpolation float viewDepth : TEXCOORD8;
    // A surfel's rows, as splat_draw.slang's VertexOut.
    nointerpolation float3 surfelX : TEXCOORD9;
    nointerpolation float3 surfelY : TEXCOORD10;
    nointerpolation float3 surfelW : TEXCOORD11;
};

float4 readSlot(uint k, uint splat) {
    uint format = attribParams.slotFormat[k];
    if (format == ATTRIB_NONE) {
        return float4(0.0);
    }
    uint base = attribParams.headerWords + splat * attribParams.strideWords + attribParams.slotWord[k];
    uint4 w = readAttribWords(attribs, base, attribParams.slotWords[k]);
    return decodeAttrib4(format, w, attribParams.slotCount[k]);
}

float3 labelColor(uint label) {
    // Golden-ratio hues, bright and well separated for small labels.
    float h = frac(float(label) * 0.61803398875 + 0.1);
    float3 rgb = saturate(abs(frac(h + float3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0);
    return lerp(float3(1.0), rgb, 0.8);
}

// World-space splat center, for the view direction in COLOR_RELIGHT.
float3 splatCenter(uint splat) {
    if (hasFlag(DRAW_EXT)) {
        return unpackSplatExtCenterAlpha(splats[2u * splat]).xyz;
    }
    float3 center, scales;
    float4 quaternion, rgba;
    unpackSplat(splats[splat], center, scales, quaternion, rgba);
    return center;
}

float3 shade(float3 rgb, float4 a, float4 b, uint splat) {
    switch (attribParams.colorMode) {
    case COLOR_LABEL: {
        uint label = uint(max(a.x, 0.0) + 0.5);
        float3 c = labelColor(label);
        if (attribParams.highlight != 0u) {
            c = label + 1u == attribParams.highlight ? float3(1.0, 0.9, 0.2) : c * 0.35;
        }
        return c * (0.35 + 0.65 * dot(rgb, float3(0.299, 0.587, 0.114)) / 0.5);
    }
    case COLOR_RELIGHT: {
        float3 n = a.xyz;
        float len = length(n);
        if (len < 1e-6) {
            return rgb;
        }
        n /= len;
        // Camera position: renderToView maps world to view, so invert it.
        float4 q = params.renderToViewQuat;
        float3 eye = quatVec(float4(-q.xyz, q.w), -params.renderToViewPos.xyz);
        // Splat normals have no inside or outside; face them to the camera.
        if (dot(n, eye - splatCenter(splat)) < 0.0) {
            n = -n;
        }
        float diffuse = max(dot(n, normalize(attribParams.light.xyz)), 0.0);
        return rgb * (attribParams.light.w + (1.0 - attribParams.light.w) * diffuse) * 1.3;
    }
    case COLOR_PROJECT:
        return saturate(attribParams.projectOffset.xyz + float3(
            dot(attribParams.projectR0, a) + dot(attribParams.projectR1, b),
            dot(attribParams.projectG0, a) + dot(attribParams.projectG1, b),
            dot(attribParams.projectB0, a) + dot(attribParams.projectB1, b)));
    case COLOR_VALUE:
        return a.xyz;
    default:
        return rgb;
    }
}

[shader("vertex")]
AttribVertexOut splatAttribVertex(uint vertexId: SV_VertexID, uint instanceId: SV_InstanceID) {
    SplatVertex base = computeSplatVertex(vertexId, instanceId);
    AttribVertexOut o;
    o.position = base.position;
    o.rgba = base.rgba;
    o.splatUv = base.splatUv;
    o.adjustedStdDev = base.adjustedStdDev;
    o.splatIndex = base.splatIndex;
    o.surfelX = base.surfelX;
    o.surfelY = base.surfelY;
    o.surfelW = base.surfelW;
    float2 ndc = base.position.xy / base.position.w;
    o.viewRay = float3(ndc.x / params.proj0.x, ndc.y / params.proj1.y, -1.0);
    o.viewDepth = base.position.w;
    uint splat = base.splatIndex;
    o.slot0 = readSlot(0u, splat);
    o.slot1 = readSlot(1u, splat);
    o.slot2 = readSlot(2u, splat);
    o.slot3 = readSlot(3u, splat);
    if (attribParams.colorMode != COLOR_SPLAT) {
        float4 slots[4] = { o.slot0, o.slot1, o.slot2, o.slot3 };
        uint k = min(attribParams.colorSlot, 2u);
        o.rgba.rgb = shade(o.rgba.rgb, slots[k], slots[k + 1u], splat);
    }
    return o;
}

struct AttribFragmentOut {
    float4 color : SV_Target0;
    float4 id : SV_Target1;
    float4 attrib0 : SV_Target2;
    float4 attrib1 : SV_Target3;
};

[shader("fragment")]
AttribFragmentOut splatAttribFragment(AttribVertexOut v) {
    if (hasFlag(DRAW_DISK_CLIP) && params.diskCenter.w != 0.0 && diskClipped(v.viewRay, v.viewDepth)) {
        discard;
    }
    SplatVertex base;
    base.position = v.position;
    base.rgba = v.rgba;
    base.splatUv = v.splatUv;
    base.adjustedStdDev = v.adjustedStdDev;
    base.splatIndex = v.splatIndex;
    base.surfelX = v.surfelX;
    base.surfelY = v.surfelY;
    base.surfelW = v.surfelW;
    AttribFragmentOut o;
    o.color = computeSplatFragment<SPLAT_BLEND>(base);
    // Premultiplied or not, the colour's alpha is the splat's coverage here.
    float alpha = o.color.a;
    uint id = v.splatIndex + 1u;
    float3 bytes = float3(id & 0xffu, (id >> 8u) & 0xffu, (id >> 16u) & 0xffu) / 255.0;
    o.id = float4(bytes, alpha >= attribParams.pickAlpha ? 1.0 : 0.0);
    o.attrib0 = float4(v.slot0.xyz * alpha, alpha);
    o.attrib1 = float4(v.slot1.xyz * alpha, alpha);
    return o;
}

// athenea's display transform (athenea/technique/display.slang, vendored) as
// an output stage over a picture: the aofx layout the fx chain uses (a flat
// float4 buffer, premultiplied linear Rec.709, `stride` pixels a row), so
// it runs as an Effect (src/webgpu/athenea/AtheneaDisplay.ts) after the
// splats were blended in linear light.
//
//   colour   premultiplied over a background, times 2^exposure, through a
//            view transform (Standard, AgX, ACES 2.0), into the display's
//            gamut and transfer (sRGB, BT.1886, Display P3, linear P3)
//
// `displayColourOf` is display.slang's `displayColour` line for line, with
// the two things it reads from display.slang's own globals (the ACES tables
// and the headroom) passed in, so this kernel binds a picture rather than
// display.slang's texture. The view transforms themselves -- `agx`,
// `outputTransformFwd` -- are athenea's functions, called as they are; a
// Dawn test checks this against `displayColour` itself
// (slang/tests/athenea_display.slang).
//
// The ACES 2.0 tables are built here by athenea's three prepare passes
// (aces2_prepare.slang, whose kernels are vendored and so not compiled:
// the same bodies, over this module's buffers).
module display;

import athenea.technique.aces2;
import athenea.technique.display;

public struct AtheneaDisplayParams {
    public uint  width;        // pixels written
    public uint  height;
    public uint  srcStride;    // picture rows, in pixels
    public uint  dstStride;
    public int   srcOffsetX;   // where the target's pixel 0 sits in the source
    public int   srcOffsetY;
    public uint  srcWidth;
    public uint  srcHeight;
    public uint  view;         // 0 Standard, 1 AgX, 2 ACES 2.0
    public uint  display;      // 0 sRGB, 1 Rec.709 (BT.1886), 2 Display P3, 3 linear P3, extended
    public float exposure;     // stops
    public float peakScale;    // display 3: headroom over reference white
    public float backgroundR;  // display-referred linear, under what the picture leaves uncovered
    public float backgroundG;
    public float backgroundB;
    /// 0: the encoded value (a unorm target shows it as it is). 1: the value
    /// an sRGB-encoding output (three's canvas, an -srgb texture) encodes to
    /// that code value: the sRGB curve undone, so display 0 hands on the
    /// display-referred linear colour and the others their own code values.
    public uint  forSrgbOutput;
};

public struct AcesPrepareParams {
    public float peakLuminance;
    public uint  limiting;   // 0 Rec.709, 1 P3-D65
    public uint  pad0;
    public uint  pad1;
};

StructuredBuffer<float4>              src;
RWStructuredBuffer<float4>            dst;
// Named apart from display.slang's own acesParams and acesTables, which an
// import brings into scope.
StructuredBuffer<Aces2Params>         displayAces;        // one, built by the prepare passes
StructuredBuffer<float>               displayAcesTables;  // kTableWords
ConstantBuffer<AtheneaDisplayParams>  displayParams;

/// display.slang's `transferEncode`, with the headroom given.
public float3 transferEncodeOf(float3 c, uint display, float peakScale) {
    if (display == 3) {
        return clamp(c, float3(0.0), float3(peakScale));
    }
    c = saturate(c);
    if (display == 1) {
        return pow(c, float3(1.0 / 2.4));
    }
    return float3(srgbEncodeOf(c.r), srgbEncodeOf(c.g), srgbEncodeOf(c.b));
}

public float srgbEncodeOf(float x) {
    return x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

public float srgbDecodeOf(float x) {
    return x <= 0.04045 ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4);
}

/// display.slang's `displayColour`, with its ACES tables and headroom given.
public float3 displayColourOf(float4 premultiplied, float3 background, float exposure, uint view, uint display,
                              float peakScale, StructuredBuffer<Aces2Params> aces, StructuredBuffer<float> tables) {
    float3 c = premultiplied.rgb + background * (1.0 - saturate(premultiplied.a));
    c *= pow(2.0, exposure);
    if (view == 1) {
        c = agx(c);
    }
    if (view == 2) {
        const Aces2Params p = aces[0];
        return transferEncodeOf(outputTransformFwd(mul(c, p.sceneToAP0), p, tables), display, peakScale);
    }
    if (display == 2 || display == 3) {
        c = mul(kRec709ToP3, c);
    }
    return transferEncodeOf(c, display, peakScale);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void atheneaDisplay(uint3 tid: SV_DispatchThreadID) {
    const AtheneaDisplayParams p = displayParams;
    if (tid.x >= p.width || tid.y >= p.height) {
        return;
    }
    const int sx = int(tid.x) + p.srcOffsetX;
    const int sy = int(tid.y) + p.srcOffsetY;
    const bool inside = sx >= 0 && sy >= 0 && sx < int(p.srcWidth) && sy < int(p.srcHeight);
    const float4 c = inside ? src[uint(sy) * p.srcStride + uint(sx)] : float4(0.0);
    float3 shown = displayColourOf(c, float3(p.backgroundR, p.backgroundG, p.backgroundB), p.exposure, p.view,
                                   p.display, p.peakScale, displayAces, displayAcesTables);
    if (p.forSrgbOutput != 0 && p.display != 3) {
        shown = float3(srgbDecodeOf(shown.r), srgbDecodeOf(shown.g), srgbDecodeOf(shown.b));
    }
    // Opaque: the background took whatever the picture left uncovered.
    dst[tid.y * p.dstStride + tid.x] = float4(shown, 1.0);
}

// --- ACES 2.0 tables: aces2_prepare.slang's three passes ---------------------

ConstantBuffer<AcesPrepareParams> prepare;
RWStructuredBuffer<Aces2Params>   acesParamsOut;
RWStructuredBuffer<float>         acesTablesOut;

Chromaticities limitingPrimaries() { return prepare.limiting == 1 ? kP3D65 : kRec709; }

[shader("compute")]
[numthreads(1, 1, 1)]
void acesPrepareParams(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) return;
    Aces2Params p = initAces2Params(prepare.peakLuminance, limitingPrimaries(), kRec709);
    float3 reachJMh[kTotalCorners];
    float3 limitRGB[kTotalCorners];
    float3 limitJMh[kTotalCorners];
    findReachCorners(reachJMh, p.reach, p.limit_J_max, p.ts.forward_limit);
    buildLimitingCuspCorners(limitRGB, limitJMh, p.limit, p.peakLuminance);
    float sorted[kMaxSortedCorners];
    extractSortedCubeHues(sorted, reachJMh, limitJMh);
    float hues[kTotalTableSize];
    buildHueTable(hues, sorted);
    for (int i = 0; i < kTotalTableSize; ++i) {
        acesTablesOut[kHuesOffset + i] = hues[i];
    }
    // The reference's determine_hue_linearity_search_range.
    int lo = 0;
    int hi = 1;
    for (int i = kBaseIndex; i < kBaseIndex + kTableSize; ++i) {
        const int pos = int(fmod(max(hues[i], 0.0), 360.0) / 360.0 * float(kTotalTableSize));
        const int delta = i - pos;
        lo = min(lo, delta + 0);
        hi = max(hi, delta + 1);
    }
    p.searchLo = lo;
    p.searchHi = hi;
    acesParamsOut[0] = p;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void acesPrepareTables(uint3 tid: SV_DispatchThreadID) {
    const int i = int(tid.x);
    if (i >= kTableSize) return;
    const Aces2Params p = acesParamsOut[0];
    acesTablesOut[kReachOffset + kBaseIndex + i] =
        reachMForHue(float(i) * 360.0 / float(kTableSize), p.reach, p.limit_J_max);
    float3 limitRGB[kTotalCorners];
    float3 limitJMh[kTotalCorners];
    buildLimitingCuspCorners(limitRGB, limitJMh, p.limit, p.peakLuminance);
    const float hue = acesTablesOut[kHuesOffset + kBaseIndex + i];
    const float2 JM = findDisplayCuspForHue(hue, limitRGB, limitJMh, p.limit);
    const float2 cusp = float2(JM.x, JM.y * (1.0 + 0.27 * 0.12));   // smooth_m * smooth_cusps
    acesTablesOut[kCuspJOffset + kBaseIndex + i] = cusp.x;
    acesTablesOut[kCuspMOffset + kBaseIndex + i] = cusp.y;
    acesTablesOut[kGammaOffset + kBaseIndex + i] = upperHullGammaForCusp(cusp, hue, p);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void acesPrepareWrap(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) return;
    const int last = kBaseIndex + kTableSize;   // 361
    acesTablesOut[kReachOffset + 0] = acesTablesOut[kReachOffset + kTableSize];
    acesTablesOut[kReachOffset + last] = acesTablesOut[kReachOffset + kBaseIndex];
    acesTablesOut[kCuspJOffset + 0] = acesTablesOut[kCuspJOffset + kTableSize];
    acesTablesOut[kCuspMOffset + 0] = acesTablesOut[kCuspMOffset + kTableSize];
    acesTablesOut[kCuspJOffset + last] = acesTablesOut[kCuspJOffset + kBaseIndex];
    acesTablesOut[kCuspMOffset + last] = acesTablesOut[kCuspMOffset + kBaseIndex];
    acesTablesOut[kGammaOffset + 0] = acesTablesOut[kGammaOffset + kTableSize];
    acesTablesOut[kGammaOffset + last] = acesTablesOut[kGammaOffset + kBaseIndex];
}

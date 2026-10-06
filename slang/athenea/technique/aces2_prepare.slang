// Copyright (c) 2026 jesus luque.
//
// The kernels that build an ACES 2.0 output transform's parameters and
// tables for one peak luminance and one set of limiting primaries: what
// the reference computes once at init, here on the device. Three passes:
// the parameters and the hue table (one thread, since the hue table is
// built from the sorted corner hues in order), then the reach, cusp and
// upper hull gamma for each hue (a thread each), then the wrap entries.
import athenea.technique.aces2;

struct PrepareParams {
    float peakLuminance;
    uint  limiting;   // 0 Rec.709, 1 P3-D65
    uint  pad0;
    uint  pad1;
};

ConstantBuffer<PrepareParams>   prepare;
RWStructuredBuffer<Aces2Params> params;   // one
RWStructuredBuffer<float>       tables;   // kTableWords

Chromaticities limitingPrimaries() { return prepare.limiting == 1 ? kP3D65 : kRec709; }

[shader("compute")]
[numthreads(1, 1, 1)]
void aces2Params(uint3 tid: SV_DispatchThreadID) {
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
        tables[kHuesOffset + i] = hues[i];
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
    params[0] = p;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void aces2Tables(uint3 tid: SV_DispatchThreadID) {
    const int i = int(tid.x);
    if (i >= kTableSize) return;
    const Aces2Params p = params[0];
    // The reach at the uniform hue i.
    tables[kReachOffset + kBaseIndex + i] = reachMForHue(float(i) * 360.0 / float(kTableSize), p.reach, p.limit_J_max);
    // The cusp at the hue table's entry, and the upper hull's gamma there.
    float3 limitRGB[kTotalCorners];
    float3 limitJMh[kTotalCorners];
    buildLimitingCuspCorners(limitRGB, limitJMh, p.limit, p.peakLuminance);
    const float hue = tables[kHuesOffset + kBaseIndex + i];
    const float2 JM = findDisplayCuspForHue(hue, limitRGB, limitJMh, p.limit);
    const float2 cusp = float2(JM.x, JM.y * (1.0 + 0.27 * 0.12));   // smooth_m * smooth_cusps
    tables[kCuspJOffset + kBaseIndex + i] = cusp.x;
    tables[kCuspMOffset + kBaseIndex + i] = cusp.y;
    tables[kGammaOffset + kBaseIndex + i] = upperHullGammaForCusp(cusp, hue, p);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void aces2Wrap(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) return;
    const int last = kBaseIndex + kTableSize;   // 361
    tables[kReachOffset + 0] = tables[kReachOffset + kTableSize];
    tables[kReachOffset + last] = tables[kReachOffset + kBaseIndex];
    tables[kCuspJOffset + 0] = tables[kCuspJOffset + kTableSize];
    tables[kCuspMOffset + 0] = tables[kCuspMOffset + kTableSize];
    tables[kCuspJOffset + last] = tables[kCuspJOffset + kBaseIndex];
    tables[kCuspMOffset + last] = tables[kCuspMOffset + kBaseIndex];
    tables[kGammaOffset + 0] = tables[kGammaOffset + kTableSize];
    tables[kGammaOffset + last] = tables[kGammaOffset + kBaseIndex];
}

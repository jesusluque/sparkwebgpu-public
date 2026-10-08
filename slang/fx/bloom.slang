// Copyright (c) 2026 aopenfx contributors.
//
// Bloom and glow: light brighter than a threshold spreads into its
// surroundings, as it does through a real lens.
//
// A pyramid, not one wide blur. The picture's bright part is halved again and
// again (bloomDown), each level a 13-tap filter of the one above, and the
// levels are summed back up from the smallest (bloomUp), each upsample a
// 3x3 tent. The result is a sum of gaussians of growing width: a sharp halo
// near the source and a long soft tail, for a few taps a pixel whatever the
// reach. A single blur that wide would cost hundreds of taps per pixel and
// have no core.
//
// Scene-referred: run it on linear light before the display transform, where
// a lamp at 50 is fifty times brighter than white and blooms accordingly. On a
// display-referred picture everything above white is already 1 and every
// highlight blooms alike.
//
// THE FIRST LEVEL
//
// bloomDown's first pass also thresholds (a soft knee, so the bloom fades in
// rather than switching on) and weights its five groups of taps by
// 1 / (1 + luma) (Karis's average). Without that a single splat at 10 000
// flickers as a large square of light whenever it crosses a pixel.
//
// Nothing is imported and nothing is groupshared: aofx kernel rules.

/// One struct for both entries; each reads what it needs. Four-byte members
/// in declaration order, no padding.
struct BloomParams {
    /// The source picture (`src`).
    uint  srcWidth;
    uint  srcHeight;
    uint  srcStride;
    /// Destination pixel (0,0) of the full-size level is at this source
    /// pixel. Zero below the first level.
    int   srcOffsetX;
    int   srcOffsetY;
    /// The coarse level being upsampled (`aux`, bloomUp only).
    uint  auxWidth;
    uint  auxHeight;
    uint  auxStride;
    /// The destination.
    uint  dstWidth;
    uint  dstHeight;
    uint  dstStride;
    /// bloomDown: nonzero on the first level (threshold and Karis average).
    uint  prefilter;
    /// The light that starts to bloom, and the width of the soft knee below
    /// it, in the picture's own units (linear).
    float threshold;
    float knee;
    /// No sample above this enters the bloom (a guard against infinities).
    float clampMax;
    /// bloomUp: dst = src * srcScale + tent(aux) * auxScale * tint.
    float srcScale;
    float auxScale;
    float tintR;
    float tintG;
    float tintB;
};

StructuredBuffer<float4>    src;
StructuredBuffer<float4>    aux;
RWStructuredBuffer<float4>  dst;
ConstantBuffer<BloomParams> params;

/// A pixel of the source, clamped to its edge: the bloom of a lamp at the
/// border should not fade towards it.
float4 srcPixel(int x, int y) {
    const int cx = clamp(x, 0, int(params.srcWidth) - 1);
    const int cy = clamp(y, 0, int(params.srcHeight) - 1);
    return src[uint(cy) * params.srcStride + uint(cx)];
}

/// The source at a continuous position in its pixels (centres at +0.5).
float4 srcBilinear(float2 p) {
    const float2 q = p - 0.5;
    const float2 f = floor(q);
    const float2 t = q - f;
    const int x = int(f.x);
    const int y = int(f.y);
    return lerp(lerp(srcPixel(x, y), srcPixel(x + 1, y), t.x),
                lerp(srcPixel(x, y + 1), srcPixel(x + 1, y + 1), t.x), t.y);
}

float4 auxPixel(int x, int y) {
    const int cx = clamp(x, 0, int(params.auxWidth) - 1);
    const int cy = clamp(y, 0, int(params.auxHeight) - 1);
    return aux[uint(cy) * params.auxStride + uint(cx)];
}

float4 auxBilinear(float2 p) {
    const float2 q = p - 0.5;
    const float2 f = floor(q);
    const float2 t = q - f;
    const int x = int(f.x);
    const int y = int(f.y);
    return lerp(lerp(auxPixel(x, y), auxPixel(x + 1, y), t.x),
                lerp(auxPixel(x, y + 1), auxPixel(x + 1, y + 1), t.x), t.y);
}

float luma(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

/// The average of four taps, weighted by 1 / (1 + luma) on the first level.
float3 group4(float4 a, float4 b, float4 c, float4 d) {
    if (params.prefilter == 0u) {
        return (a.rgb + b.rgb + c.rgb + d.rgb) * 0.25;
    }
    const float wa = 1.0 / (1.0 + luma(a.rgb));
    const float wb = 1.0 / (1.0 + luma(b.rgb));
    const float wc = 1.0 / (1.0 + luma(c.rgb));
    const float wd = 1.0 / (1.0 + luma(d.rgb));
    return (a.rgb * wa + b.rgb * wb + c.rgb * wc + d.rgb * wd) /
           (wa + wb + wc + wd);
}

/// The soft-knee threshold: nothing below threshold - knee, a quadratic ramp
/// across the knee, and everything above the threshold less the threshold.
float3 thresholded(float3 c) {
    const float bright = max(c.r, max(c.g, c.b));
    const float knee = max(params.knee, 1e-5);
    float soft = clamp(bright - params.threshold + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee);
    const float keep = max(soft, bright - params.threshold) / max(bright, 1e-5);
    return c * keep;
}

/// Half the size: each destination pixel the 13-tap filter of the source
/// block under it (Jimenez, "Next generation post processing in Call of
/// Duty: Advanced Warfare"), five overlapping groups of four.
[shader("compute")]
[numthreads(16, 16, 1)]
void bloomDown(uint3 tid: SV_DispatchThreadID) {
    const uint2 xy = tid.xy;
    if (xy.x >= params.dstWidth || xy.y >= params.dstHeight) {
        return;
    }
    // The destination pixel's centre in source pixels: 2x, plus where the
    // full-size level starts in the source.
    const float2 c = float2(xy) * 2.0 + 1.0 +
                     float2(float(params.srcOffsetX), float(params.srcOffsetY));
    const float4 a = srcBilinear(c + float2(-2.0, -2.0));
    const float4 b = srcBilinear(c + float2(0.0, -2.0));
    const float4 d = srcBilinear(c + float2(2.0, -2.0));
    const float4 e = srcBilinear(c + float2(-1.0, -1.0));
    const float4 f = srcBilinear(c + float2(1.0, -1.0));
    const float4 g = srcBilinear(c + float2(-2.0, 0.0));
    const float4 h = srcBilinear(c);
    const float4 i = srcBilinear(c + float2(2.0, 0.0));
    const float4 j = srcBilinear(c + float2(-1.0, 1.0));
    const float4 k = srcBilinear(c + float2(1.0, 1.0));
    const float4 l = srcBilinear(c + float2(-2.0, 2.0));
    const float4 m = srcBilinear(c + float2(0.0, 2.0));
    const float4 n = srcBilinear(c + float2(2.0, 2.0));
    float3 sum = group4(e, f, j, k) * 0.5 +
                 (group4(a, b, g, h) + group4(b, d, h, i) +
                  group4(g, h, l, m) + group4(h, i, m, n)) * 0.125;
    sum = min(max(sum, float3(0.0, 0.0, 0.0)),
              float3(params.clampMax, params.clampMax, params.clampMax));
    if (params.prefilter != 0u) {
        sum = thresholded(sum);
    }
    dst[xy.y * params.dstStride + xy.x] = float4(sum, 0.0);
}

/// Twice the size: the level below (`src`, at the destination's size) plus
/// a 3x3 tent of the coarser one (`aux`). The last pass, onto the picture,
/// scales the bloom by its intensity and tint and keeps the picture's alpha.
[shader("compute")]
[numthreads(16, 16, 1)]
void bloomUp(uint3 tid: SV_DispatchThreadID) {
    const uint2 xy = tid.xy;
    if (xy.x >= params.dstWidth || xy.y >= params.dstHeight) {
        return;
    }
    // The destination pixel's centre in the coarse level's pixels.
    const float2 p = (float2(xy) + 0.5) *
                     float2(float(params.auxWidth) / float(params.dstWidth),
                            float(params.auxHeight) / float(params.dstHeight));
    float3 up = auxBilinear(p).rgb * 4.0;
    up = up + (auxBilinear(p + float2(-1.0, 0.0)).rgb +
               auxBilinear(p + float2(1.0, 0.0)).rgb +
               auxBilinear(p + float2(0.0, -1.0)).rgb +
               auxBilinear(p + float2(0.0, 1.0)).rgb) * 2.0;
    up = up + auxBilinear(p + float2(-1.0, -1.0)).rgb +
         auxBilinear(p + float2(1.0, -1.0)).rgb +
         auxBilinear(p + float2(-1.0, 1.0)).rgb +
         auxBilinear(p + float2(1.0, 1.0)).rgb;
    up = up * (1.0 / 16.0);

    const int sx = int(xy.x) + params.srcOffsetX;
    const int sy = int(xy.y) + params.srcOffsetY;
    const float4 base = srcPixel(sx, sy);
    const float3 tint = float3(params.tintR, params.tintG, params.tintB);
    dst[xy.y * params.dstStride + xy.x] =
        float4(base.rgb * params.srcScale + up * params.auxScale * tint, base.a);
}

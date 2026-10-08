// The 32-byte extended splat (ExtSplats): two uint4 per splat.
//   a = center.xyz as f32 | alpha half (and blue half in the cov form)
//   b = rg half | b half, ln scale.x half | ln scale.yz half | quat 10-10-12
// Ported from src/shaders/splatDefines.glsl.

module ext;

import core.math;
import core.quat_codec;
import core.splat_types;

// A surfel's flat axis (a scale of exactly 0, 2DGS) as a finite f16 ln:
// log(0) = -inf, and packHalf2x16 of a value outside f16's finite range is
// indeterminate in WGSL. exp(-65504) is exactly 0 in f32 (as the CPU's
// athc::SURFEL_LN), so unpackSplatExt gives the 0 back.
public static const float kSurfelLnScale = -65504.0;

public float3 lnScales(float3 scales) {
    return select(scales > 0.0, log(max(scales, float3(1.0e-30))), float3(kSurfelLnScale));
}

public void packSplatExt(out uint4 a, out uint4 b, float3 center, float3 scales, float4 quaternion, float4 rgba) {
    a = uint4(asuint(center), packHalf2x16(float2(rgba.a, 0.0)));
    b.x = packHalf2x16(rgba.rg);
    const float3 ln = lnScales(scales);
    b.y = packHalf2x16(float2(rgba.b, ln.x));
    b.z = packHalf2x16(ln.yz);
    b.w = encodeQuatOctXy1010R12(quaternion);
}

public void unpackSplatExt(uint4 a, uint4 b, out float3 center, out float3 scales, out float4 quaternion, out float4 rgba) {
    center = asfloat(a.xyz);
    float2 bs = unpackHalf2x16(b.y);
    rgba = float4(unpackHalf2x16(b.x), bs.x, unpackHalf2x16(a.w).x);
    scales = float3(exp(bs.y), exp(unpackHalf2x16(b.z)));
    quaternion = decodeQuatOctXy1010R12(b.w);
}

public float4 unpackSplatExtCenterAlpha(uint4 a) {
    return float4(asfloat(a.xyz), unpackHalf2x16(a.w).x);
}

public Gsplat readExtSplat(uint4 a, uint4 b, int index) {
    Gsplat s;
    unpackSplatExt(a, b, s.center, s.scales, s.quaternion, s.rgba);
    s.index = index;
    s.flags = all(s.scales == float3(0.0)) ? 0u : GSPLAT_FLAG_ACTIVE;
    return s;
}

public void packSplatExtCov(out uint4 a, out uint4 b, float3 center, float4 rgba, float3 xxyyzz, float3 xyxzyz) {
    a = uint4(asuint(center), packHalf2x16(float2(rgba.a, rgba.b)));
    b.x = packHalf2x16(rgba.rg);
    float3 cor = float3(
        clamp(xyxzyz.x / sqrt(xxyyzz.x * xxyyzz.y), -1.0, 1.0),
        clamp(xyxzyz.y / sqrt(xxyyzz.x * xxyyzz.z), -1.0, 1.0),
        clamp(xyxzyz.z / sqrt(xxyyzz.y * xxyyzz.z), -1.0, 1.0));
    cor = sign(cor) * clamp(log(abs(cor)), -100.0, -0.0000001);
    float3 lnDiag = log(xxyyzz);
    b.y = packHalf2x16(lnDiag.xy);
    b.z = packHalf2x16(float2(lnDiag.z, cor.x));
    b.w = packHalf2x16(cor.yz);
}

public void unpackSplatExtCov(uint4 a, uint4 b, out float3 center, out float4 rgba, out float3 xxyyzz, out float3 xyxzyz) {
    center = asfloat(a.xyz);
    float2 ab = unpackHalf2x16(a.w);
    rgba = float4(unpackHalf2x16(b.x), ab.y, ab.x);
    float2 xxyy = unpackHalf2x16(b.y);
    float2 zzxy = unpackHalf2x16(b.z);
    float2 xzyz = unpackHalf2x16(b.w);
    xxyyzz = exp(float3(xxyy, zzxy.x));
    xyxzyz = float3(zzxy.y, xzyz);
    xyxzyz = -sign(xyxzyz) * exp(-abs(xyxzyz));
    xyxzyz *= float3(
        sqrt(xxyyzz.x * xxyyzz.y),
        sqrt(xxyyzz.x * xxyyzz.z),
        sqrt(xxyyzz.y * xxyyzz.z));
}

// RGB with a shared exponent and per-channel signs (SH-ready colours).
public uint encodeExtRgb(float3 rgb) {
    float3 absRgb = abs(rgb);
    float maxAbs = max(absRgb.r, max(absRgb.g, absRgb.b));
    int base = clamp(int(ceil(log2(maxAbs))) + 15, 0, 31);
    float divisor = exp2(float(base - 15)) / 255.0;
    uint3 u = uint3(round(clamp(absRgb / divisor, 0.0, 255.0)));
    uint signs = (rgb.r < 0.0 ? 1u : 0u) | (rgb.g < 0.0 ? 2u : 0u) | (rgb.b < 0.0 ? 4u : 0u);
    return u.r | (u.g << 8u) | (u.b << 16u) | (((uint(base) << 3u) | signs) << 24u);
}

public float3 decodeExtRgb(uint encoded) {
    float divisor = exp2(float(int((encoded >> 27u) & 0x1fu) - 15)) / 255.0;
    float3 rgb = float3(encoded & 0xffu, (encoded >> 8u) & 0xffu, (encoded >> 16u) & 0xffu) * divisor;
    return float3(
        ((encoded & 0x1000000u) != 0u) ? -rgb.r : rgb.r,
        ((encoded & 0x2000000u) != 0u) ? -rgb.g : rgb.g,
        ((encoded & 0x4000000u) != 0u) ? -rgb.b : rgb.b);
}

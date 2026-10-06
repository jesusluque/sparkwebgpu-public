// Copyright (c) 2026 jesus luque.
//
// The microfacet and Fresnel terms of MaterialX's pbrlib, as its GLSL writes
// them (libraries/pbrlib/genglsl/lib/mx_microfacet*.glsl, 1.39.5), so a
// material here reflects what it does there; with the sampling and densities a
// path tracer needs beside them. Directions are local: z is the normal.
module microfacet;
import sampling;

public static const float kEpsilon = 1e-8;

public float square(float x) { return x * x; }
public float pow5(float x) { return square(square(x)) * x; }
public float pow6(float x) { const float x2 = x * x; return x2 * x2 * x2; }

// ---- GGX ------------------------------------------------------------------

public float ggxD(float3 h, float2 alpha) {
    const float2 he = h.xy / alpha;
    const float denominator = dot(he, he) + h.z * h.z;
    return 1.0 / (kPi * alpha.x * alpha.y * denominator * denominator);
}

/// Smith's Lambda for the anisotropic GGX.
public float ggxLambda(float3 w, float2 alpha) {
    const float z2 = w.z * w.z;
    if (z2 <= 0.0) {
        return 0.0;
    }
    const float a2 = (square(alpha.x * w.x) + square(alpha.y * w.y)) / z2;
    return 0.5 * (-1.0 + sqrt(1.0 + a2));
}

public float ggxG1(float3 w, float2 alpha) {
    return 1.0 / (1.0 + ggxLambda(w, alpha));
}

/// MaterialX's height-correlated G2, isotropic in the average alpha.
public float ggxG2(float NdotL, float NdotV, float alpha) {
    const float alpha2 = alpha * alpha;
    const float lambdaL = sqrt(alpha2 + (1.0 - alpha2) * NdotL * NdotL);
    const float lambdaV = sqrt(alpha2 + (1.0 - alpha2) * NdotV * NdotV);
    return 2.0 * NdotL * NdotV / (lambdaL * NdotV + lambdaV * NdotL);
}

/// Dupuy and Benyoub 2023: a visible normal from `wo` (wo.z > 0).
public float3 ggxSampleVisibleNormal(float3 wo, float2 alpha, float2 u) {
    const float3 v = normalize(float3(wo.xy * alpha, wo.z));
    const float phi = 2.0 * kPi * u.x;
    const float z = (1.0 - u.y) * (1.0 + v.z) - v.z;
    const float sinTheta = sqrt(saturate(1.0 - z * z));
    const float3 c = float3(sinTheta * cos(phi), sinTheta * sin(phi), z);
    const float3 h = c + v;
    return normalize(float3(h.xy * alpha, max(h.z, 0.0)));
}

/// The density of visible normals: G1(wo) max(0, wo.h) D(h) / wo.z.
public float ggxVisibleNormalPdf(float3 wo, float3 h, float2 alpha) {
    return ggxG1(wo, alpha) * max(0.0, dot(wo, h)) * ggxD(h, alpha) / wo.z;
}

public float averageAlpha(float2 alpha) { return sqrt(alpha.x * alpha.y); }

// ---- directional albedo fits (MaterialX's analytic ones) -------------------

public float3 ggxDirectionalAlbedo(float NdotV, float alpha, float3 F0, float3 F90) {
    const float x = NdotV;
    const float y = alpha;
    const float x2 = x * x;
    const float y2 = y * y;
    const float4 r = float4(0.1003, 0.9345, 1.0, 1.0) +
                     float4(-0.6303, -2.323, -1.765, 0.2281) * x +
                     float4(9.748, 2.229, 8.263, 15.94) * y +
                     float4(-2.038, -3.748, 11.53, -55.83) * x * y +
                     float4(29.34, 1.424, 28.96, 13.08) * x2 +
                     float4(-8.245, -0.7684, -7.507, 41.26) * y2 +
                     float4(-26.44, 1.436, -36.11, 54.9) * x2 * y +
                     float4(19.99, 0.2913, 15.86, 300.2) * x * y2 +
                     float4(-5.448, 0.6286, 33.37, -285.1) * x2 * y2;
    const float2 ab = saturate(r.xy / r.zw);
    return F0 * ab.x + F90 * ab.y;
}

/// Turquin 2019: single scattering scaled up to what multiple scattering would add.
public float3 ggxEnergyCompensation(float NdotV, float alpha, float3 Fss) {
    const float Ess = ggxDirectionalAlbedo(NdotV, alpha, float3(1.0), float3(1.0)).x;
    return 1.0 + Fss * (1.0 - Ess) / Ess;
}

// ---- Fresnel ---------------------------------------------------------------

public float iorToF0(float ior) { return square((ior - 1.0) / (ior + 1.0)); }

public float f0ToIor(float F0) {
    const float s = sqrt(clamp(F0, 0.01, 0.99));
    return (1.0 + s) / (1.0 - s);
}

public float fresnelDielectric(float cosTheta, float ior) {
    const float c = cosTheta;
    const float g2 = ior * ior + c * c - 1.0;
    if (g2 < 0.0) {
        return 1.0;   // total internal reflection
    }
    const float g = sqrt(g2);
    return 0.5 * square((g - c) / (g + c)) * (1.0 + square(((g + c) * c - 1.0) / ((g - c) * c + 1.0)));
}

/// A METAL'S INDEX, FROM THE COLOUR AN ARTIST PICKED.
///
/// MaterialX's `artistic_ior` (Gulbrandsen 2014): a reflectivity at normal
/// incidence and a colour at grazing become the complex index a conductor's
/// Fresnel wants. It is what `athenea_usd_preview_surface.mtlx` puts in front of
/// its `conductor_bsdf`, which is what a metallic USD material compiles to,
/// so anything else reflecting that same metal has to start here or the two
/// will not agree.
public void mxArtisticIor(float3 reflectivity, float3 edge, out float3 ior, out float3 extinction) {
    const float3 r = clamp(reflectivity, float3(0.0), float3(0.99));
    const float3 root = sqrt(r);
    const float3 low = (float3(1.0) - r) / (float3(1.0) + r);
    const float3 high = (float3(1.0) + root) / (float3(1.0) - root);
    ior = lerp(high, low, clamp(edge, float3(0.0), float3(1.0)));
    const float3 up = ior + float3(1.0);
    const float3 down = ior - float3(1.0);
    extinction = sqrt(max((up * up * r - down * down) / max(float3(1.0) - r, float3(1.0e-6)), float3(0.0)));
}

public float3 fresnelConductor(float cosTheta, float3 n, float3 k) {
    const float cosTheta2 = square(saturate(cosTheta));
    const float sinTheta2 = 1.0 - cosTheta2;
    const float3 n2 = n * n;
    const float3 k2 = k * k;
    const float3 t0 = n2 - k2 - sinTheta2;
    const float3 a2plusb2 = sqrt(t0 * t0 + 4.0 * n2 * k2);
    const float3 t1 = a2plusb2 + cosTheta2;
    const float3 a = sqrt(max(0.5 * (a2plusb2 + t0), 0.0));
    const float3 t2 = 2.0 * a * cosTheta;
    const float3 rs = (t1 - t2) / (t1 + t2);
    const float3 t3 = cosTheta2 * a2plusb2 + sinTheta2 * sinTheta2;
    const float3 t4 = t2 * sinTheta2;
    const float3 rp = rs * (t3 - t4) / (t3 + t4);
    return 0.5 * (rp + rs);
}

/// Hoffman's generalization of Schlick with an F82 tint.
public float3 fresnelHoffmanSchlick(float cosTheta, float3 F0, float3 F82, float3 F90, float exponent) {
    const float cosThetaMax = 1.0 / 7.0;
    const float factor = 1.0 / (cosThetaMax * pow6(1.0 - cosThetaMax));
    const float x = saturate(cosTheta);
    const float3 a = lerp(F0, F90, pow(1.0 - cosThetaMax, exponent)) * (1.0 - F82) * factor;
    return lerp(F0, F90, pow(1.0 - x, exponent)) - a * x * pow6(1.0 - x);
}

public float fresnelSchlick(float cosTheta, float F0, float F90) {
    return lerp(F0, F90, pow5(saturate(1.0 - cosTheta)));
}

// ---- diffuse -----------------------------------------------------------------

public static const float kFujii1 = 0.5 - 2.0 / (3.0 * kPi);
public static const float kFujii2 = 2.0 / 3.0 - 28.0 / (15.0 * kPi);

public float orenNayar(float NdotV, float NdotL, float LdotV, float roughness) {
    const float s = LdotV - NdotL * NdotV;
    const float stinv = s > 0.0 ? s / max(NdotL, NdotV) : 0.0;
    const float sigma2 = roughness * roughness;
    const float A = 1.0 - 0.5 * (sigma2 / (sigma2 + 0.33));
    const float B = 0.45 * sigma2 / (sigma2 + 0.09);
    return A + B * stinv;
}

public float orenNayarDirectionalAlbedo(float NdotV, float roughness) {
    const float2 r = float2(1.0, 1.0) + float2(-0.4297, -0.6076) * roughness +
                     float2(-0.7632, -0.4993) * NdotV * roughness + float2(1.4385, 2.0315) * roughness * roughness;
    return saturate(r.x / r.y);
}

public float fujiiDirectionalAlbedo(float cosTheta, float roughness) {
    const float A = 1.0 / (1.0 + kFujii1 * roughness);
    const float B = roughness * A;
    const float Si = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    const float G = Si * (acos(clamp(cosTheta, -1.0, 1.0)) - Si * cosTheta) +
                    2.0 * ((Si / cosTheta) * (1.0 - Si * Si * Si) - Si) / 3.0;
    return A + B * G * kInvPi;
}

public float fujiiAverageAlbedo(float roughness) {
    const float A = 1.0 / (1.0 + kFujii1 * roughness);
    return A * (1.0 + kFujii2 * roughness);
}

/// OpenPBR's energy-compensated Oren-Nayar (EON), without the 1/pi and cosine.
public float3 orenNayarCompensated(float NdotV, float NdotL, float LdotV, float roughness, float3 colour) {
    const float s = LdotV - NdotL * NdotV;
    const float stinv = s > 0.0 ? s / max(NdotL, NdotV) : s;
    const float A = 1.0 / (1.0 + kFujii1 * roughness);
    const float3 single = colour * A * (1.0 + roughness * stinv);
    const float albedoV = fujiiDirectionalAlbedo(NdotV, roughness);
    const float albedoL = fujiiDirectionalAlbedo(NdotL, roughness);
    const float average = fujiiAverageAlbedo(roughness);
    const float3 multiColour = colour * colour * average / (1.0 - colour * max(0.0, 1.0 - average));
    const float3 multi = multiColour * max(kEpsilon, 1.0 - albedoV) * max(kEpsilon, 1.0 - albedoL) /
                         max(kEpsilon, 1.0 - average);
    return single + multi;
}

public float3 orenNayarCompensatedDirectionalAlbedo(float cosTheta, float roughness, float3 colour) {
    const float albedo = fujiiDirectionalAlbedo(cosTheta, roughness);
    const float average = fujiiAverageAlbedo(roughness);
    const float3 multiColour = colour * colour * average / (1.0 - colour * max(0.0, 1.0 - average));
    return lerp(multiColour, colour, albedo);
}

public float burley(float NdotV, float NdotL, float LdotH, float roughness) {
    const float F90 = 0.5 + 2.0 * roughness * LdotH * LdotH;
    return fresnelSchlick(NdotL, 1.0, F90) * fresnelSchlick(NdotV, 1.0, F90);
}

public float burleyDirectionalAlbedo(float NdotV, float roughness) {
    const float x = NdotV;
    const float fit0 = 0.97619 - 0.488095 * pow5(1.0 - x);
    const float fit1 = 1.55754 + (-2.02221 + (2.56283 - 1.06244 * x) * x) * x;
    return lerp(fit0, fit1, roughness);
}

// ---- sheen -----------------------------------------------------------------

public float imageworksSheenD(float NdotH, float roughness) {
    const float inverse = 1.0 / max(roughness, 0.005);
    const float sin2 = 1.0 - NdotH * NdotH;
    return (2.0 + inverse) * pow(max(sin2, 0.0), inverse * 0.5) / (2.0 * kPi);
}

public float imageworksSheen(float NdotL, float NdotV, float NdotH, float roughness) {
    return imageworksSheenD(NdotH, roughness) / (4.0 * (NdotL + NdotV - NdotL * NdotV));
}

public float imageworksSheenDirectionalAlbedo(float NdotV, float roughness) {
    const float2 r = float2(13.67300, 1.0) + float2(-68.78018, 61.57746) * NdotV +
                     float2(799.08825, 442.78211) * roughness + float2(-905.00061, 2597.49308) * NdotV * roughness +
                     float2(60.28956, 121.81241) * NdotV * NdotV + float2(1086.96473, 3045.55075) * roughness * roughness;
    return saturate(r.x / r.y);
}

public float zeltnerSheenDirectionalAlbedo(float x, float y) {
    const float s = y * (0.0206607 + 1.58491 * y) / (0.0379424 + y * (1.32227 + y));
    const float m = y * (-0.193854 + y * (-1.14885 + y * (1.7932 - 0.95943 * y * y))) / (0.046391 + y);
    const float o = y * (0.000654023 + (-0.0207818 + 0.119681 * y) * y) / (1.26264 + y * (-1.92021 + y));
    return exp(-0.5 * square((x - m) / s)) / (s * sqrt(2.0 * kPi)) + o;
}

public float zeltnerLtcAInv(float x, float y) {
    return (2.58126 * x + 0.813703 * y) * y / (1.0 + 0.310327 * x * x + 2.60994 * x * y);
}

public float zeltnerLtcBInv(float x, float y) {
    return sqrt(1.0 - x) * (y - 1.0) * y * y * y / (0.0000254053 + 1.71228 * x - 1.71506 * x * y + 1.34174 * y * y);
}

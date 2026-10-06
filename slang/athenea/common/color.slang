// Copyright (c) 2026 jesus luque.
module color;

/// The sRGB curve, extended past 1: a trained cloud's colours are not bounded
/// to the display, and clamping here clips every highlight the scene has
/// (a sky at 1.6 came out 0.93). Negative input is taken as zero.
public float srgbToLinear(float c) {
    c = max(c, 0.0);
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

public float3 srgbToLinear(float3 c) {
    return float3(srgbToLinear(c.x), srgbToLinear(c.y), srgbToLinear(c.z));
}

/// The way back, extended past 1 for the same reason and exactly inverse to
/// it. What it is for: an image shown on an sRGB display, a code value a
/// test compares in, a file written for a program that expects encoded
/// colours. Not a splat: every splat is blended in linear light.
public float linearToSrgb(float c) {
    c = max(c, 0.0);
    return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

public float3 linearToSrgb(float3 c) {
    return float3(linearToSrgb(c.x), linearToSrgb(c.y), linearToSrgb(c.z));
}

/// WHAT A SPLAT'S EVALUATED HARMONICS ARE IN LINEAR LIGHT.
///
/// Every gaussian is blended in the working space, linear Rec.709, and sRGB
/// appears only where an image is shown. A cloud says which space its
/// colours were written in: a capture (PLY, SPZ, SOG, .splat, a
/// ParticleField that does not say) was trained on sRGB photographs, so its
/// colours are sRGB and are decoded here, one splat at a time; a cloud this
/// engine wrote (`athenea mesh2splat`, an export, a bake) holds light and is
/// taken as it is (`GpuSplats::linear`, `primvars:athenea:splat:linear`).
/// Negative values -- harmonics ringing below zero -- are no light either way.
public float3 cloudLight(float3 c, bool linearCloud) {
    return linearCloud ? max(c, float3(0.0)) : srgbToLinear(c);
}

public float hash01(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return float((word >> 22u) ^ word) / 4294967295.0;
}

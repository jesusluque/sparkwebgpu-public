// Copyright (c) 2026 jesus luque.
//
// A DOME, PREPARED ONCE SO THAT A KERNEL WITHOUT A TEXTURE TABLE CAN READ IT.
//
// A dome's image lives in the material texture table, and the splat
// projection sits below material in the module order: it cannot sample it.
// Until this, that meant a cloud under an HDRI was lit by the dome's constant
// colour -- the sky's mean, with none of its direction -- while the mesh
// beside it reflected the sky itself.
//
// So the image is prepared into two things a buffer can hold, by kernels that
// do have the table (athenea/technique/env_project, athenea/technique/env_prefilter):
//
//   the harmonics   nine coefficients of the sky's radiance, degree 2, which
//                   is what a diffuse surface needs: the irradiance of any
//                   environment is captured by nine to about a percent
//                   (Ramamoorthi and Hanrahan 2001).
//   the levels      the sky convolved with a GGX lobe, one level a roughness,
//                   in an octahedral map -- what a reflection of a given
//                   sharpness sees, which is the half a colour cannot hold.
//
// Both are read here, by whoever has the buffers. The writer's side of the
// layout is the only thing that must agree with this file.
module environment;
__exported import athenea.common.octahedral;

/// THE MAP IS A MIP CHAIN OF THE SKY, AND ROUGHNESS WALKS IT.
///
/// Level 0 is the sky resampled, one tap a texel, and each level halves its
/// side and takes the next roughness: eight of them, `level / 7`, so level 1
/// is 0.143 rather than the 0.2 that six gave. Both halves of that matter.
///
/// The side, because no prefiltered map can be sharper than its base: a fixed
/// 256 put a quarter of a degree of blur on every mirror, every smooth
/// reflection and the far face of every glass ball, and a refraction came
/// back soft beside the mesh's. It now follows the dome's own image --
/// `Environment::baseSideFor`, so a texel of the map stands for what a texel
/// of the lat-long stands for, 2048 for a 4k sky -- and reaches the shader as
/// a uniform, since it is no longer the same for every stage.
///
/// The count, because a lookup blends the two levels that bracket it: glass
/// at roughness 0.02 used to be a tenth of a level convolved to 0.2, and that
/// tenth was enough haze to lose the workshop through the ball. At eight
/// levels it is a seventh of a level convolved to 0.143, read at half the
/// base rather than at 256.
///
/// A level never goes below `kEnvFloorSide`: the roughest is the whole sky
/// averaged, and 16 a side is already the convolution's own answer rather
/// than the coarseness's.
public static const uint  kEnvLevels = 8;
/// The floor under level 0, and what a dome that is only a colour takes.
public static const uint  kEnvCoarseSide = 256;
/// The floor under every level.
public static const uint  kEnvFloorSide = 16;
/// Coefficients kept: degree 3, which a TX transfer's sixteen are dotted
/// with (docs/decisions.md, task TX, step 3). The irradiance and the first
/// transfer read the first nine (`kEnvIrradianceCoefficients`), whose values
/// are what they were: each coefficient is its own sum.
public static const uint  kEnvCoefficients = 16;
/// Degree 2: what a diffuse surface's irradiance needs, the clamped cosine's
/// band 3 being zero.
public static const uint  kEnvIrradianceCoefficients = 9;
/// No environment prepared for this light.
public static const uint  kEnvNone = 0xFFFFFFFFu;

public uint envSideOf(uint level, uint baseSide) {
    const uint base = max(baseSide, kEnvCoarseSide);
    return max(base >> min(level, kEnvLevels - 1), kEnvFloorSide);
}

/// Where a level's texels start, counted in texels from the dome's own start.
public uint envLevelOffset(uint level, uint baseSide) {
    uint at = 0;
    for (uint k = 0; k < min(level, kEnvLevels); ++k) {
        const uint side = envSideOf(k, baseSide);
        at += side * side;
    }
    return at;
}

/// Texels one dome takes: every level of its octahedral map.
public uint envTexelsPerDome(uint baseSide) {
    return envLevelOffset(kEnvLevels, baseSide);
}

/// Two words a texel: rgb as halves, the fourth half unused. A word pair
/// rather than three halves so a texel is addressed by a shift.
public static const uint kEnvWordsPerTexel = 2;

public uint envWordAt(uint dome, uint level, uint2 uv, uint baseSide) {
    const uint side = envSideOf(level, baseSide);
    const uint texel = envLevelOffset(level, baseSide) + uv.y * side + uv.x;
    return (dome * envTexelsPerDome(baseSide) + texel) * kEnvWordsPerTexel;
}

/// One texel of a prepared dome.
public float3 envTexelAt(StructuredBuffer<uint> texels, uint dome, uint level, uint2 uv, uint baseSide) {
    const uint at = envWordAt(dome, level, uv, baseSide);
    const uint lo = texels[at];
    const uint hi = texels[at + 1];
    return float3(f16tof32(lo & 0xFFFFu), f16tof32(lo >> 16), f16tof32(hi & 0xFFFFu));
}

/// Bilinear over one level of the octahedral square. The seam runs along the
/// square's edges, so a lookup is clamped rather than wrapped: at the seam the
/// two sides hold the same direction and the clamp reads the nearer of them.
float3 envLevelAt(StructuredBuffer<uint> texels, uint dome, uint level, float3 dir, uint baseSide) {
    const float side = float(envSideOf(level, baseSide));
    const float2 at = clamp(octEncode(dir) * side - 0.5, 0.0, side - 1.0);
    const uint2 lo = uint2(floor(at));
    const uint2 hi = min(lo + 1, uint2(uint(side) - 1, uint(side) - 1));
    const float2 f = at - float2(lo);
    const float3 v00 = envTexelAt(texels, dome, level, lo, baseSide);
    const float3 v10 = envTexelAt(texels, dome, level, uint2(hi.x, lo.y), baseSide);
    const float3 v01 = envTexelAt(texels, dome, level, uint2(lo.x, hi.y), baseSide);
    const float3 v11 = envTexelAt(texels, dome, level, hi, baseSide);
    return lerp(lerp(v00, v10, f.x), lerp(v01, v11, f.x), f.y);
}

/// What a lobe of this roughness sees along `dir`: the two levels that
/// bracket it, blended. Level 0 is the mirror and level kEnvLevels-1 is the
/// whole sky averaged, so a rough surface costs the same as a sharp one.
///
/// ROUGHNESS WALKS THE CHAIN AS ITS SQUARE ROOT, not straight, because a
/// level's side halves while the lobe it should hold widens far faster than
/// that. Straight, level k stood for k/7 of roughness: gold at 0.08 came out
/// as half a mirror blended with half of a lobe convolved to 0.143, which is
/// neither, and it read soft and dark beside the mesh. As a square root the
/// levels stand for (k/7)^2 -- 0, 0.020, 0.082, 0.184, 0.327, 0.510, 0.735, 1
/// -- so that same 0.08 lands on the level convolved to 0.082, and each
/// level's texels are about as wide as its own lobe. `Environment` writes the
/// same curve; the two must agree or a lookup reads a blur it did not ask for.
public float3 envRadiance(StructuredBuffer<uint> texels, uint dome, float3 dir, float roughness,
                          uint baseSide) {
    const float level = sqrt(saturate(roughness)) * float(kEnvLevels - 1);
    const uint lo = uint(floor(level));
    const uint hi = min(lo + 1, kEnvLevels - 1);
    const float f = level - float(lo);
    return lerp(envLevelAt(texels, dome, lo, dir, baseSide),
                envLevelAt(texels, dome, hi, dir, baseSide), f);
}

/// THE CONVOLUTION A DIFFUSE SURFACE IS, in the coefficients rather than in
/// the map: the clamped cosine lobe is a zonal harmonic whose own three
/// bands are pi, 2pi/3 and pi/4, so the irradiance arriving at a normal is
/// the sky's coefficients weighed by those and read at `n`.
///
/// The coefficients are stored as the sky's own radiance, not as irradiance
/// already convolved, because the transfer a gaussian carries has the cosine
/// in it already: one source, two consumers.
public static const float kEnvBandA0 = 3.14159265358979;
public static const float kEnvBandA1 = 2.0943951023932;    // 2 pi / 3
public static const float kEnvBandA2 = 0.78539816339745;   // pi / 4

public float envBandWeight(uint k) {
    return k == 0 ? kEnvBandA0 : (k < 4 ? kEnvBandA1 : kEnvBandA2);
}

/// Irradiance at `n`, from nine coefficients of radiance.
public float3 envIrradiance(StructuredBuffer<float4> sh, uint dome, float3 n) {
    const uint base = dome * kEnvCoefficients;
    float3 e = float3(0.0);
    [unroll] for (uint k = 0; k < kEnvIrradianceCoefficients; ++k) {
        e += sh[base + k].rgb * (envBandWeight(k) * envBasisValue(k, n));
    }
    return e;
}

/// The real spherical harmonics to degree 3, in the order this engine keeps
/// everywhere else (`athenea/common/sh.slang`'s `shBasisValue`, terms 0..15). It
/// is repeated here rather than imported so that this module stays free of
/// anything a splat kernel cannot afford; the test holds the two equal.
public float envBasisValue(uint k, float3 d) {
    const float x = d.x, y = d.y, z = d.z;
    switch (k) {
    case 0: return 0.28209479177387814;
    case 1: return -0.4886025119029199 * y;
    case 2: return 0.4886025119029199 * z;
    case 3: return -0.4886025119029199 * x;
    case 4: return 1.0925484305920792 * x * y;
    case 5: return -1.0925484305920792 * y * z;
    case 6: return 0.31539156525252005 * (2.0 * z * z - x * x - y * y);
    case 7: return -1.0925484305920792 * x * z;
    case 8: return 0.5462742152960396 * (x * x - y * y);
    // Degree 3, as `shBasisValue` has it: what a TX transfer's direct half,
    // its indirect one at degree 3 and its reflected field are projected on.
    case 9: return -0.5900435899266435 * y * (3.0 * x * x - y * y);
    case 10: return 2.890611442640554 * x * y * z;
    case 11: return -0.4570457994644658 * y * (4.0 * z * z - x * x - y * y);
    case 12: return 0.3731763325901154 * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y);
    case 13: return -0.4570457994644658 * x * (4.0 * z * z - x * x - y * y);
    case 14: return 1.445305721320277 * z * (x * x - y * y);
    case 15: return -0.5900435899266435 * x * (x * x - 3.0 * y * y);
    default: return 0.0;
    }
}

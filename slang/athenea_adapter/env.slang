// A dome prepared as athenea prepares it (technique::Environment, txf
// 89a04d9), for the relight plugin (src/webgpu/athenea/AtheneaSky.ts): the
// buffers athenea/light/environment.slang reads, written by the same passes
// in the same order.
//
//   mips      the lat-long's area-weighted chain (athenea/algo/mips.slang,
//             latLong): each row weighed by the solid angle it stands for
//   sun       env_sun.slang's envSun: the disc found by the sky's median and
//             peak, summed over the projection's own texels -> envSun
//   project   env_project.slang's envProject: sixteen coefficients of the
//             sky's radiance, the sun's texels skipped -> envSh
//   residual  env_residual.slang's envResidual: the sky without its sun, at
//             the level the sun was measured on, with its own area mips
//   prefilter env_prefilter.slang's envPrefilter: the residual convolved
//             with GGX, eight levels of an octahedral map -> envTexels
//
// Those kernels read the dome through athenea's texture table (gTextures),
// which WGSL has no form of, so they are not compiled from the vendored
// files (athenea/technique is not vendored): the bodies below are theirs,
// line for line, reading `skySource` -- the dome's lat-long as an rgba32float
// texture with its mip chain -- where they read the table, and taking the
// dome's LightRecord from `envDome[0]` (one dome, slice 0). The one
// difference is the residual's filtered read: rgba32float is not filterable
// in WebGPU, so `residualAt` is the sampler's trilinear filter (wrap in u,
// clamp in v, linear between levels) written out over textureLoad.
module env;

import athenea.light.lights;
import athenea.light.environment;
import athenea.material.microfacet;

public static const uint kAtheneaEnvGroup = 64;
public static const uint kAtheneaMipGroup = 16;

public struct AtheneaEnvParams {
    public uint  domes;          ///< 1
    public uint  level;          ///< sun: the search level; prefilter: the roughness level written
    public uint  projectLevel;   ///< the level env_project sums over
    public uint  samples;        ///< prefilter: directions a texel takes
    public uint  sourceLevel;    ///< prefilter: the coarsest level a sample may read (projectLevel)
    public uint  baseSide;       ///< the octahedral side of level 0
    public float roughness;      ///< prefilter: of this level
    public uint  residualLevel;  ///< the source level the residual copies (the sun's level)
    public uint  sourceWidth;    ///< the lat-long at level 0
    public uint  sourceHeight;
    public uint  residualLevels; ///< mips of the residual
    public uint  width;          ///< residual / mips: the level written
    public uint  height;
    public uint  sourceMipWidth; ///< mips: the level read
    public uint  sourceMipHeight;
    public uint  rowThreads;     ///< prefilter: threads a grid row (a level past 65535 groups takes rows)
};

StructuredBuffer<LightRecord>          envDome;
Texture2D<float4>                      skySource;
Texture2D<float4>                      residual;
[format("rgba32f")] WTexture2D<float4> mipTarget;
[format("rgba32f")] WTexture2D<float4> residualOut;
RWStructuredBuffer<float4>             sunOut;
StructuredBuffer<float4>               sun;
RWStructuredBuffer<float4>             shOut;
RWStructuredBuffer<uint>               texelsOut;
ConstantBuffer<AtheneaEnvParams>       envParams;

static const float kEnvPi = 3.14159265358979;

// ---------------------------------------------------------------------------
// athenea/algo/mips.slang, mipDownsample with latLong and a float target.

struct Taps {
    uint   first;
    uint   count;
    float3 weights;
};

Taps tapsAlong(uint x, uint sourceSize, uint size) {
    Taps taps;
    if (sourceSize == size) {
        taps.first = x;
        taps.count = 1;
        taps.weights = float3(1.0, 0.0, 0.0);
    } else if (sourceSize == 2 * size) {
        taps.first = 2 * x;
        taps.count = 2;
        taps.weights = float3(0.5, 0.5, 0.0);
    } else {
        const float n = float(size);
        const float inverse = 1.0 / (2.0 * n + 1.0);
        taps.first = 2 * x;
        taps.count = 3;
        taps.weights = float3(n - float(x), n, float(x) + 1.0) * inverse;
    }
    return taps;
}

/// One level of a lat-long chain from the level above it (`skySource`, a
/// one-level view), into `mipTarget`.
[shader("compute")]
[numthreads(16, 16, 1)]
void atheneaEnvMip(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= envParams.width || tid.y >= envParams.height) {
        return;
    }
    const Taps tx = tapsAlong(tid.x, envParams.sourceMipWidth, envParams.width);
    const Taps ty = tapsAlong(tid.y, envParams.sourceMipHeight, envParams.height);
    float4 sum = float4(0.0);
    float rowWeights[3] = { ty.weights[0], ty.weights[1], ty.weights[2] };
    float total = 0.0;
    for (uint j = 0; j < ty.count; ++j) {
        const float step = 3.14159265358979 / float(envParams.sourceMipHeight);
        const float row = float(ty.first + j);
        rowWeights[j] *= cos(row * step) - cos((row + 1.0) * step);
        total += rowWeights[j];
    }
    for (uint j = 0; j < ty.count; ++j) {
        rowWeights[j] = total > 0.0 ? rowWeights[j] / total : ty.weights[j];
    }
    for (uint j = 0; j < ty.count; ++j) {
        for (uint i = 0; i < tx.count; ++i) {
            const float4 texel = skySource.Load(int3(int(tx.first + i), int(ty.first + j), 0));
            sum += tx.weights[i] * rowWeights[j] * texel;
        }
    }
    mipTarget.Store(tid.xy, sum);
}

// ---------------------------------------------------------------------------
// The dome's texels, as gTextures.textures[record.slot].Load reads them.

float3 sourceLoad(uint x, uint y, uint level) {
    return skySource.Load(int3(int(x), int(y), int(level))).rgb;
}

uint sourceWidthAt(uint level) {
    return max(envParams.sourceWidth >> level, 1u);
}

uint sourceHeightAt(uint level) {
    return max(envParams.sourceHeight >> level, 1u);
}

// ---------------------------------------------------------------------------
// env_sun.slang

public static const float3 kSunGrey = float3(0.2126, 0.7152, 0.0722);
public static const float kSunOverMedian = 16.0;
public static const float kSunOfPeak = 0.02;
public static const float kSunPeakOverMedian = 32.0;
public static const float kSunWidest = 0.175;

public struct SunRegion {
    public bool   found;
    public float3 axis;
    public float  threshold;
};

public SunRegion sunRegionOf(LightRecord light, uint level) {
    SunRegion region;
    region.found = false;
    region.axis = float3(0.0, 1.0, 0.0);
    region.threshold = 0.0;
    if (light.texture == 0xFFFFFFFFu) {
        return region;
    }
    const uint w = sourceWidthAt(level);
    const uint h = sourceHeightAt(level);
    float histogram[80];
    for (uint b = 0; b < 80; ++b) {
        histogram[b] = 0.0;
    }
    float total = 0.0;
    float peak = -1.0;
    uint2 peakAt = uint2(0);
    for (uint y = 0; y < h; ++y) {
        const float v = (float(y) + 0.5) / float(h);
        const float weight = sin((1.0 - v) * kEnvPi);
        if (weight <= 0.0) {
            continue;
        }
        for (uint x = 0; x < w; ++x) {
            const float lum = dot(sourceLoad(x, y, level), kSunGrey);
            const uint bin = uint(clamp((log2(max(lum, 1.0e-9)) + 20.0) * 2.0, 0.0, 79.0));
            histogram[bin] += weight;
            total += weight;
            if (lum > peak) {
                peak = lum;
                peakAt = uint2(x, y);
            }
        }
    }
    float below = 0.0;
    float median = 0.0;
    for (uint b = 0; b < 80; ++b) {
        below += histogram[b];
        if (below >= 0.5 * total) {
            median = exp2(float(b) * 0.5 - 20.0 + 0.25);
            break;
        }
    }
    region.axis = domeDirection(light, (float2(peakAt) + 0.5) / float2(float(w), float(h)));
    region.threshold = max(kSunOverMedian * median, kSunOfPeak * peak);
    region.found = peak > kSunPeakOverMedian * max(median, 1.0e-9);
    return region;
}

public bool sunHolds(SunRegion region, float3 dir, float3 rgb) {
    return region.found && dot(rgb, kSunGrey) >= region.threshold && dot(dir, region.axis) >= cos(kSunWidest);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void atheneaEnvSun(uint3 tid: SV_DispatchThreadID) {
    const uint dome = tid.x;
    if (dome >= envParams.domes) {
        return;
    }
    const LightRecord light = envDome[dome];
    if (light.texture == 0xFFFFFFFFu) {
        sunOut[dome * 2] = float4(0.0, 1.0, 0.0, 0.0);
        sunOut[dome * 2 + 1] = float4(0.0);
        return;
    }
    const float3 emission = lightEmission(light);
    const SunRegion region = sunRegionOf(light, envParams.projectLevel);
    float3 delivered = float3(0.0);
    float3 centre = float3(0.0);
    float solid = 0.0;
    float weight = 0.0;
    if (region.found) {
        const uint w = sourceWidthAt(envParams.projectLevel);
        const uint h = sourceHeightAt(envParams.projectLevel);
        const float area = 2.0 * kEnvPi * kEnvPi / float(w * h);
        for (uint y = 0; y < h; ++y) {
            const float v = (float(y) + 0.5) / float(h);
            const float sinTheta = sin((1.0 - v) * kEnvPi);
            if (sinTheta <= 0.0) {
                continue;
            }
            for (uint x = 0; x < w; ++x) {
                const float u = (float(x) + 0.5) / float(w);
                const float3 dir = domeDirection(light, float2(u, v));
                const float3 rgb = sourceLoad(x, y, envParams.projectLevel);
                if (!sunHolds(region, dir, rgb)) {
                    continue;
                }
                const float omega = sinTheta * area;
                delivered += rgb * omega;
                centre += dir * (dot(rgb, kSunGrey) * omega);
                weight += dot(rgb, kSunGrey) * omega;
                solid += omega;
            }
        }
    }
    const float discCone = solid > 0.0 ? acos(saturate(1.0 - solid / (2.0 * kEnvPi))) : 0.0;
    const float spread = weight > 0.0 ? 2.0 * max(1.0 - length(centre) / weight, 0.0) : 0.0;
    const bool compact = spread <= 2.0 * 0.5 * discCone * discCone + 1.0e-6;
    const bool isSun = region.found && solid > 0.0 && dot(centre, centre) > 0.0 && compact;
    const float3 direction = isSun ? normalize(centre) : region.axis;
    const float cone = isSun ? discCone : 0.0;
    sunOut[dome * 2] = float4(direction, isSun ? max(cone, 1.0e-4) : 0.0);
    sunOut[dome * 2 + 1] = float4(delivered * emission, region.threshold);
}

// ---------------------------------------------------------------------------
// env_project.slang

static const float kTwoPiSquared = 2.0 * 3.14159265358979 * 3.14159265358979;

[shader("compute")]
[numthreads(64, 1, 1)]
void atheneaEnvProject(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    const uint dome = at / kEnvCoefficients;
    const uint k = at % kEnvCoefficients;
    if (dome >= envParams.domes) {
        return;
    }
    const LightRecord light = envDome[dome];
    const float3 emission = lightEmission(light);
    float3 sum = float3(0.0);
    if (light.texture != 0xFFFFFFFFu) {
        const uint level = envParams.projectLevel;
        const uint w = sourceWidthAt(level);
        const uint h = sourceHeightAt(level);
        const float area = kTwoPiSquared / float(w * h);
        const bool cutSun = sun[dome * 2].w > 0.0;
        const SunRegion region = sunRegionOf(light, level);
        for (uint y = 0; y < h; ++y) {
            const float v = (float(y) + 0.5) / float(h);
            const float sinTheta = sin((1.0 - v) * 3.14159265358979);
            if (sinTheta <= 0.0) {
                continue;
            }
            for (uint x = 0; x < w; ++x) {
                const float u = (float(x) + 0.5) / float(w);
                const float3 dir = domeDirection(light, float2(u, v));
                const float3 rgb = sourceLoad(x, y, level);
                if (cutSun && sunHolds(region, dir, rgb)) {
                    continue;
                }
                sum += rgb * (envBasisValue(k, dir) * sinTheta * area);
            }
        }
        sum *= emission;
    } else if (k == 0) {
        sum = emission * (4.0 * 3.14159265358979 * envBasisValue(0, float3(0.0, 1.0, 0.0)));
    }
    shOut[dome * kEnvCoefficients + k] = float4(sum, 0.0);
}

// ---------------------------------------------------------------------------
// env_residual.slang

[shader("compute")]
[numthreads(16, 16, 1)]
void atheneaEnvResidual(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= envParams.width || tid.y >= envParams.height) {
        return;
    }
    const uint dome = 0;
    const LightRecord light = envDome[dome];
    if (light.texture == 0xFFFFFFFFu) {
        residualOut.Store(tid.xy, float4(1.0, 1.0, 1.0, 1.0));
        return;
    }
    const float3 rgb = sourceLoad(tid.x, tid.y, envParams.residualLevel);
    const float4 disc = sun[dome * 2];
    bool sunlit = false;
    if (disc.w > 0.0) {
        const float2 uv = (float2(tid.xy) + 0.5) / float2(float(envParams.width), float(envParams.height));
        const float3 dir = domeDirection(light, uv);
        sunlit = dot(rgb, kSunGrey) >= sun[dome * 2 + 1].w && dot(dir, disc.xyz) >= cos(kSunWidest);
    }
    residualOut.Store(tid.xy, float4(sunlit ? float3(0.0) : rgb, 1.0));
}

// ---------------------------------------------------------------------------
// env_prefilter.slang

/// The residual's filtered read at `uv` and fractional `level`: a sampler
/// with linear min/mag/mip filtering, u wrapped and v clamped
/// (Environment::create's residualSampler), over textureLoad.
float3 residualBilinear(float2 uv, uint level) {
    const int w = int(max((envParams.sourceWidth >> envParams.residualLevel) >> level, 1u));
    const int h = int(max((envParams.sourceHeight >> envParams.residualLevel) >> level, 1u));
    const float2 at = uv * float2(float(w), float(h)) - 0.5;
    const float2 lo = floor(at);
    const float2 f = at - lo;
    const int x0 = ((int(lo.x) % w) + w) % w;
    const int x1 = (x0 + 1) % w;
    const int y0 = clamp(int(lo.y), 0, h - 1);
    const int y1 = clamp(int(lo.y) + 1, 0, h - 1);
    const float3 v00 = residual.Load(int3(x0, y0, int(level))).rgb;
    const float3 v10 = residual.Load(int3(x1, y0, int(level))).rgb;
    const float3 v01 = residual.Load(int3(x0, y1, int(level))).rgb;
    const float3 v11 = residual.Load(int3(x1, y1, int(level))).rgb;
    return lerp(lerp(v00, v10, f.x), lerp(v01, v11, f.x), f.y);
}

float3 residualAt(float2 uv, float level) {
    const float last = float(envParams.residualLevels - 1u);
    const float l = clamp(level, 0.0, last);
    const uint lo = uint(floor(l));
    const uint hi = min(lo + 1u, envParams.residualLevels - 1u);
    const float f = l - float(lo);
    const float3 a = residualBilinear(uv, lo);
    return f > 0.0 ? lerp(a, residualBilinear(uv, hi), f) : a;
}

float3 skyResidualAt(LightRecord light, float3 dir, float level) {
    if (light.texture == 0xFFFFFFFFu) {
        return float3(1.0);
    }
    return residualAt(domeUv(light, dir), max(level - float(envParams.residualLevel), 0.0));
}

void frameAbout(float3 z, out float3 x, out float3 y) {
    const float sign = z.z >= 0.0 ? 1.0 : -1.0;
    const float a = -1.0 / (sign + z.z);
    const float b = z.x * z.y * a;
    x = float3(1.0 + sign * z.x * z.x * a, sign * b, -sign * z.x);
    y = float3(b, sign + z.y * z.y * a, -z.y);
}

float2 hammersley(uint i, uint n) {
    uint bits = i;
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555u) << 1) | ((bits & 0xAAAAAAAAu) >> 1);
    bits = ((bits & 0x33333333u) << 2) | ((bits & 0xCCCCCCCCu) >> 2);
    bits = ((bits & 0x0F0F0F0Fu) << 4) | ((bits & 0xF0F0F0F0u) >> 4);
    bits = ((bits & 0x00FF00FFu) << 8) | ((bits & 0xFF00FF00u) >> 8);
    return float2(float(i) / float(max(n, 1u)), float(bits) * 2.3283064365386963e-10);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void atheneaEnvPrefilter(uint3 tid: SV_DispatchThreadID) {
    const uint side = envSideOf(envParams.level, envParams.baseSide);
    const uint perDome = side * side;
    const uint at = tid.y * envParams.rowThreads + tid.x;
    const uint dome = at / perDome;
    const uint texel = at % perDome;
    if (tid.x >= envParams.rowThreads || dome >= envParams.domes) {
        return;
    }
    const LightRecord light = envDome[dome];
    const uint2 uv = uint2(texel % side, texel / side);
    const float3 normal = octDecode((float2(uv) + 0.5) / float(side));

    float sourceTexels = 1.0;
    float coarsest = float(envParams.sourceLevel);
    if (light.texture != 0xFFFFFFFFu) {
        sourceTexels = float(envParams.sourceWidth * envParams.sourceHeight);
        coarsest = floor(log2(float(max(envParams.sourceWidth, envParams.sourceHeight))));
    }
    const float texelSolidAngle = 4.0 * kEnvPi / max(sourceTexels, 1.0);

    float3 sum = float3(0.0);
    if (envParams.level == 0 || envParams.samples <= 1) {
        const float mapSolidAngle = 4.0 * kEnvPi / float(side * side);
        const float level = clamp(0.5 * log2(mapSolidAngle / texelSolidAngle), 0.0,
                                  float(envParams.sourceLevel));
        sum = skyResidualAt(light, normal, level);
    } else {
        const float alpha = max(envParams.roughness * envParams.roughness, 1.0e-3);
        const float2 alpha2 = float2(alpha, alpha);
        float3 tangent, bitangent;
        frameAbout(normal, tangent, bitangent);
        float weight = 0.0;
        const uint seed = (texel * 747796405u + dome * 2891336453u + envParams.level * 277803737u) ^ 0x9E3779B9u;
        const uint mixed = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
        const float2 texelShift = float2(float(mixed & 0xFFFFu), float(mixed >> 16)) * (1.0 / 65536.0);
        for (uint s = 0; s < envParams.samples; ++s) {
            const float2 u = frac(hammersley(s, envParams.samples) + texelShift);
            const float3 h = ggxSampleVisibleNormal(float3(0.0, 0.0, 1.0), alpha2, u);
            const float3 hWorld = normalize(tangent * h.x + bitangent * h.y + normal * h.z);
            const float3 wi = reflect(-normal, hWorld);
            const float nDotL = dot(normal, wi);
            if (nDotL <= 0.0) {
                continue;
            }
            const float pdf = max(ggxVisibleNormalPdf(float3(0.0, 0.0, 1.0), h, alpha2) /
                                      (4.0 * max(h.z, 1.0e-6)),
                                  1.0e-9);
            const float sampleSolidAngle = 1.0 / (float(envParams.samples) * pdf);
            const float level = clamp(0.5 * log2(sampleSolidAngle / texelSolidAngle) + 1.0, 0.0, coarsest);
            sum += skyResidualAt(light, wi, level) * nDotL;
            weight += nDotL;
        }
        sum = weight > 0.0 ? sum / weight
                           : skyResidualAt(light, normal, float(envParams.sourceLevel));
    }
    sum *= lightEmission(light);

    const uint word = envWordAt(dome, envParams.level, uv, envParams.baseSide);
    texelsOut[word] = (f32tof16(sum.g) << 16) | f32tof16(sum.r);
    texelsOut[word + 1] = f32tof16(sum.b);
}

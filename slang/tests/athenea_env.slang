// athenea's environment checks (shaders/athenea/test/environment_check.slang
// at txf 89a04d9), over a dome prepared by slang/athenea_adapter/env.slang
// (test/gpu/atheneaSky.test.ts). The same kernels with the counters written
// by their single thread instead of InterlockedAdd, which a plain uint
// buffer has no form of in WGSL; they read the map through athenea's own
// envIrradiance, envRadiance and envTexelAt.
import athenea.light.environment;
import athenea.common.octahedral;

struct EnvCheckParams {
    uint  dome;
    uint  directions;
    float expected;
    float tolerance;

    float upExpected;
    float downExpected;
    float roughExpected;
    uint  baseSide;

    uint  level;          ///< peaks, energy
    uint  count;          ///< read energy: directions
    float readRoughness;  ///< read energy
    uint  pad0;
};

StructuredBuffer<uint>         envTexels;
StructuredBuffer<float4>       envSh;
RWStructuredBuffer<uint>       stats;
RWStructuredBuffer<float>      worst;
ConstantBuffer<EnvCheckParams> params;

float3 spiralDirection(uint i, uint n) {
    const float golden = 2.39996322972865332;
    const float z = 1.0 - 2.0 * (float(i) + 0.5) / float(max(n, 1u));
    const float r = sqrt(max(1.0 - z * z, 0.0));
    const float phi = golden * float(i);
    return float3(r * cos(phi), r * sin(phi), z);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void environmentCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint beyond = 0;
    uint checked = 0;
    float worstRelative = 0.0;
    if (params.expected > 0.0) {
        for (uint i = 0; i < params.directions; ++i) {
            const float3 n = spiralDirection(i, params.directions);
            const float3 e = envIrradiance(envSh, params.dome, n);
            if (i == 0) {
                worst[1] = e.r;
                worst[2] = e.g;
                worst[3] = e.b;
            }
            const float mean = (e.r + e.g + e.b) / 3.0;
            const float relative = abs(mean - params.expected) / max(params.expected, 1.0e-6);
            worstRelative = max(worstRelative, relative);
            if (relative > params.tolerance) {
                beyond += 1u;
            }
            checked += 1u;
        }
    }
    stats[0] = beyond;
    stats[1] = checked;
    worst[0] = worstRelative;

    const float3 up = envRadiance(envTexels, params.dome, float3(0.0, 1.0, 0.0), 0.0, params.baseSide);
    const float3 down = envRadiance(envTexels, params.dome, float3(0.0, -1.0, 0.0), 0.0, params.baseSide);
    const float3 rough = envRadiance(envTexels, params.dome, float3(0.0, 1.0, 0.0), 1.0, params.baseSide);
    worst[4] = (up.r + up.g + up.b) / 3.0;
    worst[5] = (down.r + down.g + down.b) / 3.0;
    worst[6] = (rough.r + rough.g + rough.b) / 3.0;
    stats[2] = params.upExpected >= 0.0 && abs(worst[4] - params.upExpected) > params.tolerance ? 1u : 0u;
    stats[3] = params.downExpected >= 0.0 && abs(worst[5] - params.downExpected) > params.tolerance ? 1u : 0u;
    stats[4] = params.roughExpected >= 0.0 && abs(worst[6] - params.roughExpected) > params.tolerance ? 1u : 0u;
}

float peakLum(uint2 uv) {
    return dot(envTexelAt(envTexels, params.dome, params.level, uv, params.baseSide), float3(0.2126, 0.7152, 0.0722));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void environmentPeaks(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    const uint side = envSideOf(params.level, params.baseSide);
    float brightest = 0.0;
    float darkest = 3.0e38;
    for (uint y = 0; y < side; ++y) {
        for (uint x = 0; x < side; ++x) {
            brightest = max(brightest, peakLum(uint2(x, y)));
            darkest = min(darkest, peakLum(uint2(x, y)));
        }
    }
    const float above = darkest + 0.25 * (brightest - darkest);
    uint found = 0;
    for (uint y = 1; y + 1 < side; ++y) {
        for (uint x = 1; x + 1 < side; ++x) {
            const float v = peakLum(uint2(x, y));
            if (v < above) {
                continue;
            }
            bool top = true;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if ((dx != 0 || dy != 0) && peakLum(uint2(uint(int(x) + dx), uint(int(y) + dy))) >= v) {
                        top = false;
                    }
                }
            }
            found += top ? 1u : 0u;
        }
    }
    stats[0] = found;
    stats[1] = side * side;
    worst[0] = brightest;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void environmentEnergy(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    const uint side = envSideOf(params.level, params.baseSide);
    float sum = 0.0;
    float carry = 0.0;
    for (uint y = 0; y < side; ++y) {
        for (uint x = 0; x < side; ++x) {
            const float2 uv = (float2(float(x), float(y)) + 0.5) / float(side);
            const float h = 0.25 / float(side);
            const float3 du = octDecode(uv + float2(h, 0.0)) - octDecode(uv - float2(h, 0.0));
            const float3 dv = octDecode(uv + float2(0.0, h)) - octDecode(uv - float2(0.0, h));
            const float omega = length(cross(du, dv)) / (4.0 * h * h) / float(side * side);
            const float term = peakLum(uint2(x, y)) * omega;
            const float t = sum + term;
            carry += abs(sum) >= abs(term) ? (sum - t) + term : (term - t) + sum;
            sum = t;
        }
    }
    worst[0] = sum + carry;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void environmentReadEnergy(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    float sum = 0.0;
    float carry = 0.0;
    for (uint i = 0; i < params.count; ++i) {
        const float3 d = spiralDirection(i, params.count);
        const float term =
            dot(envRadiance(envTexels, params.dome, d, params.readRoughness, params.baseSide),
                float3(0.2126, 0.7152, 0.0722));
        const float t = sum + term;
        carry += abs(sum) >= abs(term) ? (sum - t) + term : (term - t) + sum;
        sum = t;
    }
    worst[0] = (sum + carry) * (4.0 * 3.14159265358979 / float(params.count));
}

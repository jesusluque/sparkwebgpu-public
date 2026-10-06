// Copyright (c) 2026 jesus luque.
//
// What `athenea flatten` solves once, before any gaussian is fitted
// (flatten_basis.slang says what the fit is):
//
// - `flattenGram`: for each class -- a surface, seen from in front and
//   weighted `backWeight` behind, and a thin wall, seen from everywhere --
//   the normal matrix Y^T W Y + lambda R over the N Fibonacci directions,
//   and its Cholesky factor;
// - `flattenSolve`: M = (Y^T W Y + lambda R)^-1 Y^T W, a column a thread;
// - `flattenRoughnessFloor`: the least roughness whose GGX reflection lobe
//   keeps all but `floorShare` of its energy in bands 0 to 3. Below it a
//   reflection degree 3 harmonics are fitted to rings; at it, it blurs, which
//   is the honest answer a standard file can give and keeps the highlight's
//   energy.
module flatten_fit;
import flatten_basis;

struct FitParams {
    uint  directions;   // N
    float lambda;       // Tikhonov weight, times l^2 (l + 1)^2
    float backWeight;   // a surface's weight behind it
    float floorShare;   // the energy a lobe may leave above band 3
};
ConstantBuffer<FitParams> params;
RWStructuredBuffer<float> gram;        // 2 x 256: the lower Cholesky factor, by class
RWStructuredBuffer<float> basisRows;   // 2 x N x 16: M's columns, by class
RWStructuredBuffer<uint>  floorIndex;  // [0]: the floor's candidate (kFloorCandidates where none)

static const float kFourPi = 12.566370614359172;

float directionWeight(uint cls, float3 l) {
    return (kFourPi / float(params.directions)) * ((cls == 1u || l.z > 0.0) ? 1.0 : params.backWeight);
}

[shader("compute")]
[numthreads(2, 1, 1)]
void flattenGram(uint3 tid: SV_DispatchThreadID) {
    const uint cls = tid.x;
    if (cls >= 2) {
        return;
    }
    float g[256];
    for (uint e = 0; e < 256; ++e) {
        g[e] = 0.0;
    }
    for (uint k = 0; k < params.directions; ++k) {
        const float3 l = fibonacciDirection(k, params.directions);
        const float w = directionWeight(cls, l);
        float y[16];
        flatBasis(-l, y);
        for (uint a = 0; a < 16; ++a) {
            for (uint b = 0; b <= a; ++b) {
                g[a * 16 + b] += w * y[a] * y[b];
            }
        }
    }
    for (uint a = 0; a < 16; ++a) {
        const float band = float(flatBand(a));
        g[a * 16 + a] += params.lambda * band * band * (band + 1.0) * (band + 1.0);
    }
    // Cholesky, lower, in place: the matrix is symmetric positive definite
    // (the regularisation and the back weight see to the surface's).
    for (uint j = 0; j < 16; ++j) {
        float d = g[j * 16 + j];
        for (uint k = 0; k < j; ++k) {
            d -= g[j * 16 + k] * g[j * 16 + k];
        }
        d = sqrt(max(d, 1.0e-20));
        g[j * 16 + j] = d;
        for (uint i = j + 1; i < 16; ++i) {
            float s = g[i * 16 + j];
            for (uint k = 0; k < j; ++k) {
                s -= g[i * 16 + k] * g[j * 16 + k];
            }
            g[i * 16 + j] = s / d;
        }
    }
    for (uint e = 0; e < 256; ++e) {
        gram[cls * 256 + e] = g[e];
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void flattenSolve(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.directions;
    if (tid.x >= 2 * n) {
        return;
    }
    const uint cls = tid.x / n;
    const uint k = tid.x % n;
    const float3 l = fibonacciDirection(k, n);
    const float w = directionWeight(cls, l);
    float y[16];
    flatBasis(-l, y);
    // L L^T x = w y: forward, then back.
    float x[16] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    for (uint i = 0; i < 16; ++i) {
        float s = w * y[i];
        for (uint j = 0; j < i; ++j) {
            s -= gram[cls * 256 + i * 16 + j] * x[j];
        }
        x[i] = s / gram[cls * 256 + i * 16 + i];
    }
    for (int i = 15; i >= 0; --i) {
        float s = x[i];
        for (uint j = uint(i) + 1; j < 16; ++j) {
            s -= gram[cls * 256 + j * 16 + uint(i)] * x[j];
        }
        x[i] = s / gram[cls * 256 + uint(i) * 16 + uint(i)];
    }
    for (uint i = 0; i < 16; ++i) {
        basisRows[(cls * n + k) * 16 + i] = x[i];
    }
}

/// THE SHARE OF A GGX REFLECTION LOBE'S ENERGY ABOVE BAND 3, at roughness
/// `roughness` (alpha its square, as the splat lobes take it), seen head on:
/// the lobe about the mirror direction is D(h) / (4 h.l) with h halfway, a
/// zonal function of the angle from the mirror direction; its Legendre
/// coefficients to band 3 against its whole square integral.
float lobeShareAbove3(float roughness) {
    const float alpha = max(roughness * roughness, 1.0e-3);
    const float a2 = alpha * alpha;
    const uint steps = 4096;
    float total = 0.0;
    float p[4] = { 0.0, 0.0, 0.0, 0.0 };
    for (uint s = 0; s < steps; ++s) {
        const float mu = -1.0 + (float(s) + 0.5) * (2.0 / float(steps));
        const float cosH = sqrt(max(0.5 * (1.0 + mu), 1.0e-8));
        const float d = cosH * cosH * (a2 - 1.0) + 1.0;
        const float f = a2 / (3.14159265358979 * d * d) / (4.0 * cosH);
        const float p2 = 0.5 * (3.0 * mu * mu - 1.0);
        const float p3 = 0.5 * (5.0 * mu * mu * mu - 3.0 * mu);
        total += f * f;
        p[0] += f;
        p[1] += f * mu;
        p[2] += f * p2;
        p[3] += f * p3;
    }
    const float dmu = 2.0 / float(steps);
    total *= dmu;
    float kept = 0.0;
    for (uint l = 0; l < 4; ++l) {
        // f = sum a_l P_l, a_l = (2l + 1)/2 int f P_l, int f^2 = sum a_l^2 2/(2l + 1).
        const float a = 0.5 * float(2 * l + 1) * p[l] * dmu;
        kept += a * a * 2.0 / float(2 * l + 1);
    }
    return total > 0.0 ? saturate(1.0 - kept / total) : 0.0;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void flattenRoughnessFloor(uint3 tid: SV_DispatchThreadID) {
    const uint index = tid.x;
    if (index >= kFloorCandidates) {
        return;
    }
    if (lobeShareAbove3(flatFloorCandidate(index)) <= params.floorShare) {
        InterlockedMin(floorIndex[0], index);
    }
}

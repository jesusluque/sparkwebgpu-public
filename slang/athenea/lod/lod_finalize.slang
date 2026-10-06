// Copyright (c) 2026 jesus luque.
//
// Moments -> one packed Gaussian per group (common/packing.slang's layout).
module lod_finalize;
import athenea.common.packing;
import lod_common;

StructuredBuffer<float>    moments;
RWStructuredBuffer<float4> positions;
RWStructuredBuffer<uint>   shape;
RWStructuredBuffer<uint>   sh;
RWStructuredBuffer<uint>   normals;   // where `params.normals` says the cloud keeps them
RWStructuredBuffer<uint>   emission;  // where `params.emission` says the cloud keeps it
ConstantBuffer<LodParams>  params;

/// Cyclic Jacobi on a symmetric 3x3: `a` ends diagonal (the eigenvalues) and
/// `v`'s columns are the eigenvectors. Six sweeps is past convergence for a
/// 3x3 in single precision.
void jacobi(inout float3x3 a, out float3x3 v) {
    v = float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
    for (int sweep = 0; sweep < 6; ++sweep) {
        for (int pq = 0; pq < 3; ++pq) {
            const int p = pq == 2 ? 1 : 0;
            const int q = pq == 0 ? 1 : 2;
            const float apq = a[p][q];
            if (abs(apq) < 1e-30) {
                continue;
            }
            const float theta = (a[q][q] - a[p][p]) / (2.0 * apq);
            const float t = sign(theta + 1e-30) / (abs(theta) + sqrt(theta * theta + 1.0));
            const float c = 1.0 / sqrt(t * t + 1.0);
            const float s = t * c;
            float3x3 j = float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
            j[p][p] = c;
            j[q][q] = c;
            j[p][q] = s;
            j[q][p] = -s;
            a = mul(transpose(j), mul(a, j));
            v = mul(v, j);
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void lodFinalize(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.groups) {
        return;
    }
    const uint at = g * params.stride;
    const float weight = max(moments[at], 1e-30);
    const float3 mu = float3(moments[at + 1], moments[at + 2], moments[at + 3]) / weight;
    float3x3 cov = float3x3(moments[at + 4], moments[at + 5], moments[at + 6],
                            moments[at + 5], moments[at + 7], moments[at + 8],
                            moments[at + 6], moments[at + 8], moments[at + 9]) / weight;
    cov = cov - float3x3(mu.x * mu, mu.y * mu, mu.z * mu);
    float3x3 axes;
    jacobi(cov, axes);
    if (determinant(axes) < 0.0) {
        axes[0][2] = -axes[0][2];
        axes[1][2] = -axes[1][2];
        axes[2][2] = -axes[2][2];
    }
    const float3 s = sqrt(max(float3(cov[0][0], cov[1][1], cov[2][2]), float3(1e-14)));
    const float smallest = min(min(s.x, s.y), s.z);
    const float area = s.x * s.y * s.z / max(smallest, 1e-20);
    const float opacity = min(moments[at] / max(area, 1e-30), 0.99);
    const float3 base = float3(moments[at + 10], moments[at + 11], moments[at + 12]) / weight;

    positions[g] = float4(mu, opacity);
    const uint row = g * 4;
    const float3 ls = log(s);
    shape[row] = encodeQuaternion(quaternionOfAxes(axes));
    shape[row + 1] = packHalves(halfSafe(ls.x), halfSafe(ls.y));
    shape[row + 2] = packHalves(halfSafe(ls.z), halfSafe(base.x));
    shape[row + 3] = packHalves(halfSafe(base.y), halfSafe(base.z));
    if (params.normals != 0) {
        // The weighted sum's direction, made a unit again. Where the children
        // face every way at once and the sum is nothing, the merged
        // Gaussian's own shortest axis stands in.
        const uint nb = at + kMomentsHead + params.keep * 3;
        const float3 sum = float3(moments[nb], moments[nb + 1], moments[nb + 2]);
        const uint shortest = s.x <= s.y ? (s.x <= s.z ? 0u : 2u) : (s.y <= s.z ? 1u : 2u);
        const float3 axis = float3(axes[0][shortest], axes[1][shortest], axes[2][shortest]);
        normals[g] = packNormal(dot(sum, sum) > 1.0e-24 * weight * weight ? normalize(sum) : axis);
    }
    if (params.emission != 0) {
        // The mean, as the base colour's is.
        const uint eb = at + emissionMoments(params);
        emission[g] = packRgb9e5(float3(moments[eb], moments[eb + 1], moments[eb + 2]) / weight);
    }
    if (params.keep == 0) {
        return;
    }
    const uint shRow = g * params.shWords;
    uint word = 0;
    uint filled = 0;
    uint written = 0;
    for (uint half = 0; half < params.keep * 3; ++half) {
        const uint h = f32tof16(halfSafe(moments[at + kMomentsHead + half] / weight)) & 0xFFFFu;
        word |= h << (filled * 16);
        filled += 1;
        if (filled == 2) {
            sh[shRow + written] = word;
            written += 1;
            word = 0;
            filled = 0;
        }
    }
    if (filled == 1) {
        sh[shRow + written] = word;
    }
}

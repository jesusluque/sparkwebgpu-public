// Copyright (c) 2026 jesus luque.
//
// The finest merged level's moments, from the splats of each group.
module lod_leaf_moments;
import athenea.common.packing;
import lod_common;

StructuredBuffer<float4>   positions;   // Morton order
StructuredBuffer<uint>     shape;
StructuredBuffer<uint>     sh;
StructuredBuffer<uint>     starts;
StructuredBuffer<uint>     normals;     // Morton order, where `params.normals` says
StructuredBuffer<uint>     emission;    // Morton order, where `params.emission` says
RWStructuredBuffer<float>  moments;
ConstantBuffer<LodParams>  params;       // count: splats; groups

float halfAt(uint row, uint index) {
    const uint word = sh[row + index / 2];
    return (index & 1u) == 0u ? lowHalf(word) : highHalf(word);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void lodLeafMoments(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.groups) {
        return;
    }
    const uint first = starts[g];
    const uint end = g + 1 < params.groups ? starts[g + 1] : params.count;
    const uint at = g * params.stride;
    for (uint k = 0; k < params.stride; ++k) {
        moments[at + k] = 0.0;
    }
    for (uint i = first; i < end; ++i) {
        const float4 p = positions[i];
        const uint row = i * 4;
        const float3 s = exp(float3(lowHalf(shape[row + 1]), highHalf(shape[row + 1]), lowHalf(shape[row + 2])));
        const float4 q = decodeQuaternion(shape[row]);
        const float x = q.x, y = q.y, z = q.z, w = q.w;
        const float3x3 r = float3x3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
                                    2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
                                    2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y));
        const float3x3 m = float3x3(r[0] * s, r[1] * s, r[2] * s);
        const float3x3 cov = mul(m, transpose(m));
        // The area it covers face on: its two longest axes.
        const float smallest = min(min(s.x, s.y), s.z);
        const float area = s.x * s.y * s.z / max(smallest, 1e-20);
        const float weight = p.w * area;
        moments[at] += weight;
        moments[at + 1] += weight * p.x;
        moments[at + 2] += weight * p.y;
        moments[at + 3] += weight * p.z;
        moments[at + 4] += weight * (cov[0][0] + p.x * p.x);
        moments[at + 5] += weight * (cov[0][1] + p.x * p.y);
        moments[at + 6] += weight * (cov[0][2] + p.x * p.z);
        moments[at + 7] += weight * (cov[1][1] + p.y * p.y);
        moments[at + 8] += weight * (cov[1][2] + p.y * p.z);
        moments[at + 9] += weight * (cov[2][2] + p.z * p.z);
        // The colour is averaged in the space the cloud keeps it in -- a
        // capture's sRGB, a conversion's linear light (GpuSplats::linear) --
        // and the merged Gaussian is in that space too.
        moments[at + 10] += weight * highHalf(shape[row + 2]);
        moments[at + 11] += weight * lowHalf(shape[row + 3]);
        moments[at + 12] += weight * highHalf(shape[row + 3]);
        for (uint half = 0; half < params.keep * 3; ++half) {
            moments[at + kMomentsHead + half] += weight * halfAt(i * params.shWords, half);
        }
        if (params.normals != 0) {
            // Weighed as the colour is: a group's normal is where its
            // gaussians, by what they cover, face.
            const float3 n = unpackNormal(normals[i]);
            const uint nb = at + kMomentsHead + params.keep * 3;
            moments[nb] += weight * n.x;
            moments[nb + 1] += weight * n.y;
            moments[nb + 2] += weight * n.z;
        }
        if (params.emission != 0) {
            // Weighed as the colour is: what a group gives off is what its
            // gaussians give off, by what they cover.
            const float3 e = unpackRgb9e5(emission[i]);
            const uint eb = at + emissionMoments(params);
            moments[eb] += weight * e.x;
            moments[eb + 1] += weight * e.y;
            moments[eb + 2] += weight * e.z;
        }
    }
}

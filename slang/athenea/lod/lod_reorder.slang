// Copyright (c) 2026 jesus luque.
//
// The cloud's packed splats gathered into Morton order: slot k takes splat
// order[k].
module lod_reorder;
import lod_common;

StructuredBuffer<uint>     order;
StructuredBuffer<float4>   srcPositions;
StructuredBuffer<uint>     srcShape;
StructuredBuffer<uint>     srcSh;
RWStructuredBuffer<float4> positions;
RWStructuredBuffer<uint>   shape;
RWStructuredBuffer<uint>   sh;
// The shading normals, where `params.normals` says the cloud keeps them.
StructuredBuffer<uint>     srcNormals;
RWStructuredBuffer<uint>   normals;
// The radiance given off, where `params.emission` says the cloud keeps it.
StructuredBuffer<uint>     srcEmission;
RWStructuredBuffer<uint>   emission;
ConstantBuffer<LodParams>  params;

[shader("compute")]
[numthreads(256, 1, 1)]
void lodReorder(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= params.count) {
        return;
    }
    const uint i = order[k];
    positions[k] = srcPositions[i];
    for (uint w = 0; w < 4; ++w) {
        shape[k * 4 + w] = srcShape[i * 4 + w];
    }
    for (uint w = 0; w < params.shWords; ++w) {
        sh[k * params.shWords + w] = srcSh[i * params.shWords + w];
    }
    if (params.normals != 0) {
        normals[k] = srcNormals[i];
    }
    if (params.emission != 0) {
        emission[k] = srcEmission[i];
    }
}

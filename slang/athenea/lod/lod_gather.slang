// Copyright (c) 2026 jesus luque.
//
// The selected Gaussians of one level (or the splats) copied into this frame's
// cloud, compacted: element i goes to base + dest[i] when selected[i].
module lod_gather;
import lod_common;

StructuredBuffer<uint>     selected;
StructuredBuffer<uint>     dest;
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
void lodGather(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count || selected[i] == 0) {
        return;
    }
    const uint j = params.base + dest[i];
    const uint k = params.offset + i;   // chunks share their cloud's buffers from `offset`
    positions[j] = srcPositions[k];
    for (uint w = 0; w < 4; ++w) {
        shape[j * 4 + w] = srcShape[k * 4 + w];
    }
    for (uint w = 0; w < params.shWords; ++w) {
        sh[j * params.shWords + w] = srcSh[k * params.shWords + w];
    }
    if (params.normals != 0) {
        normals[j] = srcNormals[k];
    }
    if (params.emission != 0) {
        emission[j] = srcEmission[k];
    }
}

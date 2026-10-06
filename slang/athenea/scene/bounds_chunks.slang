// Copyright (c) 2026 jesus luque.
//
// Per-chunk min and max of float4 xyz, then folded by bounds_reduce. The
// cloud's extent, found where the cloud is.
module bounds_chunks;

struct BoundsParams {
    uint count;
    uint chunkSize;
    uint chunkCount;
    uint pad0;
};

StructuredBuffer<float4>     positions;
RWStructuredBuffer<float4>   extents;   // chunkCount * 2: min, max
ConstantBuffer<BoundsParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void boundsChunks(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= params.chunkCount) {
        return;
    }
    const uint begin = c * params.chunkSize;
    const uint end = min(begin + params.chunkSize, params.count);
    float3 lo = float3(3.4e38);
    float3 hi = float3(-3.4e38);
    for (uint i = begin; i < end; ++i) {
        lo = min(lo, positions[i].xyz);
        hi = max(hi, positions[i].xyz);
    }
    extents[c * 2] = float4(lo, 0.0);
    extents[c * 2 + 1] = float4(hi, 0.0);
}

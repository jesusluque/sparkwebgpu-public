// Copyright (c) 2026 jesus luque.
module bounds_reduce;

struct BoundsParams {
    uint count;
    uint chunkSize;
    uint chunkCount;
    uint pad0;
};

StructuredBuffer<float4>     extents;
RWStructuredBuffer<float4>   result;   // 2: min, max
ConstantBuffer<BoundsParams> params;

[shader("compute")]
[numthreads(1, 1, 1)]
void boundsReduce(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    float3 lo = float3(3.4e38);
    float3 hi = float3(-3.4e38);
    for (uint c = 0; c < params.chunkCount; ++c) {
        lo = min(lo, extents[c * 2].xyz);
        hi = max(hi, extents[c * 2 + 1].xyz);
    }
    result[0] = float4(lo, 0.0);
    result[1] = float4(hi, 0.0);
}

// Copyright (c) 2026 jesus luque.
module lod_morton;
import athenea.common.morton;
import lod_common;

StructuredBuffer<float4>  positions;
RWStructuredBuffer<uint>  keys;
RWStructuredBuffer<uint>  values;
ConstantBuffer<LodParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void lodMorton(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const float3 lo = float3(params.boundsLoX, params.boundsLoY, params.boundsLoZ);
    keys[i] = morton30((positions[i].xyz - lo) / max(params.extent, 1e-20));
    values[i] = i;
}

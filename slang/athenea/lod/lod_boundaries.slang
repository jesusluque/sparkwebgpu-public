// Copyright (c) 2026 jesus luque.
//
// Where a level's cells begin in the sorted splats: 1 where the code's top 3r
// bits differ from the splat before.
module lod_boundaries;
import lod_common;

StructuredBuffer<uint>    keys;
RWStructuredBuffer<uint>  boundary;
ConstantBuffer<LodParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void lodBoundaries(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint shift = levelShift(params.level);
    boundary[i] = (i == 0 || (keys[i] >> shift) != (keys[i - 1] >> shift)) ? 1u : 0u;
}

// Copyright (c) 2026 jesus luque.
//
// A coarser level's moments: each group's children at the finer level,
// summed. A group's children are the finer groups of its splats -- contiguous,
// from the finer group of its first splat to that of the next group's first.
module lod_merge_moments;
import lod_common;

StructuredBuffer<uint>     starts;       // this level, per group
StructuredBuffer<uint>     fineGroup;    // finer level, per splat
StructuredBuffer<float>    fineMoments;
RWStructuredBuffer<float>  moments;
ConstantBuffer<LodParams>  params;       // count: splats; groups; fineGroups

[shader("compute")]
[numthreads(64, 1, 1)]
void lodMergeMoments(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.groups) {
        return;
    }
    const uint first = fineGroup[starts[g]];
    const uint end = g + 1 < params.groups ? fineGroup[starts[g + 1]] : params.fineGroups;
    const uint at = g * params.stride;
    for (uint k = 0; k < params.stride; ++k) {
        float sum = 0.0;
        for (uint h = first; h < end; ++h) {
            sum += fineMoments[h * params.stride + k];
        }
        moments[at + k] = sum;
    }
}

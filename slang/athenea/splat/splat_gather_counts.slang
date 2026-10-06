// Copyright (c) 2026 jesus luque.
// Tiles touched, in depth order: the input to the pairs' prefix sum.
module splat_gather_counts;
import frame;

StructuredBuffer<uint>      order;
StructuredBuffer<uint>      tilesTouched;
RWStructuredBuffer<uint>    sortedCounts;
ConstantBuffer<FrameParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void splatGatherCounts(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= params.count) {
        return;
    }
    sortedCounts[k] = tilesTouched[order[k]];
}

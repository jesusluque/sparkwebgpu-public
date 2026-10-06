// Copyright (c) 2026 jesus luque.
//
// The visible splats, packed: only they are depth-sorted. A frame of a large
// scene culls most of what it holds, and sorting what no tile will ever read
// was the second-largest cost of the frame.
module splat_compact;
import frame;

StructuredBuffer<uint>      visible;
StructuredBuffer<uint>      visibleOffsets;
StructuredBuffer<uint>      depthKeys;
RWStructuredBuffer<uint>    sortKeys;
RWStructuredBuffer<uint>    sortOrder;
ConstantBuffer<FrameParams> params;   // count: all splats

[shader("compute")]
[numthreads(256, 1, 1)]
void splatCompact(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count || visible[i] == 0) {
        return;
    }
    const uint at = visibleOffsets[i];
    sortKeys[at] = depthKeys[i];
    sortOrder[at] = i;
}

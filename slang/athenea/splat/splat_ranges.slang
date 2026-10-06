// Copyright (c) 2026 jesus luque.
// Where each tile's run of sorted pairs starts and ends. Each boundary is
// written by exactly one invocation.
module splat_ranges;
import frame;

StructuredBuffer<uint>      pairTiles;
RWStructuredBuffer<uint>    ranges;
ConstantBuffer<FrameParams> params;   // count: pairs

[shader("compute")]
[numthreads(256, 1, 1)]
void splatRanges(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint t = pairTiles[i];
    if (i == 0 || pairTiles[i - 1] != t) {
        ranges[t * 2] = i;
    }
    if (i + 1 == params.count || pairTiles[i + 1] != t) {
        ranges[t * 2 + 1] = i + 1;
    }
}

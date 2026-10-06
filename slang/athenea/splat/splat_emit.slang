// Copyright (c) 2026 jesus luque.
//
// One (tile, splat) pair per touched tile, splats in depth order. Repeats the
// project stage's test exactly, so it writes exactly the count it was given.
module splat_emit;
import frame;

StructuredBuffer<uint>      order;
StructuredBuffer<uint>      offsets;
StructuredBuffer<uint>      tilesTouched;
StructuredBuffer<uint>      tileRects;
StructuredBuffer<ProjRecord> proj;
RWStructuredBuffer<uint>    pairTiles;
RWStructuredBuffer<uint>    pairSplats;
ConstantBuffer<FrameParams> params;   // count: splats; base: pair capacity

[shader("compute")]
[numthreads(256, 1, 1)]
void splatEmit(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= params.count) {
        return;
    }
    const uint s = order[k];
    const uint expected = tilesTouched[s];
    if (expected == 0) {
        return;
    }
    const uint first = offsets[k];
    const ProjRecord record = proj[s];
    const float2 centre = record.screen.xy;
    const float cutoff = record.screen.w;
    const float3 conic = record.conic.xyz;
    const uint4 rect = uint4(tileRects[s * 4], tileRects[s * 4 + 1], tileRects[s * 4 + 2],
                             tileRects[s * 4 + 3]);
    uint written = 0;
    for (uint ty = rect.y; ty <= rect.w && written < expected; ++ty) {
        for (uint tx = rect.x; tx <= rect.z && written < expected; ++tx) {
            if (!tileTouched(params, centre, conic, cutoff, tx, ty)) {
                continue;
            }
            const uint at = first + written;
            if (at < params.base) {
                pairTiles[at] = ty * params.tilesX + tx;
                pairSplats[at] = s;
            }
            written += 1;
        }
    }
}

// Copyright (c) 2026 jesus luque.
module splat_tiles_clear;
import frame;

RWStructuredBuffer<uint>    ranges;   // 2 per tile: start, end
ConstantBuffer<FrameParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void splatTilesClear(uint3 tid: SV_DispatchThreadID) {
    const uint t = tid.x;
    if (t >= params.tilesX * params.tilesY) {
        return;
    }
    ranges[t * 2] = 0;
    ranges[t * 2 + 1] = 0;
}

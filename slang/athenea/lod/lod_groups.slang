// Copyright (c) 2026 jesus luque.
//
// Every splat's group at a level, every group's first splat and its cell code
// -- from the boundaries and their exclusive prefix sum. Each group's start is
// written by the one splat that begins it: disjoint writes.
module lod_groups;
import lod_common;

StructuredBuffer<uint>    keys;
StructuredBuffer<uint>    boundary;
StructuredBuffer<uint>    before;    // exclusive prefix sum of boundary
RWStructuredBuffer<uint>  group;     // per splat
RWStructuredBuffer<uint>  starts;    // per group
RWStructuredBuffer<uint>  cells;     // per group: the level's cell code
ConstantBuffer<LodParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void lodGroups(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint g = before[i] + boundary[i] - 1;
    group[i] = g;
    if (boundary[i] != 0) {
        starts[g] = i;
        cells[g] = keys[i] >> levelShift(params.level);
    }
}

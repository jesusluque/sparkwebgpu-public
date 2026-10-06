// Copyright (c) 2026 jesus luque.
//
// 30-bit Morton codes: 10 bits an axis, x in the highest of each triple. A
// code's top 3r bits name the octree cell at level r that holds the point.
module morton;

uint spread(uint v) {
    v &= 0x3FFu;
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v << 8)) & 0x0300F00Fu;
    v = (v | (v << 4)) & 0x030C30C3u;
    v = (v | (v << 2)) & 0x09249249u;
    return v;
}

uint compact(uint v) {
    v &= 0x09249249u;
    v = (v | (v >> 2)) & 0x030C30C3u;
    v = (v | (v >> 4)) & 0x0300F00Fu;
    v = (v | (v >> 8)) & 0x030000FFu;
    v = (v | (v >> 16)) & 0x000003FFu;
    return v;
}

/// The code of a point at `unit` (0..1 on each axis of the cloud's bounds).
public uint morton30(float3 unit) {
    const uint3 q = min(uint3(saturate(unit) * 1024.0), uint3(1023u));
    return (spread(q.x) << 2) | (spread(q.y) << 1) | spread(q.z);
}

/// The cell index on each axis of a level-r cell code (the top 3r bits).
public uint3 cellIndex(uint cell) {
    return uint3(compact(cell >> 2), compact(cell >> 1), compact(cell));
}

/// The other way: the code of the cell at `index` on each axis.
public uint cellCode(uint3 index) {
    return (spread(index.x) << 2) | (spread(index.y) << 1) | spread(index.z);
}

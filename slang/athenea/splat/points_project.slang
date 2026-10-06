// Copyright (c) 2026 jesus luque.
//
// Points as opaque discs in the splat pipeline: the route that works on every
// backend (CUDA has no rasteriser), and the one that mixes points and splats in
// a single depth order. A disc is a record like a splat's, with an isotropic
// conic of its radius, a cut at one radius, alpha one, and colour.w = 1 so the
// blend draws a hard edge and stops there.
module points_project;
import athenea.common.packing;
import athenea.points.point_frame;
import frame;

StructuredBuffer<float4>       positions;
StructuredBuffer<uint>         colours;
RWStructuredBuffer<ProjRecord> proj;
RWStructuredBuffer<uint>       visible;
RWStructuredBuffer<uint>       tileRects;
RWStructuredBuffer<uint>       tilesTouched;
RWStructuredBuffer<uint>       depthKeys;
RWStructuredBuffer<uint>       cryptoIds;   // a point carries no id: 0 where the frame keeps a matte
ConstantBuffer<FrameParams>    params;   // m*, count, base, ...
ConstantBuffer<PointParams>    points;   // sizeMode, size, colour override

void cull(uint slot) {
    tilesTouched[slot] = 0;
    visible[slot] = 0;
    depthKeys[slot] = 0x00FFFFFFu;
    tileRects[slot * 4] = 1;
    tileRects[slot * 4 + 1] = 1;
    tileRects[slot * 4 + 2] = 0;
    tileRects[slot * 4 + 3] = 0;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void pointsProject(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint slot = params.base + i;
    const float3 t = toView(params, positions[i].xyz);
    if (!(t.z > params.nearZ) || t.z > params.farZ) {
        cull(slot);
        return;
    }
    const float radius = pointRadius(points, t.z);
    const float invZ = params.orthographic != 0 ? 1.0 : 1.0 / t.z;
    const float2 centre = float2(params.focalX * t.x * invZ + params.centreX,
                                 params.focalY * t.y * invZ + params.centreY);
    const float3 conic = float3(1.0 / (radius * radius), 0.0, 1.0 / (radius * radius));
    const float cutoff = 1.0;
    uint4 rect;
    tileRect(params, centre, float2(radius, radius), rect);
    if (rect.x > rect.z || rect.y > rect.w) {
        cull(slot);
        return;
    }
    uint touched = 0;
    for (uint ty = rect.y; ty <= rect.w; ++ty) {
        for (uint tx = rect.x; tx <= rect.z; ++tx) {
            if (tileTouched(params, centre, conic, cutoff, tx, ty)) {
                touched += 1;
            }
        }
    }
    if (touched == 0) {
        cull(slot);
        return;
    }
    float3 rgb = float3(points.colourR, points.colourG, points.colourB);
    if (points.colourMode == 0) {
        rgb = float3(lowHalf(colours[i * 2]), highHalf(colours[i * 2]), lowHalf(colours[i * 2 + 1]));
    }
    ProjRecord record;
    record.screen = float4(centre, t.z, cutoff);
    record.conic = float4(conic, 1.0);
    record.colour = float4(rgb, 1.0);
    proj[slot] = record;
    if (params.cryptoMode != 0) {
        cryptoIds[slot] = 0u;
    }
    tileRects[slot * 4] = rect.x;
    tileRects[slot * 4 + 1] = rect.y;
    tileRects[slot * 4 + 2] = rect.z;
    tileRects[slot * 4 + 3] = rect.w;
    tilesTouched[slot] = touched;
    visible[slot] = 1;
    depthKeys[slot] = bit_cast<uint>(t.z) >> 7;
}

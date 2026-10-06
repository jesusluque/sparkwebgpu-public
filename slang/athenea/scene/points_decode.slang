// Copyright (c) 2026 jesus luque.
//
// Raw point records -> positions (float4 xyz, 1) and colours (two uint words
// of fp16: r|g, b|a), linear, compacted.
module points_decode;
import athenea.common.color;
import athenea.common.packing;

struct PointParams {
    uint  count;
    uint  base;
    uint  colourKind;
    float detail;
};

StructuredBuffer<float>     raw;
StructuredBuffer<uint>      valid;
StructuredBuffer<uint>      dest;
RWStructuredBuffer<float4>  positions;
RWStructuredBuffer<uint>    colours;   // count * 2
ConstantBuffer<PointParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void pointsDecode(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count || valid[i] == 0) {
        return;
    }
    const uint j = params.base + dest[i];
    const uint at = i * 6;
    positions[j] = float4(raw[at], raw[at + 1], raw[at + 2], 1.0);
    float3 rgb = float3(1.0);
    if (params.colourKind == 1) {
        rgb = srgbToLinear(float3(raw[at + 3], raw[at + 4], raw[at + 5]) / 255.0);
    } else if (params.colourKind == 2) {
        rgb = float3(raw[at + 3], raw[at + 4], raw[at + 5]);
    }
    colours[j * 2] = packHalves(rgb.x, rgb.y);
    colours[j * 2 + 1] = packHalves(rgb.z, 1.0);
}

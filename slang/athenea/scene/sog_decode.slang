// Copyright (c) 2026 jesus luque.
//
// SOG's image bytes (athenea::io::RawSog) -> raw splat records in the engine's own
// float encoding, which then take the same validate and decode every other
// format takes. What each byte means is PlayCanvas's (MIT: playcanvas/engine
// gsplat-sog-data.js), as openFXplayer's SogReader ported it.
//
//   record  0..2 position   3 opacity (linear)   4..6 log scale
//           7 w  8 x  9 y  10 z   11..13 SH DC   14.. rest, rgb per basis
module sog_decode;

struct SogParams {
    uint  count;          // splats in this slice
    uint  first;          // index of the slice's first splat
    uint  stride;         // floats per output record: 14 + 3 * keep
    uint  keep;           // rest basis functions written (0, 3, 8, 15)
    uint  version;        // 1: ranges, 2: codebooks
    uint  coefficients;   // per palette entry (0: no higher harmonics)
    uint  centroidsWidth;
    uint  pad0;
    float meansMinX; float meansMinY; float meansMinZ;
    float meansMaxX; float meansMaxY; float meansMaxZ;
    float scalesMinX; float scalesMinY; float scalesMinZ;
    float scalesMaxX; float scalesMaxY; float scalesMaxZ;
    float sh0MinR; float sh0MinG; float sh0MinB; float sh0MinA;
    float sh0MaxR; float sh0MaxG; float sh0MaxB; float sh0MaxA;
    float shNMin; float shNMax;
    float pad1; float pad2;
};

StructuredBuffer<uint>  meansL;
StructuredBuffer<uint>  meansU;
StructuredBuffer<uint>  quats;
StructuredBuffer<uint>  scales;
StructuredBuffer<uint>  sh0;
StructuredBuffer<uint>  shLabels;     // one placeholder texel when there are no harmonics
StructuredBuffer<uint>  shCentroids;
StructuredBuffer<float> codebooks;    // scales 0..255, sh0 256..511, shN 512..767 (version 2)
RWStructuredBuffer<float> records;
ConstantBuffer<SogParams> params;

uint channel(uint texel, uint c) {
    return (texel >> (8 * c)) & 0xFFu;
}

float unlerp(float lo, float hi, uint byte) {
    return lo + (hi - lo) * (float(byte) / 255.0);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sogDecode(uint3 tid: SV_DispatchThreadID) {
    const uint slot = tid.x;
    if (slot >= params.count) {
        return;
    }
    const uint i = params.first + slot;
    const uint at = slot * params.stride;

    // Position: 16 bits an axis over two images, in a signed log domain:
    // p = sign(n) (exp(|n|) - 1).
    const float3 lo = float3(params.meansMinX, params.meansMinY, params.meansMinZ);
    const float3 hi = float3(params.meansMaxX, params.meansMaxY, params.meansMaxZ);
    for (uint axis = 0; axis < 3; ++axis) {
        const uint q = (channel(meansU[i], axis) << 8) | channel(meansL[i], axis);
        const float n = lo[axis] + (hi[axis] - lo[axis]) * (float(q) / 65535.0);
        records[at + axis] = sign(n) * (exp(abs(n)) - 1.0);
    }

    // Colour and opacity.
    const uint c = sh0[i];
    if (params.version == 2) {
        for (uint k = 0; k < 3; ++k) {
            records[at + 11 + k] = codebooks[256 + channel(c, k)];
        }
        records[at + 3] = float(channel(c, 3)) / 255.0;
    } else {
        records[at + 11] = unlerp(params.sh0MinR, params.sh0MaxR, channel(c, 0));
        records[at + 12] = unlerp(params.sh0MinG, params.sh0MaxG, channel(c, 1));
        records[at + 13] = unlerp(params.sh0MinB, params.sh0MaxB, channel(c, 2));
        records[at + 3] = 1.0 / (1.0 + exp(-unlerp(params.sh0MinA, params.sh0MaxA, channel(c, 3))));
    }

    // Log scales, by codebook or by range.
    const float3 slo = float3(params.scalesMinX, params.scalesMinY, params.scalesMinZ);
    const float3 shi = float3(params.scalesMaxX, params.scalesMaxY, params.scalesMaxZ);
    for (uint axis = 0; axis < 3; ++axis) {
        const uint byte = channel(scales[i], axis);
        records[at + 4 + axis] = params.version == 2 ? codebooks[byte]
                                                     : slo[axis] + (shi[axis] - slo[axis]) * (float(byte) / 255.0);
    }

    // Quaternion, smallest three: components in [-sqrt2/2, sqrt2/2], the
    // alpha byte (252..255) naming the one left out.
    const float norm = 1.41421356237;
    const uint qt = quats[i];
    const float a = (float(channel(qt, 0)) / 255.0 - 0.5) * norm;
    const float b = (float(channel(qt, 1)) / 255.0 - 0.5) * norm;
    const float d3 = (float(channel(qt, 2)) / 255.0 - 0.5) * norm;
    const float rest = sqrt(max(0.0, 1.0 - (a * a + b * b + d3 * d3)));
    float4 q = float4(0.0, 0.0, 0.0, 1.0);   // x y z w
    const int mode = int(channel(qt, 3)) - 252;
    if (mode == 0) {
        q = float4(a, b, d3, rest);
    } else if (mode == 1) {
        q = float4(rest, b, d3, a);
    } else if (mode == 2) {
        q = float4(b, rest, d3, a);
    } else if (mode == 3) {
        q = float4(b, d3, rest, a);
    }
    records[at + 7] = q.w;
    records[at + 8] = q.x;
    records[at + 9] = q.y;
    records[at + 10] = q.z;

    // Higher harmonics through the palette: the label names an entry, a
    // palette row holds 64 entries of `coefficients` texels each, one colour
    // per channel.
    if (params.keep == 0) {
        return;
    }
    const uint label = shLabels[i];
    const uint entry = channel(label, 0) | (channel(label, 1) << 8);
    const uint u = (entry % 64) * params.coefficients;
    const uint v = entry / 64;
    for (uint basis = 0; basis < params.keep; ++basis) {
        const uint texel = shCentroids[(u + basis) + v * params.centroidsWidth];
        for (uint colour = 0; colour < 3; ++colour) {
            const uint byte = channel(texel, colour);
            records[at + 14 + basis * 3 + colour] =
                params.version == 2 ? codebooks[512 + byte] : unlerp(params.shNMin, params.shNMax, byte);
        }
    }
}

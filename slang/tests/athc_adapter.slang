// Reads every .athc stream of every splat through slang/athenea_adapter, so
// a test can check athenea's decoders on the GPU against the CPU's.

import core.attrib;
import athenea_adapter.athc;
import athenea.common.packing;

struct AthcTestParams {
    uint count;
    uint normalOct;
    uint emission;
    uint pbr;
    uint lobes;
    uint transfer;
    uint shadowBits;
    uint transferCount;
    uint shadowWords;
    uint pad0;
    uint pad1;
    uint pad2;
};

StructuredBuffer<uint> pool;
// Per splat: normal (3), emission (3), pbr (4), lobes (13), transfer[0],
// transfer[last], shadow word 0, 0: 28 floats.
RWStructuredBuffer<float> decoded;
ConstantBuffer<AthcTestParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void athcDecode(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    AthcIds ids;
    ids.normalOct = params.normalOct;
    ids.emission = params.emission;
    ids.pbr = params.pbr;
    ids.lobes = params.lobes;
    ids.transfer = params.transfer;
    ids.shadowBits = params.shadowBits;
    ids.transferCount = params.transferCount;
    ids.shadowWords = params.shadowWords;
    const uint o = i * 28u;
    const float3 n = athcNormal(pool, ids, i, float3(0.0, 0.0, 1.0));
    const float3 e = athcEmission(pool, ids, i);
    const float4 m = athcPbr(pool, ids, i);
    const SplatLobes l = athcLobes(pool, ids, i);
    decoded[o + 0u] = n.x;
    decoded[o + 1u] = n.y;
    decoded[o + 2u] = n.z;
    decoded[o + 3u] = e.x;
    decoded[o + 4u] = e.y;
    decoded[o + 5u] = e.z;
    decoded[o + 6u] = m.x;
    decoded[o + 7u] = m.y;
    decoded[o + 8u] = m.z;
    decoded[o + 9u] = m.w;
    decoded[o + 10u] = l.specularWeight;
    decoded[o + 11u] = l.specularColour.x;
    decoded[o + 12u] = l.specularColour.y;
    decoded[o + 13u] = l.specularColour.z;
    decoded[o + 14u] = l.specularIor;
    decoded[o + 15u] = l.coatWeight;
    decoded[o + 16u] = l.coatRoughness;
    decoded[o + 17u] = l.coatIor;
    decoded[o + 18u] = l.coatDarkening;
    decoded[o + 19u] = l.sheenColour.x;
    decoded[o + 20u] = l.sheenColour.y;
    decoded[o + 21u] = l.sheenColour.z;
    decoded[o + 22u] = l.sheenRoughness;
    decoded[o + 23u] = athcTransfer(pool, ids, i, 0u);
    decoded[o + 24u] = athcTransfer(pool, ids, i, params.transferCount > 0u ? params.transferCount - 1u : 0u);
    decoded[o + 25u] = asfloat(athcShadowWord(pool, ids, i, 0u));
    decoded[o + 26u] = 0.0;
    decoded[o + 27u] = 0.0;
}

// The glue between athenea's Slang (vendored, never edited: slang/athenea)
// and our attribute pool (core/attrib.slang). A .athc's streams arrive as
// attributes (rust/spark-lib/src/athc.rs `attrib_specs`):
//
//   normalOct   u32       packing.slang packNormal
//   emission    u32       packing.slang packRgb9e5 (linear)
//   pbr         u32       splat_encoding.slang packPbr
//   lobes       u32 x 3   packing.slang packLobes
//   transfer    f16 x N   transfer_layout.slang (N = 9, 36, 84, 16, 64, 112, or 10)
//   shadowBits  u32 x W   open directions, a bit a cell
//
// athenea's kernels read these from one StructuredBuffer each; here each is
// an attribute of the one pool, by id (resolved by name on the host,
// ATTRIB_NONE when the cloud does not carry it), decoded with athenea's own
// functions so the web and the native read the same bits the same way.
module athc;

import core.attrib;
import athenea.common.packing;
import athenea.scene.splat_encoding;

/// Attribute ids of a .athc's streams in a pool (ATTRIB_NONE: absent).
public struct AthcIds {
    public uint normalOct;
    public uint emission;
    public uint pbr;
    public uint lobes;
    public uint transfer;
    public uint shadowBits;
    public uint transferCount;
    public uint shadowWords;
};

public bool athcHas(uint id) { return id != ATTRIB_NONE; }

/// The shading normal in the cloud's space, or `fallback` (the frame's
/// short axis, say) where the cloud keeps none.
public float3 athcNormal(StructuredBuffer<uint> pool, AthcIds ids, uint splat, float3 fallback) {
    return athcHas(ids.normalOct) ? unpackNormal(readAttribU(pool, ids.normalOct, splat, 0u)) : fallback;
}

/// Linear radiance the splat gives off by itself; black where none.
public float3 athcEmission(StructuredBuffer<uint> pool, AthcIds ids, uint splat) {
    return athcHas(ids.emission) ? unpackRgb9e5(readAttribU(pool, ids.emission, splat, 0u)) : float3(0.0);
}

/// metallic, roughness, transmission, and flags (1 thin-walled, 2 Schlick
/// metal); a capture's (0, 1, 0, 0) where the cloud carries no material, as
/// splat_project.slang falls back.
public float4 athcPbr(StructuredBuffer<uint> pool, AthcIds ids, uint splat) {
    if (!athcHas(ids.pbr)) {
        return float4(0.0, 1.0, 0.0, 0.0);
    }
    const uint w = readAttribU(pool, ids.pbr, splat, 0u);
    return float4(pbrMetallic(w), pbrRoughness(w), pbrTransmission(w),
                  float((pbrThinWalled(w) ? 1u : 0u) | (pbrSchlickMetal(w) ? 2u : 0u)));
}

/// The layers over the base, or `plainLobes()` where the cloud names none.
public SplatLobes athcLobes(StructuredBuffer<uint> pool, AthcIds ids, uint splat) {
    if (!athcHas(ids.lobes)) {
        return plainLobes();
    }
    return unpackLobes(uint3(readAttribU(pool, ids.lobes, splat, 0u), readAttribU(pool, ids.lobes, splat, 1u),
                             readAttribU(pool, ids.lobes, splat, 2u)));
}

/// Transfer value k (0 .. transferCount - 1), as athenea's halves.
public float athcTransfer(StructuredBuffer<uint> pool, AthcIds ids, uint splat, uint k) {
    if (!athcHas(ids.transfer) || k >= ids.transferCount) {
        return 0.0;
    }
    return f16tof32(readAttribU(pool, ids.transfer, splat, k));
}

/// Word w of the open-direction bits; all open where the cloud has none.
public uint athcShadowWord(StructuredBuffer<uint> pool, AthcIds ids, uint splat, uint w) {
    return athcHas(ids.shadowBits) && w < ids.shadowWords ? readAttribU(pool, ids.shadowBits, splat, w)
                                                            : 0xFFFFFFFFu;
}

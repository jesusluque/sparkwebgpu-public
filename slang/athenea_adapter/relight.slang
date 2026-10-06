// athenea's relighting of a cloud under a dome and lights (splat_relight's
// relitSplat, vendored and called as it is), as a pass before generate:
// src/webgpu/athenea/relightPlugin.ts runs it per mesh over the source
// splats, and the ISplatColour of relight_colour.slang hands generate each
// splat's relit colour.
//
// Per splat this is what athenea's splat_project does for a relit cloud
// (projectSplat -> relitToward -> relitSplat), with its inputs read from
// the mesh: the splat from generate's `src` (packed or ext), the .athc
// streams from the attribute pool (slang/athenea_adapter/athc.slang's
// layout: normalOct, emission, pbr, lobes, transfer, shadowBits), the
// object's place as rows, and the sky the env passes prepared (env.slang).
//
// The pool is handed to athenea's functions as its `transfer` and
// `cellBits` buffers: a gaussian's transfer starts at its attribute's word
// times two (halves, low first, as transferAt reads them) and its cells at
// the shadowBits attribute's word, so athenea indexes the pool directly.
//
// What it does not do here: per-pixel slope and the sharp coat need the
// cloud's curvature, which a .athc does not carry (noSlope: every lobe is
// read at the splat's centre); the colour's harmonics beyond the constant
// term are not read (relit clouds from meshes are degree 0); no ray is
// traced (no shadow factors, reflections or glass exits).
module relight;

import core.math;
import core.splat_types;
import core.packed;
import core.ext;
import core.attrib;
import athenea.splat.splat_relight;
import athenea.common.packing;
import athenea.common.color;
import athenea.common.octahedral;
import athenea.scene.splat_encoding;

public static const uint kRelightExt = 1u;         // src is ext (else packed)
public static const uint kRelightLodOpacity = 2u;  // packed src keeps alpha / 2
public static const uint kRelightLinear = 4u;      // stored colours are linear light (else sRGB)
public static const uint kRelightLit = 8u;         // the body is baked radiance (athenea's litBody)
public static const uint kRelightIndirect = 16u;   // use the transfer's indirect half
public static const uint kRelightCache = 32u;      // relit reads the kept view-independent terms
public static const uint kRelightGroup = 256;
/// Two uint4 a splat in `relit`: the colour (asuint float4: linear rgb,
/// alpha or -1 to keep the splat's), then the kept terms (txCache).
public static const uint kRelightStride = 2;

public struct AtheneaRelightParams {
    public uint  count;            ///< source splats
    public uint  flags;
    public uint  lightCount;
    public uint  envLights;
    public uint  envBaseSide;
    public uint  normalOct;        ///< attribute ids in the pool (ATTRIB_NONE: absent)
    public uint  emission;
    public uint  pbr;
    public uint  lobes;
    public uint  transfer;
    public uint  shadowBits;
    public uint  transferCount;    ///< values a gaussian (9, 36, 84, 16, 64, 112, 10), 0: none
    public uint  shadowWords;      ///< words a gaussian (2, 8, 32), 0: none
    public float ior;              ///< the cloud's index (0: nothing bends)
    public float emissionScale;
    public uint  pad0;
    public float4 encoding;        ///< packed src: rgbMin, rgbMax, lnScaleMin, lnScaleMax
    public float4 row0;            ///< object to world, rows of a 3x4
    public float4 row1;
    public float4 row2;
    public float4 eyeObject;       ///< the eye in the cloud's space
    public float4 eyeWorld;
};

StructuredBuffer<uint4>              relightSrc;
StructuredBuffer<uint>               relightPool;
StructuredBuffer<LightRecord>        relightLights;
StructuredBuffer<float4>             envSh;
StructuredBuffer<float4>             envSun;
StructuredBuffer<uint>               envTexels;
StructuredBuffer<uint>               envOfLight;
RWStructuredBuffer<uint4>            relit;
ConstantBuffer<AtheneaRelightParams> relightParams;

bool hasFlag(uint f) { return (relightParams.flags & f) != 0u; }
bool hasAttr(uint id) { return id != ATTRIB_NONE; }

Gsplat readSplat(uint i) {
    if (hasFlag(kRelightExt)) {
        return readExtSplat(relightSrc[2u * i], relightSrc[2u * i + 1u], int(i));
    }
    Gsplat s = readPackedSplat(relightSrc[i], int(i), relightParams.encoding);
    if (hasFlag(kRelightLodOpacity)) {
        s.rgba.a *= 2.0;
    }
    return s;
}

float4 rowOf(uint k) {
    return k == 0u ? relightParams.row0 : (k == 1u ? relightParams.row1 : relightParams.row2);
}

/// Where a gaussian's transfer starts in the pool, in halves.
uint transferBaseOf(uint i) {
    return attribBase(relightPool, attribDesc(relightPool, relightParams.transfer), i) * 2u;
}

/// Where its open-direction bits start in the pool, in words.
uint cellBaseOf(uint i) {
    return attribBase(relightPool, attribDesc(relightPool, relightParams.shadowBits), i);
}

uint poolWord(uint id, uint i, uint k) {
    return readAttribU(relightPool, id, i, k);
}

/// splat_project's storedFacing: the face the transfer was measured for.
float3 storedFacing(float3x3 r, float3 scale, uint i) {
    if (hasAttr(relightParams.normalOct)) {
        return unpackNormal(poolWord(relightParams.normalOct, i, 0u));
    }
    const uint shortest = scale.x <= scale.y ? (scale.x <= scale.z ? 0u : 2u) : (scale.y <= scale.z ? 1u : 2u);
    return normalize(float3(r[0][shortest], r[1][shortest], r[2][shortest]));
}

uint transferCount() {
    return hasAttr(relightParams.transfer) ? relightParams.transferCount : 0u;
}

uint shadowWords() {
    return hasAttr(relightParams.shadowBits) ? relightParams.shadowWords : 0u;
}

/// The surface splat_project's relitToward builds, from this mesh's buffers.
SplatSurface surfaceOf(uint i, Gsplat g, float3x3 r, float3 cloudNormal, float3 colour) {
    const float4 w0 = relightParams.row0;
    const float4 w1 = relightParams.row1;
    const float4 w2 = relightParams.row2;
    const bool hasPbr = hasAttr(relightParams.pbr);
    const uint pbr = hasPbr ? poolWord(relightParams.pbr, i, 0u) : 0u;
    SplatSurface surface;
    surface.position = relightToWorld(w0, w1, w2, g.center);
    surface.normal = relightNormalToWorld(w0, w1, w2, cloudNormal);
    surface.albedo = colour;
    surface.metallic = hasPbr ? pbrMetallic(pbr) : 0.0;
    surface.roughness = hasPbr ? pbrRoughness(pbr) : 1.0;
    surface.transmission = hasPbr ? pbrTransmission(pbr) : 0.0;
    surface.thinWalled = hasPbr && pbrThinWalled(pbr);
    surface.schlickMetal = hasPbr && pbrSchlickMetal(pbr);
    surface.ior = relightParams.ior;
    surface.backFacing = splatFacing(r, g.scales, g.center, relightParams.eyeObject.xyz) < 0.0;
    surface.hasBehind = false;
    surface.behindColour = float3(0.0);
    surface.exitThrough = float3(1.0);
    surface.hasReflection = false;
    surface.reflectionColour = float3(0.0);
    surface.reflectionTraced = false;
    surface.hasExit = false;
    surface.exitDirection = float3(0.0, 0.0, 1.0);
    surface.lit = hasFlag(kRelightLit);
    surface.emission = hasAttr(relightParams.emission)
                           ? relightParams.emissionScale * unpackRgb9e5(poolWord(relightParams.emission, i, 0u))
                           : float3(0.0);
    const bool hasLobes = hasAttr(relightParams.lobes);
    surface.lobes = hasLobes ? unpackLobes(uint3(poolWord(relightParams.lobes, i, 0u),
                                                 poolWord(relightParams.lobes, i, 1u),
                                                 poolWord(relightParams.lobes, i, 2u)))
                             : plainLobes();
    if (hasLobes && surface.ior > 1.0 && surface.transmission > 0.0) {
        surface.ior = max(surface.lobes.specularIor, 1.0);
    }
    const uint count = transferCount();
    surface.transferCount = count;
    surface.transferBase = count != 0u ? transferBaseOf(i) : 0u;
    splatTransferFrame(surface, relightPool, r, w0, w1, w2);
    const uint words = shadowWords();
    surface.hasShadowBits = words != 0u && words < 8u && count >= 9u;
    surface.shadowBits = surface.hasShadowBits ? uint2(poolWord(relightParams.shadowBits, i, 0u),
                                                       poolWord(relightParams.shadowBits, i, 1u))
                                               : uint2(0u);
    surface.cellWords = words >= 8u && count >= 9u ? words : 0u;
    surface.cellBase = surface.cellWords != 0u ? cellBaseOf(i) : 0u;
    return surface;
}

/// The first dome with a prepared sky, as splatTransferViewless finds it.
uint firstDome() {
    for (uint k = 0; k < relightParams.lightCount && k < relightParams.envLights; ++k) {
        if (relightLights[k].kind == kLightDome && envOfLight[k] != kEnvNone) {
            return envOfLight[k];
        }
    }
    return kEnvNone;
}

/// splat_project's splatTransferViewless, a splat: the kept terms, or zero.
uint4 viewless(uint i, Gsplat g) {
    const uint dome = firstDome();
    const uint count = transferCount();
    if (count < 9u || count == kTransferZonalCount || dome == kEnvNone || hasFlag(kRelightLit)) {
        return uint4(0u);
    }
    const float3x3 r = splatRotation(g.quaternion);
    const float4 w0 = relightParams.row0;
    const float4 w1 = relightParams.row1;
    const float4 w2 = relightParams.row2;
    SplatSurface s = {};
    s.normal = relightNormalToWorld(w0, w1, w2, storedFacing(r, g.scales, i));
    s.transferCount = count;
    s.transferBase = transferBaseOf(i);
    const uint words = shadowWords();
    s.hasShadowBits = words != 0u && words < 8u;
    s.shadowBits = s.hasShadowBits ? uint2(poolWord(relightParams.shadowBits, i, 0u),
                                           poolWord(relightParams.shadowBits, i, 1u))
                                   : uint2(0u);
    s.cellWords = words >= 8u ? words : 0u;
    s.cellBase = s.cellWords != 0u ? cellBaseOf(i) : 0u;
    s.lit = false;
    float3 front;
    float3 coupling;
    float  sunShare;
    transferViewless<2>(s, relightPool, relightPool, envSh, envSun, dome, hasFlag(kRelightIndirect), front, coupling,
                        sunShare);
    return uint4(packHalves(front.x, front.y), packHalves(front.z, coupling.x),
                 packHalves(coupling.y, coupling.z), packHalves(sunShare, float(dome + 1u)));
}

/// What a relit splat shows the eye (splat_project's projectSplat colour
/// path for a relit cloud, kTransfer 0, 1 or 2), and its alpha: athenea's
/// thin sheet held to kSheetAlpha with its colour scaled down to match, or
/// -1 where the splat keeps its own.
float4 relitColour<let kTransfer : int>(uint i) {
    const Gsplat g = readSplat(i);
    if (!isActive(g.flags)) {
        return float4(0.0, 0.0, 0.0, -1.0);
    }
    const float3x3 r = splatRotation(g.quaternion);
    const float3 colour = cloudLight(g.rgba.rgb, hasFlag(kRelightLinear));
    const bool hasNormals = hasAttr(relightParams.normalOct);
    const float3 cloudNormal = splatShadingNormal(r, g.scales, g.center, relightParams.eyeObject.xyz, hasNormals,
                                                  hasNormals ? poolWord(relightParams.normalOct, i, 0u) : 0u);
    SplatSurface surface = surfaceOf(i, g, r, cloudNormal, colour);
    SplatSlope slope = noSlope();
    if (kTransfer >= 2 && hasFlag(kRelightCache) && surface.transferCount != kTransferZonalCount &&
        dot(cloudNormal, storedFacing(r, g.scales, i)) > 0.0) {
        const uint4 kept = relit[i * kRelightStride + 1u];
        if (kept.w != 0u) {
            slope.cached = true;
            slope.cachedDome = uint(highHalf(kept.w) + 0.5) - 1u;
            slope.cachedFront = float3(lowHalf(kept.x), highHalf(kept.x), lowHalf(kept.y));
            slope.cachedCoupling = float3(highHalf(kept.y), lowHalf(kept.z), highHalf(kept.z));
            slope.cachedSunShare = lowHalf(kept.w);
        }
    }
    // No shadow was measured: relightParams' shadow lights are 0, and the
    // factors are never read (iesValues stands in for the buffer).
    float3 light = relitSplat<kTransfer>(surface, normalize(relightParams.eyeWorld.xyz - surface.position),
                                         relightLights, relightParams.lightCount, 0xFFFFFFFFu, 0xFFFFFFFFu,
                                         iesValues, 0u, 0u, envTexels, envSh, envOfLight, relightParams.envLights,
                                         relightParams.envBaseSide, envSun, relightPool, relightPool,
                                         hasFlag(kRelightIndirect), slope);
    float alpha = -1.0;
    const bool hasPbr = hasAttr(relightParams.pbr);
    const uint pbr = hasPbr ? poolWord(relightParams.pbr, i, 0u) : 0u;
    // splat_project's thin sheet, with Spark's alpha as the paid opacity.
    if (hasPbr && pbrThinWalled(pbr) && pbrTransmission(pbr) > 0.5) {
        const float alphaOwn = max(g.rgba.a - 1.0 / 255.0, 0.0);
        alpha = max(alphaOwn, 0.1);
        light *= alphaOwn / alpha;
    }
    return float4(light, alpha);
}

void relightAt<let kTransfer : int>(uint i) {
    if (i >= relightParams.count) {
        return;
    }
    relit[i * kRelightStride] = asuint(relitColour<kTransfer>(i));
}

/// The kept terms (splatTransferViewless), when the sky, the lights, the
/// cloud or its place changed.
[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaRelightViewless(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= relightParams.count) {
        return;
    }
    const Gsplat g = readSplat(i);
    relit[i * kRelightStride + 1u] = isActive(g.flags) ? viewless(i, g) : uint4(0u);
}

/// A cloud with a TX transfer (cells): everything.
[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaRelight(uint3 tid: SV_DispatchThreadID) {
    relightAt<2>(tid.x);
}

/// The first transfer (no cells).
[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaRelightFirst(uint3 tid: SV_DispatchThreadID) {
    relightAt<1>(tid.x);
}

/// No transfer: the lights and the dome's harmonics and map alone.
[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaRelightPlain(uint3 tid: SV_DispatchThreadID) {
    relightAt<0>(tid.x);
}

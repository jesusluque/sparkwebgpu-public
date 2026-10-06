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
// A shadow catcher (atheneaRelightCatcher) is drawn black at the opacity
// splat_project's catcherOpacity gives it.
//
// Where the cloud keeps athenea's curvature (.athc v3 section CURV, the
// attribute `curvature`), the pass does what splat_project does with it:
// the mean curvature for a solid glass's far face (lensExit: a pawn's head
// is a lens), and, with the frame's projection (relightParams' view rows,
// focal and viewport), the normal a pixel right and a pixel down
// (SplatSlope) -- relitSplat then hands back the reflection's slope across
// the footprint, or its sharp coat and polish to be read per pixel. Those
// leave in the three uint4 after the colour and the kept terms (the record
// splat_project writes to `slopes` and `sharpPolish`, and the centre in
// pixels with what the colour holds of the centre's sharp reading), for
// relight_colour.slang to hand to the draw (AtheneaRelightBlend).
//
// A splat whose transfer is not resident (a paged pool that has not brought
// its page) keeps its captured colour (alpha -2: generate keeps the rgba).
//
// What it does not do here: the colour's harmonics beyond the constant
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
public static const uint kRelightSlope = 64u;      // the viewport is known: per-pixel slope and sharp lobes
public static const uint kRelightGroup = 256;
/// Five uint4 a splat in `relit`: the colour (asuint float4: linear rgb,
/// alpha, -1 to keep the splat's alpha, -2 to keep its whole rgba), the
/// kept terms (txCache), then what the draw reads per pixel: athenea's
/// `slopes` record, its `sharpPolish` record, and (centre x, centre y in
/// pixels as f32; halves: the centre's sharp reading the colour still holds,
/// rgb, and the mark: 0 none, 1 slope, 2 sharp).
public static const uint kRelightStride = 5;
public static const uint kRelightPixel = 2;   // the first of the three
public static const float kRelightKeepAll = -2.0;

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
    public uint  curvature;        ///< the curvature attribute (ATTRIB_NONE: absent)
    public float4 encoding;        ///< packed src: rgbMin, rgbMax, lnScaleMin, lnScaleMax
    public float4 row0;            ///< object to world, rows of a 3x4
    public float4 row1;
    public float4 row2;
    public float4 eyeObject;       ///< the eye in the cloud's space
    public float4 eyeWorld;
    public float4 view0;           ///< the cloud's space to the eye's (x right, y up, z depth ahead), rows
    public float4 view1;
    public float4 view2;
    public float4 focal;           ///< pixels: focal x, focal y (negative: rows go down), centre x, centre y
    public float4 viewport;        ///< width, height in pixels
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
    if (count < 9u || count == kTransferZonalCount || dome == kEnvNone || hasFlag(kRelightLit) ||
        !attribResident(relightPool, relightParams.transfer, i)) {
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

/// What the draw reads of a splat per pixel (kRelightPixel's three uint4).
struct RelightPixel {
    uint4 slopes;
    uint4 polish;
    uint4 centre;
};

RelightPixel noPixel() {
    RelightPixel p;
    p.slopes = uint4(0u);
    p.polish = uint4(0u);
    p.centre = uint4(0u);
    return p;
}

/// The curvature (xx, xy, yy) of splat i, where the cloud keeps it.
bool curvatureOf(uint i, out float3 k) {
    const uint id = relightParams.curvature;
    k = float3(0.0);
    if (!hasAttr(id) || !attribResident(relightPool, id, i)) {
        return false;
    }
    k = float3(f16tof32(readAttribU(relightPool, id, i, 0u)), f16tof32(readAttribU(relightPool, id, i, 1u)),
               f16tof32(readAttribU(relightPool, id, i, 2u)));
    return all(isfinite(k));
}

/// splat_project's projection of a gaussian, as far as SplatSlope needs it:
/// the linearisation a pixel right and down is (`tx3`, `ty3`, in the
/// cloud's space), the centre in pixels and the 2D covariance (a, b, c).
bool projectForSlope(Gsplat g, float3x3 r, out float3 tx3, out float3 ty3, out float2 centre, out float3 cov) {
    tx3 = float3(0.0);
    ty3 = float3(0.0);
    centre = float2(0.0);
    cov = float3(0.0);
    const float3x3 wv = float3x3(relightParams.view0.xyz, relightParams.view1.xyz, relightParams.view2.xyz);
    const float3 t = mul(wv, g.center) + float3(relightParams.view0.w, relightParams.view1.w, relightParams.view2.w);
    if (!(t.z > 1.0e-6)) {
        return false;
    }
    const float focalX = relightParams.focal.x;
    const float focalY = relightParams.focal.y;
    const float centreX = relightParams.focal.z;
    const float centreY = relightParams.focal.w;
    const float width = relightParams.viewport.x;
    const float height = relightParams.viewport.y;
    const float invZ = 1.0 / t.z;
    centre = float2(focalX * t.x * invZ + centreX, focalY * t.y * invZ + centreY);
    // splat_project's clamp of the linearisation to 1.3x the frustum.
    const float2 lo = (float2(-0.15 * width, -0.15 * height) - float2(centreX, centreY)) / float2(focalX, focalY);
    const float2 hi = (float2(1.15 * width, 1.15 * height) - float2(centreX, centreY)) / float2(focalX, focalY);
    const float tx = clamp(t.x * invZ, min(lo.x, hi.x), max(lo.x, hi.x)) * t.z;
    const float ty = clamp(t.y * invZ, min(lo.y, hi.y), max(lo.y, hi.y)) * t.z;
    const float3 jx = float3(focalX * invZ, 0.0, -focalX * tx * invZ * invZ);
    const float3 jy = float3(0.0, focalY * invZ, -focalY * ty * invZ * invZ);
    tx3 = mul(jx, wv);
    ty3 = mul(jy, wv);
    const float3 scale = g.scales;
    const float3x3 m = float3x3(r[0] * scale, r[1] * scale, r[2] * scale);
    const float3 ax = mul(tx3, m);
    const float3 ay = mul(ty3, m);
    cov = float3(dot(ax, ax), dot(ax, ay), dot(ay, ay));
    return true;
}

/// What a relit splat shows the eye (splat_project's projectSplat colour
/// path for a relit cloud, kTransfer 0, 1 or 2), and its alpha: athenea's
/// thin sheet held to kSheetAlpha with its colour scaled down to match, or
/// -1 where the splat keeps its own, -2 where it keeps its whole captured
/// colour (its transfer is not resident). `pixel`: what the draw reads per
/// pixel (splat_project's slope and sharp records).
float4 relitColour<let kTransfer : int>(uint i, out RelightPixel pixel) {
    pixel = noPixel();
    const Gsplat g = readSplat(i);
    if (!isActive(g.flags)) {
        return float4(0.0, 0.0, 0.0, -1.0);
    }
    // A paged pool without this splat's transfer page: as captured.
    if (kTransfer != 0 && !attribResident(relightPool, relightParams.transfer, i)) {
        return float4(0.0, 0.0, 0.0, kRelightKeepAll);
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
    // splat_project: WHICH WAY THE SURFACE FACES A PIXEL AWAY, from the
    // cloud's curvature (the shape operator in the splat's first two axes,
    // measured on the mesh's normal).
    float3 shapeUV;
    const bool curved = kTransfer >= 2 && curvatureOf(i, shapeUV);
    float3 cov = float3(0.0);
    float2 centre = float2(0.0);
    if (curved) {
        const float3 axisN = float3(r[0][2], r[1][2], r[2][2]);
        const float faces = dot(cloudNormal, axisN) >= 0.0 ? 1.0 : -1.0;
        slope.curvature = faces * 0.5 * (shapeUV.x + shapeUV.z);
    }
    float3 tx3, ty3;
    if (curved && hasFlag(kRelightSlope) && shadowWords() >= 8u && transferCount() >= 9u &&
        projectForSlope(g, r, tx3, ty3, centre, cov)) {
        const float4 w0 = relightParams.row0;
        const float4 w1 = relightParams.row1;
        const float4 w2 = relightParams.row2;
        const float3 u = float3(r[0][0], r[1][0], r[2][0]);
        const float3 v = float3(r[0][1], r[1][1], r[2][1]);
        const float2x2 toPixels = float2x2(dot(tx3, u), dot(tx3, v), dot(ty3, u), dot(ty3, v));
        const float det = toPixels[0][0] * toPixels[1][1] - toPixels[0][1] * toPixels[1][0];
        if (abs(det) > 1.0e-12) {
            const float2 stepX = float2(toPixels[1][1], -toPixels[1][0]) / det;
            const float2 stepY = float2(-toPixels[0][1], toPixels[0][0]) / det;
            const float2x2 shapeOf = float2x2(shapeUV.x, shapeUV.y, shapeUV.y, shapeUV.z);
            const float3 axis = float3(r[0][2], r[1][2], r[2][2]);
            const float side = dot(cloudNormal, axis) >= 0.0 ? 1.0 : -1.0;
            const float2 turnX = mul(shapeOf, stepX) * side;
            const float2 turnY = mul(shapeOf, stepY) * side;
            const float most = 0.5;
            const float3 dx = u * turnX.x + v * turnX.y;
            const float3 dy = u * turnY.x + v * turnY.y;
            const float3 hx = dx * min(1.0, most / max(length(dx), 1.0e-9));
            const float3 hy = dy * min(1.0, most / max(length(dy), 1.0e-9));
            slope.on = true;
            slope.normalX = relightNormalToWorld(w0, w1, w2, normalize(cloudNormal + hx));
            slope.normalY = relightNormalToWorld(w0, w1, w2, normalize(cloudNormal + hy));
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
    float sheetScale = 1.0;
    if (hasPbr && pbrThinWalled(pbr) && pbrTransmission(pbr) > 0.5) {
        const float alphaOwn = max(g.rgba.a - 1.0 / 255.0, 0.0);
        alpha = max(alphaOwn, 0.1);
        sheetScale = alphaOwn / alpha;
    }
    // splat_project's records for the blend. The colour keeps the centre's
    // reading of a sharp lobe (athenea takes it out at the projection; here
    // the draw takes it out, so the colour Spark carries is not negative).
    float mark = 0.0;
    float3 held = float3(0.0);
    if (kTransfer >= 2 && slope.sharpOn) {
        held = slope.sharpWeight * slope.sharpSeen + slope.sharpPolish * slope.sharpPolishSeen + slope.sharpSunCentre;
        const float2 at = octEncode(slope.sharpMirror);
        float2 stepX = octEncode(slope.sharpMirrorX) - at;
        float2 stepY = octEncode(slope.sharpMirrorY) - at;
        stepX = all(abs(stepX) < 0.25) ? stepX : float2(0.0);
        stepY = all(abs(stepY) < 0.25) ? stepY : float2(0.0);
        const float scale = sheetScale;
        pixel.slopes = uint4(packHalves(at.x, at.y), packHalves(stepX.x, stepX.y), packHalves(stepY.x, stepY.y),
                             packHalves(slope.sharpWeight * scale, slope.sharpRoughness));
        const float3 polish = slope.sharpPolish * scale;
        const float3 polishSun = slope.sharpSunPolish * scale;
        pixel.polish = uint4(packHalves(polish.x, polish.y), packHalves(polish.z, slope.sharpPolishRoughness),
                             packHalves(polishSun.x, polishSun.y), packHalves(polishSun.z, slope.sharpSunCoat * scale));
        mark = 2.0;
    } else if (kTransfer >= 2 && slope.on) {
        // splat_project: the ramp never goes below black inside the
        // footprint (its largest fall over three sigma is at most the colour).
        const float3 fall = 3.0 * sqrt(max(cov.x * slope.slopeX * slope.slopeX +
                                               2.0 * cov.y * slope.slopeX * slope.slopeY +
                                               cov.z * slope.slopeY * slope.slopeY,
                                           float3(0.0)));
        const float3 keep = min(float3(1.0), max(light, float3(0.0)) / max(fall, float3(1.0e-9)));
        const float3 sx = slope.slopeX * keep * sheetScale;
        const float3 sy = slope.slopeY * keep * sheetScale;
        pixel.slopes = uint4(packHalves(sx.x, sx.y), packHalves(sx.z, sy.x), packHalves(sy.y, sy.z), 0u);
        mark = 1.0;
    }
    light *= sheetScale;
    if (mark > 0.0) {
        held *= sheetScale;
        pixel.centre = uint4(asuint(centre.x), asuint(centre.y), packHalves(held.x, held.y), packHalves(held.z, mark));
    }
    return float4(light, alpha);
}

void relightAt<let kTransfer : int>(uint i) {
    if (i >= relightParams.count) {
        return;
    }
    RelightPixel pixel;
    relit[i * kRelightStride] = asuint(relitColour<kTransfer>(i, pixel));
    relit[i * kRelightStride + kRelightPixel] = pixel.slopes;
    relit[i * kRelightStride + kRelightPixel + 1u] = pixel.polish;
    relit[i * kRelightStride + kRelightPixel + 2u] = pixel.centre;
}

/// splat_project's catcherOpacity (a shadow catcher, athenea mesh2splat
/// --shadow-catcher): what the object took of the light reaching a ground
/// gaussian, as the opacity that takes as much out of what is drawn behind
/// it. Over the cells of its open directions, the sky's radiance from the
/// harmonics times the cosine and the cell's solid angle, with the object
/// (the open cells) and without it (every cell above the surface), and the
/// sun through the same bits on the first side alone.
static const float kCatcherDepth = 6.2831853;

float catcherOpacity(uint i, float3x3 r, float3 scale, float opacity) {
    const uint words = shadowWords();
    if (words < 8u) {
        return 0.0;
    }
    const uint dome = firstDome();
    if (dome == kEnvNone) {
        return 0.0;
    }
    const float3 n = relightNormalToWorld(relightParams.row0, relightParams.row1, relightParams.row2,
                                          storedFacing(r, scale, i));
    const uint side = splatCellSide(words);
    const uint base = cellBaseOf(i);
    const float3 luma = float3(0.2126, 0.7152, 0.0722);
    float sky[16];
    [unroll] for (uint k = 0; k < 16u; ++k) {
        sky[k] = dot(envSh[dome * kEnvCoefficients + k].rgb, luma);
    }
    float with = 0.0;
    float without = 0.0;
    for (uint cell = 0; cell < side * side; ++cell) {
        const float2 uv = (float2(float(cell % side), float(cell / side)) + 0.5) / float(side);
        const float3 d = octDecode(uv);
        const float cosine = dot(n, d);
        if (!(cosine > 0.0)) {
            continue;
        }
        float radiance = 0.0;
        [unroll] for (uint k = 0; k < 16u; ++k) {
            radiance += sky[k] * envBasisValue(k, d);
        }
        const float l1 = abs(d.x) + abs(d.y) + abs(d.z);
        const float worth = max(radiance, 0.0) * cosine * (4.0 / float(side * side)) * l1 * l1 * l1;
        without += worth;
        if (((relightPool[base + (cell >> 5u)] >> (cell & 31u)) & 1u) != 0u) {
            with += worth;
        }
    }
    const float4 disc = envSun[dome * 2];
    if (disc.w > 0.0) {
        const float facing = max(dot(n, disc.xyz), 0.0);
        const float sun = dot(envSun[dome * 2 + 1].rgb, luma) * facing;
        const float2 at = octEncode(disc.xyz) * float(side) - 0.5;
        const float2 lo = floor(at);
        const float2 f = at - lo;
        float open = 0.0;
        float weight = 0.0;
        [unroll] for (uint k = 0; k < 4; ++k) {
            const int2 c = clamp(int2(lo) + int2(k & 1, k >> 1), int2(0), int2(int(side) - 1));
            const float wk = ((k & 1) != 0 ? f.x : 1.0 - f.x) * ((k >> 1) != 0 ? f.y : 1.0 - f.y);
            if (wk <= 0.0 || dot(octDecode((float2(c) + 0.5) / float(side)), n) <= 0.0) {
                continue;
            }
            const uint index = uint(c.y) * side + uint(c.x);
            open += wk * float((relightPool[base + (index >> 5u)] >> (index & 31u)) & 1u);
            weight += wk;
        }
        with += sun * (weight > 0.0 ? open / weight : 1.0);
        without += sun;
    }
    const float ratio = saturate(with / max(without, 1.0e-6));
    return opacity * saturate(-log(max(ratio, 1.0e-4)) / kCatcherDepth);
}

/// A shadow catcher's splats: black, as opaque as what the object took.
// A cloud over 65 535 workgroups wide (16 776 960 splats, a 256-page pool)
// is dispatched as rows of that many threads: the splat is x + y * row.
public static const uint kRelightDispatchRow = 65535u * 256u;
uint relightIndex(uint3 tid) { return tid.x + tid.y * kRelightDispatchRow; }

[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaRelightCatcher(uint3 tid: SV_DispatchThreadID) {
    const uint i = relightIndex(tid);
    if (i >= relightParams.count) {
        return;
    }
    const Gsplat g = readSplat(i);
    const float alpha = isActive(g.flags) ? catcherOpacity(i, splatRotation(g.quaternion), g.scales, g.rgba.a) : 0.0;
    relit[i * kRelightStride] = asuint(float4(0.0, 0.0, 0.0, alpha));
    relit[i * kRelightStride + kRelightPixel + 2u] = uint4(0u);
}

/// The kept terms (splatTransferViewless), when the sky, the lights, the
/// cloud or its place changed.
[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaRelightViewless(uint3 tid: SV_DispatchThreadID) {
    const uint i = relightIndex(tid);
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
    relightAt<2>(relightIndex(tid));
}

/// The first transfer (no cells).
[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaRelightFirst(uint3 tid: SV_DispatchThreadID) {
    relightAt<1>(relightIndex(tid));
}

/// No transfer: the lights and the dome's harmonics and map alone.
[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaRelightPlain(uint3 tid: SV_DispatchThreadID) {
    relightAt<0>(relightIndex(tid));
}

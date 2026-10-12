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
import athenea.common.transfer_layout;
import athenea.scene.splat_encoding;

public static const uint kRelightExt = 1u;         // src is ext (else packed)
public static const uint kRelightLodOpacity = 2u;  // packed src keeps alpha / 2
public static const uint kRelightLinear = 4u;      // stored colours are linear light (else sRGB)
public static const uint kRelightLit = 8u;         // the body is baked radiance (athenea's litBody)
public static const uint kRelightIndirect = 16u;   // use the transfer's indirect half
public static const uint kRelightCache = 32u;      // relit reads the kept view-independent terms
public static const uint kRelightSlope = 64u;      // the viewport is known: per-pixel slope and sharp lobes
public static const uint kRelightNoCells = 128u;   // debug: the open-direction cells are not read
public static const uint kRelightDebugPlace = 512u;      // debug view: a smooth colour of the place
public static const uint kRelightDebugNormal = 1024u;    // debug view: the shading normal
public static const uint kRelightDebugFootprint = 2048u; // debug view: the footprint's spread, the coat roughness
public static const uint kRelightCullBacks = 4096u;      // opaque splats whose stored face looks away are not drawn
public static const uint kRelightFootprintPixel = 8192u; // the footprint prefilter knows the frame: its pixel term
public static const uint kRelightCpca = 16384u;          // the transfer is rebuilt in envOfLight's scratch (cpca.slang)
// athenea's joint CPCA under a dome alone (proposal 094, cpca_joint.slang):
// the kept terms from the clusters' projections (jointTable) and the
// gaussian's int8 weights, no transfer rebuilt; the eye pass reads only the
// openness and the reflected block, which cpca_joint.slang rebuilt.
public static const uint kRelightJoint = 32768u;
public static const uint kRelightSheetDraw = 65536u;     // thin sheets marked (kRelightSheet): the draw floors them after its AA
// A merged node's second opacity (.athc LODM, a.w's high half of an ext
// source) while it is at most opacitySwitchPx on screen: generate's
// GEN_OPACITY_SWITCH, the same test, so a sheet's alpha is the one drawn.
public static const uint kRelightOpacitySwitch = 131072u;
// athenea-pieces b26d276 / 35ae3cb (slang/athenea_patches/0001): the bake
// widened each gaussian's roughness by the surface's turn across it
// (mesh2splat's m2sSpread, Toksvig: alpha'^2 = alpha^2 + 2 spread^2, spread =
// the shape operator's largest principal curvature times the wider size).
// kRelightUnwiden: the sharp per-pixel lobes (kRelightSlope) read the sky at
// the roughness before that widening, their pixel's footprint term added on
// top. kRelightCoatFold: a coat's widening held to kCoatFoldRoughness (0.3)
// where the spread is large. kRelightPolishField: the material's own
// roughness (pbr bits 26..31) is read, and a polish at least
// kPolishFieldRoughness rough has the reflected field answer its closed part.
public static const uint kRelightUnwiden = 262144u;
public static const uint kRelightCoatFold = 524288u;
public static const uint kRelightPolishField = 1048576u;
public static const uint kRelightGroup = 256;
/// Five uint4 a splat in `relit`: the colour (asuint float4: linear rgb,
/// alpha, -1 to keep the splat's alpha, -2 to keep its whole rgba), the
/// kept terms (txCache), then what the draw reads per pixel: athenea's
/// `slopes` record, its `sharpPolish` record, and (centre x, centre y in
/// pixels as f32; halves: the centre's sharp reading the colour still holds,
/// rgb, and the mark: 0 none, 1 slope, 2 sharp).
// A splat's records in `relit`: its colour, then the kept terms (when
// kRelightCache can be set), then the three the draw reads per pixel (only
// while it does: pixelDetail). relightParams.stride is how many a buffer
// holds; the host sizes it to what the options need (a phone's paged pool
// with neither keeps one uint4 a splat, not five).
public static const uint kRelightStride = 5;       // the most: colour, kept, the draw's three
public static const uint kRelightStrideKept = 2;   // colour and kept terms
public static const uint kRelightStrideColour = 1; // the colour alone
public static const uint kRelightPixel = 2;   // the first of the three
public static const float kRelightKeepAll = -2.0;
/// A thin sheet's colour word under kRelightSheetDraw: alpha kRelightSheet +
/// its alpha (held to kSheetAlpha, the colour scaled to match). Without the
/// flag the sheet's alpha alone, and the draw's compensation then takes it
/// under kSheetAlpha (mass kept: a grazing or distant sheet lets more
/// through than athenea's). The colour plugin hands it on
/// as a negative alpha, generate marks the splat as a sheet in the
/// accumulator, and the draw floors the alpha at kSheetAlpha again AFTER
/// its anti-alias compensation (athenea's splat_project order).
public static const float kRelightSheet = 8.0;

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
    public float footprint;        ///< the footprint prefilter's gain (0: none; footprintRoughness)
    public uint  listCount;        ///< > 0: relight only these splats, the list at listBase (below)
    public uint  listBase;         ///< uint4 index in `relit` of the list (four splat indices each)
    public uint  threadFirst;      ///< the dispatch's first thread (a batch of the list or of the cloud)
    public uint  cpcaScratch;      ///< kRelightCpca: where the batch's transfer starts in envOfLight, words
    public uint  cpcaWords;        ///< kRelightCpca: words a slot of it
    public uint  jointTable;       ///< kRelightJoint: where the clusters' projections start in envOfLight, words
    public float opacitySwitchPx;  ///< kRelightOpacitySwitch: the size on screen (2 max sigma, px) up to which
                                   ///< a merged node takes its second opacity
    public uint  stride;           ///< uint4 records a splat in `relit` (kRelightStride*): 1 the colour
                                   ///< alone, 2 with the kept terms, 5 with the draw's three as well
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
    public float4 filter;          ///< x: the footprint's clamp (kappa), y: the facing below which a back is
                                   ///< culled (kRelightCullBacks), z: the pixel filter's variance (px^2),
                                   ///< w: a solid lens's cover (glassCover; at most 1: athenea's alpha)
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
        Gsplat e = readExtSplat(relightSrc[2u * i], relightSrc[2u * i + 1u], int(i));
        if (hasFlag(kRelightOpacitySwitch)) {
            const float second = unpackHalf2x16(relightSrc[2u * i].w).y;
            const float sigma = max(e.scales.x, max(e.scales.y, e.scales.z));
            const float dist = max(length(e.center - relightParams.eyeObject.xyz), 1.0e-6);
            if (second > 0.0 && 2.0 * sigma * abs(relightParams.focal.y) <= relightParams.opacitySwitchPx * dist) {
                e.rgba.a = second;
            }
        }
        return e;
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

/// The thread's slot in its batch (threadFirst on): where cpca.slang left
/// its transfer in envOfLight under kRelightCpca.
static uint gSlot = 0u;

/// Where a gaussian's transfer starts in the pool, in halves (under
/// kRelightCpca, in envOfLight's scratch: the buffer the kCpca variants
/// hand athenea as its transfer).
uint transferBaseOf(uint i) {
    if (hasFlag(kRelightCpca)) {
        return (relightParams.cpcaScratch + gSlot * relightParams.cpcaWords) * 2u;
    }
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
    return hasAttr(relightParams.shadowBits) && !hasFlag(kRelightNoCells) ? relightParams.shadowWords : 0u;
}

/// The material's own roughness (athenea's pbrMaterialRoughness, pbr bits
/// 26..31), -1 where the cloud does not carry it.
float ownRoughnessOf(uint i) {
    return hasAttr(relightParams.pbr) ? pbrMaterialRoughness(poolWord(relightParams.pbr, i, 0u)) : -1.0;
}

/// The surface splat_project's relitToward builds, from this mesh's buffers.
SplatSurface surfaceOf<let kCpca : int>(uint i, Gsplat g, float3x3 r, float3 cloudNormal, float3 colour) {
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
    surface.materialRoughness = hasPbr && hasFlag(kRelightPolishField) ? pbrMaterialRoughness(pbr) : -1.0;
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
    if (kCpca != 0) {
        splatTransferFrame(surface, envOfLight, r, w0, w1, w2);
    } else {
        splatTransferFrame(surface, relightPool, r, w0, w1, w2);
    }
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
uint4 viewless<let kCpca : int>(uint i, Gsplat g) {
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
    if (kCpca == 2) {
        jointViewless(s, i, dome, hasFlag(kRelightIndirect), front, coupling, sunShare);
    } else if (kCpca != 0) {
        transferViewless<2>(s, envOfLight, relightPool, envSh, envSun, dome, hasFlag(kRelightIndirect), front,
                            coupling, sunShare);
    } else {
        transferViewless<2>(s, relightPool, relightPool, envSh, envSun, dome, hasFlag(kRelightIndirect), front,
                            coupling, sunShare);
    }
    return uint4(packHalves(front.x, front.y), packHalves(front.z, coupling.x),
                 packHalves(coupling.y, coupling.z), packHalves(sunShare, float(dome + 1u)));
}

/// Term `term` of a cluster's projections (cpca_joint.slang
/// atheneaJointProject) for a block's 8 int8 weights (words w0, w1).
float3 jointTerm(uint cluster, uint term, uint w0, uint w1) {
    const uint at = relightParams.jointTable + cluster * 144u + term * 36u;
    float3 v = float3(asfloat(envOfLight[at]), asfloat(envOfLight[at + 1u]), asfloat(envOfLight[at + 2u]));
    [unroll] for (uint k = 0; k < 8u; ++k) {
        const uint w = k < 4u ? w0 : w1;
        const float q = float(int(w << (24u - 8u * (k & 3u))) >> 24);
        const uint d = at + 4u * (1u + k);
        v += q * float3(asfloat(envOfLight[d]), asfloat(envOfLight[d + 1u]), asfloat(envOfLight[d + 2u]));
    }
    return v;
}

/// transferViewless (a TX transfer with cells) from the joint CPCA: the
/// body's two halves, the sun's bounce and the white sky's gathering out of
/// the cluster's projections, the rest as athenea writes it.
void jointViewless(SplatSurface s, uint i, uint dome, bool withIndirect, out float3 front, out float3 coupling,
                   out float sunShare) {
    const uint base = attribBase(relightPool, attribDesc(relightPool, relightParams.transfer), i);
    const uint c = relightPool[base + 6u];
    const uint i0 = relightPool[base + 2u];
    const uint i1 = relightPool[base + 3u];
    front = jointTerm(c, 0u, relightPool[base], relightPool[base + 1u]);
    const float3 gathered = jointTerm(c, 1u, i0, i1);
    if (withIndirect) {
        front += gathered;
    }
    const float4 disc = envSun[dome * 2];
    const float3 irradiance = disc.w > 0.0 ? envSun[dome * 2 + 1].rgb : float3(0.0);
    const float cosine = max(dot(s.normal, disc.xyz), 0.0);
    sunShare = disc.w > 0.0 ? splatOpenToward(s, relightPool, relightPool, disc.xyz) : 1.0;
    front += irradiance * (cosine * sunShare * kOverPiSun);
    float3 along = float3(0.0);
    if (withIndirect && disc.w > 0.0) {
        along = max(jointTerm(c, 2u, i0, i1), float3(0.0));
        front += irradiance * along;
    }
    coupling = float3(0.0);
    if (withIndirect && transferHasField(s.transferCount)) {
        float3 all = gathered;
        float3 mean = envSh[dome * kEnvCoefficients].rgb * 0.28209479177387814;
        if (disc.w > 0.0) {
            all += irradiance * along;
            mean += irradiance * (1.0 / (4.0 * 3.14159265358979));
        }
        const float3 white = jointTerm(c, 3u, i0, i1) * 3.5449077018110318;
        coupling = float3(white.x > 1.0e-4 ? max(all.x / white.x, 0.0) : mean.x,
                          white.y > 1.0e-4 ? max(all.y / white.y, 0.0) : mean.y,
                          white.z > 1.0e-4 ? max(all.z / white.z, 0.0) : mean.z);
    }
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

/// The variance of a merged splat's normals (the curvature's fourth half;
/// 0 for a splat, or a cloud written without it).
float normalSpreadOf(uint i) {
    const uint id = relightParams.curvature;
    return hasAttr(id) ? f16tof32(readAttribU(relightPool, id, i, 3u)) : 0.0;
}

/// How far the bake widened a gaussian's roughness (athenea's
/// slope.bakeSpread, splat_project): the largest principal curvature of the
/// shape operator (xx, xy, yy) times the wider of the first two sizes.
public float bakeSpreadOf(float3 shapeUV, float3 scales) {
    const float a = shapeUV.x;
    const float b = shapeUV.y;
    const float c = shapeUV.z;
    const float kMax = abs(0.5 * (a + c)) + sqrt(0.25 * (a - c) * (a - c) + b * b);
    const float s = kMax * max(scales.x, scales.y);
    return isfinite(s) ? s : 0.0;
}

/// splat_project's projection of a gaussian, as far as SplatSlope needs it:
/// the linearisation a pixel right and down is (`tx3`, `ty3`, in the
/// cloud's space), the centre in pixels and the 2D covariance (a, b, c)
/// after splat_project's third of a pixel of dilation, which is what the
/// slope's fall is held by (without it a footprint of a pixel or two kept
/// half as much again of a crease's ramp as athenea's).
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
    cov = float3(dot(ax, ax) + 0.3, dot(ax, ay), dot(ay, ay) + 0.3);
    return true;
}

/// THE FOOTPRINT PREFILTER: a splat is shaded once, at its centre, and
/// drawn as one colour over its whole gaussian; on a curved panel the normal
/// turns across that gaussian, so a sharp reflection (the horizon in the
/// paint's coat) is point-sampled once a splat, and where it changes faster
/// than the splats are spaced the draw shows the splats' layout as a fixed
/// moire (the light Corvette's 9 mm merged cells: the doors' reflected bands
/// combed across, 2.5 px apart at 4k). So the lobes are filtered over the
/// spread of the normal, as geometric specular anti-aliasing does over a
/// pixel (Kaplanyan et al. 2016, Tokuyoshi & Kaplanyan 2019): in slope
/// space the NDF is convolved with the footprint's slope distribution, and
/// for GGX/Beckmann that adds gain^2 times its variance to alpha^2 (alpha =
/// roughness^2; filtered GGX's gain is sqrt(2)), clamped by kappa (theirs
/// 0.18: a crease widens to a rough lobe, not a diffuse one), alpha^2 at
/// most 1.
///
/// The variance is the largest of three footprints': the splat's (`shapeUV`,
/// the shape operator S in its first two axes: the normal's variance over
/// the gaussian is su^2 |S e_u|^2 + sv^2 |S e_v|^2); a merged gaussian's
/// members' normals (`spread`, LEAN's second moment: .athc's curvature
/// fourth half, athc.rs store_normal_variance), which a merged cell over a
/// crease has and the curvature does not see; and the pixel's
/// (`pixelVariance`: the pixel filter's variance times |dn/dx|^2 +
/// |dn/dy|^2, what a splat smaller than a pixel stands for; 0 when the frame
/// is not known). A splat small against the curvature and the pixel (the
/// detailed cloud close up) keeps its roughness.
public float footprintRoughness(float roughness, float3 shapeUV, float3 scales, float gain, float kappa,
                                float spread, float pixelVariance) {
    const float su = scales.x;
    const float sv = scales.y;
    const float splatVariance = su * su * (shapeUV.x * shapeUV.x + shapeUV.y * shapeUV.y) +
                                sv * sv * (shapeUV.y * shapeUV.y + shapeUV.z * shapeUV.z);
    const float variance = max(max(splatVariance, isfinite(spread) ? spread : 0.0),
                               max(isfinite(pixelVariance) ? pixelVariance : 0.0, 0.0));
    const float alpha = roughness * roughness;
    const float add = min(gain * gain * variance, kappa);
    const float widened = min(alpha * alpha + add, 1.0);
    const float r = sqrt(sqrt(widened));
    return isfinite(r) ? max(r, roughness) : roughness;
}

/// The same without a clamp (kappa 1) or a pixel term: AM's form, `gain`
/// in quadrature.
public float footprintRoughness(float roughness, float3 shapeUV, float3 scales, float gain) {
    return footprintRoughness(roughness, shapeUV, scales, gain, 1.0, 0.0, 0.0);
}

/// The same with the merged normals' variance (`spread`).
public float footprintRoughness(float roughness, float3 shapeUV, float3 scales, float gain, float spread) {
    return footprintRoughness(roughness, shapeUV, scales, gain, 1.0, spread, 0.0);
}

/// How the normal turns a pixel right and a pixel down (in the cloud's
/// space, from the shape operator), splat_project's SplatSlope steps:
/// false where the splat is edge-on to the eye.
bool normalTurnPerPixel(float3x3 r, float3 shapeUV, float3 cloudNormal, float3 tx3, float3 ty3, out float3 dx,
                        out float3 dy) {
    dx = float3(0.0);
    dy = float3(0.0);
    const float3 u = float3(r[0][0], r[1][0], r[2][0]);
    const float3 v = float3(r[0][1], r[1][1], r[2][1]);
    const float2x2 toPixels = float2x2(dot(tx3, u), dot(tx3, v), dot(ty3, u), dot(ty3, v));
    const float det = toPixels[0][0] * toPixels[1][1] - toPixels[0][1] * toPixels[1][0];
    if (!(abs(det) > 1.0e-12)) {
        return false;
    }
    const float2 stepX = float2(toPixels[1][1], -toPixels[1][0]) / det;
    const float2 stepY = float2(-toPixels[0][1], toPixels[0][0]) / det;
    const float2x2 shapeOf = float2x2(shapeUV.x, shapeUV.y, shapeUV.y, shapeUV.z);
    const float3 axis = float3(r[0][2], r[1][2], r[2][2]);
    const float side = dot(cloudNormal, axis) >= 0.0 ? 1.0 : -1.0;
    const float2 turnX = mul(shapeOf, stepX) * side;
    const float2 turnY = mul(shapeOf, stepY) * side;
    dx = u * turnX.x + v * turnX.y;
    dy = u * turnY.x + v * turnY.y;
    return true;
}

/// THE RUNTIME BACK-FACE REJECT (kRelightCullBacks): a solidified panel is
/// baked as a shell with two faces (the Corvette's paint: 3 mm, the inner
/// face dark), and where the splats are as wide as the shell is thick the
/// centre sort interleaves them, so the inner face shows through the outer
/// one as a moire (what usd-athc --drop-backs, athc_build::drop_hidden_backs,
/// removes offline). An opaque splat (no transmission, not a thin sheet)
/// whose stored face -- the one its transfer was measured for, the mesh's
/// normal where the cloud keeps it -- looks away from the eye by more than
/// relightParams.filter.y (a cosine, below 0: some slack for normals that
/// merged cells over a crease turned) is a back face of a closed surface,
/// hidden by the front one: it is not drawn (alpha 0; generate's cull then
/// gives it no sort slot). athenea draws it (its backFacing only bends glass).
bool culledBack(uint i, Gsplat g, float3x3 r) {
    if (hasAttr(relightParams.pbr)) {
        const uint pbr = poolWord(relightParams.pbr, i, 0u);
        if (pbrTransmission(pbr) > 0.0 || pbrThinWalled(pbr)) {
            return false;
        }
    }
    const float3 toEye = relightParams.eyeObject.xyz - g.center;
    const float far = length(toEye);
    if (!(far > 0.0)) {
        return false;
    }
    return dot(storedFacing(r, g.scales, i), toEye / far) < relightParams.filter.y;
}

/// A solid glass's alpha under glassCover `n`: n of its gaussians stacked,
/// as Spark's LoD opacity draws a coverage past 1 (splat_shape's splatAlpha:
/// 1 - (1 - g)^ratio, wider than the gaussian). Its own coverage is kept in
/// it: a splat's opacity, or a merged node's ratio past 1 (the LoD opacity
/// the decoder gave it, athc::spark_lod_opacity). Holding a merged node to a
/// plain gaussian under 1 instead opened the gaps between the cells of a
/// level: drawn from its merged levels 4-5 (the pawn a little way off), the
/// head let 3-5% of the room straight through, a bright window enough.
public float coveredGlassAlpha(float a, float n) {
    const float e = 2.718281828459045;
    float ratio = a;
    if (a > 1.0) {
        const float d = min(4.0 * a - 3.0, 5.0);
        ratio = exp((d * d - 1.0) / e);
    }
    const float covered = n * ratio;
    if (covered <= 1.0) {
        return 1.0 - pow(max(1.0 - min(a, 1.0), 0.0), n);
    }
    const float d = min(sqrt(1.0 + e * log(covered)), 5.0);
    return 1.0 + 0.25 * (d - 1.0);
}

/// What a relit splat shows the eye (splat_project's projectSplat colour
/// path for a relit cloud, kTransfer 0, 1 or 2), and its alpha: athenea's
/// thin sheet held to kSheetAlpha with its colour scaled down to match, or
/// -1 where the splat keeps its own, -2 where it keeps its whole captured
/// colour (its transfer is not resident). `pixel`: what the draw reads per
/// pixel (splat_project's slope and sharp records).
float4 relitColour<let kTransfer : int, let kCpca : int>(uint i, out RelightPixel pixel) {
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
    if (hasFlag(kRelightCullBacks) && culledBack(i, g, r)) {
        return float4(0.0);
    }
    const float3 colour = cloudLight(g.rgba.rgb, hasFlag(kRelightLinear));
    const bool hasNormals = hasAttr(relightParams.normalOct);
    const float3 cloudNormal = splatShadingNormal(r, g.scales, g.center, relightParams.eyeObject.xyz, hasNormals,
                                                  hasNormals ? poolWord(relightParams.normalOct, i, 0u) : 0u);
    SplatSurface surface = surfaceOf<kCpca>(i, g, r, cloudNormal, colour);
    SplatSlope slope = noSlope();
    if (kTransfer >= 2 && hasFlag(kRelightCache) && surface.transferCount != kTransferZonalCount &&
        dot(cloudNormal, storedFacing(r, g.scales, i)) > 0.0) {
        const uint4 kept = relit[i * relightParams.stride + 1u];
        if (kept.w != 0u) {
            slope.cached = true;
            slope.cachedDome = uint(highHalf(kept.w) + 0.5) - 1u;
            slope.cachedFront = float3(lowHalf(kept.x), highHalf(kept.x), lowHalf(kept.y));
            slope.cachedCoupling = float3(highHalf(kept.y), lowHalf(kept.z), highHalf(kept.z));
            slope.cachedSunShare = lowHalf(kept.w);
        }
    }
    // The joint CPCA keeps no direct or indirect block for this pass: what
    // relitSplat would compute from them (transferViewless, for a splat the
    // kept terms do not cover) comes from the clusters' projections.
    if (kCpca == 2 && kTransfer >= 2 && !slope.cached) {
        const uint dome = firstDome();
        if (dome != kEnvNone && !surface.lit) {
            float3 front;
            float3 coupling;
            float sunShare;
            jointViewless(surface, i, dome, hasFlag(kRelightIndirect), front, coupling, sunShare);
            slope.cached = true;
            slope.cachedDome = dome;
            slope.cachedFront = front;
            slope.cachedCoupling = coupling;
            slope.cachedSunShare = sunShare;
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
    // The frame's projection, where the pass knows it: the per-pixel slope
    // (kRelightSlope) and the footprint prefilter's pixel term.
    float3 tx3, ty3;
    const bool projected = curved && (hasFlag(kRelightSlope) || hasFlag(kRelightFootprintPixel)) &&
                           projectForSlope(g, r, tx3, ty3, centre, cov);
    float3 dx = float3(0.0);
    float3 dy = float3(0.0);
    const bool turns = projected && normalTurnPerPixel(r, shapeUV, cloudNormal, tx3, ty3, dx, dy);
    // The bake's spread (splat_project's slope.bakeSpread): the shape
    // operator's largest principal curvature times the wider size.
    const float spread = curved ? bakeSpreadOf(shapeUV, g.scales) : 0.0;
    // The roughnesses as stored (the bake's), before the footprint
    // prefilter: the sharp lobes are unwidened from these.
    const float storedPolish = surface.roughness;
    const float storedCoat = surface.lobes.coatRoughness;
    // The pixel's own turn (athenea's splat_project pixelTurn2), with the
    // fold's hold: what widens the sharp lobes per pixel.
    float sharpPixelTurn2 = 0.0;
    // The footprint prefilter: the lobes widened by how the normal turns
    // over what the splat's one colour stands for. Without the per-pixel
    // slope that is its gaussian, or the pixel where the splat is smaller;
    // with it (the draw reads the sharp lobes per pixel) the pixel alone.
    if (curved && relightParams.footprint > 0.0) {
        const float gain = relightParams.footprint;
        const float kappa = relightParams.filter.x;
        const float pixelVariance = turns ? relightParams.filter.z * (dot(dx, dx) + dot(dy, dy)) : 0.0;
        const bool splatTerms = !hasFlag(kRelightSlope);
        const float3 shape = splatTerms ? shapeUV : float3(0.0);
        const float spread = splatTerms ? normalSpreadOf(i) : 0.0;
        if (splatTerms || pixelVariance > 0.0) {
            surface.roughness =
                footprintRoughness(surface.roughness, shape, g.scales, gain, kappa, spread, pixelVariance);
            surface.lobes.coatRoughness =
                footprintRoughness(surface.lobes.coatRoughness, shape, g.scales, gain, kappa, spread, pixelVariance);
        }
    }
    // A FOLD'S COAT (athenea's kCoatFoldRoughness, splat_project): the
    // bake's widening held to 0.3 where the surface turns faster than a
    // gaussian is wide. After the footprint prefilter, so the prefilter's
    // own widening is held too (before it, the splat term re-widened a fold
    // past 0.3 again without pixelDetail).
    // Only on a lacquer whose material is as rough as a paint's
    // (kPolishFieldRoughness, its own roughness off the pbr word, not the
    // stored one; athenea-pieces' splat_project): on chrome, glass and the
    // rough metals, already at or past the path trace, it lit what the trace
    // shows dark (thread CV, by region: chrome -1.4 dB, rim spokes -0.5). A
    // cloud without the mark keeps its coat.
    if (curved && hasFlag(kRelightCoatFold) && surface.lobes.coatWeight > 0.0 &&
        ownRoughnessOf(i) >= kPolishFieldRoughness) {
        const float coat = surface.lobes.coatRoughness;
        surface.lobes.coatRoughness = max(unwidened(coat, spread), min(coat, kCoatFoldRoughness));
    }
    if (turns && hasFlag(kRelightSlope) && shadowWords() >= 8u && transferCount() >= 9u) {
        const float4 w0 = relightParams.row0;
        const float4 w1 = relightParams.row1;
        const float4 w2 = relightParams.row2;
        {
            // A TUBE TURNS, A FOLD WIDENS (athenea f9c9a2b, splat_project): t,
            // the turn across +-2 sigma, is twice the bake's spread; on a
            // surface's own curvature the pixels turn in full, on normals
            // smoothed across a fold the turn goes and the sharp lobes widen
            // by it (foldWidened, below).
            const float held = 1.0 - smoothstep(kFoldTurnLow, kFoldTurnHigh, 2.0 * spread);
            sharpPixelTurn2 = held * held * (dot(dx, dx) + dot(dy, dy));
            const float most = 0.5;
            const float3 hx = held * dx * min(1.0, most / max(held * length(dx), 1.0e-9));
            const float3 hy = held * dy * min(1.0, most / max(held * length(dy), 1.0e-9));
            slope.on = true;
            slope.normalX = relightNormalToWorld(w0, w1, w2, normalize(cloudNormal + hx));
            slope.normalY = relightNormalToWorld(w0, w1, w2, normalize(cloudNormal + hy));
        }
    }
    // No shadow was measured: relightParams' shadow lights are 0, and the
    // factors are never read (iesValues stands in for the buffer).
    float3 light;
    if (kCpca != 0) {
        light = relitSplat<kTransfer>(surface, normalize(relightParams.eyeWorld.xyz - surface.position),
                                      relightLights, relightParams.lightCount, 0xFFFFFFFFu, 0xFFFFFFFFu, iesValues,
                                      0u, 0u, envTexels, envSh, envOfLight, relightParams.envLights,
                                      relightParams.envBaseSide, envSun, envOfLight, relightPool,
                                      hasFlag(kRelightIndirect), slope);
    } else {
        light = relitSplat<kTransfer>(surface, normalize(relightParams.eyeWorld.xyz - surface.position),
                                      relightLights, relightParams.lightCount, 0xFFFFFFFFu, 0xFFFFFFFFu, iesValues,
                                      0u, 0u, envTexels, envSh, envOfLight, relightParams.envLights,
                                      relightParams.envBaseSide, envSun, relightPool, relightPool,
                                      hasFlag(kRelightIndirect), slope);
    }
    // THE SHARP LOBES, UNWIDENED (athenea's `unwidened`): they are drawn
    // per pixel along a mirror that already turns with the surface, so the
    // bake's widening by that turn counted it twice. Read at the stored
    // roughness less the bake's spread, with only the pixel's own footprint
    // term on top -- which is what the per-pixel mirror does not resolve --
    // and not the stored roughness's widening unwidened (the same unless the
    // spread passes the stored roughness).
    if (kTransfer >= 2 && slope.sharpOn && hasFlag(kRelightUnwiden) && spread > 0.0) {
        if (slope.sharpWeight > 0.0) {
            slope.sharpRoughness = unwidened(min(storedCoat, slope.sharpRoughness), spread);
        }
        if (any(slope.sharpPolish > float3(0.0))) {
            slope.sharpPolishRoughness = unwidened(min(storedPolish, slope.sharpPolishRoughness), spread);
        }
    }
    // A FOLD WIDENS the sharp lobes by the turn the pixels no longer take
    // (athenea f9c9a2b: alpha' = sqrt(alpha^2 + w (t/2)^2), held under
    // kFoldRoughness), after the unwidening; then THE PIXEL'S OWN SPREAD, as
    // athenea's splat_project: the read is at the pixel's centre and the
    // normal turns by dx, dy across it, a box of slopes whose variance, twice
    // for the mirror, widens the lobe: alpha^2 += 2 (|dx|^2 + |dy|^2) / 12
    // (the compact path's, the same).
    if (kTransfer >= 2 && slope.sharpOn) {
        const float turn = 2.0 * spread;
        const float widen = spread > 0.0 ? smoothstep(kFoldTurnLow, kFoldTurnHigh, turn) * 0.25 * turn * turn : 0.0;
        if (slope.sharpWeight > 0.0) {
            const float a = foldWidened(slope.sharpRoughness, widen);
            slope.sharpRoughness = sqrt(sqrt(min(a * a * a * a + sharpPixelTurn2 / 6.0, 1.0)));
        }
        if (any(slope.sharpPolish > float3(0.0))) {
            const float a = foldWidened(slope.sharpPolishRoughness, widen);
            slope.sharpPolishRoughness = sqrt(sqrt(min(a * a * a * a + sharpPixelTurn2 / 6.0, 1.0)));
        }
    }
    float alpha = -1.0;
    bool sheet = false;
    const bool hasPbr = hasAttr(relightParams.pbr);
    const uint pbr = hasPbr ? poolWord(relightParams.pbr, i, 0u) : 0u;
    // splat_project's thin sheet, with Spark's alpha as the paid opacity.
    float sheetScale = 1.0;
    if (hasPbr && pbrThinWalled(pbr) && pbrTransmission(pbr) > 0.5) {
        // A merged sheet's coverage past 1 (LoD opacity) is drawn as one
        // sheet: the thin sheet's opacity is at most 1.
        const float alphaOwn = max(min(g.rgba.a, 1.0) - 1.0 / 255.0, 0.0);
        alpha = max(alphaOwn, 0.1);
        sheetScale = alphaOwn / alpha;
        sheet = true;
    } else if (hasPbr && !pbrThinWalled(pbr) && pbrTransmission(pbr) > 0.5 && relightParams.filter.w > 1.0) {
        // A solid glass covered whole (glassCover): what it lets through is
        // its lens, not the room straight behind it.
        alpha = coveredGlassAlpha(g.rgba.a, relightParams.filter.w);
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
    // Debug views (AtheneaRelightOptions.debug), in place of the light: what
    // the draw makes of a smooth colour, the normals, and the footprint's
    // spread (red: the roughness a gain of 1 adds to a mirror, green: the
    // coat's own roughness; blue: no curvature).
    if (hasFlag(kRelightDebugPlace)) {
        const float3 p = surface.position;
        light = float3(0.01 * (1.5 + sin(9.0 * p.x + 7.0 * p.y + 5.0 * p.z)));
    } else if (hasFlag(kRelightDebugNormal)) {
        light = 0.02 * (surface.normal * 0.5 + 0.5);
    } else if (hasFlag(kRelightDebugFootprint)) {
        light = curved ? 0.02 * float3(footprintRoughness(0.0, shapeUV, g.scales, 1.0, normalSpreadOf(i)),
                                       surface.lobes.coatRoughness, 0.0)
                       : float3(0.0, 0.0, 0.02);
    }
    if (mark > 0.0) {
        held *= sheetScale;
        pixel.centre = uint4(asuint(centre.x), asuint(centre.y), packHalves(held.x, held.y), packHalves(held.z, mark));
    }
    return float4(light, sheet && hasFlag(kRelightSheetDraw) ? kRelightSheet + alpha : alpha);
}

/// The splat thread t relights: the t-th of the list after the records
/// (listCount > 0: the splats a LoD cut draws, the host's copy of the
/// mesh's lodIndices), else splat t; count when there is none.
uint relightTarget(uint t) {
    if (relightParams.listCount == 0u) {
        return t;
    }
    if (t >= relightParams.listCount) {
        return relightParams.count;
    }
    return relit[relightParams.listBase + t / 4u][t % 4u];
}

void relightAt<let kTransfer : int, let kCpca : int>(uint t) {
    gSlot = t - relightParams.threadFirst;
    const uint i = relightTarget(t);
    if (i >= relightParams.count) {
        return;
    }
    RelightPixel pixel;
    const uint at = i * relightParams.stride;
    relit[at] = asuint(relitColour<kTransfer, kCpca>(i, pixel));
    // The draw's three only where the buffer keeps them (pixelDetail). It
    // reads the slope and polish records only under a mark in the centre's
    // (kRelightSlope): without it they are left as they are.
    if (relightParams.stride >= kRelightStride) {
        if (hasFlag(kRelightSlope)) {
            relit[at + kRelightPixel] = pixel.slopes;
            relit[at + kRelightPixel + 1u] = pixel.polish;
        }
        relit[at + kRelightPixel + 2u] = pixel.centre;
    }
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
    const float took = saturate(-log(max(ratio, 1.0e-4)) / kCatcherDepth);
    // A merged gaussian's coverage past 1 (Spark's LoD opacity, 1..2: see
    // athc.rs coverage_ratios) is not a factor to scale: whole where the
    // object took everything, else the plain gaussian it was capped to.
    return opacity > 1.0 ? (took >= 1.0 ? opacity : took) : opacity * took;
}

/// A shadow catcher's splats: black, as opaque as what the object took.
// A cloud over 65 535 workgroups wide (16 776 960 splats, a 256-page pool)
// is dispatched as rows of that many threads: the splat is x + y * row.
public static const uint kRelightDispatchRow = 65535u * 256u;
uint relightIndex(uint3 tid) { return tid.x + tid.y * kRelightDispatchRow + relightParams.threadFirst; }

[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaRelightCatcher(uint3 tid: SV_DispatchThreadID) {
    const uint i = relightIndex(tid);
    if (i >= relightParams.count) {
        return;
    }
    const Gsplat g = readSplat(i);
    const float alpha = isActive(g.flags) ? catcherOpacity(i, splatRotation(g.quaternion), g.scales, g.rgba.a) : 0.0;
    relit[i * relightParams.stride] = asuint(float4(0.0, 0.0, 0.0, alpha));
    if (relightParams.stride >= kRelightStride) {
        relit[i * relightParams.stride + kRelightPixel + 2u] = uint4(0u);
    }
}

void viewlessAt<let kCpca : int>(uint i) {
    gSlot = i - relightParams.threadFirst;
    if (i >= relightParams.count) {
        return;
    }
    const Gsplat g = readSplat(i);
    relit[i * relightParams.stride + 1u] = isActive(g.flags) ? viewless<kCpca>(i, g) : uint4(0u);
}

/// The kept terms (splatTransferViewless), when the sky, the lights, the
/// cloud or its place changed.
[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaRelightViewless(uint3 tid: SV_DispatchThreadID) {
    viewlessAt<0>(relightIndex(tid));
}

/// The kept terms with the transfer rebuilt by cpca.slang (kRelightCpca).
[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaRelightViewlessCpca(uint3 tid: SV_DispatchThreadID) {
    viewlessAt<1>(relightIndex(tid));
}

/// The kept terms from the joint CPCA's projections (kRelightJoint).
[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaRelightViewlessJoint(uint3 tid: SV_DispatchThreadID) {
    viewlessAt<2>(relightIndex(tid));
}

/// A TX cloud under a dome with the joint CPCA (kRelightJoint): the eye's
/// blocks rebuilt by cpca_joint.slang (mode 1), the rest projected.
[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaRelightJoint(uint3 tid: SV_DispatchThreadID) {
    relightAt<2, 2>(relightIndex(tid));
}

/// A cloud with a TX transfer (cells): everything.
[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaRelight(uint3 tid: SV_DispatchThreadID) {
    relightAt<2, 0>(relightIndex(tid));
}

/// A TX cloud whose transfer cpca.slang rebuilt (kRelightCpca).
[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaRelightCpca(uint3 tid: SV_DispatchThreadID) {
    relightAt<2, 1>(relightIndex(tid));
}

/// The first transfer (no cells).
[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaRelightFirst(uint3 tid: SV_DispatchThreadID) {
    relightAt<1, 0>(relightIndex(tid));
}

/// The first transfer, rebuilt by cpca.slang (kRelightCpca).
[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaRelightFirstCpca(uint3 tid: SV_DispatchThreadID) {
    relightAt<1, 1>(relightIndex(tid));
}

/// No transfer: the lights and the dome's harmonics and map alone.
[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaRelightPlain(uint3 tid: SV_DispatchThreadID) {
    relightAt<0, 0>(relightIndex(tid));
}

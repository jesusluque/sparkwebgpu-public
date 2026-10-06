// Copyright (c) 2026 jesus luque.
//
// What this scene's lights put on a splat, for both routes that draw one.
//
// WHY IT LIVES IN A MODULE OF ITS OWN
//
// Relighting used to be written into `splat_project.slang`, which is the tile
// rasteriser's -- so a cloud that asked to be relit was relit when rasterised
// and shown exactly as it was baked when path traced. The flag meant two
// things, and with a cloud converted from a mesh (whose colours are an albedo
// and not radiance anybody captured) the traced frame was the albedo itself:
// a white material came back at 1.0 and a black marble at 0.003, both with no
// light on them at all. This engine's rule everywhere else is that the routes
// agree; here they now share the function rather than a description of it.
//
// WHAT IT IS, AND WHAT IT IS NOT
//
// A splat is not a surface. The normal is its shortest axis -- the direction a
// flattened Gaussian faces -- turned towards the eye, or where the cloud keeps a
// shading normal of its own (`GpuSplats::normals`, a conversion's normal map)
// that one, turned to the same side (`splatShadingNormal`); its albedo is the
// harmonics' constant term, the colour it shows on average. On top of that
// there is now a specular lobe, because without one a dark glossy material is
// black: the chess set's marble is `0.003` of diffuse albedo and everything
// you see of it in a render of the mesh is reflection. The lobe is GGX with
// Smith's height-correlated shadowing and Schlick's Fresnel, the same shape
// the surface lobes use, driven by a metallic and a roughness the splat
// carries where its cloud was converted from a mesh.
//
// WHAT A MATERIAL LAYERS OVER ITS BASE (`SplatSurface::lobes`, proposal 026).
// Where the conversion read them, a gaussian also carries what OpenPBR,
// standard_surface and UsdPreviewSurface put over the base: a weight, colour
// and index for the dielectric reflection, a clear coat, and a sheen. None of
// it is a new lobe -- the coat is the same GGX dielectric the polish is, at
// its own roughness and index, and the sheen is the lobe library's Imageworks
// sheen -- and each is layered the way MaterialX's `layer` node layers: what
// lies on top takes from what reaches the layer below it the share it
// reflects at the eye (`SplatLayers`). A gaussian that carries none of it
// reflects with `plainLobes()`, which is this module as it was before.
//
// One sample at the middle of each light, no second sample, and a shadow only
// where the caller measured one beforehand. That is what a splat gets: it has
// no BSDF to importance sample and no ray of its own in the rasteriser.
module splat_relight;
__exported import athenea.light.lights;
__exported import athenea.light.environment;
import athenea.common.color;
import athenea.common.octahedral;
import athenea.common.packing;
import athenea.common.transfer_layout;
// The DFG fit the surface lobes use: one definition for both routes. It
// declares no binding, which is what lets a module below material import it.
import athenea.material.microfacet;

/// What a splat reflects with.
/// One value of a gaussian's transfer, f16 pairs, `base` counted in halves.
public float transferAt(StructuredBuffer<uint> transfer, uint base, uint k) {
    const uint at = base + k;
    const uint word = transfer[at >> 1];
    return f16tof32((at & 1u) != 0u ? (word >> 16) : (word & 0xFFFFu));
}

/// THE VALUES A ZONAL TRANSFER IS: two lobes of five, each the octahedral
/// square's two coordinates of its axis in the gaussian's frame
/// (common/octahedral.slang) and its three zonal coefficients, bands 0 to 2.
public static const uint kTransferZonalCount = 10;
public static const uint kZonalLobeValues = 5;

/// `sqrt(4 pi / (2 l + 1))` for the band harmonic `k` is in: a zonal lobe of
/// coefficients `z_l` about the axis `a` is, as harmonics, `z_l` times this
/// times `Y_k(a)` (Sloan, "Stupid Spherical Harmonics Tricks", 2008).
public float zonalBandScale(uint k) {
    return k == 0 ? 3.5449077018110318 : (k < 4 ? 2.0466534158929770 : 1.5853309190424043);
}

public uint zonalBandOf(uint k) {
    return k == 0 ? 0u : (k < 4 ? 1u : 2u);
}

public struct SplatSurface {
    public float3 position;
    public float3 normal;      ///< already turned towards the eye
    public float3 albedo;
    /// HOW MUCH OF AN ENVIRONMENT REACHES THIS GAUSSIAN, where its cloud
    /// carries it: `transferBase` is where its values start in the buffer,
    /// counted in halves, and `transferCount` is 0, 9 (the direct half), 10
    /// (the direct half as zonal lobes, below) or 36 (both halves). With it a
    /// cloud is lit by a sky it was never baked under.
    public uint   transferBase;
    public uint   transferCount;
    /// A ZONAL TRANSFER (`transferCount` 10, `kTransferZonalCount`): two
    /// lobes, each an axis in the gaussian's own frame and three zonal
    /// coefficients, rather than nine harmonics in the world. The axes turn
    /// with the gaussian, so a cloud a skeleton carries keeps its transfer
    /// through every pose (proposal 014 B). `splatTransferFrame` fills both
    /// fields below: the gaussian's three axes in the world, and the lobes put
    /// back as the nine harmonics everything after it reads.
    public float3x3 transferFrame;
    public float    transferSh[9];
    /// WHICH WAYS OUT ARE OPEN, where the cloud carries them: sixty-four bits
    /// of an 8 x 8 octahedral grid, set where the bake's ray left the scene.
    /// What the sun is shadowed by when they are there (`splatSunOpen`). In
    /// the world beside nine harmonics, in the gaussian's own frame beside a
    /// zonal transfer.
    public bool   hasShadowBits;
    public uint2  shadowBits;
    /// THE OPEN DIRECTIONS OF A TX TRANSFER (docs/decisions.md, task TX):
    /// `cellWords` words a gaussian of a finer grid over the whole sphere --
    /// 8 for 16 x 16 cells, 32 for 32 x 32 -- starting at `cellBase` in the
    /// buffer the caller passes beside the transfer. 0 where the cloud carries
    /// the 64 bits above or none, and then nothing below reads them.
    public uint   cellWords;
    public uint   cellBase;
    /// A THIN WALL: a glass sheet the conversion made as transparent as the
    /// sheet is (its opacity is what the sheet reflects head on), so the
    /// blend is its transmission and what it shades is its reflection alone.
    public bool   thinWalled;
    public float  metallic;    ///< 0 for a capture, which carries no such thing
    /// ITS METAL IS A SCHLICK -- OpenPBR's `generalized_schlick_bsdf`, glTF's
    /// -- from its colour head on to the specular colour at grazing, rather
    /// than a conductor of the artistic index standard_surface's is
    /// (`ggxEnvConductorEdged`). For a dark metal the two part by twice at
    /// sixty degrees: a car's paint read matte and bright as a conductor.
    public bool   schlickMetal;
    public float  roughness;   ///< 1 is the Lambert-only answer this had before
    /// What the material lets through, and the reason a splat can be
    /// translucent at all. A gaussian cannot refract; what it can do is send
    /// on the light that arrives from **behind** it, which is what glass lit
    /// from the far side does and what a purely reflective lobe never does.
    /// So the body of the material is split: `1 - transmission` of it faces
    /// the light, and `transmission` of it faces away. What stands behind the
    /// splat is seen through its own opacity, which the conversion lowers for
    /// the same reason.
    public float  transmission;
    /// WHETHER THIS GAUSSIAN IS THE FAR FACE OF WHAT IT STANDS ON.
    ///
    /// A normal is turned towards the eye before anything is shaded, which is
    /// right for a body and wrong for glass: a ball is a shell, the eye sees
    /// both of its faces through each other, and each of them handed over the
    /// image behind it -- the same sky twice, and a ball twice as bright as
    /// the mesh beside it. What a solid transmits is what its near face
    /// transmits; the far one is where the ray left, and it was counted
    /// there.
    public bool   backFacing;
    /// WHERE THE GLASS ACTUALLY SENDS IT, when a ray went and looked.
    ///
    /// A ball is a thick lens and one bend is not a lens: what turns the room
    /// over is the second interface. Where the route can trace the cloud
    /// (rt_glass.slang finds the far face), the direction that comes back
    /// from bending in and out again is handed over here, and the single bend
    /// below is what is left for a route that cannot.
    public bool   hasExit;
    public float3 exitDirection;
    /// WHAT THE FAR FACE LETS OUT, where a ray found one: what is not
    /// reflected back into the glass there, and the material's tint a second
    /// time, since a ray through a solid crosses two interfaces and the mesh
    /// tints at each. One where nothing was found.
    public float3 exitThrough;
    /// WHAT THE MIRROR DIRECTION ACTUALLY MEETS, where a ray went and looked.
    ///
    /// The polish reads the prepared sky along the mirror direction, which is
    /// right for a gaussian with nothing in front of it and wrong for one
    /// facing its own asset: a collar under a ball reflects the ball, not the
    /// ceiling. Where the route can trace the cloud, what the mirror ray meets
    /// is handed over here -- the colour of that particle, which the pass
    /// before this one has already shaded.
    public bool   hasReflection;
    public float3 reflectionColour;
    /// WHETHER A RAY WENT AND LOOKED AT ALL.
    ///
    /// Where one did and met nothing, nothing is in the way and the whole sky
    /// reaches the mirror direction: the fit below is not asked. Where none
    /// was cast, the fit is all there is. Leaving the fit in after a ray had
    /// already answered took a tenth off a gold collar under a sky of one
    /// colour, which is a surface that should read exactly what the mesh
    /// beside it reads.
    public bool   reflectionTraced;
    /// WHAT THE RAY MET AFTER LEAVING, where it met anything: the colour of
    /// the cloud's own particle behind the glass, already linear. A ball
    /// shows the room, and a pawn's ball shows its collar; the sky is only
    /// what is left where nothing stands between.
    public bool   hasBehind;
    public float3 behindColour;
    /// THE INDEX THAT HALF BENDS BY, 0 where nothing bends it.
    ///
    /// With one, the transmitted half stops being a cosine average of what
    /// is behind and becomes the sky along the direction Snell gives at this
    /// gaussian's own normal: a ball shows the room turned, which is the one
    /// thing that says glass. It is the entry face alone -- a cloud has no
    /// far face to find without a ray -- so a thick lens is not what this
    /// draws.
    public float  ior;
    /// True where `albedo` is not an albedo at all but light: a cloud whose
    /// body was baked (`athenea mesh2splat`, which asks the path tracer for the
    /// light on every gaussian). What is left for a frame to add is the
    /// polish -- the reflection, which is the part of a surface that one
    /// colour cannot hold and which a bake therefore left out.
    public bool   lit;
    /// WHAT THE SURFACE GIVES OFF BY ITSELF, linear radiance (the cloud's
    /// `emission`, a converted lamp shade or screen; zero where it has none).
    /// Added as it is: no light reaches it and no shadow takes it away, and
    /// it is the same from both sides of the disc. Not where the body is
    /// `lit` -- a radiance bake met the emission at its first vertex and
    /// holds it already, and adding it again would be it twice.
    public float3 emission;
    /// WHAT THE MATERIAL LAYERED OVER ITS BASE: the specular's weight,
    /// colour and index, the coat, the sheen (packing.slang's `SplatLobes`).
    /// `plainLobes()` where the cloud carries none (`splatLobesOf`).
    public SplatLobes lobes;
};

/// Whether the gaussian's transfer is zonal lobes in its own frame.
public bool splatTransferZonal(SplatSurface s) {
    return s.transferCount == kTransferZonalCount;
}

/// Harmonic `k` of a gaussian's transfer, in the world: read as it was
/// stored, or out of the lobes `splatTransferFrame` turned into the world.
public float transferValue(SplatSurface s, StructuredBuffer<uint> transfer, uint k) {
    return splatTransferZonal(s) ? s.transferSh[k] : transferAt(transfer, s.transferBase, k);
}

/// THE GAUSSIAN'S FRAME, AND ITS ZONAL TRANSFER TURNED BY IT.
///
/// `rotation` is the frame the gaussian has now -- the one its `shape` holds,
/// which for a cloud a skeleton carries is what `splat_skin` made of the
/// blend's Jacobian this frame -- and the rows take the cloud into the world.
/// Each lobe's axis is a direction of that frame, so it is taken to the world
/// by the frame and then by the rows: one 3 x 3 a lobe. A lobe about an axis
/// `a` with zonal coefficients `z` is, as harmonics, `z_l sqrt(4 pi / (2l +
/// 1)) Y_k(a)`, closed, so what the frame reads afterwards -- the body's dot
/// with the sky, the sun's share, the openness -- is the same nine numbers it
/// always read, only rebuilt here rather than read.
///
/// A cloud whose transfer is nine harmonics gets its frame and nothing else.
public void splatTransferFrame(inout SplatSurface s, StructuredBuffer<uint> transfer, float3x3 rotation,
                               float4 row0, float4 row1, float4 row2) {
    const float3 u = relightDirectionToWorld(row0, row1, row2, float3(rotation[0][0], rotation[1][0], rotation[2][0]));
    const float3 v = relightDirectionToWorld(row0, row1, row2, float3(rotation[0][1], rotation[1][1], rotation[2][1]));
    const float3 n = relightDirectionToWorld(row0, row1, row2, float3(rotation[0][2], rotation[1][2], rotation[2][2]));
    s.transferFrame = float3x3(u.x, v.x, n.x, u.y, v.y, n.y, u.z, v.z, n.z);
    [unroll] for (uint k = 0; k < 9; ++k) {
        s.transferSh[k] = 0.0;
    }
    if (!splatTransferZonal(s)) {
        return;
    }
    [unroll] for (uint lobe = 0; lobe < 2; ++lobe) {
        const uint base = s.transferBase + lobe * kZonalLobeValues;
        const float2 square = float2(transferAt(transfer, base, 0), transferAt(transfer, base, 1));
        const float3 axis = normalize(mul(s.transferFrame, octDecode(square)));
        const float3 z = float3(transferAt(transfer, base, 2), transferAt(transfer, base, 3),
                                transferAt(transfer, base, 4));
        [unroll] for (uint k = 0; k < 9; ++k) {
            s.transferSh[k] += z[zonalBandOf(k)] * zonalBandScale(k) * envBasisValue(k, axis);
        }
    }
}

/// A direction of the world in the gaussian's own frame, where its transfer
/// keeps one (a zonal transfer's shadow bits); the world's itself otherwise.
float3 transferLocal(SplatSurface s, float3 d) {
    return splatTransferZonal(s) ? normalize(mul(d, s.transferFrame)) : d;
}

/// A gaussian's layers, or the plain ones where its cloud carries none.
public SplatLobes splatLobesOf(bool has, StructuredBuffer<uint> lobes, uint i) {
    return has ? unpackLobes(uint3(lobes[i * 3], lobes[i * 3 + 1], lobes[i * 3 + 2])) : plainLobes();
}

/// A dielectric's reflectivity head on, at an index: 0.04 at 1.5 exactly,
/// which is the number every cloud before an index was reflected with.
float dielectricF0(float ior) {
    return abs(ior - 1.5) < 1.0e-6 ? 0.04 : iorToF0(max(ior, 1.0));
}

/// What tints the dielectric reflection: the specular's weight times its
/// colour, one for a gaussian that carries neither.
float3 specularTint(SplatSurface s) {
    return s.lobes.specularWeight * s.lobes.specularColour;
}

/// GGX's D times Smith's height-correlated V for a light in `wi`, at
/// `roughness`: the shape every reflection here shares, the coat's as much
/// as the polish's. Zero below the surface.
float ggxShape(float3 n, float3 wo, float3 wi, float roughness) {
    const float3 h = normalize(wo + wi);
    const float nDotV = max(dot(n, wo), 1.0e-4);
    const float nDotL = max(dot(n, wi), 0.0);
    const float nDotH = max(dot(n, h), 0.0);
    if (nDotL <= 0.0) {
        return 0.0;
    }
    // A perfectly smooth splat would be a mirror the size of a texel, which
    // at one sample a light is a spike nothing resolves; the floor is the
    // same one an interactive frame puts on a surface.
    const float alpha = max(roughness * roughness, 1.0e-3);
    const float a2 = alpha * alpha;
    const float d = nDotH * nDotH * (a2 - 1.0) + 1.0;
    const float distribution = a2 / max(3.14159265358979 * d * d, 1.0e-9);
    const float lambdaV = nDotL * sqrt(nDotV * nDotV * (1.0 - a2) + a2);
    const float lambdaL = nDotV * sqrt(nDotL * nDotL * (1.0 - a2) + a2);
    return distribution * 0.5 / max(lambdaV + lambdaL, 1.0e-9);
}

/// GGX, as the surface lobes write it: D, Smith's height-correlated V, and
/// Schlick's F folded into one weight for a light in direction `wi`.
public float3 specularLobe(SplatSurface s, float3 wo, float3 wi) {
    const float shape = ggxShape(s.normal, wo, wi, s.roughness);
    if (shape <= 0.0) {
        return float3(0.0);
    }
    const float vDotH = max(dot(wo, normalize(wo + wi)), 0.0);
    const float p5 = pow(1.0 - vDotH, 5.0);
    // A dielectric reflects 4% head on (at 1.5), tinted by the specular's
    // weight and colour; a metal reflects its own colour, turning to the
    // specular colour at grazing -- except where the metal is already baked,
    // and then there is nothing of it left to add. With the plain lobes both
    // are what they were: `f0 + (1 - f0) (1 - v.h)^5`.
    const float3 tint = specularTint(s);
    const float f0d = dielectricF0(s.lobes.specularIor);
    float3 fresnel;
    if (s.lit) {
        const float3 f0 = f0d * tint * (1.0 - s.metallic);
        fresnel = f0 + (tint - f0) * p5;
    } else {
        const float3 dielectric = f0d * tint + (tint - f0d * tint) * p5;
        const float3 metal = s.albedo + (s.lobes.specularColour - s.albedo) * p5;
        fresnel = lerp(dielectric, metal, s.metallic);
    }
    return shape * fresnel;
}

/// The coat's reflection of a light in `wi`: the same GGX at the coat's
/// roughness, with a dielectric's Schlick at the coat's index, by its weight.
float3 coatLobe(SplatSurface s, float3 wo, float3 wi) {
    if (s.lobes.coatWeight <= 0.0) {
        return float3(0.0);
    }
    const float shape = ggxShape(s.normal, wo, wi, s.lobes.coatRoughness);
    const float vDotH = max(dot(wo, normalize(wo + wi)), 0.0);
    const float f0 = dielectricF0(s.lobes.coatIor);
    return float3(s.lobes.coatWeight * shape * (f0 + (1.0 - f0) * pow(1.0 - vDotH, 5.0)));
}

/// The sheen's reflection of a light in `wi`, Imageworks' (Conty and
/// Kulla's) as the lobe library has it: its colour times the BRDF.
float3 sheenLobe(SplatSurface s, float3 wo, float3 wi) {
    if (all(s.lobes.sheenColour <= float3(0.0))) {
        return float3(0.0);
    }
    const float nDotL = dot(s.normal, wi);
    if (nDotL <= 0.0) {
        return float3(0.0);
    }
    const float nDotV = max(dot(s.normal, wo), 1.0e-4);
    const float nDotH = max(dot(s.normal, normalize(wo + wi)), 0.0);
    return s.lobes.sheenColour * imageworksSheen(nDotL, nDotV, nDotH, max(s.lobes.sheenRoughness, 0.01));
}

/// What a lobe of this roughness reflects of everything around it, as
/// Lazarov's analytic fit of the split-sum environment term gives it. It is
/// what a dome needs: a dome is not somewhere, so one sample of it is not a
/// highlight but a fraction of a whole sky, and the fraction is this.
/// WHAT A GAUSSIAN SENDS BACK OF AN ENVIRONMENT, AS THE SURFACE DOES IT.
///
/// This was Lazarov's fit, which is cheap and old, while the surface lobes
/// beside it used MaterialX's `ggxDirectionalAlbedo`. Asked the same
/// question over a grid of roughness and `N.V` the two came apart by 76 % --
/// worst where a surface grazes, which is every silhouette in a frame -- so
/// a cloud and the mesh next to it under one sky could not have agreed.
///
/// What is answered here is what `lobes.slang` answers for the two lobes a
/// metallic-roughness material is, mixed by the metalness exactly as the
/// `mix` node in `athenea_usd_preview_surface.mtlx` mixes them:
///
///   a dielectric   `F0` of 0.04 against a grazing one, Fresnel taken
///                  exactly (`fresnelDielectric`), which is `kLobeDielectric`
///   a conductor    its own colour at both ends, which is what a conductor's
///                  Fresnel does and what `kLobeConductor` passes
///
/// and both are scaled by `ggxEnergyCompensation`, the light that leaves a
/// rough surface after bouncing between its microfacets. Without it a rough
/// gaussian is darker than the rough mesh beside it by everything the second
/// bounce carries, which at alpha 0.5 is a tenth and climbing.
public float3 ggxEnvDielectric(float roughness, float nDotV) {
    return ggxEnvDielectricAt(roughness, nDotV, 1.5);
}

/// The same at an index of its own: a specular's `specular_ior`, a coat's.
public float3 ggxEnvDielectricAt(float roughness, float nDotV, float ior) {
    const float alpha = max(roughness * roughness, 1.0e-4);
    const float3 F = float3(fresnelDielectric(nDotV, max(ior, 1.0)));
    return ggxDirectionalAlbedo(nDotV, alpha, float3(dielectricF0(ior)), float3(1.0)) *
           ggxEnergyCompensation(nDotV, alpha, F);
}

/// WHAT A SMOOTH DIELECTRIC REFLECTS OF A SKY, with its Fresnel taken
/// exactly. The fit above is Schlick's between its two ends, and Schlick is
/// low through the middle of the curve -- 0.070 at sixty degrees where a
/// lacquer at 1.45 reflects 0.081 -- which a uniform sky shows over a whole
/// ball: a coat over black read 0.084 against the path traced 0.099, whose
/// coat is sampled with the exact Fresnel. So the fit is scaled by the exact
/// Fresnel over Schlick's at the eye: exact on a mirror, and fading as the
/// lobe widens and averages the curve over directions the ratio does not
/// hold for (alpha, from a quadrature of the lobe at 0.1, 0.3 and 0.6).
/// For a reflection only: what passes under it is the lobe library's
/// complement of the fit (`lobeAlbedo`), as the path tracer's layer takes it.
public float3 ggxEnvDielectricExactAt(float roughness, float nDotV, float ior) {
    const float alpha = max(roughness * roughness, 1.0e-4);
    const float index = max(ior, 1.0);
    const float f0 = dielectricF0(index);
    const float schlick = f0 + (1.0 - f0) * pow(1.0 - saturate(nDotV), 5.0);
    const float exact = fresnelDielectric(saturate(nDotV), index);
    // A mirror reflects its Fresnel, which the fit is not at the ends (7%
    // over it head on at 1.45).
    if (alpha < 1.0e-3) {
        return float3(exact);
    }
    const float ratio = exact / max(schlick, 1.0e-6);
    return ggxEnvDielectricAt(roughness, nDotV, ior) * lerp(ratio, 1.0, saturate(alpha));
}

float3 ggxEnvConductor(float3 albedo, float roughness, float nDotV) {
    return ggxEnvConductorEdged(albedo, float3(1.0), roughness, nDotV);
}

float3 ggxEnvConductorEdged(float3 albedo, float3 edge, float roughness, float nDotV) {
    const float alpha = max(roughness * roughness, 1.0e-4);
    // A conductor's Fresnel stands for both ends of the fit, and it is taken
    // the way the surface takes it: MaterialX's artistic index out of the
    // colour, then the conductor's own Fresnel. Schlick between the colour
    // and one was a per cent off in the middle of the curve and six per cent
    // off on gold, which is a gold collar six per cent darker than the mesh
    // beside it under a sky of one colour.
    // The edge is white, not the colour: `standard_surface`'s metal takes its
    // grazing tint from `specular_color`, which is white unless an artist says
    // otherwise, and a gold that stays gold at grazing came back a fifth dark
    // in blue against the mesh beside it.
    float3 index;
    float3 extinction;
    mxArtisticIor(saturate(albedo), saturate(edge), index, extinction);
    const float3 F = fresnelConductor(nDotV, index, extinction);
    return ggxDirectionalAlbedo(nDotV, alpha, F, F) * ggxEnergyCompensation(nDotV, alpha, F);
}

/// The two, mixed by the metalness.
public float3 environmentBrdf(float3 albedo, float metallic, float roughness, float nDotV) {
    return lerp(ggxEnvDielectric(roughness, nDotV), ggxEnvConductor(albedo, roughness, nDotV),
                saturate(metallic));
}

/// The two with a gaussian's own layers: the dielectric at the specular's
/// index and tinted by its weight and colour, the metal turning to the
/// specular colour at grazing (standard_surface's `specular_color` on a
/// metal). The plain lobes make it `environmentBrdf`.
public float3 environmentBrdfOf(SplatSurface s, float nDotV) {
    // Two returns, not one with the metal chosen inside: the conductor's is
    // the expression every cloud read, to the bit (lobes_check holds it so).
    if (s.schlickMetal) {
        return lerp(ggxEnvDielectricAt(s.roughness, nDotV, s.lobes.specularIor) * specularTint(s),
                    ggxEnvSchlickMetal(s.albedo, s.lobes.specularColour, s.roughness, nDotV), saturate(s.metallic));
    }
    return lerp(ggxEnvDielectricAt(s.roughness, nDotV, s.lobes.specularIor) * specularTint(s),
                ggxEnvConductorEdged(s.albedo, s.lobes.specularColour, s.roughness, nDotV), saturate(s.metallic));
}

/// A SCHLICK METAL'S SHARE OF AN ENVIRONMENT: the same DFG fit with its own
/// two ends, the colour head on and the specular colour at grazing (which for
/// OpenPBR's F82 tint of white is exactly Schlick, and otherwise its nearest),
/// and the multiple-scattering compensation of that Fresnel.
float3 ggxEnvSchlickMetal(float3 albedo, float3 edge, float roughness, float nDotV) {
    const float alpha = max(roughness * roughness, 1.0e-4);
    const float3 f0 = saturate(albedo);
    const float3 f90 = saturate(edge);
    const float3 F = f0 + (f90 - f0) * pow(1.0 - saturate(nDotV), 5.0);
    return ggxDirectionalAlbedo(nDotV, alpha, f0, f90) * ggxEnergyCompensation(nDotV, alpha, F);
}

/// WHAT THE LAYERS OVER THE BASE TAKE AND GIVE AT THE EYE.
///
/// MaterialX's `layer` weighs the base by the top's throughput at the view
/// direction, `1 - weight x directional albedo`, and the bake's lobe stack
/// does the same (material_runtime.slang): so a coat of weight one at 1.5
/// takes 4% from everything under it head on and most of it at grazing, and
/// gives it back as its own reflection. The coat is over the sheen and the
/// sheen over the base, as standard_surface stacks them (OpenPBR puts its
/// fuzz over the coat; with both on one gaussian the order is
/// standard_surface's). The sheen's weight is its
/// colour's largest channel -- exact for the white sheen every surface
/// defaults to -- since a gaussian carries the two multiplied.
public struct SplatLayers {
    public float3 coat;       ///< the coat's directional albedo, times its weight
    public float3 sheen;      ///< the sheen's, times its colour
    public float3 under;      ///< what reaches the base: through the coat and the sheen
    public float3 belowCoat;  ///< what reaches the sheen: through the coat
};

public SplatLayers splatLayers(SplatSurface s, float nDotV) {
    SplatLayers l;
    // What the coat reflects, with its Fresnel exact (ggxEnvDielectricExactAt),
    // and what it lets through, the complement of the fit, as the path
    // tracer's layer weighs the base.
    const float3 fitted = s.lobes.coatWeight > 0.0
                              ? s.lobes.coatWeight * ggxEnvDielectricAt(s.lobes.coatRoughness, nDotV, s.lobes.coatIor)
                              : float3(0.0);
    l.coat = s.lobes.coatWeight > 0.0
                 ? s.lobes.coatWeight * ggxEnvDielectricExactAt(s.lobes.coatRoughness, nDotV, s.lobes.coatIor)
                 : float3(0.0);
    const float sheenAlbedo = any(s.lobes.sheenColour > float3(0.0))
                                  ? imageworksSheenDirectionalAlbedo(nDotV, max(s.lobes.sheenRoughness, 0.01))
                                  : 0.0;
    l.sheen = s.lobes.sheenColour * sheenAlbedo;
    const float sheenWeight = max(max(s.lobes.sheenColour.x, s.lobes.sheenColour.y), s.lobes.sheenColour.z);
    l.belowCoat = saturate(float3(1.0) - fitted);
    l.under = l.belowCoat * saturate(1.0 - sheenWeight * sheenAlbedo);
    // OPENPBR'S COAT DARKENING. Light that enters a coat and comes back up
    // from the base is partly reflected down again at the coat's underside,
    // and the base absorbs some of it each time: the base comes out darker
    // than the coat's own throughput says, `(1 - K) / (1 - E K)` with
    // `K = 1 - (1 - F0) / n^2` the coat's internal reflection and `E` the
    // base's albedo -- the graph's own estimate (open_pbr_surface.mtlx,
    // `base_darkening`): the base colour, times the specular weight for a
    // metal. Under a lacquer at 1.45 a dark metal keeps 0.47 of itself; the
    // car's paint read 1.5 times the mesh at the centre of its ball without
    // it. A baked body is darkened already (the bake's stack is the graph),
    // so only the polish the frame adds is, with its albedo taken as none.
    if (s.lobes.coatWeight > 0.0 && s.lobes.coatDarkening >= 0.5) {
        const float n = max(s.lobes.coatIor, 1.0);
        const float K = 1.0 - (1.0 - dielectricF0(n)) / (n * n);
        const float3 E = s.lit ? float3(0.0)
                               : saturate(lerp(s.albedo, s.albedo * s.lobes.specularWeight, saturate(s.metallic)));
        l.under *= lerp(float3(1.0), (1.0 - K) / max(float3(1.0) - E * K, float3(1.0e-4)), s.lobes.coatWeight);
    }
    return l;
}

/// A dome, answered whole rather than sampled.
///
/// Every other light is somewhere: one sample at its centre is a direction
/// worth taking, and the lobe evaluated there is the highlight. A dome is
/// everywhere, and a single sample of it gives a splat one spike where a
/// surface would show a broad sheen -- which is exactly what the chess set's
/// polished marble is made of. So it is integrated instead: uniform radiance
/// over the hemisphere is `albedo * L` of diffuse, and `environmentBrdf * L`
/// along the mirror direction of specular, both closed forms.
/// WHAT THE SKY IS, WHERE ONE WAS PREPARED.
///
/// A dome's image goes through the material texture table and this kernel has
/// none, so until `technique::Environment` a cloud under an HDRI was lit by
/// the dome's constant colour: the sky's mean, with none of its direction,
/// while the mesh beside it reflected the sky itself. What the environment
/// prepares is that image in two forms a buffer can hold -- nine harmonics of
/// radiance for the body, and the sky convolved with a GGX lobe for the
/// reflection -- and both already carry the light's own emission.
///
/// Without one (`kEnvNone`), every line below falls back to what it read
/// before: a constant dome's irradiance is `pi * L`, so the body reads `L`
/// either way, and the two are the same picture.
/// THE BODY, FROM A TRANSFER RATHER THAN FROM AN IRRADIANCE.
///
/// A gaussian that carries a transfer carries how much of an environment
/// reaches it -- visibility times the cosine, its own albedo taken as one --
/// so what it reflects under any sky is that dotted with the sky's own
/// harmonics, times its albedo. The direct half is a scalar a coefficient
/// (geometry alone); the indirect one is coloured by what the light bounced
/// off on the way, so it is an rgb triple a coefficient.
float3 transferredBody(SplatSurface s, StructuredBuffer<uint> transfer, StructuredBuffer<float4> envSh,
                       uint dome, bool withIndirect) {
    const uint base = dome * kEnvCoefficients;
    float3 sum = float3(0.0);
    // DEGREE 3 (task TX, step 3): sixteen of each half against sixteen of
    // the sky's.
    if (transferDirectCount(s.transferCount) == 16u) {
        [unroll] for (uint k = 0; k < 16u; ++k) {
            sum += envSh[base + k].rgb * transferAt(transfer, s.transferBase, k);
        }
        if (withIndirect && transferIndirectCount(s.transferCount) == 48u) {
            [unroll] for (uint k = 0; k < 16u; ++k) {
                sum += envSh[base + k].rgb * transferRgbAt(transfer, s.transferBase + 16u + k * 3u);
            }
        }
        return sum;
    }
    [unroll] for (uint k = 0; k < kEnvIrradianceCoefficients; ++k) {
        sum += envSh[base + k].rgb * transferValue(s, transfer, k);
    }
    if (withIndirect && s.transferCount >= 36) {
        [unroll] for (uint k = 0; k < kEnvIrradianceCoefficients; ++k) {
            const uint at = s.transferBase + 9 + k * 3;
            const float3 indirect = float3(transferAt(transfer, at, 0), transferAt(transfer, at, 1),
                                           transferAt(transfer, at, 2));
            sum += envSh[base + k].rgb * indirect;
        }
    }
    return sum;
}

/// THREE VALUES OF A TRANSFER, an rgb, `at` counted in halves.
float3 transferRgbAt(StructuredBuffer<uint> transfer, uint at) {
    return float3(transferAt(transfer, at, 0), transferAt(transfer, at, 1), transferAt(transfer, at, 2));
}

/// THE INDIRECT HALF READ ALONG ONE DIRECTION, rgb: how much a delta of unit
/// irradiance arriving from `d` brings to this gaussian after bouncing off
/// what stands around it. What the sun's bounce is, and a light's, which the
/// first transfer left out because the sun was taken out of the harmonics the
/// indirect half is dotted with. Zero where the cloud keeps no indirect half.
public float3 splatIndirectAlong(SplatSurface s, StructuredBuffer<uint> transfer, float3 d) {
    const uint coefficients = transferIndirectCount(s.transferCount) / 3u;
    const uint base = s.transferBase + transferDirectCount(s.transferCount);
    float3 sum = float3(0.0);
    for (uint k = 0; k < coefficients; ++k) {
        sum += transferRgbAt(transfer, base + k * 3u) * envBasisValue(k, d);
    }
    return max(sum, float3(0.0));
}

/// HOW MUCH MORE THE SCENE AROUND A GAUSSIAN BOUNCES UNDER THIS SKY THAN UNDER
/// A WHITE ONE, per channel (task TX, step 2).
///
/// The reflected field was baked under a white sky of radiance one: its
/// pattern is what stands around the gaussian -- which way, how far, of what
/// colour -- and its brightness is that sky's. Under another sky the pattern
/// stands and the brightness is the ratio of what the indirect half gathers
/// under it, the sun's bounce included, to what it gathers under the white
/// one, `T_ind,0 sqrt(4 pi)`. A rank-one coupling: it is exact for a
/// gaussian surrounded by one surface lit by one sky, and it puts a lit
/// ground's colour, not its shape, under a sky lit from one side. Where the
/// gaussian bounces nothing at all the field is nothing too, and the sky's
/// mean radiance stands in.
public float3 splatFieldCoupling(SplatSurface s, StructuredBuffer<uint> transfer, StructuredBuffer<float4> envSh,
                                 uint dome, float4 disc, float3 irradiance) {
    const uint coefficients = min(transferIndirectCount(s.transferCount) / 3u, kEnvCoefficients);
    const uint base = s.transferBase + transferDirectCount(s.transferCount);
    float3 gathered = float3(0.0);
    for (uint k = 0; k < coefficients; ++k) {
        gathered += transferRgbAt(transfer, base + k * 3u) * envSh[dome * kEnvCoefficients + k].rgb;
    }
    float3 mean = envSh[dome * kEnvCoefficients].rgb * 0.28209479177387814;
    if (disc.w > 0.0) {
        gathered += irradiance * splatIndirectAlong(s, transfer, disc.xyz);
        mean += irradiance * (1.0 / (4.0 * 3.14159265358979));
    }
    const float3 white = transferRgbAt(transfer, base) * 3.5449077018110318;   // sqrt(4 pi)
    return float3(white.x > 1.0e-4 ? max(gathered.x / white.x, 0.0) : mean.x,
                  white.y > 1.0e-4 ? max(gathered.y / white.y, 0.0) : mean.y,
                  white.z > 1.0e-4 ? max(gathered.z / white.z, 0.0) : mean.z);
}

/// WHAT THE CLOSED DIRECTIONS SHOW ALONG `d`, under a white sky (task TX,
/// step 2): the reflected field read there, its bands narrowed to a lobe of
/// this roughness -- a GGX lobe reflected is about a von Mises-Fisher of
/// concentration `1 / (2 alpha^2)`, whose band l is `exp(-l (l + 1) alpha^2)`.
/// Times `splatFieldCoupling` it is radiance.
/// The field along one direction at two roughnesses -- the polish's and the
/// coat's -- out of one reading of its forty-eight values: the frame's most
/// costly reads (24 of the TX car's 85 ms of projection were the field's).
public void splatFieldAlongPair(SplatSurface s, StructuredBuffer<uint> transfer, float3 d, float roughness,
                                float coatRoughness, out float3 polish, out float3 coat) {
    const uint base = s.transferBase + transferFieldOffset(s.transferCount);
    const float a = roughness * roughness;
    const float b = coatRoughness * coatRoughness;
    const float a2 = a * a, b2 = b * b;
    polish = float3(0.0);
    coat = float3(0.0);
    [unroll] for (uint j = 0; j < 16u; ++j) {
        const float band = j < 1u ? 0.0 : (j < 4u ? 2.0 : (j < 9u ? 6.0 : 12.0));   // l (l + 1)
        const float3 value = transferRgbAt(transfer, base + j * 3u) * envBasisValue(j, d);
        polish += value * exp(-band * a2);
        coat += value * exp(-band * b2);
    }
    polish = max(polish, float3(0.0));
    coat = max(coat, float3(0.0));
}

public float3 splatFieldAlong(SplatSurface s, StructuredBuffer<uint> transfer, float3 d, float roughness) {
    const uint base = s.transferBase + transferFieldOffset(s.transferCount);
    const float alpha = roughness * roughness;
    const float a2 = alpha * alpha;
    float3 sum = float3(0.0);
    [unroll] for (uint j = 0; j < 16u; ++j) {
        const float band = j < 1u ? 0.0 : (j < 4u ? 2.0 : (j < 9u ? 6.0 : 12.0));   // l (l + 1)
        sum += transferRgbAt(transfer, base + j * 3u) * (exp(-band * a2) * envBasisValue(j, d));
    }
    return max(sum, float3(0.0));
}

/// HOW MUCH OF THE SUN REACHES THIS GAUSSIAN, out of its own transfer.
///
/// The disc was taken out of the harmonics (`env_sun.slang`) and comes back
/// as a direction, so what it needs is a visibility along that direction --
/// and a transfer is exactly a visibility, projected. Evaluated at the sun it
/// gives the cosine-weighted, visibility-weighted answer; the same sum with
/// an open hemisphere's own coefficients gives what it would be with nothing
/// in the way, and the ratio of the two is the share that gets through.
///
/// Degree 2, so it is a soft shadow: a smooth falloff where a hard edge
/// belongs, and a little light where there should be none. It is
/// nevertheless the first shadow a relit cloud has ever had that knows which
/// way the light came from, and it costs no ray and no byte. A per-gaussian
/// bitmask of traced directions is what would make the edge hard, and is the
/// next thing.
///
/// Numerator and denominator share the truncation, which is what makes the
/// ratio one where nothing occludes. `T` is the degree-2 projection of
/// `V cos/pi`; with `V` one that function is the clamped cosine over pi,
/// whose own coefficients are `{1, 2/3, 1/4} Y_k(n)` -- the convolution
/// weights `environment.slang` keeps, over pi. Comparing the truncation
/// against the *exact* cosine instead reads 1.06 at the pole and 5.9 at
/// eighty-nine degrees, which is the truncation showing, not an occlusion.
///
/// Past about eighty-seven degrees the denominator is small enough to be
/// noise and the cosine below has almost closed the term anyway, so the share
/// goes to one rather than to whatever the division says.
public static const float kOverPiSun = 1.0 / 3.14159265358979;

public float splatSunShare(SplatSurface s, StructuredBuffer<uint> transfer, float3 sunDirection) {
    if (s.transferCount < 9) {
        return 1.0;
    }
    float lit = 0.0;
    float open = 0.0;
    const float3 n = s.normal;
    [unroll] for (uint k = 0; k < kEnvIrradianceCoefficients; ++k) {
        const float basis = envBasisValue(k, sunDirection);
        const float band = k == 0 ? 1.0 : (k < 4 ? (2.0 / 3.0) : 0.25);
        lit += transferValue(s, transfer, k) * basis;
        open += band * envBasisValue(k, n) * basis;
    }
    return open > 0.04 ? saturate(lit / open) : 1.0;
}

/// HOW MUCH OF THE SUN GETS THROUGH, out of the bits the bake traced.
///
/// The transfer's own estimate (`splatSunShare`) is degree 2 and can only
/// shadow softly. Where the cloud carries the sixty-four bits, the sun's
/// direction is looked up in them instead: the four cells around it on the
/// octahedral grid, weighed bilinearly, so an edge falls over one cell -- a
/// quarter of a radian -- rather than over a hemisphere. A cell below the
/// gaussian's own horizon was never traced (its bit is clear because nothing
/// was asked, not because something was in the way), so it is left out of
/// the weights rather than counted as shadow. The grid is not wrapped across
/// the octahedral seam; a sun on the seam reads the cells on its own side.
public float splatSunOpen(SplatSurface s, StructuredBuffer<uint> transfer, float3 sunDirection) {
    if (!s.hasShadowBits) {
        return splatSunShare(s, transfer, sunDirection);
    }
    // Beside a zonal transfer the bits were laid out in the gaussian's own
    // frame, so the sun and the horizon are looked up there.
    const float3 sun = transferLocal(s, sunDirection);
    const float3 normal = transferLocal(s, s.normal);
    // In the frame a cell on the equator of a normal along an axis sits at a
    // rounding's distance from the horizon, on one side or the other as the
    // frame turns: held off it, so a turned gaussian reads the cells the
    // unturned one does. In the world it stays exactly zero, as it was.
    const float horizon = splatTransferZonal(s) ? 1.0e-4 : 0.0;
    const float2 at = octEncode(sun) * 8.0 - 0.5;
    const float2 lo = floor(at);
    const float2 f = at - lo;
    float open = 0.0;
    float weight = 0.0;
    [unroll] for (uint k = 0; k < 4; ++k) {
        const int2 cell = clamp(int2(lo) + int2(k & 1, k >> 1), int2(0), int2(7));
        const float w = ((k & 1) != 0 ? f.x : 1.0 - f.x) * ((k >> 1) != 0 ? f.y : 1.0 - f.y);
        const float3 towards = octDecode((float2(cell) + 0.5) / 8.0);
        if (dot(towards, normal) <= horizon || w <= 0.0) {
            continue;
        }
        const uint index = uint(cell.y) * 8u + uint(cell.x);
        const uint word = index < 32u ? s.shadowBits.x : s.shadowBits.y;
        open += w * float((word >> (index & 31u)) & 1u);
        weight += w;
    }
    // No traced cell near the sun at all: it grazes this gaussian's horizon,
    // where the cosine has already closed the term.
    return weight > 0.0 ? open / weight : 1.0;
}

/// THE GRID A TX TRANSFER'S BITS STAND FOR: 16 x 16 octahedral cells over
/// the whole sphere for eight words a gaussian, 32 x 32 for thirty-two.
public uint splatCellSide(uint words) {
    return words >= 32u ? 32u : 16u;
}

bool splatCellBit(SplatSurface s, StructuredBuffer<uint> cellBits, uint index) {
    return ((cellBits[s.cellBase + (index >> 5u)] >> (index & 31u)) & 1u) != 0u;
}

/// HOW OPEN ONE DIRECTION IS, out of a TX transfer's bits: the four cells
/// around it on the octahedral grid, weighed bilinearly, as the sun's 64 bits
/// are read (`splatSunOpen`). A cell whose centre lies behind `axis` is left
/// out of the weights -- its ray left from the far side of the surface, where
/// a solid's own body stands -- so a direction grazing the horizon reads the
/// cells above it rather than the body below; a zero `axis` keeps every cell.
/// -1 where no cell was weighed at all.
public float splatCellsOpen(SplatSurface s, StructuredBuffer<uint> cellBits, float3 d, float3 axis) {
    const uint side = splatCellSide(s.cellWords);
    const float sideF = float(side);
    const float2 at = octEncode(d) * sideF - 0.5;
    const float2 lo = floor(at);
    const float2 f = at - lo;
    const bool judged = dot(axis, axis) > 0.0;
    float open = 0.0;
    float weight = 0.0;
    [unroll] for (uint k = 0; k < 4; ++k) {
        const int2 cell = clamp(int2(lo) + int2(k & 1, k >> 1), int2(0), int2(int(side) - 1));
        const float w = ((k & 1) != 0 ? f.x : 1.0 - f.x) * ((k >> 1) != 0 ? f.y : 1.0 - f.y);
        if (w <= 0.0) {
            continue;
        }
        if (judged && dot(octDecode((float2(cell) + 0.5) / sideF), axis) <= 0.0) {
            continue;
        }
        open += w * (splatCellBit(s, cellBits, uint(cell.y) * side + uint(cell.x)) ? 1.0 : 0.0);
        weight += w;
    }
    return weight > 0.0 ? open / weight : -1.0;
}

/// HOW MUCH OF A REFLECTION'S LOBE SEES THE SKY, out of the same bits.
///
/// The Lagarde fit narrows a reflection by how much of the whole hemisphere
/// is open, one number for every direction: a door's mirror read the open
/// sky where the ground stands in it. Here the lobe itself is asked -- its
/// centre and a ring of six around it at the angle GGX reflects about half
/// its energy within (twice the half-vector's `atan(alpha)`, a little under),
/// a third of the weight on the centre and the rest on the ring. A ring
/// narrower than half a cell would read the centre's cells again, and is
/// left out. Where nothing was judged (a lobe wholly behind `axis`) it is
/// open, as the sun is at the horizon: the cosine has closed it already.
public float splatLobeOpen(SplatSurface s, StructuredBuffer<uint> cellBits, float3 m, float roughness,
                           float3 axis) {
    const float alpha = max(roughness * roughness, 1.0e-3);
    return splatConeOpen(s, cellBits, m, min(1.4 * atan(alpha), 1.2), axis);
}

/// THE SAME OVER A CONE OF HALF ANGLE `spread`: a reflection's lobe, or the
/// disc a sphere, a rect or a sun subtends, whose penumbra this is (step 4).
public float splatConeOpen(SplatSurface s, StructuredBuffer<uint> cellBits, float3 m, float spreadIn, float3 axis) {
    const float centre = splatCellsOpen(s, cellBits, m, axis);
    const float halfCell = 3.14159265358979 / float(splatCellSide(s.cellWords) * 2u);
    // NEVER NARROWER THAN A CELL: a cone inside one cell read that cell's one
    // bit, and a mirror's reflection came out speckled bit by bit across the
    // gaussians (the Corvette's paint, 0.393 against the path traced frame;
    // 0.298 with the cone held to a cell).
    const float spread = max(spreadIn, 2.0 * halfCell);
    const float3 helper = abs(m.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    const float3 t = normalize(cross(helper, m));
    const float3 b = cross(m, t);
    const float c = cos(spread);
    const float sn = sin(spread);
    float open = centre >= 0.0 ? centre / 3.0 : 0.0;
    float weight = centre >= 0.0 ? 1.0 / 3.0 : 0.0;
    [unroll] for (uint k = 0; k < 6; ++k) {
        const float phi = float(k) * (3.14159265358979 / 3.0);
        const float3 d = normalize(m * c + (t * cos(phi) + b * sin(phi)) * sn);
        const float o = splatCellsOpen(s, cellBits, d, axis);
        if (o >= 0.0) {
            open += o / 9.0;
            weight += 1.0 / 9.0;
        }
    }
    return weight > 0.0 ? open / weight : 1.0;
}

/// THE SUN, OR ANY LIGHT THAT IS A DIRECTION, out of a TX transfer's bits,
/// and the old ways where the cloud carries those instead.
public float splatOpenToward(SplatSurface s, StructuredBuffer<uint> transfer, StructuredBuffer<uint> cellBits,
                             float3 direction) {
    if (s.cellWords >= 8u) {
        const float open = splatCellsOpen(s, cellBits, direction, s.normal);
        return open >= 0.0 ? open : 1.0;
    }
    return splatSunOpen(s, transfer, direction);
}

/// WHERE THE TRANSMITTED HALF LOOKS, WITH AN INDEX.
///
/// Snell at the gaussian's own normal, entering: the normal is already turned
/// towards the eye, so the ratio is `1 / ior`. Past the critical angle there
/// is no direction to look along and the straight one is what is left, which
/// is what this did before an index existed.
/// WHERE A RAY LEAVES A SOLID GLASS, the far face guessed from the near one's
/// curvature `curvature` (1/radius): a ray bent in by the index, across the
/// chord of a sphere of that radius, and bent out again at the far side --
/// which turns the room upside down through a ball, as a lens does, and
/// leaves it straight through a slab (curvature 0: the exit parallel to the
/// way in). A ray the far face holds by total reflection leaves reflected.
public float3 lensExit(SplatSurface s, float3 wo, float curvature) {
    const float index = max(s.ior, 1.0);
    const float3 n = s.normal;
    const float3 inside = refract(-wo, n, 1.0 / index);
    if (!(dot(inside, inside) > 1.0e-8)) {
        return -wo;
    }
    const float3 t1 = normalize(inside);
    if (!(curvature > 1.0e-6)) {
        return -wo;
    }
    const float radius = 1.0 / curvature;
    const float chord = 2.0 * radius * max(dot(t1, -n), 0.0);
    // The far point, from the near one, and the outward normal there.
    const float3 farNormal = normalize(t1 * (chord / radius) + n);
    const float3 out = refract(t1, -farNormal, index);
    return dot(out, out) > 1.0e-8 ? normalize(out) : reflect(t1, -farNormal);
}

/// WHAT THE FAR FACE OF `lensExit`'S SPHERE LETS OUT, for a route that found
/// no far face (`exitThrough` is one there): the Fresnel share the second
/// interface reflects back into the glass, and the tint a second time -- the
/// mesh's dielectric lobe tints at every crossing and the colour carries it
/// once (mesh2splat folds the transmission colour into it), as rt_shade's
/// traced exit does. The pawn's head read a tenth bright and grey where the
/// path traced one is warm green. A ray the far face holds keeps what it had.
public float3 lensExitThrough(SplatSurface s, float3 wo, float curvature) {
    const float3 tint = lerp(float3(1.0), saturate(s.albedo), s.transmission);
    const float index = max(s.ior, 1.0);
    const float3 n = s.normal;
    const float3 inside = refract(-wo, n, 1.0 / index);
    if (!(dot(inside, inside) > 1.0e-8)) {
        return tint;
    }
    const float3 t1 = normalize(inside);
    float3 farNormal = -n;
    if (curvature > 1.0e-6) {
        const float radius = 1.0 / curvature;
        const float chord = 2.0 * radius * max(dot(t1, -n), 0.0);
        farNormal = normalize(t1 * (chord / radius) + n);
    }
    const float3 out = refract(t1, -farNormal, index);
    if (!(dot(out, out) > 1.0e-8)) {
        return tint;
    }
    const float reflectedBack = fresnelDielectric(saturate(dot(t1, farNormal)), 1.0 / index);
    return (1.0 - reflectedBack) * tint;
}

public float3 bentThrough(SplatSurface s, float3 wo) {
    const float3 bent = refract(-wo, s.normal, 1.0 / max(s.ior, 1.0));
    return dot(bent, bent) > 1.0e-8 ? normalize(bent) : -wo;
}

/// HOW MUCH OF ITS SKY A GAUSSIAN CAN SEE AT ALL, out of its own transfer.
///
/// The constant term of a transfer is the clamped cosine lobe weighed by
/// visibility: over an open hemisphere it is exactly `A0 * Y0 / pi`, which is
/// 0.28209, so the ratio is the fraction of the sky that reaches this
/// gaussian. It is an ambient occlusion nobody had to bake twice.
public float splatOpenness(SplatSurface s, StructuredBuffer<uint> transfer) {
    if (s.transferCount < 9) {
        return 1.0;
    }
    return saturate(transferValue(s, transfer, 0) / 0.28209479177387814);
}

/// AND HOW MUCH OF IT THE REFLECTION CAN SEE.
///
/// The reflection was added with nothing in front of it: a gaussian in a
/// pocket reflected the whole sky, and the measurement said so -- the pawn's
/// marble disc, whose stem stands over it, read a tenth bright against the
/// mesh path traced beside it. What a ray would answer exactly, an occlusion
/// answers cheaply: Lagarde and de Rousiers' fit, which narrows the opening
/// as the lobe sharpens, so a mirror in a pocket goes dark and a rough one
/// hardly moves. Where a cloud carries no transfer, openness is one and this
/// is what it always was.
public float specularOpenness(float nDotV, float openness, float roughness) {
    return saturate(pow(nDotV + openness, exp2(-16.0 * roughness - 1.0)) - 1.0 + openness);
}

/// WHAT A FRAME STILL OWES A GAUSSIAN, once its body is accounted for.
///
/// A cloud that was not baked owes the whole reflection, metal and all. One
/// whose body was baked (`lit`) already carries the metal's, because a bake
/// keeps a conductor's reflection in the body it fits -- what is left is the
/// dielectric polish, and only as much of it as the surface is not metal.
public float3 splatPolish(SplatSurface s, float nDotV) {
    return s.lit ? ggxEnvDielectricAt(s.roughness, nDotV, s.lobes.specularIor) * specularTint(s) * (1.0 - s.metallic)
                 : environmentBrdfOf(s, nDotV);
}

/// What a thin wall's reflection is scaled by: the sheet reflects `2/(1+R0)`
/// of one face's Fresnel, and its gaussians were given `2R0/(1+R0)` of
/// opacity, so the blend leaves the reflection whole at `1/R0` of a face's.
/// At the sheet's own index, as the conversion gave it its opacity
/// (m2sGlassCovers): held to 1.5 at least, a headlight's cover at 1.15 was
/// given 0.0097 of opacity and a gain of 25 where it needs 205, and every
/// sheet under 1.5 reflected that much too little. 1.5 only where no index
/// was kept.
float thinWallGain(SplatSurface s) {
    if (!s.thinWalled) {
        return 1.0;
    }
    const float index = s.ior > 1.0 ? s.ior : 1.5;
    const float r = (index - 1.0) / (index + 1.0);
    return 1.0 / max(r * r, 1.0e-3);
}

/// How much of the body is not a thin wall's transmission, which is the blend's.
float transmittedWeight(SplatSurface s) {
    return s.thinWalled ? 0.0 : s.transmission;
}

/// HOW SOFT THE ROOM IS THROUGH THE GLASS, as the roughness of a reflection.
///
/// The prepared sky's levels are reflection lobes: a microfacet tilted by
/// `theta` turns a mirror ray by `2 theta`. Through an interface it turns the
/// ray by far less -- `(1 - 1/ior) theta` going in, `(ior - 1) theta` coming
/// out, and the first of those is opened by `ior` again where the ray leaves
/// -- so a ball whose two faces are as rough as its polish shows the room
/// `sqrt(2) (ior - 1) / 2` as wide as it reflects it, a third at 1.5. Read at
/// the polish's own roughness, the pawn's head (0.23) was a haze where the
/// mesh beside it shows the workshop: relMSE over the head 0.081 to 0.048.
/// A level's width goes as the square of the roughness, so the roughness goes
/// as the root of that ratio.
///
/// Only where both faces were found. The single bend a route with no tree
/// makes is not the image a lens forms at all, and sharpening it made the
/// rasterised head worse against the mesh (0.138 to 0.211): there the blur
/// is what hides that it is the wrong picture.
public float transmittedRoughness(float roughness, float ior) {
    const float index = max(ior, 1.0);
    return roughness * sqrt(saturate(0.70710678 * (index - 1.0)));
}

/// WHAT THE TRANSMITTED HALF SEES: what a ray found behind the gaussian, the
/// sky along the direction the glass sends it where the route traced one or
/// an index bends it, and the sky's irradiance from behind -- translucency --
/// where there is no index. The same for a cloud with a transfer and one
/// without: an index is a property of the material, not of the bake.
///
/// All of it leaves through the far face, so all of it is weighed by what
/// that face lets out (`exitThrough`): what a ray met behind the glass as
/// much as the sky. `behindColour` is the colour the particle it met was
/// shaded with, which knows nothing of the glass in front of it.
float3 transmittedBehind(SplatSurface s, float3 wo, uint dome, StructuredBuffer<uint> envTexels,
                         StructuredBuffer<float4> envSh, uint envBaseSide) {
    const float3 seen =
        s.hasBehind ? s.behindColour
                    : (s.hasExit ? envRadiance(envTexels, dome, s.exitDirection,
                                               transmittedRoughness(s.roughness, s.ior), envBaseSide)
                                 : (s.ior > 1.0
                                        ? envRadiance(envTexels, dome, bentThrough(s, wo), s.roughness, envBaseSide)
                                        : envIrradiance(envSh, dome, -s.normal) * (1.0 / 3.14159265358979)));
    return seen * s.exitThrough;
}

/// How much of it passes: what the polish does not reflect, and nothing on
/// the far face of a solid, whose share the near face already sent on.
float3 transmittedShare(SplatSurface s, float nDotV) {
    return s.backFacing && s.ior > 1.0
               ? float3(0.0)
               : saturate(float3(1.0) - ggxEnvDielectricAt(s.roughness, nDotV, s.lobes.specularIor) *
                                            s.lobes.specularWeight);
}

/// WHAT A TX TRANSFER GIVES A GAUSSIAN WHATEVER THE EYE: the body's light
/// under this sky (the transfer dotted with it, the sun's exact cosine at
/// the share the cells let through, the sun's bounce), that share, and the
/// field's coupling to the sky. The frame keeps these a gaussian while the
/// sky and the lights stand (splat_project's splatTransferViewless) and
/// reads them back; what a frame computes is then what the eye changes.
public void transferViewless<let kCells : int>(SplatSurface s, StructuredBuffer<uint> transfer, StructuredBuffer<uint> cellBits,
                             StructuredBuffer<float4> envSh, StructuredBuffer<float4> envSun, uint dome,
                             bool withIndirect, out float3 front, out float3 coupling, out float sunShare) {
    front = transferredBody(s, transfer, envSh, dome, withIndirect);
    // THE DISC THE HARMONICS NO LONGER HOLD. `env_sun` took it out of
    // them and handed over its direction and the irradiance it delivers,
    // so it is added back here as the light it is: an exact cosine, and a
    // share of it that the transfer itself estimates along that
    // direction. A solid angle of zero says this dome had no sun and
    // nothing was taken, and then there is nothing to add.
    //
    // Added to the body only. The prefiltered map kept its own sun, so the
    // polish already has it, and what the polish owes is the shadowed
    // share of it taken back out (relitByDome).
    const float4 disc = envSun[dome * 2];
    const float3 irradiance = disc.w > 0.0 ? envSun[dome * 2 + 1].rgb : float3(0.0);
    const float cosine = max(dot(s.normal, disc.xyz), 0.0);
    sunShare = disc.w > 0.0 ? splatOpenToward(s, transfer, cellBits, disc.xyz) : 1.0;
    front += irradiance * (cosine * sunShare * kOverPiSun);
    // WHAT A TX TRANSFER ADDS (docs/decisions.md, task TX): the open
    // directions by the lobe (step 1), and what the closed ones show
    // (step 2) -- the reflected field, scaled to this sky -- and the sun's
    // bounce, which the first transfer lost with the sun it took out of
    // the harmonics.
    const bool cells = kCells >= 2 && s.cellWords >= 8u;
    const bool field = cells && withIndirect && transferHasField(s.transferCount);
    if (cells && withIndirect && disc.w > 0.0) {
        front += irradiance * splatIndirectAlong(s, transfer, disc.xyz);
    }
    coupling = field ? splatFieldCoupling(s, transfer, envSh, dome, disc, irradiance) : float3(0.0);
}

/// HOW THE REFLECTION TURNS ACROSS A GAUSSIAN'S FOOTPRINT (task TX). The
/// caller says which way the surface faces a pixel to the right of the
/// gaussian's centre and a pixel below it (`normalX`, `normalY`, in the
/// world; `on` where it knows, from the cloud's curvature), and the dome's
/// reflection answers how much its colour changes over those two steps:
/// the sky read along the two mirrors the turned normals give, less what
/// it reads along the centre's, times what the reflection is weighed by.
/// A rasteriser adds that slope over the footprint, so a gaussian draws the
/// highlight moving across it rather than its centre's flat over all of it.
/// THE SUN IN A REFLECTION, ANALYTIC. The prefiltered sky is the sky without
/// its sun (env_prefilter's residual): the split sum is right for a sky that
/// changes slowly across a lobe and a quarter short for a compact sun -- a
/// floating metal ball under one read 0.57-0.74 of the path traced at every
/// level the map holds (s90). So the sun comes back as what it is, a light
/// of irradiance E from one direction, through the lobe's own distribution of
/// reflected directions: with the eye along the normal, as the map assumes,
/// that is D(h) / 4 for the half vector between the mirror and the sun, and
/// it integrates to one over the sphere -- so a lobe's weight (its
/// directional albedo, the split sum's other half) times E times this is the
/// sun's whole reflection, put where the lobe sends it. The disc is folded
/// into the lobe's width (its half angle added in quadrature to alpha).
public float sunReflectPdf(float3 mirror, float3 toSun, float roughness, float discHalfAngle) {
    const float alpha = max(roughness * roughness, 1.0e-3);
    const float a2 = alpha * alpha + discHalfAngle * discHalfAngle;
    const float cosTheta = clamp(dot(mirror, toSun), -1.0, 1.0);
    const float cosHalf2 = saturate(0.5 * (1.0 + cosTheta));   // cos^2 of the half angle
    const float d = cosHalf2 * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265358979 * d * d) * 0.25;
}

/// The roughest coat drawn per pixel (SplatSlope's sharp lobe): past it the
/// prefiltered sky is smooth enough over a footprint for the slope to hold.
public static const float kSharpRoughness = 0.2;
/// The roughest base polish drawn per pixel: the paint's metal flake (0.34)
/// and every polish smoother, whose sky -- and its sun -- changes across a
/// curved panel faster than a gaussian is wide.
public static const float kSharpPolishRoughness = 0.5;

public struct SplatSlope {
    public bool   on;
    public float3 normalX;
    public float3 normalY;
    public float3 slopeX;   ///< the colour's change a pixel to the right
    public float3 slopeY;   ///< and a pixel down
    /// What the frame kept of `transferViewless` for this gaussian under
    /// dome `cachedDome`, where `cached`.
    /// The surface's mean curvature under the gaussian (the shape operator's
    /// half trace, on the face the eye sees), where the cloud keeps it: what
    /// a solid glass's far face is guessed from (`lensExit`). 0: unknown.
    public float  curvature;
    public bool   cached;
    public uint   cachedDome;
    public float3 cachedFront;
    public float3 cachedCoupling;
    public float  cachedSunShare;
    /// THE SHARP LOBE, DRAWN PER PIXEL (splat_blend, kSharpMark). A coat
    /// smooth enough that what it mirrors changes faster than a gaussian is
    /// wide is not answered at the centre and spread over the footprint: its
    /// weight (`sharpWeight`, the coat's albedo times its openness), its
    /// roughness, and the mirror at the centre and a pixel right and down
    /// are handed back, the centre's reading of the sky (`sharpSeen`) is
    /// taken out of the colour, and the blend reads the sky again along the
    /// mirror each pixel turns to. 0 in `sharpWeight`: nothing handed back.
    public float  sharpWeight;
    public float  sharpRoughness;
    public float3 sharpSeen;
    public float3 sharpMirror;
    public float3 sharpMirrorX;
    public float3 sharpMirrorY;
    /// THE BASE'S POLISH, PER PIXEL TOO: a metal flake's lobe at the paint's
    /// 0.34 reads the sun in the prefiltered sky as a bright spot, and read
    /// once per gaussian it drew the fender's blotches (s74: the polish off
    /// took two thirds of them away; the coat, the sun's share and its bounce
    /// none). Its weight (rgb), roughness and the centre's reading.
    public float3 sharpPolish;
    public float  sharpPolishRoughness;
    public float3 sharpPolishSeen;
    /// The sun through each sharp lobe, per pixel too (sunReflectPdf): the
    /// coat's weight as a luminance and the polish's as rgb, each times the
    /// share of the sun the cells let through, and what the centre's own sun
    /// came to, taken out of the colour with the centre's sky.
    public float  sharpSunCoat;
    public float3 sharpSunPolish;
    public float3 sharpSunCentre;
    /// Either lobe handed back: the mirror and its turn are written.
    public bool   sharpOn;
};

public SplatSlope noSlope() {
    SplatSlope g;
    g.on = false;
    g.normalX = float3(0.0, 0.0, 1.0);
    g.normalY = float3(0.0, 0.0, 1.0);
    g.slopeX = float3(0.0);
    g.slopeY = float3(0.0);
    g.curvature = 0.0;
    g.cached = false;
    g.cachedDome = 0xFFFFFFFFu;
    g.cachedFront = float3(0.0);
    g.cachedCoupling = float3(0.0);
    g.cachedSunShare = 1.0;
    g.sharpWeight = 0.0;
    g.sharpRoughness = 1.0;
    g.sharpSeen = float3(0.0);
    g.sharpMirror = float3(0.0, 0.0, 1.0);
    g.sharpMirrorX = float3(0.0, 0.0, 1.0);
    g.sharpMirrorY = float3(0.0, 0.0, 1.0);
    g.sharpPolish = float3(0.0);
    g.sharpPolishRoughness = 1.0;
    g.sharpPolishSeen = float3(0.0);
    g.sharpOn = false;
    g.sharpSunCoat = 0.0;
    g.sharpSunPolish = float3(0.0);
    g.sharpSunCentre = float3(0.0);
    return g;
}

float3 relitByDome<let kTransfer : int>(SplatSurface s, float3 wo, LightRecord light, uint dome,
                   StructuredBuffer<uint> envTexels, StructuredBuffer<float4> envSh, uint envBaseSide,
                   StructuredBuffer<float4> envSun, StructuredBuffer<uint> transfer,
                   StructuredBuffer<uint> cellBits, bool withIndirect, inout SplatSlope slope) {
    const float nDotV = max(dot(s.normal, wo), 1.0e-4);
    const float3 mirror = reflect(-wo, s.normal);
    const float3 body = s.lit ? float3(0.0) : s.albedo * (1.0 - s.metallic);
    // The layers over the base: what they take from what reaches it, and
    // what they give back. A baked body was baked under them already (the
    // bake's stack weighs its body by the same throughput), so it is the
    // polish and the albedo alone that pass through them here.
    const SplatLayers layers = splatLayers(s, nDotV);
    const bool coated = s.lobes.coatWeight > 0.0;
    // A dome is on both sides of a splat at once, so the translucent half
    // sees exactly what the facing half sees and the two sum to the whole
    // body: a glass ball under a sky is its own colour, not a dark shell.
    // A cloud that carries a transfer answers the body from it, and answers
    // it for any sky: what it keeps is the geometry, not the light.
    if (kTransfer != 0 && dome != kEnvNone && s.transferCount >= 9 && !s.lit) {
        // What does not depend on the eye: the body's light, the sun's share,
        // the field's coupling to this sky -- kept from an earlier frame where
        // the sky and the lights have not changed (splatTransferViewless).
        const float4 disc = envSun[dome * 2];
        const float3 irradiance = disc.w > 0.0 ? envSun[dome * 2 + 1].rgb : float3(0.0);
        const float cosine = max(dot(s.normal, disc.xyz), 0.0);
        // Level 2 is a TX transfer's kernel, level 1 the first transfer's:
        // the cells, the field and the slope compiled out of the second, so
        // its kernel is the size it was (a kernel past a size draws threads
        // wrong on Metal: the first transfer's ball came back blotched).
        const bool cells = kTransfer >= 2 && s.cellWords >= 8u;
        const bool field = cells && withIndirect && transferHasField(s.transferCount);
        float3 front;
        float3 coupling;
        float  sunShare;
        if (slope.cached && slope.cachedDome == dome) {
            front = slope.cachedFront;
            coupling = slope.cachedCoupling;
            sunShare = slope.cachedSunShare;
        } else {
            transferViewless<kTransfer>(s, transfer, cellBits, envSh, envSun, dome, withIndirect, front, coupling,
                                        sunShare);
        }
        // The field along the mirror, at the polish's roughness and the
        // coat's, read once.
        float3 fieldPolish = float3(0.0);
        float3 fieldCoat = float3(0.0);
        if (field) {
            splatFieldAlongPair(s, transfer, reflect(-wo, s.normal), s.roughness,
                                coated ? s.lobes.coatRoughness : s.roughness, fieldPolish, fieldCoat);
        }
        // The translucent half faces the other way, and a transfer was
        // measured over the hemisphere the normal faces alone: what arrives
        // behind is answered by the sky's harmonics with no occlusion, which
        // is what every side of a cloud was answered with before a transfer
        // existed. Without this a glass ball that carries one is a dark
        // shell, because half its body has nowhere to take its light from.
        float3 behind = transmittedBehind(s, wo, dome, envTexels, envSh, envBaseSide);
        // WHAT A TX TRANSFER'S GLASS SHOWS THROUGH (step 5). The rasteriser
        // traces nothing, and one bend at the entry face showed the sky where
        // a windscreen shows the cabin. A transmitting gaussian's cells were
        // traced behind it as well and its field drawn over the whole sphere,
        // so along the way out -- straight on, as a slab sends it, since one
        // bend at one face is not what a sheet of glass does -- the sky
        // stands where the bits are open and what the field holds where they
        // are not. A solid's own far face closes its bits, so through a slab
        // it is the field that answers, and it saw through that face.
        if (field && !s.hasBehind && !s.hasExit && transmittedWeight(s) > 0.0 && s.ior > 1.0) {
            // A SOLID GLASS IS A LENS: the sky along the ray it bends, sharp,
            // as before the field existed -- a clear ball refracts the room.
            // Its own far face closes every cell behind it, so the cells say
            // nothing; the field does: it holds what the eye's ray finds
            // through both faces under a white sky (the bake bent it in), a
            // glass's own throughput where the way out is open, less where
            // the scene stands in it. So the sharp sky stands where the field
            // reads open and the field coupled to the sky where it reads
            // closed. Filling the whole lens with the field (degree 3) drew
            // the pawn's clear head milky and flat with a horizon across it:
            // 0.58 against the mesh path traced, where the first transfer
            // drew it sharp.
            const float3 white = splatFieldAlong(s, transfer, -wo, s.roughness);
            const float open = smoothstep(0.5, 0.85, dot(white, float3(0.2126, 0.7152, 0.0722)));
            // Through both faces where the cloud keeps its curvature (a ball
            // shows the room inverted, a pawn's head did not), one bend
            // otherwise.
            const float3 sharp =
                slope.curvature > 0.0
                    ? envRadiance(envTexels, dome, lensExit(s, wo, slope.curvature),
                                  transmittedRoughness(s.roughness, s.ior), envBaseSide) *
                          lensExitThrough(s, wo, slope.curvature)
                    : transmittedBehind(s, wo, dome, envTexels, envSh, envBaseSide);
            behind = open * sharp + (1.0 - open) * coupling * white * s.exitThrough;
        } else if (field && !s.hasBehind && !s.hasExit && transmittedWeight(s) > 0.0) {
            const float3 through = -wo;
            const float alpha = max(s.roughness * s.roughness, 1.0e-3);
            const float openBehind = splatConeOpen(s, cellBits, through, min(1.4 * atan(alpha), 1.2), -s.normal);
            const float open = openBehind >= 0.0 ? openBehind : 0.0;
            behind = (open * envRadiance(envTexels, dome, through, s.roughness, envBaseSide) +
                      (1.0 - open) * coupling * splatFieldAlong(s, transfer, through, s.roughness)) *
                     s.exitThrough;
        }
        // WHAT PASSES IS WHAT IS NOT REFLECTED. The reflection is added
        // below whatever the surface does, so a transmitted half at full
        // strength is more light than arrived: a glass ball came back half
        // again as bright as the mesh beside it, with a rim two and a half
        // times too hot. `lobes.slang` writes the same complement for a
        // transmitting lobe (`kScatterTransmit`), so the two agree.
        // The far face of a solid transmits nothing of its own: its share of
        // the light is what the near face already sent on.
        const float3 passes = transmittedShare(s, nDotV);
        const float3 body2 = layers.under * s.albedo * (1.0 - s.metallic) *
                             ((1.0 - s.transmission) * front + transmittedWeight(s) * passes * behind);
        // What the reflection sees: the cloud itself where a ray found it,
        // and the sky narrowed by how much of it this gaussian can see where
        // nothing was traced.
        const float3 seen = s.hasReflection
                                ? s.reflectionColour
                                : envRadiance(envTexels, dome, mirror, s.roughness, envBaseSide);
        // A TX transfer asks its lobe which of its directions are open; the
        // first transfer had one number for the whole hemisphere.
        // THE BASE'S POLISH IS OPEN BY THE HARMONICS, as the first transfer
        // had it, not by the cells: on the Corvette's chrome and rough metal
        // -- small parts in a crowded scene -- the cells' openness read them
        // 1.64 and 0.221 against the path traced frame, the harmonics' 1.05
        // and 0.128, for the same mean. The coat keeps the cells (below).
        const float narrowed =
            s.reflectionTraced ? 1.0 : specularOpenness(nDotV, splatOpenness(s, transfer), s.roughness);
        // THE SUN IN THE POLISH, analytic (sunReflectPdf): the map no longer
        // holds it, so what the polish reflects of it is its lobe's weight
        // times the irradiance through the lobe, by the share the cells let
        // through. Before, the map kept the sun and its shadowed share was
        // taken back out; a compact sun read through the split sum came out a
        // quarter short whatever the level (s90). A traced reflection replaced
        // the map and brings what it found.
        const float3 sunPolish = s.hasReflection || !(disc.w > 0.0)
                                     ? float3(0.0)
                                     : splatPolish(s, nDotV) * irradiance *
                                           (sunShare * sunReflectPdf(mirror, disc.xyz, s.roughness, disc.w));
        // THE COAT: the same reflection at its own roughness and index, over
        // everything, narrowed by what this gaussian sees of the sky at that
        // roughness, and its own analytic sun by the share the cells let through.
        float3 coat = float3(0.0);
        if (coated) {
            const float3 coatSeen = s.hasReflection
                                        ? s.reflectionColour
                                        : envRadiance(envTexels, dome, mirror, s.lobes.coatRoughness, envBaseSide);
            const float coatNarrowed =
                s.reflectionTraced ? 1.0
                : cells            ? splatLobeOpen(s, cellBits, mirror, s.lobes.coatRoughness, s.normal)
                                   : specularOpenness(nDotV, splatOpenness(s, transfer), s.lobes.coatRoughness);
            const float3 coatSun = s.hasReflection || !(disc.w > 0.0)
                                       ? float3(0.0)
                                       : layers.coat * irradiance *
                                             (sunShare * sunReflectPdf(mirror, disc.xyz, s.lobes.coatRoughness, disc.w));
            // A coat smooth enough is drawn per pixel (SplatSlope's sharp
            // lobe): the first dome's, where the cloud keeps the curvature
            // the mirror turns by.
            if (kTransfer >= 2 && slope.on && !s.hasReflection && dome == 0u &&
                s.lobes.coatRoughness < kSharpRoughness) {
                slope.sharpWeight = dot(layers.coat, float3(0.2126, 0.7152, 0.0722)) * coatNarrowed;
                slope.sharpRoughness = s.lobes.coatRoughness;
                slope.sharpSeen = coatSeen;
                slope.sharpMirror = mirror;
                slope.sharpMirrorX = reflect(-wo, slope.normalX);
                slope.sharpMirrorY = reflect(-wo, slope.normalY);
                slope.sharpOn = true;
                slope.sharpSunCoat = dot(layers.coat, float3(0.2126, 0.7152, 0.0722)) * sunShare;
                slope.sharpSunCentre += coatSun;
            }
            coat = (field ? max(layers.coat * (coatNarrowed * coatSeen +
                                               (1.0 - coatNarrowed) * coupling * fieldCoat),
                                float3(0.0))
                          : layers.coat * coatNarrowed * coatSeen) +
                   coatSun;
        }
        // THE SHEEN: a lobe as broad as a body's, so it is answered as one --
        // its albedo at this eye times the light the transfer says reaches
        // the gaussian, shadow and sun included -- under the coat.
        const float3 sheen = layers.belowCoat * layers.sheen * front;
        // THE SLOPE: the open part of the reflection, read again along the
        // mirrors a pixel away (SplatSlope).
        if (kTransfer >= 2 && slope.on && !s.hasReflection) {
            const float3 mirrorX = reflect(-wo, slope.normalX);
            const float3 mirrorY = reflect(-wo, slope.normalY);
            const float3 polishWeight = layers.under * thinWallGain(s) * splatPolish(s, nDotV) * narrowed;
            if (dome == 0u && s.roughness < kSharpPolishRoughness && any(polishWeight > float3(0.0))) {
                // The polish goes to the blend per pixel (SplatSlope's sharp
                // polish), along the same mirror and turn as the coat's.
                slope.sharpPolish = polishWeight;
                slope.sharpPolishRoughness = s.roughness;
                slope.sharpPolishSeen = seen;
                slope.sharpSunPolish = layers.under * thinWallGain(s) * splatPolish(s, nDotV) * sunShare;
                slope.sharpSunCentre += layers.under * thinWallGain(s) * sunPolish;
                slope.sharpMirror = mirror;
                slope.sharpMirrorX = mirrorX;
                slope.sharpMirrorY = mirrorY;
                slope.sharpOn = true;
                slope.slopeX = float3(0.0);
                slope.slopeY = float3(0.0);
            } else {
                slope.slopeX = polishWeight * (envRadiance(envTexels, dome, mirrorX, s.roughness, envBaseSide) - seen);
                slope.slopeY = polishWeight * (envRadiance(envTexels, dome, mirrorY, s.roughness, envBaseSide) - seen);
            }
            if (coated && !(slope.sharpWeight > 0.0)) {
                const float r = s.lobes.coatRoughness;
                const float3 coatSeen = envRadiance(envTexels, dome, mirror, r, envBaseSide);
                const float coatOpen = splatLobeOpen(s, cellBits, mirror, r, s.normal);
                const float3 coatWeight = layers.coat * (cells ? coatOpen : 1.0);
                slope.slopeX += coatWeight * (envRadiance(envTexels, dome, mirrorX, r, envBaseSide) - coatSeen);
                slope.slopeY += coatWeight * (envRadiance(envTexels, dome, mirrorY, r, envBaseSide) - coatSeen);
            }
        }
        if (field) {
            // The field is the coat's: on the base's polish it read the metals
            // a third bright (chrome 0.261 where the path traced frame reads
            // 0.197, exact without it), while the paint's closed reflections
            // are its lacquer's.
            const float3 closed = float3(0.0);
            return body2 +
                   layers.under * thinWallGain(s) *
                       (max(splatPolish(s, nDotV) * (narrowed * seen + (1.0 - narrowed) * closed), float3(0.0)) +
                        sunPolish) +
                   coat + sheen;
        }
        return body2 +
               layers.under * thinWallGain(s) *
                   (splatPolish(s, nDotV) * narrowed * seen + sunPolish) +
               coat + sheen;
    }
    float3 above, behind, reflected;
    if (dome == kEnvNone) {
        above = lightHit(light, s.position, s.normal).radiance;
        behind = lightHit(light, s.position, -s.normal).radiance;
        reflected = lightHit(light, s.position, mirror).radiance;
    } else {
        // Irradiance over pi is the radiance a Lambert body sends back, which
        // is what the constant dome above hands over directly.
        const float kOverPi = 1.0 / 3.14159265358979;
        above = envIrradiance(envSh, dome, s.normal) * kOverPi;
        behind = envIrradiance(envSh, dome, -s.normal) * kOverPi;
        // The disc `env_sun` took out of the harmonics, given back unshadowed:
        // a cloud with no transfer has nothing to shadow it by.
        const float4 disc = envSun[dome * 2];
        if (disc.w > 0.0) {
            const float3 irradiance = envSun[dome * 2 + 1].rgb;
            const float cosine = dot(s.normal, disc.xyz);
            above += irradiance * (max(cosine, 0.0) * kOverPi);
            behind += irradiance * (max(-cosine, 0.0) * kOverPi);
        }
        reflected = envRadiance(envTexels, dome, mirror, s.roughness, envBaseSide);
    }
    // A prepared sky gives the transmitted half the same answer a transfer
    // cloud gets: what stands behind, bent by the index. The complement is
    // taken only there, where the polish is the prepared sky's too; a
    // constant dome, and translucency with no index, keep what they did.
    if (dome != kEnvNone && (s.hasBehind || s.hasExit || s.ior > 1.0)) {
        behind = transmittedBehind(s, wo, dome, envTexels, envSh, envBaseSide) * transmittedShare(s, nDotV);
    }
    // The coat reflects along the same mirror, read at its own roughness.
    float3 coat = float3(0.0);
    if (coated) {
        const float3 coatSeen = s.hasReflection     ? s.reflectionColour
                                : dome == kEnvNone ? reflected
                                                   : envRadiance(envTexels, dome, mirror, s.lobes.coatRoughness,
                                                                 envBaseSide);
        coat = layers.coat * coatSeen;
    }
    return layers.under * body * ((1.0 - s.transmission) * above + transmittedWeight(s) * behind) +
           layers.under * thinWallGain(s) * splatPolish(s, nDotV) *
               (s.hasReflection ? s.reflectionColour : reflected) +
           coat + layers.belowCoat * layers.sheen * above;
}

/// The light `light` puts on `s`, seen from `wo`, with `through` of it
/// arriving.
///
/// Three terms: the body of the material facing the light, the body of it
/// facing away -- which is the translucent half, and is why a light behind a
/// glass splat lights it instead of leaving it black -- and the reflection,
/// which is on the lit side only.
public float3 relitByOne(SplatSurface s, float3 wo, LightSample ls, float3 through) {
    if (ls.pdf <= 0.0) {
        return float3(0.0);
    }
    const float nDotL = dot(s.normal, ls.wi);
    const float front = max(nDotL, 0.0);
    const float back = -nDotL;   // negative in front of the surface; the wrap below clamps it
    // Nothing to gather where the body is already lit: what was baked is the
    // light on it, and lighting it again would be that light twice.
    const float3 body = s.lit ? float3(0.0) : s.albedo * (1.0 - s.metallic) * (1.0 / 3.14159265358979);
    // Translucency is not transparency. What comes through a translucent
    // material has been scattered on the way, so it does not stop at the
    // terminator the way a reflection does and it does not draw what stands
    // behind: it wraps around the body and leaves softened. `kWrap` is how
    // far round it reaches -- half a hemisphere, which is what reads as jade
    // or wax rather than as a window.
    const float kWrap = 0.5;
    const float wrapped = saturate((back + kWrap) / (1.0 + kWrap));
    // Under the coat and the sheen, which take their share at the eye and
    // add their own lobes on the lit side.
    const SplatLayers layers = splatLayers(s, max(dot(s.normal, wo), 1.0e-4));
    const float3 diffuse = layers.under * body * ((1.0 - s.transmission) * front + transmittedWeight(s) * wrapped);
    const float3 specular = layers.under * specularLobe(s, wo, ls.wi) * front * thinWallGain(s) +
                            (coatLobe(s, wo, ls.wi) + layers.belowCoat * sheenLobe(s, wo, ls.wi)) * front;
    return through * (diffuse + specular) * ls.radiance / ls.pdf;
}

/// THE ANGLE A LIGHT SUBTENDS FROM A GAUSSIAN, half of it: a sphere's or a
/// disk's radius over the distance, half a rect's larger side, half a
/// distant light's angular diameter. What its shadow is soft over.
public float lightHalfAngle(LightRecord light, float distance) {
    if (light.kind == kLightDistant) {
        return 0.5 * light.sizeX;
    }
    const float size = light.kind == kLightRect ? 0.5 * max(light.sizeX, light.sizeY) : light.sizeX;
    return atan(max(size, 0.0) / max(distance, 1.0e-4));
}

/// WHAT A LIGHT THAT IS NOT THE SKY PUTS ON A TX GAUSSIAN AFTER A BOUNCE
/// (task TX, step 4): the indirect half read along the light, which is how
/// much of it the surroundings send on to the body, and the reflected field
/// scaled to it, which is what a reflection shows where its lobe is closed --
/// the floor a lamp lights, seen in the paint. The same terms a dome's sun
/// gets, from one sample of the light: `radiance / pdf` is what it delivers.
public float3 splatLightBounce(SplatSurface s, float3 wo, LightSample ls, StructuredBuffer<uint> transfer,
                               StructuredBuffer<uint> cellBits) {
    if (ls.pdf <= 0.0 || transferIndirectCount(s.transferCount) == 0u) {
        return float3(0.0);
    }
    const float3 delivered = ls.radiance / ls.pdf;
    const float3 along = splatIndirectAlong(s, transfer, ls.wi);
    const float nDotV = max(dot(s.normal, wo), 1.0e-4);
    const SplatLayers layers = splatLayers(s, nDotV);
    const float3 bounce = delivered * along;
    float3 out = s.lit ? float3(0.0)
                       : layers.under * s.albedo * (1.0 - s.metallic) * (1.0 - transmittedWeight(s)) * bounce +
                             layers.belowCoat * layers.sheen * bounce;
    if (transferHasField(s.transferCount) && !s.reflectionTraced) {
        const uint base = s.transferBase + transferDirectCount(s.transferCount);
        const float3 white = transferRgbAt(transfer, base) * 3.5449077018110318;
        const float3 coupling = float3(white.x > 1.0e-4 ? bounce.x / white.x : 0.0,
                                       white.y > 1.0e-4 ? bounce.y / white.y : 0.0,
                                       white.z > 1.0e-4 ? bounce.z / white.z : 0.0);
        const float3 mirror = reflect(-wo, s.normal);
        const float open = splatLobeOpen(s, cellBits, mirror, s.roughness, s.normal);
        out += layers.under * thinWallGain(s) * splatPolish(s, nDotV) * (1.0 - open) * coupling *
               splatFieldAlong(s, transfer, mirror, s.roughness);
        if (s.lobes.coatWeight > 0.0) {
            const float coatOpen = splatLobeOpen(s, cellBits, mirror, s.lobes.coatRoughness, s.normal);
            out += layers.coat * (1.0 - coatOpen) * coupling *
                   splatFieldAlong(s, transfer, mirror, s.lobes.coatRoughness);
        }
    }
    return max(out, float3(0.0));
}

/// Every light this splat is linked to, one sample each at its centre.
///
/// `shadowLights` of 0 means nothing measured a shadow, and each light is
/// taken whole; otherwise `shadowFactors[shadowBase + k]` is what reaches the
/// splat from light k (splat_shadow.slang).
public float3 relitSplat<let kTransfer : int>(SplatSurface s, float3 wo, StructuredBuffer<LightRecord> lights, uint lightCount,
                         uint categoriesLo, uint categoriesHi, StructuredBuffer<float> shadowFactors,
                         uint shadowBase, uint shadowLights, StructuredBuffer<uint> envTexels,
                         StructuredBuffer<float4> envSh, StructuredBuffer<uint> envOfLight,
                         uint envLights, uint envBaseSide, StructuredBuffer<float4> envSun,
                         StructuredBuffer<uint> transfer, StructuredBuffer<uint> cellBits, bool withIndirect,
                         inout SplatSlope slope) {
    // A baked body starts from the light it already carries -- its emission
    // in it; an albedo starts from what it gives off by itself and gathers.
    float3 lit = s.lit ? s.albedo : s.emission;
    for (uint k = 0; k < lightCount; ++k) {
        const LightRecord light = lights[k];
        // Light linking reaches a cloud as it reaches a mesh: by one bit.
        if (!lightLinked(light.lightCategory, categoriesLo, categoriesHi)) {
            continue;
        }
        const float through = k < shadowLights ? shadowFactors[shadowBase + k] : 1.0;
        if (light.kind == kLightDome) {
            // A dome is everywhere and nowhere: no distance to fade over.
            const uint dome = k < envLights ? envOfLight[k] : kEnvNone;
            lit += shadowTint(light, through, 0.0) *
                   relitByDome<kTransfer>(s, wo, light, dome, envTexels, envSh, envBaseSide, envSun, transfer, cellBits,
                               withIndirect, slope);
            continue;
        }
        const LightSample ls = sampleLight(light, s.position, s.normal, float2(0.5, 0.5));
        if (!ls.valid) {
            continue;
        }
        // A TX TRANSFER SHADOWS A LIGHT BY ITS OWN BITS (step 4), over the
        // cone the light subtends, which is its penumbra; where a ray also
        // measured a shadow the darker of the two stands. The bits say which
        // ways leave the scene, so an occluder beyond a light inside the
        // scene shadows it too: right for a sun and a lamp far off, the
        // approximation for one standing among things.
        if (kTransfer >= 2 && s.cellWords >= 8u && (light.flags & kLightShadow) != 0u) {
            const float open = splatConeOpen(s, cellBits, ls.wi, lightHalfAngle(light, ls.distance), s.normal);
            const float bits = open >= 0.0 ? open : 1.0;
            const float shadowed = k < shadowLights ? min(through, bits) : bits;
            lit += relitByOne(s, wo, ls, shadowTint(light, shadowed, ls.distance));
            if (withIndirect) {
                lit += splatLightBounce(s, wo, ls, transfer, cellBits);
            }
            continue;
        }
        lit += relitByOne(s, wo, ls, shadowTint(light, through, ls.distance));
    }
    return lit;
}

/// The rotation a packed quaternion stands for, columns in the frame's order.
public float3x3 splatRotation(float4 q) {
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    return float3x3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
                    2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
                    2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y));
}

/// The normal a splat stands for: its shortest axis, turned towards the eye.
/// `rotation` is the frame's three columns, `scale` its three sizes.
/// Which side of a gaussian the eye is on, before the normal is turned
/// towards it: positive where the eye sees the face the surface faced.
public float splatFacing(float3x3 rotation, float3 scale, float3 position, float3 eye) {
    const uint shortest = scale.x <= scale.y ? (scale.x <= scale.z ? 0u : 2u) : (scale.y <= scale.z ? 1u : 2u);
    const float3 normal = normalize(float3(rotation[0][shortest], rotation[1][shortest], rotation[2][shortest]));
    return dot(normal, normalize(eye - position));
}

public float3 splatNormal(float3x3 rotation, float3 scale, float3 position, float3 eye) {
    const uint shortest = scale.x <= scale.y ? (scale.x <= scale.z ? 0u : 2u) : (scale.y <= scale.z ? 1u : 2u);
    float3 normal = normalize(float3(rotation[0][shortest], rotation[1][shortest], rotation[2][shortest]));
    if (dot(normal, normalize(eye - position)) < 0.0) {
        normal = -normal;
    }
    return normal;
}

/// THE NORMAL A RELIT SPLAT IS SHADED WITH. Where the cloud keeps a shading
/// normal of its own (`stored`, packed, `hasStored`) it is that one: what a
/// mesh's normal map made of the surface, which the disc's short axis -- the
/// face's own normal -- never knew. A splat is two-sided, so the disc's axis
/// is turned towards the eye first (`splatNormal`), and the stored normal is
/// put on the same side of the disc: the eye on the far side of a surface sees
/// its relief from the far side. Everything geometric -- the footprint, where
/// a ray meets the disc, the plane a reflection clears -- keeps the disc's.
public float3 splatShadingNormal(float3x3 rotation, float3 scale, float3 position, float3 eye, bool hasStored,
                                 uint stored) {
    const float3 disc = splatNormal(rotation, scale, position, eye);
    if (!hasStored) {
        return disc;
    }
    const float3 n = unpackNormal(stored);
    return dot(n, disc) < 0.0 ? -n : n;
}

/// A point of the cloud in the world, by the rows a caller passes.
public float3 relightToWorld(float4 row0, float4 row1, float4 row2, float3 p) {
    return float3(dot(row0, float4(p, 1.0)), dot(row1, float4(p, 1.0)), dot(row2, float4(p, 1.0)));
}

/// A direction of the cloud in the world -- a ray, the way out of a glass --
/// by the same rows, made unit. A normal is not a direction under a scale
/// that is not uniform: relightNormalToWorld.
public float3 relightDirectionToWorld(float4 row0, float4 row1, float4 row2, float3 v) {
    const float3 turned = float3(dot(row0.xyz, v), dot(row1.xyz, v), dot(row2.xyz, v));
    return dot(turned, turned) > 1.0e-20 ? normalize(turned) : float3(0.0, 0.0, 1.0);
}

/// A NORMAL of the cloud in the world, by the same rows: not as a direction
/// but by the inverse transpose, which keeps it perpendicular to the surface
/// under any linear map. Up to a scale that is the cofactor matrix, whose
/// columns are `b x c`, `c x a` and `a x b` for the columns `a`, `b`, `c` of
/// the linear part -- what the skinner turns a normal with too. Taken as a
/// direction, a card scaled (2, 1, 1) with its normal tilted 45 degrees in
/// x tilted to 63 instead of 27. A mirror flips the cofactor against the
/// inverse transpose, so the determinant's sign is put back: the normal stays
/// on the side of the surface it was on, the eye's.
public float3 relightNormalToWorld(float4 row0, float4 row1, float4 row2, float3 n) {
    const float3 a = float3(row0.x, row1.x, row2.x);
    const float3 b = float3(row0.y, row1.y, row2.y);
    const float3 c = float3(row0.z, row1.z, row2.z);
    const float3 bc = cross(b, c);
    const float side = dot(a, bc) < 0.0 ? -1.0 : 1.0;
    const float3 turned = side * (n.x * bc + n.y * cross(c, a) + n.z * cross(a, b));
    return dot(turned, turned) > 1.0e-20 ? normalize(turned) : float3(0.0, 0.0, 1.0);
}

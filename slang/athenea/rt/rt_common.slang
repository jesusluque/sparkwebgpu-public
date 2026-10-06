// Copyright (c) 2026 jesus luque.
//
// Ray tracing Gaussians, the way 3DGRT does (Moenne-Loccoz et al. 2024):
// each Gaussian is enclosed by a stretched icosahedron, the proxies go into a
// hardware BVH, and a ray gathers the k nearest proxies it enters, integrates
// those particles at their maximum response along the ray, and continues from
// the last one. No projection, no tiles, no sort of the scene: the order is
// the ray's own.
module rt_common;
import athenea.common.packing;
__exported import athenea.common.edit;

/// A ray, as both routes carry it. Not `RayDesc`: that type exists on a
/// target only where ray tracing does, and the compute BVH route is meant to
/// run where there is none (CUDA without OptiX -- measured, nvrtc: "identifier
/// RayDesc is undefined"). The hardware route fills a RayDesc from this.
public struct AtheneaRay {
    public float3 Origin;
    public float  TMin;
    public float3 Direction;
    public float  TMax;
};

public static const uint kIcoVertices = 12;
public static const uint kIcoTriangles = 20;
/// Inradius of the unit-circumradius icosahedron: vertices are scaled by
/// cutoff / this, so the icosahedron's faces clear the cutoff ellipsoid.
public static const float kIcoInradius = 0.7946544723;

public struct RtParams {
    public uint  width;
    public uint  height;
    public float focalX;
    public float focalY;
    public float centreX;
    public float centreY;
    public float nearZ;
    public float farZ;
    public uint  orthographic;
    public uint  count;          // splats in this cloud (proxy, frames, shade)
    public uint  base;           // proxy: splats per chunk; frames, shade: first slot written
    public uint  shWords;        // shade: this cloud's words per splat
    // view -> world, rows of a 3x4 (camera to world with the kernel's +z)
    public float v00; public float v01; public float v02; public float v03;
    public float v10; public float v11; public float v12; public float v13;
    public float v20; public float v21; public float v22; public float v23;
    public uint  shLimit;
    public uint  linearCloud;    // shade: 1 this cloud's colours are linear light, 0 sRGB (common/color.slang)
    public float bgR; public float bgG; public float bgB; public float bgA;
    public uint  restPerColour;  // shade: this cloud's harmonics
    public uint  instances;      // bvh render: instances to traverse
    public uint  maxSegments;
    public uint  depthMode;      // 0 alpha-weighted mean, 1 first crossing
    public float depthThreshold;
    /// Where a query is cut in two and composed again (rtRenderSplit): what
    /// proves the segment query composes. 0: one query, as a frame draws.
    public float splitAt;
    // shade: what reaches each splat from each light, measured before the
    // frame (technique::SplatVisibility) -- `shadowFactors[(shadowBase + i) *
    // shadowLights + k]`. Zero lights: every light is taken whole.
    public uint  shadowBase; public uint shadowLights;
    // shade: what this cloud is relit with. The rows take a splat to the
    // world, where the lights are; the eye is there too.
    public uint  relight;        // 1: light these splats rather than show them as baked
    public uint  lightCount;
    public uint  categoriesLo; public uint categoriesHi;
    public uint  hasPbr;         // 1: `pbr` holds this cloud's metallic and roughness
    public uint  hasNormals;     // 1: `normals` holds this cloud's shading normals
    public uint  hasEmission;    // 1: `emission` holds the radiance this cloud's gaussians give off
    public uint  hasLobes;       // 1: `lobes` holds what this cloud's materials layered over the base
    public uint  litBody;        // 1: the colours are light already, and only the polish is added
    /// Lights an environment was prepared for: `envOfLight[k]` is the slice
    /// of light k, or `kEnvNone`. Zero: no sky prepared, and a dome is its
    /// colour as it always was.
    public uint  envLights;
    /// The octahedral side of that sky's level 0, which follows the dome's
    /// own image rather than being fixed: one number both the writer and the
    /// reader of the map must agree on (athenea/light/environment.slang).
    public uint  envBaseSide;
    /// ROWS OF `splatOverrides`, the per-prim material table a pick edits:
    /// 0 says there is none and the cloud's own values stand alone.
    public uint  overrideCount;
    /// A cloud's transfer, as the rasteriser's frame carries it.
    public uint  transferWords;
    public uint  transferCount;
    public uint  transferIndirect;
    /// The words a gaussian of which ways out are open (`shadowBits`), 0 where
    /// the cloud carries none: 2 for 8 x 8 cells, 8 or 32 for a TX transfer.
    public uint  shadowBits;
    /// The index a transmitting gaussian bends the sky by; 0 bends nothing.
    /// What it does and what it cannot is in splat_relight.slang.
    public float ior;
    /// THE CLOUD'S OWN TREE, and what a gaussian casts into it: the far face
    /// of the glass it stands on, and what its mirror direction meets.
    /// Particles of 0: no tree, and both fall back to the sky
    /// (rt_glass.slang).
    public uint  glassNodeBase;
    public uint  glassBase;
    public uint  glassParticles;
    /// 1: the second pass may look along the mirror direction for the cloud's
    /// own particles, which is a reflection of the asset in itself.
    public uint  reflectCloud;
    /// 1: this is the second pass over a cloud's glass, and the colours of
    /// everything else are already written, so glass may show what is behind
    /// it rather than the sky.
    public uint  glassPass;
    public uint  pad5;
    public float w00; public float w01; public float w02; public float w03;
    public float w10; public float w11; public float w12; public float w13;
    public float w20; public float w21; public float w22; public float w23;
    public float eyeWorldX; public float eyeWorldY; public float eyeWorldZ; public float pad6;
    public EditParams edit;      // shade: this instance's SplatEdit (common/edit.slang)
};

/// The squared Mahalanobis radius a particle is cut at: the raster's rule, so
/// the two routes draw the same extent of every particle.
public float rtCutoff(float alpha) {
    return min(2.0 * log(max(255.0 * alpha, 1.0001)), 9.0);
}

public float3x3 rotationOf(float4 q) {
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    return float3x3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
                    2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
                    2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y));
}

/// The box around a particle's cut ellipsoid, cloud-local: what the compute
/// route's BVH holds at a leaf. `q`, `s01` and `s2` are the first three words
/// of its shape record. One function for the build (bvh_leaves) and the
/// refit (bvh_refit_level), so a refitted tree bounds exactly what a built one does.
public void rtParticleBox(float4 p, uint q, uint s01, uint s2, out float3 lo, out float3 hi) {
    const float3 s = exp(float3(lowHalf(s01), highHalf(s01), lowHalf(s2)));
    const float3x3 rs = mul(rotationOf(decodeQuaternion(q)),
                            float3x3(s.x, 0.0, 0.0, 0.0, s.y, 0.0, 0.0, 0.0, s.z));
    // The ellipsoid x = c + (R S) u, |u| <= r, reaches r |row a of R S| along axis a.
    const float r = sqrt(rtCutoff(p.w));
    const float3 half = r * float3(length(rs[0]), length(rs[1]), length(rs[2]));
    lo = p.xyz - half;
    hi = p.xyz + half;
}

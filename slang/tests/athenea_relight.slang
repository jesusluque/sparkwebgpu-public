// athenea's relit colour on athenea's own buffer layout, for
// test/gpu/atheneaRelight.test.ts: splat_project's relitToward (txf 89a04d9)
// with its `params.*` and per-stream buffers -- pbr[i], lobes[i * 3],
// normals[i], emission[i], a transfer at i * transferWords * 2 halves, the
// cells at i * shadowBits words -- and relitSplat as it is. The splat's
// geometry is read from the same `src` the relight plugin reads (ext), so
// what the test compares is the plugin's mapping onto athenea: the pool
// addressing, the object's rows, the kept terms (f16) and the colour path.
//
// The streams sit in one buffer, `native`, at the offsets the params give,
// because ten storage buffers is what a stage has: athenea's functions take
// a buffer and a base, so the arithmetic is the native one either way.
import core.splat_types;
import core.ext;
import athenea.splat.splat_relight;
import athenea.common.packing;
import athenea.common.color;
import athenea.scene.splat_encoding;

struct RefParams {
    uint count;
    uint transferCount;
    uint transferWords;
    uint shadowBits;      ///< words a gaussian, as splat_project's params.shadowBits

    uint pbrAt;           ///< word offsets of the streams in `native` (0xFFFFFFFF: none)
    uint lobesAt;
    uint normalsAt;
    uint emissionAt;

    uint transferAt;
    uint cellsAt;
    uint lightCount;
    uint envLights;

    uint envBaseSide;
    uint transferIndirect;
    uint litBody;
    uint linearCloud;

    float ior;
    float curvature;      ///< every splat's shape operator (k, 0, k) on its third axis (0: none)
    uint pad1;
    uint pad2;

    float4 w0;
    float4 w1;
    float4 w2;
    float4 eyeCloud;
    float4 eyeWorld;
};

StructuredBuffer<uint4>        src;
StructuredBuffer<uint>         native;
StructuredBuffer<LightRecord>  lights;
StructuredBuffer<float4>       envSh;
StructuredBuffer<float4>       envSun;
StructuredBuffer<uint>         envTexels;
StructuredBuffer<uint>         envOfLight;
RWStructuredBuffer<float4>     outColour;
ConstantBuffer<RefParams>      params;

static const uint kNone = 0xFFFFFFFFu;

float3 relitToward<let kTransfer : int>(uint i) {
    const Gsplat g = readExtSplat(src[2u * i], src[2u * i + 1u], int(i));
    const float3 position = g.center;
    const float3 scale = g.scales;
    const float3x3 r = splatRotation(g.quaternion);
    const bool hasNormals = params.normalsAt != kNone;
    const float3 cloudNormal = splatShadingNormal(r, scale, position, params.eyeCloud.xyz, hasNormals,
                                                  hasNormals ? native[params.normalsAt + i] : 0u);
    const float3 colour = cloudLight(g.rgba.rgb, params.linearCloud != 0);
    const float4 w0 = params.w0;
    const float4 w1 = params.w1;
    const float4 w2 = params.w2;
    const bool hasPbr = params.pbrAt != kNone;
    const uint pbr = hasPbr ? native[params.pbrAt + i] : 0u;
    SplatSurface surface;
    surface.position = relightToWorld(w0, w1, w2, position);
    surface.normal = relightNormalToWorld(w0, w1, w2, cloudNormal);
    surface.albedo = colour;
    surface.metallic = hasPbr ? pbrMetallic(pbr) : 0.0;
    surface.roughness = hasPbr ? pbrRoughness(pbr) : 1.0;
    surface.transmission = hasPbr ? pbrTransmission(pbr) : 0.0;
    surface.thinWalled = hasPbr && pbrThinWalled(pbr);
    surface.schlickMetal = hasPbr && pbrSchlickMetal(pbr);
    surface.ior = params.ior;
    surface.backFacing = splatFacing(r, scale, position, params.eyeCloud.xyz) < 0.0;
    surface.hasBehind = false;
    surface.behindColour = float3(0.0);
    surface.exitThrough = float3(1.0);
    surface.hasReflection = false;
    surface.reflectionColour = float3(0.0);
    surface.reflectionTraced = false;
    surface.hasExit = false;
    surface.exitDirection = float3(0.0, 0.0, 1.0);
    surface.lit = params.litBody != 0;
    surface.emission = params.emissionAt != kNone ? unpackRgb9e5(native[params.emissionAt + i]) : float3(0.0);
    const bool hasLobes = params.lobesAt != kNone;
    surface.lobes = hasLobes ? unpackLobes(uint3(native[params.lobesAt + i * 3], native[params.lobesAt + i * 3 + 1],
                                                 native[params.lobesAt + i * 3 + 2]))
                             : plainLobes();
    if (hasLobes && surface.ior > 1.0 && surface.transmission > 0.0) {
        surface.ior = max(surface.lobes.specularIor, 1.0);
    }
    surface.transferCount = params.transferCount;
    surface.transferBase = params.transferAt * 2u + i * params.transferWords * 2;
    splatTransferFrame(surface, native, r, w0, w1, w2);
    surface.hasShadowBits = params.shadowBits != 0 && params.shadowBits < 8u && params.transferCount >= 9;
    surface.shadowBits = surface.hasShadowBits ? uint2(native[params.cellsAt + i * 2], native[params.cellsAt + i * 2 + 1])
                                               : uint2(0);
    surface.cellWords = params.shadowBits >= 8u && params.transferCount >= 9 ? params.shadowBits : 0u;
    surface.cellBase = params.cellsAt + i * surface.cellWords;
    SplatSlope slope = noSlope();
    // splat_project's mean curvature, on the face the eye sees.
    if (kTransfer >= 2 && params.curvature != 0.0) {
        const float3 axisN = float3(r[0][2], r[1][2], r[2][2]);
        const float faces = dot(cloudNormal, axisN) >= 0.0 ? 1.0 : -1.0;
        slope.curvature = faces * 0.5 * (params.curvature + params.curvature);
    }
    return relitSplat<kTransfer>(surface, normalize(params.eyeWorld.xyz - surface.position), lights,
                                 params.lightCount, 0xFFFFFFFFu, 0xFFFFFFFFu, iesValues, 0u, 0u, envTexels, envSh,
                                 envOfLight, params.envLights, params.envBaseSide, envSun, native, native,
                                 params.transferIndirect != 0, slope);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void referenceRelight(uint3 tid: SV_DispatchThreadID) {
    if (tid.x < params.count) {
        outColour[tid.x] = float4(relitToward<2>(tid.x), 1.0);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void referenceRelightPlain(uint3 tid: SV_DispatchThreadID) {
    if (tid.x < params.count) {
        outColour[tid.x] = float4(relitToward<0>(tid.x), 1.0);
    }
}

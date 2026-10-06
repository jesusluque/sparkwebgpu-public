// Copyright (c) 2026 jesus luque.
//
// How much of each light reaches each splat: one ray a splat a light, against
// the cloud's own proxies, through rt_shadow.slang's product of (1 - alpha).
// A kernel of its own because an inline ray is a feature a device may not
// have -- with the RayQuery inside splat_project the whole splat rasteriser
// became unavailable on CUDA -- and because a splat's shadow does not change
// with the pixel: it is one number a frame, read by the projection.
module splat_shadow;
import athenea.common.packing;
import athenea.light.lights;
import athenea.rt.rt_shadow;
import athenea.rt.rt_common;

struct SplatShadowParams {
    uint  count;        // splats in this cloud
    uint  base;         // its first slot in `factors`
    uint  lights;       // lights a factor is measured for
    float offset;       // where the ray starts, in the splat's own sigmas
    float cut;          // and where it gives up
    uint  categoriesLo; // this cloud's light linking
    uint  categoriesHi;
    uint  pad0;
    // object -> world, rows of a 3x4.
    float w00; float w01; float w02; float w03;
    float w10; float w11; float w12; float w13;
    float w20; float w21; float w22; float w23;
};

StructuredBuffer<float4>      positions;
StructuredBuffer<uint>        shape;
StructuredBuffer<LightRecord> lights;
StructuredBuffer<float4>      shadowFrames;
StructuredBuffer<float4>      shadowColours;
StructuredBuffer<float4>      shadowInstanceData;
StructuredBuffer<uint>        shadowInstanceIndices;
RWStructuredBuffer<float>     factors;
ConstantBuffer<SplatShadowParams> params;

uniform RaytracingAccelerationStructure scene;

float3 toWorld(float3 p) {
    return float3(params.w00 * p.x + params.w01 * p.y + params.w02 * p.z + params.w03,
                  params.w10 * p.x + params.w11 * p.y + params.w12 * p.z + params.w13,
                  params.w20 * p.x + params.w21 * p.y + params.w22 * p.z + params.w23);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void splatShadowFactors(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint slot = params.base + i;
    const float4 posOpacity = positions[i];
    const float3 position = toWorld(posOpacity.xyz);
    // The splat's own reach, so the ray starts past the crowd of Gaussians
    // this one is part of: without it every splat is shadowed by the splats
    // it is made of, and a relit capture renders black.
    const uint row = i * 4;
    const float3 scale = exp(float3(lowHalf(shape[row + 1]), highHalf(shape[row + 1]), lowHalf(shape[row + 2])));
    const float own = max(scale.x, max(scale.y, scale.z));
    for (uint k = 0; k < params.lights; ++k) {
        float through = 1.0;
        const LightRecord light = lights[k];
        if (lightLinked(light.lightCategory, params.categoriesLo, params.categoriesHi) &&
            (light.flags & kLightShadow) != 0) {
            // The direction the relight takes: the light's centre, as
            // splat_project samples it.
            const LightSample ls = sampleLight(light, position, float3(0.0, 0.0, 1.0), float2(0.5, 0.5));
            if (ls.valid) {
                const float start = max(params.offset * own, 1.0e-4);
                ShadowState state = shadowBegin();
                RayDesc traced;
                traced.Origin = position;
                traced.Direction = ls.wi;
                traced.TMin = start;
                traced.TMax = max(ls.distance - start, start);
                RayQuery<RAY_FLAG_NONE> query;
                query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, traced);
                while (query.Proceed()) {
                    if (query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) {
                        continue;
                    }
                    const uint instance = query.CandidateInstanceIndex();
                    const uint id = shadowInstanceIndices[instance * 2] +
                                    query.CandidatePrimitiveIndex() / kIcoTriangles;
                    const float4 w0 = shadowInstanceData[instance * 3];
                    const float4 w1 = shadowInstanceData[instance * 3 + 1];
                    const float4 w2 = shadowInstanceData[instance * 3 + 2];
                    const float3x3 toCloud = float3x3(w0.xyz, w1.xyz, w2.xyz);
                    const float3 origin = mul(toCloud, traced.Origin) + float3(w0.w, w1.w, w2.w);
                    const float3 direction = mul(toCloud, traced.Direction);
                    const float opacity = shadowColours[id + shadowInstanceIndices[instance * 2 + 1]].w;
                    float tPeak;
                    const float alpha = shadowAlpha(shadowFrames[id * 4], shadowFrames[id * 4 + 1],
                                                    shadowFrames[id * 4 + 2], opacity, origin, direction,
                                                    traced.TMin, traced.TMax, 0.0, 1e30, tPeak);
                    if (alpha > 0.0) {
                        shadowTake(state, id, instance, alpha);
                        if (shadowDone(state, params.cut)) {
                            query.Abort();
                            break;
                        }
                    }
                }
                through = state.transmittance;
            }
        }
        factors[slot * params.lights + k] = through;
    }
}

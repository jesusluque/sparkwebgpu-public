// Copyright (c) 2026 jesus luque.
//
// The bake of a part's visibility field, a module of its own because it is
// the one kernel here that traces an inline ray, and an inline ray is a
// feature a device may not have: on CUDA the whole module would fail to
// compile and take the read down with it. Without this module a cloud cannot
// be baked on that device, but one baked elsewhere is read there all the
// same -- the fields travel in the file (splat_visibility.slang says what
// they are).
module splat_visibility_bake;
import athenea.common.packing;
import athenea.common.octahedral;
import athenea.rt.rt_shadow;
import athenea.rt.rt_common;
import athenea.splat.splat_visibility_common;

// ----------------------------------------------------------------------------
// 2. The bake: one thread a (probe, direction), through one part's proxies.

struct BakeParams {
    uint  part;         // which VisibilityPart
    uint  threads;      // grid^3 * octave^2
    float cut;          // stop once transmittance is under this
    float start;        // the ray's TMin, in the space of the cloud: one probe cell
    uint  pad0; uint pad1; uint pad2; uint pad3;
};

StructuredBuffer<VisibilityPart> parts;
StructuredBuffer<float4>         shadowFrames;
StructuredBuffer<float4>         shadowColours;
StructuredBuffer<float4>         shadowInstanceData;
StructuredBuffer<uint>           shadowInstanceIndices;
RWStructuredBuffer<uint>         texels;     // two f16 a word: [texel/2]
ConstantBuffer<BakeParams>       bakeParams;
uniform RaytracingAccelerationStructure scene;

/// Where texel `t` of a field lives, as a word and a half of it.
uint texelWord(uint t) { return t >> 1; }

void storeHalf(uint t, float v) {
    // Two texels share a word; only this thread's half is changed. The
    // dispatch is one thread a texel, so the two halves of a word are
    // written by two threads: an atomic or-after-clear keeps them apart.
    const uint bits = f32tof16(v) & 0xFFFFu;
    const uint shifted = (t & 1u) != 0u ? (bits << 16) : bits;
    InterlockedOr(texels[texelWord(t)], shifted);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void visBake(uint3 tid: SV_DispatchThreadID) {
    const uint t = tid.x;
    if (t >= bakeParams.threads) {
        return;
    }
    const VisibilityPart part = parts[bakeParams.part];
    const uint dirs = part.octave * part.octave;
    const uint probe = t / dirs;
    const uint dir = t - probe * dirs;
    const uint g = part.grid;
    const uint x = probe % g;
    const uint y = (probe / g) % g;
    const uint z = probe / (g * g);
    const float3 unit = (float3(x, y, z) + 0.5) / float(g);
    const float3 origin = lerp(part.boxMin.xyz, part.boxMax.xyz, unit);
    const float2 f = (float2(dir % part.octave, dir / part.octave) + 0.5) / float(part.octave);
    const float3 direction = octDecode(f);

    ShadowState state = shadowBegin();
    RayDesc traced;
    traced.Origin = origin;
    traced.Direction = direction;
    traced.TMin = bakeParams.start;
    traced.TMax = 1.0e30;
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, traced);
    while (query.Proceed()) {
        if (query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) {
            continue;
        }
        const uint instance = query.CandidateInstanceIndex();
        const uint id = shadowInstanceIndices[instance * 2] + query.CandidatePrimitiveIndex() / kIcoTriangles;
        const float4 w0 = shadowInstanceData[instance * 3];
        const float4 w1 = shadowInstanceData[instance * 3 + 1];
        const float4 w2 = shadowInstanceData[instance * 3 + 2];
        const float3x3 toCloud = float3x3(w0.xyz, w1.xyz, w2.xyz);
        const float3 o = mul(toCloud, traced.Origin) + float3(w0.w, w1.w, w2.w);
        const float3 d = mul(toCloud, traced.Direction);
        const float opacity = shadowColours[id + shadowInstanceIndices[instance * 2 + 1]].w;
        float tPeak;
        const float alpha = shadowAlpha(shadowFrames[id * 4], shadowFrames[id * 4 + 1], shadowFrames[id * 4 + 2],
                                        opacity, o, d, traced.TMin, traced.TMax, 0.0, 1e30, tPeak);
        if (alpha > 0.0) {
            shadowTake(state, id, instance, alpha);
            if (shadowDone(state, bakeParams.cut)) {
                query.Abort();
                break;
            }
        }
    }
    storeHalf(part.offset + t, state.transmittance);
}


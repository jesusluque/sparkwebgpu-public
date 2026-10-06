// Copyright (c) 2026 jesus luque.
//
// WHAT A CLOUD CARRIES BESIDE ITS GAUSSIANS, THROUGH ITS LEVELS OF DETAIL
// (docs/decisions.md, task TX, "The levels of detail and .athc"): the
// material (`pbr`, the layers), the matte's ids, a transfer and its open
// directions. Each is a buffer of `words` words a gaussian, and three things
// are done to one, whatever it holds:
//
//   lodExtrasReorder  into the Morton order the store keeps (`order`)
//   lodExtrasMerge    a level's group out of the run of the order it covers:
//                     0 the first gaussian's words (an id, a material: a cell
//                     takes one), 1 f16 pairs averaged, weighed by opacity (a
//                     transfer: linear in the sky, so the mean of transfers
//                     is the transfer of the mean), 2 bits set where half the
//                     weight has them (which ways out are open)
//   lodExtrasGather   into the frame's cut, as positions and shape are
module lod_extras;

struct ExtrasParams {
    uint count;    ///< gaussians (reorder), groups (merge), candidates (gather)
    uint words;    ///< words a gaussian
    uint mode;     ///< merge: 0 first, 1 f16 mean, 2 bits by weight
    uint splats;   ///< merge: the store's gaussians, where the last run ends

    uint base;     ///< gather: the first slot of this part in the cut
    uint offset;   ///< gather: the first source gaussian of this part
    uint pad0;
    uint pad1;
};

StructuredBuffer<uint>     order;      ///< reorder: store index -> the cloud's
StructuredBuffer<uint>     starts;     ///< merge: a group's first store index
StructuredBuffer<float4>   positions;  ///< merge: the store's, w the opacity
StructuredBuffer<uint>     selected;   ///< gather
StructuredBuffer<uint>     dest;       ///< gather
StructuredBuffer<uint>     source;
RWStructuredBuffer<uint>   target;
ConstantBuffer<ExtrasParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void lodExtrasReorder(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint from = order[i];
    for (uint w = 0; w < params.words; ++w) {
        target[i * params.words + w] = source[from * params.words + w];
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void lodExtrasMerge(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    const uint first = starts[g];
    const uint end = g + 1 < params.count ? starts[g + 1] : params.splats;
    const uint out = g * params.words;
    if (params.mode == 0 || end <= first + 1) {
        for (uint w = 0; w < params.words; ++w) {
            target[out + w] = source[first * params.words + w];
        }
        return;
    }
    float total = 0.0;
    for (uint s = first; s < end; ++s) {
        total += max(positions[s].w, 0.0);
    }
    const float scale = total > 0.0 ? 1.0 / total : 1.0 / float(end - first);
    for (uint w = 0; w < params.words; ++w) {
        if (params.mode == 1) {
            float lo = 0.0;
            float hi = 0.0;
            for (uint s = first; s < end; ++s) {
                const float weight = total > 0.0 ? max(positions[s].w, 0.0) : 1.0;
                const uint word = source[s * params.words + w];
                lo += weight * f16tof32(word & 0xFFFFu);
                hi += weight * f16tof32(word >> 16);
            }
            target[out + w] = (f32tof16(hi * scale) << 16) | (f32tof16(lo * scale) & 0xFFFFu);
        } else {
            uint bits = 0u;
            for (uint b = 0; b < 32u; ++b) {
                float set = 0.0;
                for (uint s = first; s < end; ++s) {
                    const float weight = total > 0.0 ? max(positions[s].w, 0.0) : 1.0;
                    set += ((source[s * params.words + w] >> b) & 1u) != 0u ? weight : 0.0;
                }
                bits |= set * scale >= 0.5 ? (1u << b) : 0u;
            }
            target[out + w] = bits;
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void lodExtrasGather(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count || selected[i] == 0) {
        return;
    }
    const uint j = params.base + dest[i];
    const uint k = params.offset + i;
    for (uint w = 0; w < params.words; ++w) {
        target[j * params.words + w] = source[k * params.words + w];
    }
}

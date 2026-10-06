// Copyright (c) 2026 jesus luque.
//
// WHAT A CLOUD CARRIES BESIDE ITS GAUSSIANS, through a decimation: which
// record each store splat is, what each kept gaussian stands for, the keys
// and material the decimation compares, and every attribute of a kept
// gaussian made from the records it stands for (lod_decimate.slang decides
// which those are). Apart from lod_decimate because Metal binds 31 buffers
// a kernel and Slang numbers a module's globals together.
module lod_attributes;
import lod_blob;

public struct AttributeParams {
    public uint count;      // records, store splats, or kept gaussians
    public uint width;      // words a record
    public uint mode;       // how they merge (lodMergeAttribute); gather: 1 for splats; material: the component
    public uint base;       // gather: the first output slot of this part
};

StructuredBuffer<float4>          splatPositions;
StructuredBuffer<uint>            splatShape;
ConstantBuffer<AttributeParams>   params;

/// Each store splat's record: the Morton order, then the file's.
StructuredBuffer<uint>     storeOrder;   // per store splat: the cloud's splat
StructuredBuffer<uint>     origin;       // per cloud splat: its record
RWStructuredBuffer<uint>   recordOut;

[shader("compute")]
[numthreads(256, 1, 1)]
void lodRecordOf(uint3 tid: SV_DispatchThreadID) {
    const uint s = tid.x;
    if (s >= params.count) {
        return;
    }
    recordOut[s] = origin[storeOrder[s]];
}

/// What each kept gaussian stands for, as a run of store splats: a merge's
/// cell, or the one splat kept as it was.
StructuredBuffer<uint>     gatherSelected;
StructuredBuffer<uint>     gatherDest;
StructuredBuffer<uint2>    levelRange;
RWStructuredBuffer<uint2>  ranges;
StructuredBuffer<uint>     recordOf;      // per store splat: its record

[shader("compute")]
[numthreads(256, 1, 1)]
void lodGatherRanges(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count || gatherSelected[i] == 0) {
        return;
    }
    ranges[params.base + gatherDest[i]] = params.mode != 0 ? uint2(i, i + 1) : levelRange[i];
}

// A RECORD'S ATTRIBUTES: `width` words a record in `values` (and, for a rig,
// its weights in `weights`), in the file's order.
StructuredBuffer<uint>     values;
StructuredBuffer<uint>     weights;
RWStructuredBuffer<uint>   keyOut;
RWStructuredBuffer<float4> materialOut;
RWStructuredBuffer<uint>   merged;
RWStructuredBuffer<uint>   mergedWeights;

uint fold(uint key, uint value) {
    key ^= value + 0x9E3779B9u + (key << 6) + (key >> 2);
    return key * 0x85EBCA6Bu;
}

/// The joint that carries a record most, of the `width` it names.
uint dominantJoint(uint r) {
    uint best = 0;
    float most = -1.0;
    for (uint k = 0; k < params.width; ++k) {
        const float w = asfloat(weights[r * params.width + k]);
        if (w > most) {
            most = w;
            best = values[r * params.width + k];
        }
    }
    return best;
}

/// One attribute folded into each record's key: its words (mode 1), or the
/// joint that carries it most (mode 3).
[shader("compute")]
[numthreads(256, 1, 1)]
void lodAttributeKey(uint3 tid: SV_DispatchThreadID) {
    const uint r = tid.x;
    if (r >= params.count) {
        return;
    }
    uint key = keyOut[r];
    if (params.mode == 3) {
        key = fold(key, dominantJoint(r));
    } else {
        for (uint k = 0; k < params.width; ++k) {
            key = fold(key, values[r * params.width + k]);
        }
    }
    keyOut[r] = key;
}

/// One float of a record into its material vector, at component `mode`.
[shader("compute")]
[numthreads(256, 1, 1)]
void lodAttributeMaterial(uint3 tid: SV_DispatchThreadID) {
    const uint r = tid.x;
    if (r >= params.count) {
        return;
    }
    float4 m = materialOut[r];
    m[min(params.mode, 3u)] = asfloat(values[r]);
    materialOut[r] = m;
}

/// How much a store splat counts in a merge: as the moments count it, its
/// opacity times the area of its two long axes.
float weightOf(uint s) {
    const uint r = s * 4;
    const Blob b = blobOf(splatPositions[s], splatShape[r], splatShape[r + 1], splatShape[r + 2], splatShape[r + 3]);
    const float smallest = min(min(b.scale.x, b.scale.y), b.scale.z);
    return b.opacity * b.scale.x * b.scale.y * b.scale.z / max(smallest, 1.0e-20);
}

static const uint kMostJoints = 16;

/// ONE ATTRIBUTE OF EACH KEPT GAUSSIAN, from the records it stands for.
///
///   0  floats: the mean, weighted as the moments weigh -- metallic,
///      roughness, a transfer, a normal
///   1  what may not be merged across (the key saw to it): the first
///      record's -- an id, a part, a sheet
///   2  bits: each one set where records holding at least half the weight
///      have it -- which ways out are open
///   3  a rig: the weights each joint carries, summed over the records, the
///      `width` heaviest kept and made whole again
///   4  a direction: the mean of mode 0, made a unit vector again -- a
///      shading normal (`primvars:athenea:splat:normal`)
///
/// A gaussian kept as it was takes its record's words as they are, in every
/// mode.
[shader("compute")]
[numthreads(64, 1, 1)]
void lodMergeAttribute(uint3 tid: SV_DispatchThreadID) {
    const uint o = tid.x;
    if (o >= params.count) {
        return;
    }
    const uint2 run = ranges[o];
    const uint width = params.width;
    const uint out = o * width;
    if (run.y - run.x <= 1 || params.mode == 1) {
        const uint r = recordOf[run.x];
        for (uint k = 0; k < width; ++k) {
            merged[out + k] = values[r * width + k];
            if (params.mode == 3) {
                mergedWeights[out + k] = weights[r * width + k];
            }
        }
        return;
    }
    if (params.mode == 0) {
        for (uint k = 0; k < width; ++k) {
            float sum = 0.0;
            float total = 0.0;
            for (uint s = run.x; s < run.y; ++s) {
                const float w = weightOf(s);
                sum += w * asfloat(values[recordOf[s] * width + k]);
                total += w;
            }
            merged[out + k] = asuint(total > 0.0 ? sum / total : asfloat(values[recordOf[run.x] * width + k]));
        }
        return;
    }
    if (params.mode == 4) {
        float length2 = 0.0;
        for (uint k = 0; k < width; ++k) {
            float sum = 0.0;
            for (uint s = run.x; s < run.y; ++s) {
                sum += weightOf(s) * asfloat(values[recordOf[s] * width + k]);
            }
            merged[out + k] = asuint(sum);
            length2 += sum * sum;
        }
        // Children facing every way at once: the first record's, as it was.
        const bool some = length2 > 1.0e-30;
        for (uint k = 0; k < width; ++k) {
            merged[out + k] = some ? asuint(asfloat(merged[out + k]) * rsqrt(length2))
                                   : values[recordOf[run.x] * width + k];
        }
        return;
    }
    if (params.mode == 2) {
        float total = 0.0;
        for (uint s = run.x; s < run.y; ++s) {
            total += weightOf(s);
        }
        for (uint k = 0; k < width; ++k) {
            uint word = 0;
            for (uint bit = 0; bit < 32; ++bit) {
                float open = 0.0;
                for (uint s = run.x; s < run.y; ++s) {
                    if (((values[recordOf[s] * width + k] >> bit) & 1u) != 0u) {
                        open += weightOf(s);
                    }
                }
                word |= open * 2.0 >= total ? (1u << bit) : 0u;
            }
            merged[out + k] = word;
        }
        return;
    }
    // A rig: every joint any record names, and what it carries in all.
    uint joints[kMostJoints];
    float carried[kMostJoints];
    for (uint j = 0; j < kMostJoints; ++j) {
        joints[j] = 0u;
        carried[j] = 0.0;
    }
    uint named = 0;
    for (uint s = run.x; s < run.y; ++s) {
        const float w = weightOf(s);
        const uint r = recordOf[s];
        for (uint k = 0; k < width; ++k) {
            const uint joint = values[r * width + k];
            const float share = asfloat(weights[r * width + k]) * w;
            if (!(share > 0.0)) {
                continue;
            }
            uint at = named;
            for (uint j = 0; j < named; ++j) {
                if (joints[j] == joint) {
                    at = j;
                }
            }
            if (at == named) {
                if (named == kMostJoints) {
                    continue;   // past sixteen joints in one merge: the lightest go
                }
                joints[named] = joint;
                carried[named] = 0.0;
                named += 1;
            }
            carried[at] += share;
        }
    }
    float kept = 0.0;
    for (uint k = 0; k < width; ++k) {
        uint heaviest = kMostJoints;
        for (uint j = 0; j < named; ++j) {
            if (carried[j] >= 0.0 && (heaviest == kMostJoints || carried[j] > carried[heaviest])) {
                heaviest = j;
            }
        }
        if (heaviest == kMostJoints) {
            merged[out + k] = 0u;
            mergedWeights[out + k] = asuint(0.0);
            continue;
        }
        merged[out + k] = joints[heaviest];
        mergedWeights[out + k] = asuint(carried[heaviest]);
        kept += carried[heaviest];
        carried[heaviest] = -1.0;
    }
    for (uint k = 0; k < width; ++k) {
        mergedWeights[out + k] = asuint(kept > 0.0 ? asfloat(mergedWeights[out + k]) / kept : 0.0);
    }
}

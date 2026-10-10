// athenea's joint CPCA of the transfer (src/webgpu/athenea/cpcaJoint.ts):
// one cluster and 3 x 8 int8 weights a gaussian (`transferQ8`, 7 words),
// rebuilt a batch of splats at a time into the relight pass's scratch
// (relight.slang kRelightCpca), as BP's cpca.slang does for the file's
// clusters.
//
// The table (`jointBasis`, f32 as float4s): a cluster every
// kJointClusterFloats floats, its three blocks (16, 48, 48 values) one
// after the other, each its mean and then its 8 directions already scaled
// by scale / 127. A value is mean + sum_k q_k x dir_k, in f32 in that
// order, then the half.
// Under a dome alone, the relight takes athenea's proposal 094 route
// (relight.slang kRelightJoint): only the reflected block depends on the
// eye, so the eye pass rebuilds that block (and the direct block's first
// word, the openness) a gaussian (mode 1); the direct and indirect blocks
// are never rebuilt: atheneaJointProject dots each cluster's mean and
// directions with the light once (the dome's harmonics, the sun's
// direction), and the kept terms are those projections times the
// gaussian's weights.
module cpca_joint;

import core.attrib;
import athenea.light.environment;

public static const uint kJointClusterFloats = 1008;
public static const uint kJointRecordWords = 7;
public static const uint kJointDispatchRow = 65535u * 256u;
/// Words a cluster's projections (relight.slang jointTerm): 4 terms x 9
/// (the mean, then the 8 directions) rgb as float4.
public static const uint kJointProjWords = 144;
/// The eye pass's slot (mode 1): the reflected block's 24 words, then 4 free;
/// the direct block's first word sits 32 words before the reflected one, in
/// the slot two before's free words (the scratch starts 32 words late).
public static const uint kJointEyeSlotWords = 28;
public static const uint kJointEyeLead = 32;

public struct AtheneaJointParams {
    public uint count;          ///< source splats
    public uint listCount;      ///< > 0: the splats are the list's (at listBase, uint4s of jointList)
    public uint listBase;
    public uint threadFirst;    ///< the batch's first thread (list position or splat)
    public uint batch;          ///< threads in the batch (slots of the scratch)
    public uint attribute;      ///< transferQ8's id in the pool
    public uint scratchBase;    ///< words into jointOut
    public uint scratchWords;   ///< words a slot (the transfer's words)
    public uint blocks;         ///< blocks rebuilt: 1 (16 values), 2 (64), 3 (112)
    public uint mode;           ///< 0: the blocks; 1: the eye's (the direct block's first word and the reflected block)
    public uint dome;           ///< atheneaJointProject: the dome's harmonics (envSh) and sun (envSun)
    public uint tableBase;      ///< atheneaJointProject: where the projections start in jointOut, words
};

StructuredBuffer<uint>              jointPool;
StructuredBuffer<float4>            envSh;
StructuredBuffer<float4>            envSun;
StructuredBuffer<float4>            jointBasis;
StructuredBuffer<uint4>             jointList;
RWStructuredBuffer<uint>            jointOut;
ConstantBuffer<AtheneaJointParams>  jointParams;

float weightOf(uint w, uint k) {
    return float(int(w << (24u - 8u * k)) >> 24);
}

/// Block of `d4` float4s at `at4` of the table, its weights in words w0, w1,
/// written from word `out`.
void decodeBlock(uint at4, uint d4, uint w0, uint w1, uint out) {
    float q[8];
    [unroll] for (uint k = 0; k < 4u; ++k) {
        q[k] = weightOf(w0, k);
        q[4u + k] = weightOf(w1, k);
    }
    for (uint v = 0; v < d4; ++v) {
        float4 acc = jointBasis[at4 + v];
        [unroll] for (uint k = 0; k < 8u; ++k) {
            acc = acc + q[k] * jointBasis[at4 + (1u + k) * d4 + v];
        }
        jointOut[out + 2u * v] = f32tof16(acc.x) | (f32tof16(acc.y) << 16u);
        jointOut[out + 2u * v + 1u] = f32tof16(acc.z) | (f32tof16(acc.w) << 16u);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaJointDecode(uint3 tid: SV_DispatchThreadID) {
    const uint slot = tid.x + tid.y * kJointDispatchRow;
    if (slot >= jointParams.batch) {
        return;
    }
    const uint t = jointParams.threadFirst + slot;
    uint i = t;
    if (jointParams.listCount != 0u) {
        if (t >= jointParams.listCount) {
            return;
        }
        i = jointList[jointParams.listBase + t / 4u][t % 4u];
    }
    if (i >= jointParams.count) {
        return;
    }
    const AttribDesc d = attribDesc(jointPool, jointParams.attribute);
    const uint base = attribBase(jointPool, d, i);
    if (base == ATTRIB_NONE) {
        return;
    }
    const uint out = jointParams.scratchBase + slot * jointParams.scratchWords;
    const uint cluster4 = jointPool[base + 6u] * (kJointClusterFloats / 4u);
    if (jointParams.mode == 1u) {
        // The direct block's first two values (the openness reads the first).
        const uint w0 = jointPool[base];
        const uint w1 = jointPool[base + 1u];
        float2 acc = jointBasis[cluster4].xy;
        [unroll] for (uint k = 0; k < 4u; ++k) {
            acc = acc + weightOf(w0, k) * jointBasis[cluster4 + (1u + k) * 4u].xy;
            acc = acc + weightOf(w1, k) * jointBasis[cluster4 + (5u + k) * 4u].xy;
        }
        jointOut[out - kJointEyeLead] = f32tof16(acc.x) | (f32tof16(acc.y) << 16u);
        decodeBlock(cluster4 + 144u, 12u, jointPool[base + 4u], jointPool[base + 5u], out);
        return;
    }
    // Block 0 (direct, 16): floats 0 ..; block 1 (indirect, 48): 144 ..;
    // block 2 (reflected, 48): 576 .. (cpcaJoint.ts JOINT_BLOCK_FLOATS).
    decodeBlock(cluster4, 4u, jointPool[base], jointPool[base + 1u], out);
    if (jointParams.blocks > 1u) {
        decodeBlock(cluster4 + 36u, 12u, jointPool[base + 2u], jointPool[base + 3u], out + 8u);
    }
    if (jointParams.blocks > 2u) {
        decodeBlock(cluster4 + 144u, 12u, jointPool[base + 4u], jointPool[base + 5u], out + 32u);
    }
}

/// A cluster's projections on the light (thread a cluster), at tableBase:
/// for j = 0 (the mean) and j = 1..8 (the directions, scaled), as float4s,
///   term 0  sum_k envSh_k.rgb x direct_j[k]              the body's direct half
///   term 1  sum_k envSh_k.rgb (x) indirect_j[k] (rgb)     its indirect half, and what the field couples to
///   term 2  sum_k Y_k(sun) indirect_j[k] (rgb)            the sun's bounce (splatIndirectAlong, before its clamp)
///   term 3  indirect_j[0] (rgb)                           the white sky's gathering (splatFieldCoupling)
[shader("compute")]
[numthreads(64, 1, 1)]
void atheneaJointProject(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= jointParams.count) {
        return;
    }
    const uint sh = jointParams.dome * kEnvCoefficients;
    const float4 disc = envSun[jointParams.dome * 2u];
    const float3 sun = disc.w > 0.0 ? disc.xyz : float3(0.0, 0.0, 1.0);
    const uint cluster4 = c * (kJointClusterFloats / 4u);
    const uint out = jointParams.tableBase + c * kJointProjWords;
    for (uint j = 0; j < 9u; ++j) {
        // Direct: 16 values, 4 float4 a vector, from float4 0.
        float3 body = float3(0.0);
        [unroll] for (uint v = 0; v < 4u; ++v) {
            const float4 d = jointBasis[cluster4 + j * 4u + v];
            body += envSh[sh + 4u * v].rgb * d.x + envSh[sh + 4u * v + 1u].rgb * d.y +
                    envSh[sh + 4u * v + 2u].rgb * d.z + envSh[sh + 4u * v + 3u].rgb * d.w;
        }
        // Indirect: 48 values (rgb a coefficient), 12 float4 a vector, from float4 36.
        float3 gathered = float3(0.0);
        float3 along = float3(0.0);
        float3 white = float3(0.0);
        const uint in4 = cluster4 + 36u + j * 12u;
        for (uint k = 0; k < 16u; ++k) {
            const uint at = 3u * k;
            float3 rgb;
            [unroll] for (uint ch = 0; ch < 3u; ++ch) {
                const float4 q = jointBasis[in4 + (at + ch) / 4u];
                const uint lane = (at + ch) % 4u;
                rgb[ch] = lane == 0u ? q.x : (lane == 1u ? q.y : (lane == 2u ? q.z : q.w));
            }
            gathered += envSh[sh + k].rgb * rgb;
            along += envBasisValue(k, sun) * rgb;
            if (k == 0u) {
                white = rgb;
            }
        }
        const uint w = out + j * 4u;
        jointOut[w] = asuint(body.x);
        jointOut[w + 1u] = asuint(body.y);
        jointOut[w + 2u] = asuint(body.z);
        jointOut[w + 36u] = asuint(gathered.x);
        jointOut[w + 37u] = asuint(gathered.y);
        jointOut[w + 38u] = asuint(gathered.z);
        jointOut[w + 72u] = asuint(along.x);
        jointOut[w + 73u] = asuint(along.y);
        jointOut[w + 74u] = asuint(along.z);
        jointOut[w + 108u] = asuint(white.x);
        jointOut[w + 109u] = asuint(white.y);
        jointOut[w + 110u] = asuint(white.z);
    }
}

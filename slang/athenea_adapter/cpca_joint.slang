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
module cpca_joint;

import core.attrib;

public static const uint kJointClusterFloats = 1008;
public static const uint kJointRecordWords = 7;
public static const uint kJointDispatchRow = 65535u * 256u;

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
    public uint pad0;
    public uint pad1;
    public uint pad2;
};

StructuredBuffer<uint>              jointPool;
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

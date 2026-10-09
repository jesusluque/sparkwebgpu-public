// The transfer kept on the GPU as clustered PCA (.athc v3 encoding 3,
// rust/spark-lib/src/athc_cpca.rs), rebuilt a batch of splats at a time for
// the relight pass (relight.slang kRelightCpca).
//
// The pool's attribute `transferCpca` (u32 words, src/webgpu/athenea/
// cpcaResident.ts) holds a splat's transfer section by section, as halves:
//   a plain section     its D values, as the file's halves
//   a CPCA section      the entry (u16) of its cluster, then its M
//                       coefficients as i16 (the file's quantized integers;
//                       a cluster of fewer directions leaves the rest 0)
// and the table (`cpcaTable`, beside the pool, always whole) holds the
// entries: [0] their number, [1 + e] where entry e starts, then each entry
//   word 0   m (directions), bit 31: exact
//   word 1   the cluster's step (f32 bits); exact: its elements
//   word 2   D
//   word 3   0
//   then halves: CPCA, the mean (D) and the m directions (m x D);
//                exact, the D values of each of its elements (a block the
//                file keeps exact, or one this layout cannot hold: its
//                elements' first coefficient is their index in it).
//
// atheneaCpcaDecode writes the transfer of splats threadFirst ..
// threadFirst + batch - 1 (the LoD list's, or the cloud's) as the halves
// athenea's transferAt reads, `scratchWords` words a slot from scratchBase
// of `cpcaOut` (the relight pass's envOfLight buffer, after its lights),
// with athc_cpca.rs's arithmetic: acc = mean; acc = acc + (q_j x step) x
// dir_j, each product and sum in f32, then the half.
module cpca;

import core.attrib;

public static const uint kCpcaExact = 0x80000000u;
public static const uint kCpcaMaxCoefficients = 64;
public static const uint kCpcaDispatchRow = 65535u * 256u;

public struct AtheneaCpcaParams {
    public uint count;          ///< source splats
    public uint listCount;      ///< > 0: the splats are the list's (at listBase, uint4s of cpcaList)
    public uint listBase;
    public uint threadFirst;    ///< the batch's first thread (list position or splat)
    public uint batch;          ///< threads in the batch (slots of the scratch)
    public uint attribute;      ///< transferCpca's id in the pool
    public uint scratchBase;    ///< words into cpcaOut
    public uint scratchWords;   ///< words a slot (the transfer's words)
    public uint sections;       ///< 1 .. 3
    public uint pad0;
    public uint pad1;
    public uint pad2;
    public uint4 section0;      ///< values D, first value, first half in the record, 1: CPCA
    public uint4 section1;
    public uint4 section2;
};

StructuredBuffer<uint>              cpcaPool;
StructuredBuffer<uint>              cpcaTable;
StructuredBuffer<uint4>             cpcaList;
RWStructuredBuffer<uint>            cpcaOut;
ConstantBuffer<AtheneaCpcaParams>   cpcaParams;

uint recordHalf(uint base, uint h) {
    const uint w = cpcaPool[base + (h >> 1u)];
    return (h & 1u) != 0u ? (w >> 16u) : (w & 0xFFFFu);
}

float lowF(uint w) { return f16tof32(w & 0xFFFFu); }
float highF(uint w) { return f16tof32(w >> 16u); }

void decodeSection(uint4 sec, uint base, uint out) {
    const uint d = sec.x;
    const uint dw = d / 2u;
    const uint first = out + sec.y / 2u;
    if (sec.w == 0u) {
        // Plain: the halves as they are.
        for (uint w = 0; w < dw; ++w) {
            cpcaOut[first + w] = recordHalf(base, sec.z + 2u * w) | (recordHalf(base, sec.z + 2u * w + 1u) << 16u);
        }
        return;
    }
    const uint entry = recordHalf(base, sec.z);
    const uint at = cpcaTable[1u + entry];
    const uint head = cpcaTable[at];
    const uint data = at + 4u;
    if ((head & kCpcaExact) != 0u) {
        const uint index = recordHalf(base, sec.z + 1u);
        const uint from = data + index * dw;
        for (uint w = 0; w < dw; ++w) {
            cpcaOut[first + w] = cpcaTable[from + w];
        }
        return;
    }
    const uint m = min(head, kCpcaMaxCoefficients);
    const float step = asfloat(cpcaTable[at + 1u]);
    float coef[kCpcaMaxCoefficients];
    for (uint j = 0; j < m; ++j) {
        const int q = int(recordHalf(base, sec.z + 1u + j) << 16u) >> 16;
        coef[j] = float(q) * step;
    }
    for (uint w = 0; w < dw; ++w) {
        const uint meanWord = cpcaTable[data + w];
        float a0 = lowF(meanWord);
        float a1 = highF(meanWord);
        for (uint j = 0; j < m; ++j) {
            const uint dir = cpcaTable[data + (1u + j) * dw + w];
            a0 = a0 + coef[j] * lowF(dir);
            a1 = a1 + coef[j] * highF(dir);
        }
        cpcaOut[first + w] = f32tof16(a0) | (f32tof16(a1) << 16u);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaCpcaDecode(uint3 tid: SV_DispatchThreadID) {
    const uint slot = tid.x + tid.y * kCpcaDispatchRow;
    if (slot >= cpcaParams.batch) {
        return;
    }
    const uint t = cpcaParams.threadFirst + slot;
    uint i = t;
    if (cpcaParams.listCount != 0u) {
        if (t >= cpcaParams.listCount) {
            return;
        }
        i = cpcaList[cpcaParams.listBase + t / 4u][t % 4u];
    }
    if (i >= cpcaParams.count) {
        return;
    }
    const AttribDesc d = attribDesc(cpcaPool, cpcaParams.attribute);
    const uint base = attribBase(cpcaPool, d, i);
    if (base == ATTRIB_NONE) {
        return;
    }
    const uint out = cpcaParams.scratchBase + slot * cpcaParams.scratchWords;
    decodeSection(cpcaParams.section0, base, out);
    if (cpcaParams.sections > 1u) {
        decodeSection(cpcaParams.section1, base, out);
    }
    if (cpcaParams.sections > 2u) {
        decodeSection(cpcaParams.section2, base, out);
    }
}

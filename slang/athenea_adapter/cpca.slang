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
StructuredBuffer<uint4>             cpcaTable4;    ///< the same buffer, a uint4 at a time
StructuredBuffer<uint4>             cpcaList;
RWStructuredBuffer<uint>            cpcaOut;
ConstantBuffer<AtheneaCpcaParams>   cpcaParams;

uint recordHalf(uint base, uint h) {
    const uint w = cpcaPool[base + (h >> 1u)];
    return (h & 1u) != 0u ? (w >> 16u) : (w & 0xFFFFu);
}

float lowF(uint w) { return f16tof32(w & 0xFFFFu); }
float highF(uint w) { return f16tof32(w >> 16u); }

/// A CPCA section in chunks of C uint4s (8 C values) held in registers, the
/// coefficients read from the record for each chunk: every value's sum in
/// athc_cpca.rs's order.
void decodeChunks<let C : int>(uint4 sec, uint base, uint first, uint data, uint row4, uint m, float step) {
    const uint data4 = data >> 2u;
    for (uint c0 = 0; c0 < row4; c0 += uint(C)) {
        float4 lo[C];
        float4 hi[C];
        [unroll] for (int k = 0; k < C; ++k) {
            const uint4 w = cpcaTable4[data4 + c0 + uint(k)];
            lo[k] = float4(lowF(w.x), lowF(w.y), lowF(w.z), lowF(w.w));
            hi[k] = float4(highF(w.x), highF(w.y), highF(w.z), highF(w.w));
        }
        for (uint j = 0; j < m; ++j) {
            const float c = float(int(recordHalf(base, sec.z + 1u + j) << 16u) >> 16) * step;
            const uint row = data4 + (1u + j) * row4 + c0;
            [unroll] for (int k = 0; k < C; ++k) {
                const uint4 w = cpcaTable4[row + uint(k)];
                lo[k] = lo[k] + c * float4(lowF(w.x), lowF(w.y), lowF(w.z), lowF(w.w));
                hi[k] = hi[k] + c * float4(highF(w.x), highF(w.y), highF(w.z), highF(w.w));
            }
        }
        [unroll] for (int k = 0; k < C; ++k) {
            const uint at = first + 4u * (c0 + uint(k));
            cpcaOut[at] = f32tof16(lo[k].x) | (f32tof16(hi[k].x) << 16u);
            cpcaOut[at + 1u] = f32tof16(lo[k].y) | (f32tof16(hi[k].y) << 16u);
            cpcaOut[at + 2u] = f32tof16(lo[k].z) | (f32tof16(hi[k].z) << 16u);
            cpcaOut[at + 3u] = f32tof16(lo[k].w) | (f32tof16(hi[k].w) << 16u);
        }
    }
}

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
    // The transfers' sections: 48 values (TXIN, TXFD) and 16 (TXDI).
    if ((dw & 7u) == 0u) {
        decodeChunks<2>(sec, base, first, data, dw >> 2u, m, step);
        return;
    }
    float coef[kCpcaMaxCoefficients];
    for (uint j = 0; j < m; ++j) {
        const int q = int(recordHalf(base, sec.z + 1u + j) << 16u) >> 16;
        coef[j] = float(q) * step;
    }
    if ((dw & 3u) == 0u) {
        // Eight values at a time (the table's entries start on a uint4).
        const uint data4 = data >> 2u;
        const uint row4 = dw >> 2u;
        for (uint w4 = 0; w4 < row4; ++w4) {
            const uint4 meanWords = cpcaTable4[data4 + w4];
            float4 lo = float4(lowF(meanWords.x), lowF(meanWords.y), lowF(meanWords.z), lowF(meanWords.w));
            float4 hi = float4(highF(meanWords.x), highF(meanWords.y), highF(meanWords.z), highF(meanWords.w));
            for (uint j = 0; j < m; ++j) {
                const uint4 dir = cpcaTable4[data4 + (1u + j) * row4 + w4];
                const float c = coef[j];
                lo = lo + c * float4(lowF(dir.x), lowF(dir.y), lowF(dir.z), lowF(dir.w));
                hi = hi + c * float4(highF(dir.x), highF(dir.y), highF(dir.z), highF(dir.w));
            }
            const uint at = first + 4u * w4;
            cpcaOut[at] = f32tof16(lo.x) | (f32tof16(hi.x) << 16u);
            cpcaOut[at + 1u] = f32tof16(lo.y) | (f32tof16(hi.y) << 16u);
            cpcaOut[at + 2u] = f32tof16(lo.z) | (f32tof16(hi.z) << 16u);
            cpcaOut[at + 3u] = f32tof16(lo.w) | (f32tof16(hi.w) << 16u);
        }
        return;
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

/// One word (two values) of a section's rebuilt transfer, from record `base`.
uint decodeWord(uint4 sec, uint base, uint w) {
    const uint dw = sec.x / 2u;
    if (sec.w == 0u) {
        return recordHalf(base, sec.z + 2u * w) | (recordHalf(base, sec.z + 2u * w + 1u) << 16u);
    }
    const uint at = cpcaTable[1u + recordHalf(base, sec.z)];
    const uint head = cpcaTable[at];
    const uint data = at + 4u;
    if ((head & kCpcaExact) != 0u) {
        return cpcaTable[data + recordHalf(base, sec.z + 1u) * dw + w];
    }
    const uint m = min(head, kCpcaMaxCoefficients);
    const float step = asfloat(cpcaTable[at + 1u]);
    const uint meanWord = cpcaTable[data + w];
    float a0 = lowF(meanWord);
    float a1 = highF(meanWord);
    for (uint j = 0; j < m; ++j) {
        const float c = float(int(recordHalf(base, sec.z + 1u + j) << 16u) >> 16) * step;
        const uint dir = cpcaTable[data + (1u + j) * dw + w];
        a0 = a0 + c * lowF(dir);
        a1 = a1 + c * highF(dir);
    }
    return f32tof16(a0) | (f32tof16(a1) << 16u);
}

/// atheneaCpcaDecode a word a thread: batch x scratchWords threads, the
/// slot's words side by side (more threads in flight than one a splat).
[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaCpcaDecodeWords(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x + tid.y * kCpcaDispatchRow;
    const uint words = cpcaParams.scratchWords;
    const uint slot = g / words;
    const uint w = g - slot * words;
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
    uint4 sec = cpcaParams.section0;
    if (cpcaParams.sections > 1u && w >= cpcaParams.section1.y / 2u) {
        sec = cpcaParams.section1;
    }
    if (cpcaParams.sections > 2u && w >= cpcaParams.section2.y / 2u) {
        sec = cpcaParams.section2;
    }
    cpcaOut[cpcaParams.scratchBase + slot * words + w] = decodeWord(sec, base, w - sec.y / 2u);
}

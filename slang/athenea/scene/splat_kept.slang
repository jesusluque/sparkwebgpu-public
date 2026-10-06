// Copyright (c) 2026 jesus luque.
//
// A PER-RECORD ARRAY, OVER THE SPLATS VALIDATION KEPT.
//
// The decode drops what cannot be drawn and packs the rest, and writes for
// each splat the record it came from (`origin`). Anything else a file keeps a
// gaussian -- a rig's influences, a visibility's parts -- was laid out a
// record each, and has to be packed the same way before a kernel indexes it
// by splat: this gathers `words` words a splat through `origin`.
module splat_kept;

struct KeptParams {
    uint count;   ///< splats kept
    uint words;   ///< 32-bit words an element
    uint pad0; uint pad1;
};

StructuredBuffer<uint>    origin;
StructuredBuffer<uint>    byRecord;
RWStructuredBuffer<uint>  bySplat;
// The other way, for what a splat's kernel wrote and the file keeps a record
// each: the records validation dropped keep what the buffer was filled with.
StructuredBuffer<uint>    fromSplat;
RWStructuredBuffer<uint>  toRecord;
ConstantBuffer<KeptParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void splatKept(uint3 tid: SV_DispatchThreadID) {
    const uint j = tid.x;
    if (j >= params.count) {
        return;
    }
    const uint from = origin[j] * params.words;
    for (uint k = 0; k < params.words; ++k) {
        bySplat[j * params.words + k] = byRecord[from + k];
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void splatKeptScatter(uint3 tid: SV_DispatchThreadID) {
    const uint j = tid.x;
    if (j >= params.count) {
        return;
    }
    const uint to = origin[j] * params.words;
    for (uint k = 0; k < params.words; ++k) {
        toRecord[to + k] = fromSplat[j * params.words + k];
    }
}

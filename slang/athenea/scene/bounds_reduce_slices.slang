// Copyright (c) 2026 jesus luque.
//
// One box a slice of the chunk extents: what `bounds_reduce` folds over the
// whole buffer, folded here over `(first, count)` runs of it, one thread a
// run. It is how mesh2splat gets every mesh's own world-space box out of the
// per-chunk extents `streamBoundsChunks` already wrote for the model's, in
// one dispatch rather than one a mesh -- and without a field on
// `bounds_reduce` that its six other callers never set.
module bounds_reduce_slices;

struct SliceParams {
    uint slices;        // runs, one thread each
    uint resultFirst;   // where the first run's two entries go in `result`
    uint pad0;
    uint pad1;
};

StructuredBuffer<float4>     extents;   // per chunk: min, max
StructuredBuffer<uint2>      slices;    // per run: first chunk, chunk count
RWStructuredBuffer<float4>   result;    // per run: min, max, from resultFirst
ConstantBuffer<SliceParams>  params;

[shader("compute")]
[numthreads(64, 1, 1)]
void boundsReduceSlices(uint3 tid: SV_DispatchThreadID) {
    const uint m = tid.x;
    if (m >= params.slices) {
        return;
    }
    const uint2 run = slices[m];
    float3 lo = float3(3.4e38);
    float3 hi = float3(-3.4e38);
    for (uint c = run.x; c < run.x + run.y; ++c) {
        lo = min(lo, extents[c * 2].xyz);
        hi = max(hi, extents[c * 2 + 1].xyz);
    }
    result[params.resultFirst + m * 2] = float4(lo, 0.0);
    result[params.resultFirst + m * 2 + 1] = float4(hi, 0.0);
}

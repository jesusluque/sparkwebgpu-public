// Progressive accumulation: a running mean of pictures, one sample a frame.
//
// While nothing in view changes, the host renders the same frame again with
// the projection moved by a sub-pixel jitter (Halton 2, 3) and folds each
// render into `mean`: mean += (sample - mean) * weight, with weight = 1 / k
// for the k-th sample, so after k samples `mean` is their plain average
// (weight 1 on the first sample restarts it). The jitters integrate the
// pixel's box, as a supersampled render would, but over time.
//
// Pictures are float4, premultiplied and linear (the bridge's layout), so
// the mean is taken in linear light before any display transform, in float.

struct AccumulateParams {
    uint  width;      // both pictures, in pixels
    uint  height;
    uint  srcStride;  // the sample's row, in pixels
    uint  dstStride;  // the mean's row, in pixels
    float weight;     // 1 / k for the k-th sample
    uint  pad0;
    uint  pad1;
    uint  pad2;
};

StructuredBuffer<float4>         src;
RWStructuredBuffer<float4>       mean;
ConstantBuffer<AccumulateParams> params;

[shader("compute")]
[numthreads(16, 16, 1)]
void accumulateMain(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const float4 sample = src[tid.y * params.srcStride + tid.x];
    const uint at = tid.y * params.dstStride + tid.x;
    // The first sample replaces whatever was there (a NaN included).
    mean[at] = params.weight >= 1.0f ? sample
                                     : mean[at] + (sample - mean[at]) * params.weight;
}

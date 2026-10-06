// The relight pass's footprint prefilter (slang/athenea_adapter/relight.slang
// footprintRoughness) on given cases, for test/gpu/relightFootprint.test.ts.
import athenea_adapter.relight;

struct FootprintCase {
    float4 shapeAndRoughness;   ///< the shape operator (xx, xy, yy), the roughness
    float4 scalesAndGain;       ///< the gaussian's scales, the gain
};

StructuredBuffer<FootprintCase> footprintCases;
RWStructuredBuffer<float>       footprintOut;

[shader("compute")]
[numthreads(64, 1, 1)]
void footprintTest(uint3 tid: SV_DispatchThreadID) {
    uint count, stride;
    footprintCases.GetDimensions(count, stride);
    const uint i = tid.x;
    if (i >= count) {
        return;
    }
    const FootprintCase c = footprintCases[i];
    footprintOut[i] = footprintRoughness(c.shapeAndRoughness.w, c.shapeAndRoughness.xyz, c.scalesAndGain.xyz,
                                         c.scalesAndGain.w);
}

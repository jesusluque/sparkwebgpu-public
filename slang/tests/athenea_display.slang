// athenea's own `displayColour` (athenea/technique/display.slang, vendored)
// for a list of colours, with display.slang's own globals bound -- its
// `params` (the headroom `transferEncode` reads), `acesParams` and
// `acesTables`: the reference the output stage (athenea_adapter/display.slang)
// is checked against.

import athenea.technique.display;

struct RefParams {
    uint  count;
    uint  view;
    uint  display;
    float exposure;
    float backgroundR;
    float backgroundG;
    float backgroundB;
    float pad0;
};

StructuredBuffer<float4>     refSrc;
RWStructuredBuffer<float4>   refOut;
ConstantBuffer<RefParams>    refParams;

[shader("compute")]
[numthreads(64, 1, 1)]
void displayReference(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= refParams.count) {
        return;
    }
    const float3 background = float3(refParams.backgroundR, refParams.backgroundG, refParams.backgroundB);
    refOut[i] = float4(displayColour(refSrc[i], background, refParams.exposure, refParams.view, refParams.display), 1.0);
}

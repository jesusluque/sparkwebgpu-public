// Copyright (c) 2026 jesus luque.
//
// Which records become splats: opacity at least 1/255 (below it the blend
// discards every fragment the splat could make), and a finite position and
// scale (a NaN covariance fails every comparison and silently drops tiles).
// Same rule as openFXplayer's loader, on the GPU.
module splat_validate;
import splat_encoding;

StructuredBuffer<float>      raw;
RWStructuredBuffer<uint>     valid;
ConstantBuffer<DecodeParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void splatValidate(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint s = params.stride;
    const float3 p = decodePosition(raw, i, params);
    const float a = decodeOpacity(fieldOf(raw, i, s, params.opacity), params.opacityMode);
    const float3 ls = float3(decodeLogScale(fieldOf(raw, i, s, params.scale0), params.scaleMode),
                             decodeLogScale(fieldOf(raw, i, s, params.scale1), params.scaleMode),
                             decodeLogScale(fieldOf(raw, i, s, params.scale2), params.scaleMode));
    const bool finite = all(isfinite(p)) && all(isfinite(ls)) && all(abs(ls) < 60.0);
    valid[i] = (finite && a >= 1.0 / 255.0) ? 1u : 0u;
}

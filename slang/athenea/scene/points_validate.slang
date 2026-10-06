// Copyright (c) 2026 jesus luque.
//
// Which points are kept: finite ones, thinned to `detail` by a hash of the
// index -- the same subset every time, as openFXplayer's PointCloud does.
module points_validate;
import athenea.common.color;

struct PointParams {
    uint  count;
    uint  base;
    uint  colourKind;   // 0 none, 1 sRGB bytes, 2 linear floats
    float detail;
};

StructuredBuffer<float>     raw;   // count * 6
RWStructuredBuffer<uint>    valid;
ConstantBuffer<PointParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void pointsValidate(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const float3 p = float3(raw[i * 6], raw[i * 6 + 1], raw[i * 6 + 2]);
    const bool kept = params.detail >= 1.0 || hash01(i + params.base) < params.detail;
    valid[i] = (all(isfinite(p)) && kept) ? 1u : 0u;
}

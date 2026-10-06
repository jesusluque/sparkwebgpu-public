// Copyright (c) 2026 jesus luque.
//
// WHAT THE GAUSSIANS OF ONE PRIM ARE CARRYING, counted on the device.
//
// A pick hands back a Cryptomatte id. Every gaussian with that id came from
// one prim, and therefore stood on one material when `athenea mesh2splat` walked
// it -- but not necessarily on one set of numbers: a material whose roughness
// is a map gives each gaussian its own. So this answers with a range as well
// as a mean, and a host can say "roughness 0.08 to 0.31" rather than pretend
// there is a single value to show.
//
// Only counters come back, as everything that crosses back from the device
// here does: ten uints, the sums and bounds in the bytes `pbr` stores.
//
// The sums are byte values, so they hold to 16.8 million gaussians of one
// prim before a uint could wrap. No cloud here is near that, and the count
// comes back beside them for anyone who wants to check.
module splat_id_stats;
import athenea.scene.splat_encoding;

struct IdStatsParams {
    uint count;    ///< gaussians in the cloud
    uint id;       ///< the Cryptomatte id being asked about
    uint hasPbr;   ///< 1: `pbr` is this cloud's own; 0: it carries none
    uint pad0;
};

StructuredBuffer<uint>   cloudCrypto;
StructuredBuffer<uint>   pbr;
RWStructuredBuffer<uint> stats;   ///< 10 wide: count, then sum/min/max a channel
ConstantBuffer<IdStatsParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void splatIdStats(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count || cloudCrypto[i] != params.id) {
        return;
    }
    InterlockedAdd(stats[0], 1u);
    if (params.hasPbr == 0) {
        return;
    }
    const uint word = pbr[i];
    [unroll] for (uint c = 0; c < 3; ++c) {
        const uint value = (word >> (c * 8)) & 0xFFu;
        const uint at = 1 + c * 3;
        InterlockedAdd(stats[at], value);
        InterlockedMin(stats[at + 1], value);
        InterlockedMax(stats[at + 2], value);
    }
}

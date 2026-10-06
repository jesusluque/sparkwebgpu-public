// radixScatter of kernels/sort_radix.slang with subgroup ballots, for
// devices with the "subgroups" feature (a module of its own: WGSL needs
// `enable subgroups;`, which fails to compile without it). Same blocks,
// rounds and output, so it drops in after radixHistogram and the scan.
//
// The plain scatter ranks each round's keys with a 128-thread Hillis-Steele
// scan, 16 barriers a round. Here a key's rank among the same digit in its
// subgroup is a popcount of four ballots (one per digit bit), and the
// subgroups' digit counts are scanned in workgroup memory: 4 barriers a
// round, so the pass runs at the speed of its memory traffic.

import kernels.sort_radix;

// Subgroups per workgroup, at the smallest subgroup size WebGPU allows (4).
static const uint MAX_SUBGROUPS = WG / 4;

// Per subgroup and digit: the round's count, then the exclusive offset.
groupshared uint sDigitStart[MAX_SUBGROUPS * BINS];
groupshared uint sBlockBase[BINS];
groupshared Atomic<uint> sNextSubgroup;

uint popcount4(uint4 v) {
    return countbits(v.x) + countbits(v.y) + countbits(v.z) + countbits(v.w);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void radixScatterSubgroup(uint3 lid: SV_GroupThreadID, uint3 gid: SV_GroupID) {
    uint t = lid.x;
    uint lane = WaveGetLaneIndex();
    uint size = WaveGetLaneCount();
    uint subgroups = WG / size;
    if (t == 0u) {
        sNextSubgroup.store(0u);
    }
    if (t < BINS) {
        sBlockBase[t] = blockHist[t * activeBlocks() + gid.x];
    }
    GroupMemoryBarrierWithGroupSync();
    // Number the subgroups ourselves, so that slot order is lane order even
    // where subgroups don't hold consecutive invocation indices: stability
    // needs ballot ranks to follow key order.
    uint sg = 0u;
    if (WaveIsFirstLane()) {
        sg = sNextSubgroup.add(1u);
    }
    sg = WaveReadLaneFirst(sg);
    uint slot = sg * size + lane;
    // The lanes below this one, as a ballot mask.
    uint4 below;
    for (uint c = 0; c < 4; c++) {
        uint first = 32u * c;
        below[c] = lane >= first + 32u ? 0xffffffffu : (lane <= first ? 0u : (1u << (lane - first)) - 1u);
    }

    uint base = gid.x * TILE;
    uint count = activeCount();
    for (uint r = 0; r < ROUNDS; r++) {
        uint i = base + r * WG + slot;
        bool valid = i < count;
        uint key = valid ? keysIn[i] : 0u;
        uint val = valid ? valsIn[i] : 0u;
        uint digit = digitOf(key);

        uint4 same = WaveActiveBallot(valid);
        for (uint b = 0; b < 4; b++) {
            bool bit = ((digit >> b) & 1u) != 0u;
            uint4 ballot = WaveActiveBallot(bit);
            same &= bit ? ballot : ~ballot;
        }
        uint rank = popcount4(same & below);

        for (uint j = t; j < subgroups * BINS; j += WG) {
            sDigitStart[j] = 0u;
        }
        GroupMemoryBarrierWithGroupSync();
        if (valid && rank == 0u) {
            sDigitStart[sg * BINS + digit] = popcount4(same);
        }
        GroupMemoryBarrierWithGroupSync();
        if (t < BINS) {
            uint start = sBlockBase[t];
            for (uint g = 0; g < subgroups; g++) {
                uint n = sDigitStart[g * BINS + t];
                sDigitStart[g * BINS + t] = start;
                start += n;
            }
            sBlockBase[t] = start;
        }
        GroupMemoryBarrierWithGroupSync();
        if (valid) {
            uint dst = sDigitStart[sg * BINS + digit] + rank;
            keysOut[dst] = key;
            valsOut[dst] = val;
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

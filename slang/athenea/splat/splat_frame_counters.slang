// Copyright (c) 2026 jesus luque.
//
// WHAT A FRAME OF SPLATS DID, COUNTED WHERE IT HAPPENED: how many the
// projection kept and why it culled the others, the (tile, splat) pairs and
// the most tiles one splat touched, and per cloud what it kept. Read by a
// panel through an asynchronous readback (athenea::gpu::AsyncReadback), so the
// frame never waits on it; the layout is TileRasterizer.h's `SplatCounters`.
//
// Nothing is measured for this that the frame did not already write: the
// projection's `visible` and `tilesTouched`, the reason it left in a culled
// slot's depth key (frame.slang), and the prefix sums the compaction uses.
module splat_frame_counters;
import frame;

// Word offsets, as SplatCounters reads them.
static const uint kSlots = 0;
static const uint kVisible = 1;
static const uint kPairs = 2;
static const uint kMaxTiles = 3;
static const uint kCulled = 4;      // + reason, kCullReasons of them
static const uint kTally = 11;      // words a group adds up
static const uint kCloudRows = 12;  // how many clouds have a row, written by the clear
static const uint kClouds = 16;     // + 2 a cloud: visible, pairs

struct CounterParams {
    uint count;    // slots to count (the frame's splats; points follow them)
    uint words;    // for the clear
    uint base;     // a cloud's first slot
    uint cloudCount;   // the cloud's slots; for the clear, how many clouds have a row
    uint all;      // slots the prefix sums ran over
    uint index;    // the cloud's row
};

StructuredBuffer<uint>       visible;
StructuredBuffer<uint>       tilesTouched;
StructuredBuffer<uint>       depthKeys;
StructuredBuffer<uint>       visibleOffsets;
StructuredBuffer<uint>       visibleTotal;
StructuredBuffer<uint>       touchedOffsets;
StructuredBuffer<uint>       touchedTotal;
RWStructuredBuffer<uint>     counters;
ConstantBuffer<CounterParams> params;

groupshared uint tally[kTally];

[shader("compute")]
[numthreads(64, 1, 1)]
void splatCountersClear(uint3 tid: SV_DispatchThreadID) {
    if (tid.x < params.words) {
        counters[tid.x] = tid.x == kCloudRows ? params.cloudCount : 0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void splatCounters(uint3 tid: SV_DispatchThreadID, uint3 lid: SV_GroupThreadID) {
    if (lid.x < kTally) {
        tally[lid.x] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    const uint i = tid.x;
    if (i < params.count) {
        InterlockedAdd(tally[kSlots], 1u);
        if (visible[i] != 0) {
            const uint tiles = tilesTouched[i];
            InterlockedAdd(tally[kVisible], 1u);
            InterlockedAdd(tally[kPairs], tiles);
            InterlockedMax(tally[kMaxTiles], tiles);
        } else {
            const uint key = depthKeys[i];
            const uint reason = (key & 0xFF000000u) == kCulledKey ? min(key & 0xFFu, kCullReasons - 1) : 0u;
            InterlockedAdd(tally[kCulled + reason], 1u);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (lid.x < kTally && tally[lid.x] != 0) {
        if (lid.x == kMaxTiles) {
            InterlockedMax(counters[lid.x], tally[lid.x]);
        } else {
            InterlockedAdd(counters[lid.x], tally[lid.x]);
        }
    }
}

// One cloud's share, from the prefix sums the compaction already took: what
// came before its first slot, subtracted from what came before the slot
// after its last.
[shader("compute")]
[numthreads(1, 1, 1)]
void splatCountersCloud(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0 || params.cloudCount == 0) {
        return;
    }
    const uint end = params.base + params.cloudCount;
    const uint visibleEnd = end < params.all ? visibleOffsets[end] : visibleTotal[0];
    const uint touchedEnd = end < params.all ? touchedOffsets[end] : touchedTotal[0];
    counters[kClouds + params.index * 2] = visibleEnd - visibleOffsets[params.base];
    counters[kClouds + params.index * 2 + 1] = touchedEnd - touchedOffsets[params.base];
}

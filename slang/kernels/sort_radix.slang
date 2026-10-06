// GPU sort of the splats back to front: a stable LSD radix sort of 32-bit
// keys carrying splat indices, 4 bits per pass (8 passes), in three kernels
// per pass: per-block digit histograms, a scan of those (scanChunks +
// addChunkOffsets, applied level by level by the host), and a stable scatter.
//
// Stability inside a block comes from a workgroup scan of the 16 digit
// counters packed 8 bits each in a uint4 (a block round is 128 threads, so a
// counter never passes 128). No subgroup operations: it runs everywhere
// WebGPU does (kernels/sort_scatter_subgroup.slang replaces radixScatter
// where the device has subgroups).
//
// The keys are the metric's (from generate.slang) inverted float bits, so
// the largest metric (farthest) comes first. Only active splats (finite
// metric) are sorted: countActive counts them per block (and in total, into
// the instance count of the indirect draw arguments), the host scans those
// counts, and compactKeys writes their keys in order, so the radix passes
// cost what the visible splats cost (generate culls what the draw would
// skip). writeDispatch sizes the radix passes' indirect dispatches to the
// active blocks, and they read the active count from drawArgs.

static const uint WG = 128;
static const uint ROUNDS = 8;
static const uint TILE = WG * ROUNDS; // keys per block
static const uint BINS = 16;
static const uint SCAN_WG = 256;
static const uint SCAN_CHUNK = 2 * SCAN_WG;

struct SortParams {
    uint count;     // keys to sort
    uint shift;     // digit position for this pass: 0, 4, ..., 28
    uint numBlocks; // ceil(count / TILE)
    uint keyMode;   // compactKeys: KEY_FLOAT or KEY_DEPTH
    float keyLo;    // KEY_DEPTH: the metric at the near plane
    float keyScale; // KEY_DEPTH: DEPTH_KEYS / (far - near metric)
    uint pad0;
    uint pad1;
};

// compactKeys' key of a metric. KEY_FLOAT: its inverted float bits (any
// metric). KEY_DEPTH: where every active metric lies in [keyLo, keyLo +
// DEPTH_KEYS / keyScale] (generate's cull keeps the view depth between the
// near and far planes), its place in that range on DEPTH_BITS bits, in the
// key's top bits: a uniform step over the range instead of the float's
// relative one, so DEPTH_BITS sort as finely as more float bits.
static const uint KEY_FLOAT = 0u;
static const uint KEY_DEPTH = 1u;
static const uint DEPTH_BITS = 24u;
static const float DEPTH_KEYS = 16777215.0; // 2^24 - 1

struct ScanParams {
    uint count; // elements of `scanData` at this level
    uint pad0;
    uint pad1;
    uint pad2;
};

StructuredBuffer<float> sortMetric;
RWStructuredBuffer<uint> keysIn;
RWStructuredBuffer<uint> valsIn;
RWStructuredBuffer<uint> keysOut;
RWStructuredBuffer<uint> valsOut;
RWStructuredBuffer<uint> blockHist;
RWStructuredBuffer<Atomic<uint>> drawArgs; // [vertexCount, instanceCount, firstVertex, firstInstance]
StructuredBuffer<uint> sortCount;          // drawArgs, read: [1] is the active count
RWStructuredBuffer<uint> dispatchArgs;     // indirect workgroup counts of the radix passes
RWStructuredBuffer<uint> scanData;
RWStructuredBuffer<uint> chunkSums;
ConstantBuffer<SortParams> params;
ConstantBuffer<ScanParams> scanParams;

groupshared Atomic<uint> sHist[BINS];
groupshared uint4 sScan[WG];
groupshared uint sBase[BINS];
groupshared uint sScanBuf[SCAN_CHUNK];
groupshared uint sFlags[WG];

bool isActiveMetric(float m) {
    return !isinf(m) && !isnan(m);
}

uint metricKey(float m) {
    if (params.keyMode == KEY_DEPTH) {
        // Farthest first: the far end of the range is key 0.
        float t = clamp((m - params.keyLo) * params.keyScale, 0.0, DEPTH_KEYS);
        return (uint(DEPTH_KEYS) - uint(t)) << (32u - DEPTH_BITS);
    }
    return min(~asuint(max(m, 0.0)), 0xfffffffeu);
}

// Active splats per block of TILE metrics, into blockHist[block].
[shader("compute")]
[numthreads(128, 1, 1)]
void countActive(uint3 lid: SV_GroupThreadID, uint3 gid: SV_GroupID) {
    if (lid.x == 0u) {
        sHist[0].store(0u);
    }
    GroupMemoryBarrierWithGroupSync();
    uint base = gid.x * TILE;
    uint n = 0u;
    for (uint r = 0; r < ROUNDS; r++) {
        uint i = base + r * WG + lid.x;
        if (i < params.count && isActiveMetric(sortMetric[i])) {
            n++;
        }
    }
    if (n > 0u) {
        sHist[0].add(n);
    }
    GroupMemoryBarrierWithGroupSync();
    if (lid.x == 0u) {
        uint total = sHist[0].load();
        blockHist[gid.x] = total;
        drawArgs[1].add(total);
    }
}

// The active splats' keys and indices, in index order, from each block's
// scanned start in blockHist.
[shader("compute")]
[numthreads(128, 1, 1)]
void compactKeys(uint3 lid: SV_GroupThreadID, uint3 gid: SV_GroupID) {
    uint t = lid.x;
    uint start = blockHist[gid.x];
    uint base = gid.x * TILE;
    for (uint r = 0; r < ROUNDS; r++) {
        uint i = base + r * WG + t;
        float m = i < params.count ? sortMetric[i] : 0.0;
        uint mine = i < params.count && isActiveMetric(m) ? 1u : 0u;
        sFlags[t] = mine;
        GroupMemoryBarrierWithGroupSync();
        for (uint o = 1; o < WG; o <<= 1) {
            uint add = (t >= o) ? sFlags[t - o] : 0u;
            GroupMemoryBarrierWithGroupSync();
            sFlags[t] += add;
            GroupMemoryBarrierWithGroupSync();
        }
        if (mine != 0u) {
            uint dst = start + sFlags[t] - 1u;
            keysOut[dst] = metricKey(m);
            valsOut[dst] = i;
        }
        uint roundTotal = sFlags[WG - 1u];
        GroupMemoryBarrierWithGroupSync();
        start += roundTotal;
    }
}


uint activeCount() {
    return sortCount[1];
}

// Blocks the radix passes run: dispatched indirectly from dispatchArgs.
uint activeBlocks() {
    return (activeCount() + TILE - 1u) / TILE;
}

// The radix passes' workgroup counts, from the active count.
[shader("compute")]
[numthreads(1, 1, 1)]
void writeDispatch() {
    dispatchArgs[0] = activeBlocks();
    dispatchArgs[1] = 1u;
    dispatchArgs[2] = 1u;
}

uint digitOf(uint key) {
    return (key >> params.shift) & (BINS - 1u);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void radixHistogram(uint3 lid: SV_GroupThreadID, uint3 gid: SV_GroupID) {
    if (lid.x < BINS) {
        sHist[lid.x].store(0u);
    }
    GroupMemoryBarrierWithGroupSync();
    uint base = gid.x * TILE;
    uint count = activeCount();
    for (uint r = 0; r < ROUNDS; r++) {
        uint i = base + r * WG + lid.x;
        if (i < count) {
            sHist[digitOf(keysIn[i])].add(1u);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (lid.x < BINS) {
        // Bin-major, so one exclusive scan gives every (bin, block) its start.
        blockHist[lid.x * activeBlocks() + gid.x] = sHist[lid.x].load();
    }
}

// Exclusive scan of SCAN_CHUNK elements per workgroup (Blelloch), writing
// each chunk's total to chunkSums.
[shader("compute")]
[numthreads(256, 1, 1)]
void scanChunks(uint3 lid: SV_GroupThreadID, uint3 gid: SV_GroupID) {
    uint t = lid.x;
    uint base = gid.x * SCAN_CHUNK;
    uint n = scanParams.count;
    sScanBuf[2 * t] = (base + 2 * t < n) ? scanData[base + 2 * t] : 0u;
    sScanBuf[2 * t + 1] = (base + 2 * t + 1 < n) ? scanData[base + 2 * t + 1] : 0u;

    uint offset = 1;
    for (uint d = SCAN_CHUNK >> 1; d > 0; d >>= 1) {
        GroupMemoryBarrierWithGroupSync();
        if (t < d) {
            uint ai = offset * (2 * t + 1) - 1;
            uint bi = offset * (2 * t + 2) - 1;
            sScanBuf[bi] += sScanBuf[ai];
        }
        offset <<= 1;
    }
    GroupMemoryBarrierWithGroupSync();
    if (t == 0) {
        chunkSums[gid.x] = sScanBuf[SCAN_CHUNK - 1];
        sScanBuf[SCAN_CHUNK - 1] = 0u;
    }
    for (uint d = 1; d < SCAN_CHUNK; d <<= 1) {
        offset >>= 1;
        GroupMemoryBarrierWithGroupSync();
        if (t < d) {
            uint ai = offset * (2 * t + 1) - 1;
            uint bi = offset * (2 * t + 2) - 1;
            uint v = sScanBuf[ai];
            sScanBuf[ai] = sScanBuf[bi];
            sScanBuf[bi] += v;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (base + 2 * t < n) {
        scanData[base + 2 * t] = sScanBuf[2 * t];
    }
    if (base + 2 * t + 1 < n) {
        scanData[base + 2 * t + 1] = sScanBuf[2 * t + 1];
    }
}

// Adds each chunk's scanned offset (chunkSums, scanned at the next level).
[shader("compute")]
[numthreads(256, 1, 1)]
void addChunkOffsets(uint3 tid: SV_DispatchThreadID) {
    uint i = tid.x;
    if (i < scanParams.count) {
        scanData[i] += chunkSums[i / SCAN_CHUNK];
    }
}

// The one-hot of a digit in 16 packed 8-bit lanes.
uint4 oneHot(uint digit) {
    uint4 v = uint4(0u);
    v[digit >> 2] = 1u << (8u * (digit & 3u));
    return v;
}

uint lane(uint4 v, uint digit) {
    return (v[digit >> 2] >> (8u * (digit & 3u))) & 0xffu;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void radixScatter(uint3 lid: SV_GroupThreadID, uint3 gid: SV_GroupID) {
    uint t = lid.x;
    if (t < BINS) {
        sBase[t] = blockHist[t * activeBlocks() + gid.x];
    }
    uint base = gid.x * TILE;
    uint count = activeCount();
    for (uint r = 0; r < ROUNDS; r++) {
        uint i = base + r * WG + t;
        bool valid = i < count;
        uint key = valid ? keysIn[i] : 0u;
        uint val = valid ? valsIn[i] : 0u;
        uint digit = digitOf(key);
        uint4 mine = valid ? oneHot(digit) : uint4(0u);

        // Inclusive Hillis-Steele scan of the packed counters over the round.
        sScan[t] = mine;
        GroupMemoryBarrierWithGroupSync();
        for (uint o = 1; o < WG; o <<= 1) {
            uint4 add = (t >= o) ? sScan[t - o] : uint4(0u);
            GroupMemoryBarrierWithGroupSync();
            sScan[t] += add;
            GroupMemoryBarrierWithGroupSync();
        }
        uint4 inclusive = sScan[t];
        if (valid) {
            uint rank = lane(inclusive - mine, digit);
            uint dst = sBase[digit] + rank;
            keysOut[dst] = key;
            valsOut[dst] = val;
        }
        GroupMemoryBarrierWithGroupSync();
        if (t < BINS) {
            sBase[t] += lane(sScan[WG - 1], t);
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

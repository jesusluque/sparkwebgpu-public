// Copyright (c) 2026 jesus luque.
//
// A FLATTENED CLOUD IN A FORMAT'S OWN NUMBERS (athenea flatten, task TXF):
// the records splatFlatten wrote (flatten_basis.slang's layout) as the floats
// a 3DGS PLY holds, the bytes of SPZ's six streams, or the attributes of a
// glTF KHR_gaussian_splatting primitive. The processor only writes headers
// and copies these out (athenea/io/SplatWriters.h).
//
// - `flattenBounds`, then `flattenBoundsResolve`: the box the positions fill
//   (glTF wants it on POSITION, and SPZ's fixed point is chosen from it), and
//   what each format will lose: sizes under its floor, SPZ harmonics past its
//   [-1, 1], gaussians whose higher bands were shrunk to stay positive.
// - `flattenPackPly`: 62 floats a gaussian -- x y z, nx ny nz, f_dc, f_rest
//   all red then green then blue, the opacity as a logit, ln sigma, w x y z.
// - `flattenPackSpz`: one SPZ stream, four bytes a thread, each byte worked
//   out from its gaussian as Niantic's packer works it out (third_party/spz,
//   `packGaussians`), so the result reads with any SPZ reader.
// - `flattenPackGlb`: the glTF attributes, each a block of its own.
module flatten_pack;
import flatten_basis;

struct PackParams {
    uint  count;            // gaussians in `records`
    uint  stream;           // SPZ: 0 positions, 1 alphas, 2 colours, 3 scales, 4 rotations, 5 harmonics
    uint  words;            // SPZ: words of the stream
    uint  fractionalBits;   // SPZ's fixed point
    uint  sh1Bits;          // SPZ: bits kept of band 1
    uint  shRestBits;       // SPZ: bits kept of bands 2 and 3
    float logScaleFloor;    // ln sigma no size goes below (PLY, glTF)
    uint  base;             // where this cloud's gaussians start in the output (PLY, glTF)
    uint  total;            // gaussians of the whole output (glTF's blocks)
    float unit;             // the file's units a metre (1, 100, 1000): positions times it
    float logUnit;          // ln unit: what it adds to ln sigma
};
ConstantBuffer<PackParams> params;
StructuredBuffer<float>   records;
RWStructuredBuffer<float> floats;   // PLY records or glTF blocks
RWStructuredBuffer<uint>  words;    // an SPZ stream
// [0..2] min, [3..5] max of x y z as ordered bits; [6] sizes under the floor;
// [7] gaussians with an SPZ harmonic past one; [8] gaussians shrunk;
// [9] sizes under SPZ's floor; [10] sizes over the ceiling of PLY and glTF
// (Spark's); [11] sizes over SPZ's; [12] gaussians the floor of PLY and
// glTF distorts (an axis lifted past a tenth of the largest), [13] SPZ's
// floor; then, resolved: [16..21] the box as float
// bits, [22] SPZ's fractional bits. The harmonics against Spark's 8-bit
// clamp: [40 + b] band b's clamp as float bits, [44 + b] coefficients past it,
// [48 + b] the largest |c| over the clamp as float bits, [52 + b] coefficients
// of the band, and from [64] a histogram of |c|, 256 bins a band (bands 1 to 3
// as b = 0 to 2).
RWStructuredBuffer<uint>  stats;

static const float kSpzColourScale = 0.15;
static const float kSpzLogScaleFloor = -10.0;
/// The largest ln sigma SPZ's byte holds: 255 / 16 - 10.
static const float kSpzLogScaleCeiling = 5.9375;
/// The largest ln sigma Spark keeps (proposal 070 section 9): its [-12, 9].
static const float kLogScaleCeiling = 9.0;

uint orderedBits(float f) {
    const uint u = asuint(f);
    return (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u);
}

float fromOrdered(uint u) {
    return asfloat((u & 0x80000000u) != 0u ? (u & 0x7FFFFFFFu) : ~u);
}

float rec(uint g, uint k) {
    return records[g * kFlatStride + k];
}

/// A coordinate in the file's unit (`params.unit` to the metre).
float position(uint g, uint a) {
    return rec(g, kFlatPosition + a) * params.unit;
}

/// ln sigma in the file's unit.
float logScaleIn(uint g, uint a) {
    return rec(g, kFlatLogScale + a) + params.logUnit;
}

float3 coefficient(uint g, uint k) {
    const uint at = g * kFlatStride + kFlatSh + k * 3;
    return float3(records[at], records[at + 1], records[at + 2]);
}

/// What SPZ's [-1, 1] makes of a gaussian's higher bands: one where they fit,
/// and otherwise the factor that brings the largest back to one -- the whole
/// set scaled, so the shape of the view dependence is kept and only its
/// strength is lost, rather than the largest clipped alone.
float spzBox(uint g) {
    float most = 0.0;
    for (uint k = 1; k < 16; ++k) {
        const float3 c = abs(coefficient(g, k));
        most = max(most, max(c.x, max(c.y, c.z)));
    }
    return most > 1.0 ? 1.0 / most : 1.0;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void flattenBounds(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    for (uint a = 0; a < 3; ++a) {
        const uint bits = orderedBits(position(g, a));
        InterlockedMin(stats[a], bits);
        InterlockedMax(stats[3 + a], bits);
    }
    uint floored = 0, spzFloored = 0, ceiled = 0, spzCeiled = 0;
    for (uint a = 0; a < 3; ++a) {
        const float s = logScaleIn(g, a);
        floored += s < params.logScaleFloor ? 1u : 0u;
        spzFloored += s < kSpzLogScaleFloor ? 1u : 0u;
        ceiled += s > kLogScaleCeiling ? 1u : 0u;
        spzCeiled += s > kSpzLogScaleCeiling ? 1u : 0u;
    }
    if (ceiled != 0) {
        InterlockedAdd(stats[10], ceiled);
    }
    // DISTORTED, NOT ONLY THINNER: a floor that lifts an axis to more than a
    // tenth of the gaussian's largest changes what it covers. A surface's
    // gaussian is a disc whose third axis is far under any floor by design,
    // and lifting that one to a few micrometres changes nothing a viewer
    // shows face on.
    const float largest = max(logScaleIn(g, 0), max(logScaleIn(g, 1), logScaleIn(g, 2)));
    if (floored != 0 && largest < params.logScaleFloor + 2.302585) {
        InterlockedAdd(stats[12], 1u);
    }
    if (spzFloored != 0 && largest < kSpzLogScaleFloor + 2.302585) {
        InterlockedAdd(stats[13], 1u);
    }
    if (spzCeiled != 0) {
        InterlockedAdd(stats[11], spzCeiled);
    }
    if (floored != 0) {
        InterlockedAdd(stats[6], floored);
    }
    if (spzBox(g) < 1.0) {
        InterlockedAdd(stats[7], 1u);
    }
    if (rec(g, kFlatShrink) < 0.999) {
        InterlockedAdd(stats[8], 1u);
    }
    if (spzFloored != 0) {
        InterlockedAdd(stats[9], spzFloored);
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void flattenBoundsResolve(uint3 tid: SV_DispatchThreadID) {
    float most = 1.0e-6;
    for (uint a = 0; a < 6; ++a) {
        const float v = fromOrdered(stats[a]);
        stats[16 + a] = asuint(v);
        most = max(most, abs(v));
    }
    // As many fractional bits as keep the farthest coordinate inside 24
    // signed bits, with one to spare; never more than 20 (a micrometre in
    // metres), never fewer than none.
    stats[22] = uint(clamp(int(floor(22.0 - log2(most))), 0, 20));
}

float logScaleOf(uint g, uint a) {
    return max(logScaleIn(g, a), params.logScaleFloor);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void flattenPackPly(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    const uint out = (params.base + g) * 62;
    for (uint a = 0; a < 3; ++a) {
        floats[out + a] = position(g, a);
        floats[out + 3 + a] = rec(g, kFlatNormal + a);
    }
    const float3 dc = coefficient(g, 0);
    floats[out + 6] = dc.x;
    floats[out + 7] = dc.y;
    floats[out + 8] = dc.z;
    for (uint k = 1; k < 16; ++k) {
        const float3 c = coefficient(g, k);
        floats[out + 9 + (k - 1)] = c.x;
        floats[out + 9 + 15 + (k - 1)] = c.y;
        floats[out + 9 + 30 + (k - 1)] = c.z;
    }
    const float alpha = clamp(rec(g, kFlatOpacity), 1.0e-6, 1.0 - 1.0e-6);
    floats[out + 54] = log(alpha / (1.0 - alpha));
    for (uint a = 0; a < 3; ++a) {
        floats[out + 55 + a] = logScaleOf(g, a);
    }
    for (uint a = 0; a < 4; ++a) {
        floats[out + 58 + a] = rec(g, kFlatRotation + a);
    }
}

uint toByte(float x) {
    return uint(clamp(round(x), 0.0, 255.0));
}

uint quantizeSh(float x, uint bucket) {
    int q = int(round(x * 128.0) + 128.0);
    q = (q + int(bucket) / 2) / int(bucket) * int(bucket);
    return uint(clamp(q, 0, 255));
}

/// SPZ's rotation, smallest three: two bits naming the largest of x y z w,
/// then the other three as a sign and nine bits of magnitude over sqrt(1/2),
/// the largest made positive -- Niantic's `packQuaternionSmallestThree`.
uint smallestThree(uint g) {
    const float4 wxyz = float4(rec(g, kFlatRotation), rec(g, kFlatRotation + 1), rec(g, kFlatRotation + 2),
                               rec(g, kFlatRotation + 3));
    const float4 q = normalize(float4(wxyz.y, wxyz.z, wxyz.w, wxyz.x));   // x y z w
    uint largest = 0;
    for (uint i = 1; i < 4; ++i) {
        if (abs(q[i]) > abs(q[largest])) {
            largest = i;
        }
    }
    const uint negate = q[largest] < 0.0 ? 1u : 0u;
    uint comp = largest;
    for (uint i = 0; i < 4; ++i) {
        if (i != largest) {
            const uint negbit = (q[i] < 0.0 ? 1u : 0u) ^ negate;
            const uint mag = uint(511.0 * (abs(q[i]) / 0.70710678118654752) + 0.5);
            comp = (comp << 10u) | (negbit << 9u) | min(mag, 511u);
        }
    }
    return comp;
}

static const uint kSpzBytes[6] = { 9, 1, 3, 3, 4, 45 };

uint spzByte(uint g, uint o) {
    switch (params.stream) {
    case 0: {
        const float scale = float(1u << params.fractionalBits);
        const int fixed = int(round(position(g, o / 3) * scale));
        return uint(fixed >> (8 * int(o % 3))) & 0xFFu;
    }
    case 1:
        return toByte(rec(g, kFlatOpacity) * 255.0);
    case 2:
        return toByte(coefficient(g, 0)[o] * (kSpzColourScale * 255.0) + 0.5 * 255.0);
    case 3:
        return toByte((logScaleIn(g, o) + 10.0) * 16.0);
    case 4:
        return (smallestThree(g) >> (8 * o)) & 0xFFu;
    default: {
        const uint k = o / 3 + 1;
        const uint bits = k < 4 ? params.sh1Bits : params.shRestBits;
        return quantizeSh(coefficient(g, k)[o % 3] * spzBox(g), 1u << (8u - bits));
    }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void flattenPackSpz(uint3 tid: SV_DispatchThreadID) {
    const uint w = tid.x;
    if (w >= params.words) {
        return;
    }
    const uint per = kSpzBytes[min(params.stream, 5u)];
    const uint bytes = params.count * per;
    uint word = 0;
    for (uint n = 0; n < 4; ++n) {
        const uint b = w * 4 + n;
        if (b < bytes) {
            word |= spzByte(b / per, b % per) << (8 * n);
        }
    }
    words[w] = word;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void flattenPackGlb(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    const uint n = params.total;
    const uint at = params.base + g;
    for (uint a = 0; a < 3; ++a) {
        floats[at * 3 + a] = position(g, a);
    }
    // KHR_gaussian_splatting:ROTATION is x y z w, as every glTF rotation.
    floats[3 * n + at * 4] = rec(g, kFlatRotation + 1);
    floats[3 * n + at * 4 + 1] = rec(g, kFlatRotation + 2);
    floats[3 * n + at * 4 + 2] = rec(g, kFlatRotation + 3);
    floats[3 * n + at * 4 + 3] = rec(g, kFlatRotation);
    for (uint a = 0; a < 3; ++a) {
        floats[7 * n + at * 3 + a] = exp(logScaleOf(g, a));
    }
    floats[10 * n + at] = saturate(rec(g, kFlatOpacity));
    for (uint k = 0; k < 16; ++k) {
        const float3 c = coefficient(g, k);
        const uint block = 11 * n + k * 3 * n;
        floats[block + at * 3] = c.x;
        floats[block + at * 3 + 1] = c.y;
        floats[block + at * 3 + 2] = c.z;
    }
}

// ---------------------------------------------------------------------------
// THE HARMONICS AGAINST SPARK'S CLAMP. Spark keeps each band of the higher
// harmonics in 8 bits over [-m, m], m the 95th percentile of |c| over the
// band's coefficients and at least one (sysop, from Spark's source): what lies
// past m is cut. A flattened highlight is a large higher band, so how much of
// it lies past is measured: a histogram of |c| a band, the percentile from
// it, and the coefficients past it and how far.

static const uint kHistogram = 64;
static const uint kBins = 256;

uint binOf(float a) {
    // An eighth of an octave a bin, 2^-16 to 2^16.
    return uint(clamp(floor((log2(max(a, 1.0e-30)) + 16.0) * 8.0), 0.0, float(kBins - 1)));
}

uint bandOf(uint k) {
    return k < 4 ? 0u : k < 9 ? 1u : 2u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void flattenShHistogram(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    for (uint k = 1; k < 16; ++k) {
        const float3 c = abs(coefficient(g, k));
        const uint band = bandOf(k);
        for (uint ch = 0; ch < 3; ++ch) {
            InterlockedAdd(stats[kHistogram + band * kBins + binOf(c[ch])], 1u);
        }
        InterlockedAdd(stats[52 + band], 3u);
    }
}

/// The 95th percentile a band, as the upper edge of the bin it falls in, and
/// never under one.
[shader("compute")]
[numthreads(1, 1, 1)]
void flattenShPercentile(uint3 tid: SV_DispatchThreadID) {
    for (uint band = 0; band < 3; ++band) {
        const uint total = stats[52 + band];
        const uint wanted = uint(ceil(0.95 * float(total)));
        uint seen = 0;
        float edge = 0.0;
        for (uint b = 0; b < kBins; ++b) {
            seen += stats[kHistogram + band * kBins + b];
            if (seen >= wanted) {
                edge = exp2(float(b + 1) / 8.0 - 16.0);
                break;
            }
        }
        stats[40 + band] = asuint(max(edge, 1.0));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void flattenShClamp(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    for (uint k = 1; k < 16; ++k) {
        const float3 c = abs(coefficient(g, k));
        const uint band = bandOf(k);
        const float m = asfloat(stats[40 + band]);
        for (uint ch = 0; ch < 3; ++ch) {
            if (c[ch] > m) {
                InterlockedAdd(stats[44 + band], 1u);
                InterlockedMax(stats[48 + band], asuint(c[ch] / m));
            }
        }
    }
}


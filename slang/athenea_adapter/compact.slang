// THE COMPACT RELIT PATH (thread CB, research/iphone-compact.md): athenea's
// proposals 090/094/097 on WebGPU. A relit TX cloud drawn from one compact
// buffer, with the relight evaluated per drawn splat inside the frame's
// shading pass, straight from athenea's joint CPCA: no relit records, no
// per-pixel records, no accumulator, and the sort buffers (sized to what is
// drawn) reused as the draw's records once sorted.
//
// src/webgpu/athenea/compact.ts writes the buffer and runs the passes
// (CompactRelitPath). Per gaussian, byte planes (SoA, each plane 64-byte
// aligned; proposal 097 S0/S1, with the deviations iphone-compact.md lists):
//   posX, posY, posZ   u16   in its chunk's box (8192 gaussians a chunk)
//   albedo             u8 x3 the stored colour (sRGB as the cloud keeps it)
//   opacity            u8    sqrt up to 1 (codes 0..191), then 1..2 (Spark's
//                            LoD opacity) in 1/63 steps (192..255)
//   scale              u8 x3 ln scale in its chunk's range (code 0: exactly 0)
//   rotation           u32   ext's own quaternion word (oct xy 10/10, angle 12)
//   normal             u16..u32 octahedral, normalBits an axis (12: 3 bytes;
//                            097's 8 + 8 shifts the coat's reflections)
//   curvature          s8 x3 xx, xy, yy, a signed log in its chunk's range
//   material           u16   into the material table (pbr, lobes: 4 words)
//   cluster            u8    athenea's joint CPCA cluster (C = 256)
//   weights            3 planes of 8 int8 (direct, indirect, reflected), or of
//                            8 int4 (weightBits 4, 097's mode 1)
//   shadow             u16   into its piece's codebook (the open-direction
//                            cells, 8 words; a piece is 16384 gaussians)
// and tables: the chunks' ranges (12 words a chunk), the pieces' codebook
// offsets, the codebooks, the materials, and athenea's basis in f16 (504
// words a cluster: per block its mean and its 8 directions times scale/127).
//
// PAGED (slotNodes > 0; thread CN, src/webgpu/athenea/compactPages.ts): the
// gaussians are athenea's error-band pages, each in a pool slot of slotNodes
// (3072) at slotsBase + slot x slotWords: the slot's head (its page's box and
// ranges, kCompactChunkWords; word 9 its piece's codebook, word 10 its
// materials, in words) then the planes at the plane uniforms' offsets within
// the slot, slotNodes elements each. Gaussian i is slot i / slotNodes,
// element i % slotNodes; the codebooks, materials and the basis are global
// to a piece (the pool's tables, always resident).
//
// Passes a frame:
//   atheneaCompactProject  a thread a cluster: the dome's projections (as
//                          cpca_joint.slang's atheneaJointProject), when the
//                          light changes, into compactFrame
//   atheneaCompactCull     a thread a gaussian: in the frustum -> a sort key
//                          (radial distance, 24 bits) and its index, appended
//   (GpuSorter.encodeKeys) the drawn keys, back to front
//   atheneaCompactShade    a thread a drawn slot: decode, relight (athenea's
//                          relitByDome for a TX cloud under one dome, fed from
//                          the CPCA directly), project, and the slot's record
//                          into the four sort buffers:
//                            rec0  centre x, y in px (u16, 1/16 px from -1024)
//                            rec1  sigma1 (half), the major axis' angle (u16 of pi)
//                            rec2  sigma2 (half), alpha (half; negative: LoD opacity)
//                            rec3  colour, linear RGB9E5
//   atheneaCompactVertex / atheneaCompactFragment   the quads, premultiplied
module compact;

import core.math;
import core.quat_codec;
import athenea.splat.splat_relight;
import athenea.common.packing;
import athenea.common.color;
import athenea.common.octahedral;
import athenea.light.environment;
import athenea.scene.splat_encoding;

public static const uint kCompactChunk = 8192;
public static const uint kCompactChunkWords = 12;
public static const uint kCompactPiece = 16384;
public static const uint kCompactMaterialWords = 4;
public static const uint kCompactClusterWords = 504;   // 1008 halves
public static const uint kCompactProjWords = 144;
public static const uint kCompactSunWords = 8;          // the sun after the projections
public static const uint kCompactWorkgroup = 256;
public static const uint kCompactRounds = 4;           // slots a thread (the sort's TILE / 256)
public static const uint kCompactDispatchRow = 65535u * 256u;
public static const float kCompactCentreOffset = 1024.0;
public static const float kCompactCentreScale = 16.0;
public static const float kCompactCurvatureFloor = 0.01;
public static const uint kCompactKeyBits = 24;

public static const uint kCompactLinear = 1u;      // the stored colour is linear light (else sRGB)
public static const uint kCompactIndirect = 2u;    // the transfer's indirect half
public static const uint kCompactCurvature = 4u;   // the cloud keeps its curvature
public static const uint kCompactFaint = 8u;       // faint splats keep their mass (Spark's DRAW_FAINT)
public static const uint kCompactPlain = 16u;      // debug: the albedo, no relight
public static const uint kCompactCullBacks = 32u;  // opaque splats whose stored face looks away are not drawn
public static const uint kCompactDrawList = 64u;   // the cull walks a list of gaussians (a LoD cut) in compactFrame
public static const uint kCompactBinned = 128u;    // the shade walks the drawn slots by cluster (binsIn)
public static const uint kCompactBins = 256u;      // clusters a cloud may have (u8 ids)
public static const uint kCompactStableOrder = 256u; // each listed gaussian its own slot (culled: ~0), for repeatable images
public static const uint kCompactNone = 0xFFFFFFFFu;

public struct CompactParams {
    public uint  count;
    public uint  flags;
    public uint  slotNodes;    ///< paged: gaussians a slot (0: one resident cloud)
    public uint  slotWords;    ///< paged: words a slot
    public uint  slotsBase;    ///< paged: the first slot, words
    public uint  chunkTable;   ///< words, kCompactChunkWords a chunk
    public uint  pieceTable;   ///< words, a piece's codebook start (words)
    public uint  posX;         ///< plane starts, words
    public uint  posY;
    public uint  posZ;
    public uint  albedo;
    public uint  opacity;
    public uint  scale;
    public uint  rotation;
    public uint  normal;
    public uint  curvature;
    public uint  material;
    public uint  cluster;
    public uint  weights;      ///< three planes, weightsStride words apart
    public uint  weightsStride;
    public uint  shadow;
    public uint  materials;    ///< the material table, words
    public uint  basis;        ///< athenea's basis, f16, words
    public uint  clusters;
    public uint  tableBase;    ///< the projections in compactFrame, words
    public uint  envBaseSide;
    public uint  capacity;     ///< sort slots
    public uint  normalBits;   ///< bits an axis of the octahedral normal (8, 12, 16)
    public uint  weightBits;   ///< bits a CPCA weight: 8, or 4 (a word a block: q8 = round(q4 x 127 / 7))
    public float ior;
    public float footprint;
    public float footprintClamp;
    public float glassCover;
    public float minAlpha;
    public float blurAmount;
    public float maxStdDev;
    public float maxPixelRadius;
    public float minPixelRadius;
    public float clipXY;
    public float near;
    public float far;
    public float falloff;
    public float keyScale;     ///< (2^24 - 1) / the farthest distance sorted
    public float cullFacing;   ///< kCompactCullBacks: the cosine below which a back is culled
    public uint  drawList;     ///< kCompactDrawList: the list's start in compactFrame, words (count entries)
    public float4 row0;        ///< the cloud to athenea's stage, rows of a 3x4
    public float4 row1;
    public float4 row2;
    public float4 eyeObject;   ///< the eye in the cloud's space
    public float4 eyeWorld;    ///< the eye in the stage
    public float4 eyeField;    ///< the eye the reflected field is read for (eyeWorld; another for staleness checks)
    public float4 view0;       ///< the cloud to the eye's (x right, y up, z depth ahead), rows
    public float4 view1;
    public float4 view2;
    public float4 focal;       ///< pixels: focal x, focal y (negative: rows go down), centre x, centre y
    public float4 viewport;    ///< width, height
};

StructuredBuffer<uint>              compactData;
StructuredBuffer<uint>              compactFrame;
RWStructuredBuffer<uint>            compactFrameOut;
StructuredBuffer<float4>            envSh;
StructuredBuffer<float4>            envSun;
StructuredBuffer<uint>              envTexels;
RWStructuredBuffer<uint>            sortKeys;
RWStructuredBuffer<uint>            sortVals;
RWStructuredBuffer<Atomic<uint>>    sortArgs;
StructuredBuffer<uint>              sortCount;
RWStructuredBuffer<uint>            rec0;
RWStructuredBuffer<uint>            rec1;
RWStructuredBuffer<uint>            rec2;
RWStructuredBuffer<uint>            rec3;
StructuredBuffer<uint>              draw0;
StructuredBuffer<uint>              draw1;
StructuredBuffer<uint>              draw2;
StructuredBuffer<uint>              draw3;
RWStructuredBuffer<float4>          debugOut;
RWStructuredBuffer<Atomic<uint>>    bins;      ///< kCompactBinned: counts, starts (kCompactBins each), then the slots by cluster
StructuredBuffer<uint>              binsIn;    ///< the same, read by the shade
ConstantBuffer<CompactParams>       compactParams;

bool hasFlag(uint f) { return (compactParams.flags & f) != 0u; }

/// The dome's sun (envSun's disc and irradiance), copied after the
/// projections by the project pass: the shade then binds no envSun.
float4 frameFloat4(uint at) {
    return float4(asfloat(compactFrame[at]), asfloat(compactFrame[at + 1u]), asfloat(compactFrame[at + 2u]),
                  asfloat(compactFrame[at + 3u]));
}
float4 sunDisc() { return frameFloat4(compactParams.tableBase + compactParams.clusters * kCompactProjWords); }
float3 sunIrradiance() { return frameFloat4(compactParams.tableBase + compactParams.clusters * kCompactProjWords + 4u).rgb; }

// ---------------------------------------------------------------- the planes

/// Where element i of the plane at `plane` is: the plane's first word for
/// its slot, and its place in it.
uint planeOf(uint plane, uint i, out uint element) {
    const uint nodes = compactParams.slotNodes;
    if (nodes == 0u) {
        element = i;
        return plane;
    }
    const uint slot = i / nodes;
    element = i - slot * nodes;
    return compactParams.slotsBase + slot * compactParams.slotWords + plane;
}

/// `width` bytes (1..4) of element i of the plane at word `plane`.
uint fieldAt(uint plane, uint width, uint i) {
    uint e;
    const uint base = planeOf(plane, i, e);
    const uint at = e * width;
    const uint w = base + (at >> 2u);
    const uint shift = (at & 3u) * 8u;
    uint v = compactData[w] >> shift;
    if (shift + width * 8u > 32u) {
        v |= compactData[w + 1u] << (32u - shift);
    }
    return width >= 4u ? v : v & ((1u << (width * 8u)) - 1u);
}

/// Word k of gaussian i's chunk (resident) or slot head (paged).
uint chunkWord(uint i, uint k) {
    const uint nodes = compactParams.slotNodes;
    if (nodes == 0u) {
        return compactData[compactParams.chunkTable + (i / kCompactChunk) * kCompactChunkWords + k];
    }
    return compactData[compactParams.slotsBase + (i / nodes) * compactParams.slotWords + k];
}

float chunkFloat(uint i, uint k) {
    return asfloat(chunkWord(i, k));
}

public struct CompactSplat {
    public float3 center;
    public float3 scales;
    public float4 quaternion;
    public float4 rgba;        ///< the stored colour, the opacity (> 1: LoD opacity)
};

float3 positionOf(uint i) {
    const float3 lo = float3(chunkFloat(i, 0u), chunkFloat(i, 1u), chunkFloat(i, 2u));
    const float3 hi = float3(chunkFloat(i, 3u), chunkFloat(i, 4u), chunkFloat(i, 5u));
    const uint3 q = uint3(fieldAt(compactParams.posX, 2u, i), fieldAt(compactParams.posY, 2u, i),
                          fieldAt(compactParams.posZ, 2u, i));
    return lo + float3(q) * ((hi - lo) / 65535.0);
}

/// The opacity code: sqrt up to 1, then 1..2 in 1/63 steps.
public float opacityOf(uint code) {
    if (code <= 191u) {
        const float s = float(code) / 191.0;
        return s * s;
    }
    return 1.0 + float(code - 192u) / 63.0;
}

float3 scalesOf(uint i) {
    const uint w = fieldAt(compactParams.scale, 3u, i);
    const float lo = chunkFloat(i, 6u);
    const float step = (chunkFloat(i, 7u) - lo) / 254.0;
    float3 s;
    [unroll] for (uint k = 0; k < 3u; ++k) {
        const uint c = (w >> (8u * k)) & 0xFFu;
        s[k] = c == 0u ? 0.0 : exp(lo + float(c - 1u) * step);
    }
    return s;
}

public CompactSplat readCompact(uint i) {
    CompactSplat g;
    g.center = positionOf(i);
    g.scales = scalesOf(i);
    g.quaternion = decodeQuatOctXy1010R12(fieldAt(compactParams.rotation, 4u, i));
    const uint c = fieldAt(compactParams.albedo, 3u, i);
    g.rgba = float4(float(c & 0xFFu) / 255.0, float((c >> 8u) & 0xFFu) / 255.0, float((c >> 16u) & 0xFFu) / 255.0,
                    opacityOf(fieldAt(compactParams.opacity, 1u, i)));
    return g;
}

/// The stored normal (athenea's packNormal at normalBits an axis).
float3 normalOf(uint i) {
    const uint bits = compactParams.normalBits;
    const uint q = fieldAt(compactParams.normal, (2u * bits + 7u) / 8u, i);
    const uint top = (1u << bits) - 1u;
    const float2 f = float2(float(q & top), float((q >> bits) & top)) / float(top) * 2.0 - 1.0;
    float3 n = float3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    const float t = saturate(-n.z);
    n.x += n.x >= 0.0 ? -t : t;
    n.y += n.y >= 0.0 ? -t : t;
    return normalize(n);
}

/// The curvature (xx, xy, yy): a signed log of |c| / floor over its chunk's
/// largest, 127 steps.
float3 curvatureOf(uint i) {
    const uint w = fieldAt(compactParams.curvature, 3u, i);
    const float range = chunkFloat(i, 8u);   // log2(1 + max / floor)
    float3 k;
    [unroll] for (uint j = 0; j < 3u; ++j) {
        const int c = int(((w >> (8u * j)) & 0xFFu) << 24u) >> 24;
        const float m = kCompactCurvatureFloor * (exp2(float(abs(c)) / 127.0 * range) - 1.0);
        k[j] = c < 0 ? -m : m;
    }
    return k;
}

uint4 materialOf(uint i) {
    const uint table = compactParams.slotNodes == 0u ? compactParams.materials : chunkWord(i, 10u);
    const uint at = table + fieldAt(compactParams.material, 2u, i) * kCompactMaterialWords;
    return uint4(compactData[at], compactData[at + 1u], compactData[at + 2u], compactData[at + 3u]);
}

/// Where splat i's open-direction cells are in `compact` (words).
uint cellBaseOf(uint i) {
    const uint book = compactParams.slotNodes == 0u ? compactData[compactParams.pieceTable + i / kCompactPiece]
                                                    : chunkWord(i, 9u);
    return book + fieldAt(compactParams.shadow, 2u, i) * 8u;
}

// ------------------------------------------------------- the joint CPCA's terms

float basisAt(uint cluster, uint f) {
    const uint w = compactData[compactParams.basis + cluster * kCompactClusterWords + (f >> 1u)];
    return f16tof32((f & 1u) != 0u ? w >> 16u : w & 0xFFFFu);
}

float weightOf(uint w, uint k) {
    return float(int(w << (24u - 8u * k)) >> 24);
}

/// Block b's 8 int8 weights of splat i (two words).
uint2 weightsOf(uint i, uint b) {
    uint e;
    const uint plane = planeOf(compactParams.weights + b * compactParams.weightsStride, i, e);
    if (compactParams.weightBits == 4u) {
        // Back on the int8 grid the records were measured on.
        const uint w = compactData[plane + e];
        uint2 out = uint2(0u);
        [unroll] for (uint k = 0; k < 8u; ++k) {
            const int q4 = int(((w >> (4u * k)) & 0xFu) << 28u) >> 28;
            const int q8 = int(round(float(q4) * (127.0 / 7.0)));
            const uint byte = uint(q8) & 0xFFu;
            if (k < 4u) {
                out.x |= byte << (8u * k);
            } else {
                out.y |= byte << (8u * (k - 4u));
            }
        }
        return out;
    }
    return uint2(compactData[plane + 2u * e], compactData[plane + 2u * e + 1u]);
}

/// Term `term` of a cluster's projections (atheneaCompactProject) for a
/// block's weights: as relight.slang's jointTerm.
float3 jointTerm(uint cluster, uint term, uint2 w) {
    const uint at = compactParams.tableBase + cluster * kCompactProjWords + term * 36u;
    float3 v = float3(asfloat(compactFrame[at]), asfloat(compactFrame[at + 1u]), asfloat(compactFrame[at + 2u]));
    [unroll] for (uint k = 0; k < 8u; ++k) {
        const float q = weightOf(k < 4u ? w.x : w.y, k & 3u);
        const uint d = at + 4u * (1u + k);
        v += q * float3(asfloat(compactFrame[d]), asfloat(compactFrame[d + 1u]), asfloat(compactFrame[d + 2u]));
    }
    return v;
}

/// transferViewless from the projections (relight.slang jointViewless): the
/// body's light, the field's coupling to this sky, and the sun's share.
void compactViewless(SplatSurface s, uint c, uint2 wd, uint2 wi, uint dome, bool withIndirect, out float3 front,
                     out float3 coupling, out float sunShare) {
    front = jointTerm(c, 0u, wd);
    const float3 gathered = jointTerm(c, 1u, wi);
    if (withIndirect) {
        front += gathered;
    }
    const float4 disc = sunDisc();
    const float3 irradiance = disc.w > 0.0 ? sunIrradiance() : float3(0.0);
    const float cosine = max(dot(s.normal, disc.xyz), 0.0);
    sunShare = disc.w > 0.0 ? splatOpenToward(s, compactData, compactData, disc.xyz) : 1.0;
    front += irradiance * (cosine * sunShare * kOverPiSun);
    float3 along = float3(0.0);
    if (withIndirect && disc.w > 0.0) {
        along = max(jointTerm(c, 2u, wi), float3(0.0));
        front += irradiance * along;
    }
    coupling = float3(0.0);
    if (withIndirect) {
        float3 all = gathered;
        float3 mean = envSh[dome * kEnvCoefficients].rgb * 0.28209479177387814;
        if (disc.w > 0.0) {
            all += irradiance * along;
            mean += irradiance * (1.0 / (4.0 * 3.14159265358979));
        }
        const float3 white = jointTerm(c, 3u, wi) * 3.5449077018110318;
        coupling = float3(white.x > 1.0e-4 ? max(all.x / white.x, 0.0) : mean.x,
                          white.y > 1.0e-4 ? max(all.y / white.y, 0.0) : mean.y,
                          white.z > 1.0e-4 ? max(all.z / white.z, 0.0) : mean.z);
    }
}

/// The reflected block (48 values: 16 coefficients x rgb), rebuilt in
/// registers from the cluster's mean and directions (f16) and the weights.
void reflectedBlock(uint cluster, uint2 w, out float field[48]) {
    const uint at = 576u;   // the block's first float in the cluster
    [unroll] for (uint v = 0; v < 48u; ++v) {
        field[v] = basisAt(cluster, at + v);
    }
    [unroll] for (uint k = 0; k < 8u; ++k) {
        const float q = weightOf(k < 4u ? w.x : w.y, k & 3u);
        const uint dir = at + (1u + k) * 48u;
        [unroll] for (uint v = 0; v < 48u; v += 2u) {
            const uint word = compactData[compactParams.basis + cluster * kCompactClusterWords + ((dir + v) >> 1u)];
            field[v] += q * f16tof32(word & 0xFFFFu);
            field[v + 1u] += q * f16tof32(word >> 16u);
        }
    }
}

/// The direct block's first value (splatOpenness reads it).
float opennessValue(uint cluster, uint2 w) {
    float v = basisAt(cluster, 0u);
    [unroll] for (uint k = 0; k < 8u; ++k) {
        v += weightOf(k < 4u ? w.x : w.y, k & 3u) * basisAt(cluster, (1u + k) * 16u);
    }
    return v;
}

float bandOf(uint j) { return j < 1u ? 0.0 : (j < 4u ? 2.0 : (j < 9u ? 6.0 : 12.0)); }

/// splatFieldAlongPair over the rebuilt block.
void fieldAlongPair(float field[48], float3 d, float roughness, float coatRoughness, out float3 polish,
                    out float3 coat) {
    const float a = roughness * roughness;
    const float b = coatRoughness * coatRoughness;
    const float a2 = a * a, b2 = b * b;
    polish = float3(0.0);
    coat = float3(0.0);
    [unroll] for (uint j = 0; j < 16u; ++j) {
        const float3 value = float3(field[3u * j], field[3u * j + 1u], field[3u * j + 2u]) * envBasisValue(j, d);
        polish += value * exp(-bandOf(j) * a2);
        coat += value * exp(-bandOf(j) * b2);
    }
    polish = max(polish, float3(0.0));
    coat = max(coat, float3(0.0));
}

/// splatFieldAlong over the rebuilt block.
float3 fieldAlong(float field[48], float3 d, float roughness) {
    const float alpha = roughness * roughness;
    const float a2 = alpha * alpha;
    float3 sum = float3(0.0);
    [unroll] for (uint j = 0; j < 16u; ++j) {
        sum += float3(field[3u * j], field[3u * j + 1u], field[3u * j + 2u]) * (exp(-bandOf(j) * a2) * envBasisValue(j, d));
    }
    return max(sum, float3(0.0));
}

/// The reflected block read along directions without rebuilding it into
/// registers: the field's values are made two coefficients at a time (each
/// value as reflectedBlock makes it, the mean then the eight directions in
/// order) and summed straight into fieldAlongPair's polish and coat along
/// `m` and, with `through`, fieldAlong's value along `t` -- the same sums in
/// the same order as rebuilding the block then reading it, with a handful
/// of registers instead of 48 (the rebuilt block spilled).
void reflectedAlong(uint cluster, uint2 w, float3 m, float roughness, float coatRoughness, bool through, float3 t,
                    out float3 polish, out float3 coat, out float3 along) {
    const float a = roughness * roughness;
    const float b = coatRoughness * coatRoughness;
    const float a2 = a * a, b2 = b * b;
    float q[8];
    [unroll] for (uint k = 0; k < 8u; ++k) {
        q[k] = weightOf(k < 4u ? w.x : w.y, k & 3u);
    }
    const uint base = compactParams.basis + cluster * kCompactClusterWords;
    polish = float3(0.0);
    coat = float3(0.0);
    along = float3(0.0);
    // exp(-band a^2) by band (bands 0, 2, 6, 12), as fieldAlongPair takes it.
    float ga[4], gb[4];
    [unroll] for (uint n = 0; n < 4u; ++n) {
        const float band = n == 0u ? 0.0 : (n == 1u ? 2.0 : (n == 2u ? 6.0 : 12.0));
        ga[n] = exp(-band * a2);
        gb[n] = exp(-band * b2);
    }
    [unroll] for (uint jp = 0; jp < 8u; ++jp) {
        // Values 6 jp .. 6 jp + 5: coefficients 2 jp and 2 jp + 1, three words.
        float v[6];
        [unroll] for (uint h = 0; h < 3u; ++h) {
            const uint at = 576u + 6u * jp + 2u * h;
            const uint mean = compactData[base + (at >> 1u)];
            float lo = f16tof32(mean & 0xFFFFu);
            float hi = f16tof32(mean >> 16u);
            [unroll] for (uint k = 0; k < 8u; ++k) {
                const uint word = compactData[base + ((at + (1u + k) * 48u) >> 1u)];
                lo += q[k] * f16tof32(word & 0xFFFFu);
                hi += q[k] * f16tof32(word >> 16u);
            }
            v[2u * h] = lo;
            v[2u * h + 1u] = hi;
        }
        [unroll] for (uint e = 0; e < 2u; ++e) {
            const uint j = 2u * jp + e;
            const float3 f = float3(v[3u * e], v[3u * e + 1u], v[3u * e + 2u]);
            const uint n = j < 1u ? 0u : (j < 4u ? 1u : (j < 9u ? 2u : 3u));
            const float3 value = f * envBasisValue(j, m);
            polish += value * ga[n];
            coat += value * gb[n];
            if (through) {
                along += f * (ga[n] * envBasisValue(j, t));
            }
        }
    }
    polish = max(polish, float3(0.0));
    coat = max(coat, float3(0.0));
    along = max(along, float3(0.0));
}

// -------------------------------------- athenea's relitByDome, TX under a dome
// (splat_relight.slang at txf 89a04d9: its helpers below are module-internal
// there, so copied as they are; the transfer's reads come from the rebuilt
// block, the cells' from the piece's codebook in `compact`.)

float thinWallGainOf(SplatSurface s) {
    if (!s.thinWalled) {
        return 1.0;
    }
    const float index = s.ior > 1.0 ? s.ior : 1.5;
    const float r = (index - 1.0) / (index + 1.0);
    return 1.0 / max(r * r, 1.0e-3);
}

float transmittedWeightOf(SplatSurface s) {
    return s.thinWalled ? 0.0 : s.transmission;
}

float3 transmittedBehindOf(SplatSurface s, float3 wo, uint dome) {
    const float3 seen = s.ior > 1.0 ? envRadiance(envTexels, dome, bentThrough(s, wo), s.roughness,
                                                  compactParams.envBaseSide)
                                    : envIrradiance(envSh, dome, -s.normal) * (1.0 / 3.14159265358979);
    return seen * s.exitThrough;
}

float3 transmittedShareOf(SplatSurface s, float nDotV) {
    return s.backFacing && s.ior > 1.0
               ? float3(0.0)
               : saturate(float3(1.0) - ggxEnvDielectricAt(s.roughness, nDotV, s.lobes.specularIor) *
                                            s.lobes.specularWeight);
}

/// relitByDome<2> for a TX cloud (cells and field) with the kept terms in
/// hand, nothing traced, no slope (pixelDetail off): a gaussian's light.
float3 relitCompact(SplatSurface s, float3 wo, uint dome, bool withIndirect, float3 front, float3 coupling,
                    float sunShare, float3 fieldPolish, float3 fieldCoat, float3 fieldThrough, float openness,
                    float lensCurvature) {
    const uint side = compactParams.envBaseSide;
    const float nDotV = max(dot(s.normal, wo), 1.0e-4);
    const float3 mirror = reflect(-wo, s.normal);
    const SplatLayers layers = splatLayers(s, nDotV);
    const bool coated = s.lobes.coatWeight > 0.0;
    const float4 disc = sunDisc();
    const float3 irradiance = disc.w > 0.0 ? sunIrradiance() : float3(0.0);
    const bool cells = s.cellWords >= 8u;
    const bool hasField = cells && withIndirect;
    float3 behind = transmittedBehindOf(s, wo, dome);
    if (hasField && transmittedWeightOf(s) > 0.0 && s.ior > 1.0) {
        const float3 white = fieldThrough;
        const float open = smoothstep(0.5, 0.85, dot(white, float3(0.2126, 0.7152, 0.0722)));
        const float3 sharp =
            lensCurvature > 0.0
                ? envRadiance(envTexels, dome, lensExit(s, wo, lensCurvature),
                              transmittedRoughness(s.roughness, s.ior), side) *
                      lensExitThrough(s, wo, lensCurvature)
                : transmittedBehindOf(s, wo, dome);
        behind = open * sharp + (1.0 - open) * coupling * white * s.exitThrough;
    } else if (hasField && transmittedWeightOf(s) > 0.0) {
        const float3 through = -wo;
        const float alpha = max(s.roughness * s.roughness, 1.0e-3);
        const float openBehind = splatConeOpen(s, compactData, through, min(1.4 * atan(alpha), 1.2), -s.normal);
        const float open = openBehind >= 0.0 ? openBehind : 0.0;
        behind = (open * envRadiance(envTexels, dome, through, s.roughness, side) +
                  (1.0 - open) * coupling * fieldThrough) *
                 s.exitThrough;
    }
    const float3 passes = transmittedShareOf(s, nDotV);
    const float3 body2 = layers.under * s.albedo * (1.0 - s.metallic) *
                         ((1.0 - s.transmission) * front + transmittedWeightOf(s) * passes * behind);
    const float3 seen = envRadiance(envTexels, dome, mirror, s.roughness, side);
    const float openAll = saturate(openness / 0.28209479177387814);
    const float narrowed = specularOpenness(nDotV, openAll, s.roughness);
    const float3 sunPolish = !(disc.w > 0.0) ? float3(0.0)
                                             : splatPolish(s, nDotV) * irradiance *
                                                   (sunShare * sunReflectPdf(mirror, disc.xyz, s.roughness, disc.w));
    float3 coat = float3(0.0);
    if (coated) {
        const float3 coatSeen = envRadiance(envTexels, dome, mirror, s.lobes.coatRoughness, side);
        const float coatNarrowed = cells ? splatLobeOpen(s, compactData, mirror, s.lobes.coatRoughness, s.normal)
                                         : specularOpenness(nDotV, openAll, s.lobes.coatRoughness);
        const float3 coatSun = !(disc.w > 0.0)
                                   ? float3(0.0)
                                   : layers.coat * irradiance *
                                         (sunShare * sunReflectPdf(mirror, disc.xyz, s.lobes.coatRoughness, disc.w));
        coat = (hasField ? max(layers.coat * (coatNarrowed * coatSeen + (1.0 - coatNarrowed) * coupling * fieldCoat),
                               float3(0.0))
                         : layers.coat * coatNarrowed * coatSeen) +
               coatSun;
    }
    const float3 sheen = layers.belowCoat * layers.sheen * front;
    if (hasField) {
        return body2 +
               layers.under * thinWallGainOf(s) *
                   (max(splatPolish(s, nDotV) * (narrowed * seen), float3(0.0)) + sunPolish) +
               coat + sheen;
    }
    return body2 + layers.under * thinWallGainOf(s) * (splatPolish(s, nDotV) * narrowed * seen + sunPolish) + coat +
           sheen;
}

// relight.slang's footprint prefilter and solid-glass cover, as there.
float footprintOf(float roughness, float3 shapeUV, float3 scales, float gain, float kappa) {
    const float su = scales.x;
    const float sv = scales.y;
    const float splatVariance = su * su * (shapeUV.x * shapeUV.x + shapeUV.y * shapeUV.y) +
                                sv * sv * (shapeUV.y * shapeUV.y + shapeUV.z * shapeUV.z);
    const float variance = max(splatVariance, 0.0);
    const float alpha = roughness * roughness;
    const float add = min(gain * gain * variance, kappa);
    const float widened = min(alpha * alpha + add, 1.0);
    const float r = sqrt(sqrt(widened));
    return isfinite(r) ? max(r, roughness) : roughness;
}

float coveredGlass(float a, float n) {
    const float e = 2.718281828459045;
    float ratio = a;
    if (a > 1.0) {
        const float d = min(4.0 * a - 3.0, 5.0);
        ratio = exp((d * d - 1.0) / e);
    }
    const float covered = n * ratio;
    if (covered <= 1.0) {
        return 1.0 - pow(max(1.0 - min(a, 1.0), 0.0), n);
    }
    const float d = min(sqrt(1.0 + e * log(covered)), 5.0);
    return 1.0 + 0.25 * (d - 1.0);
}

/// A gaussian's surface for the relight: everything relitByDome reads that
/// is not the transfer's terms, with the footprint prefilter applied.
struct CompactSurface {
    SplatSurface surface;
    float lensCurvature;
    uint cluster;
    uint index;
};

CompactSurface surfaceOf(uint i, CompactSplat g) {
    CompactSurface o;
    const float3x3 r = splatRotation(g.quaternion);
    const float3 stored = normalOf(i);
    const float3 colour = cloudLight(g.rgba.rgb, hasFlag(kCompactLinear));
    const float3 cloudNormal = splatShadingNormal(r, g.scales, g.center, compactParams.eyeObject.xyz, false, 0u);
    // splatShadingNormal with a stored normal: put on the disc's side.
    const float3 disc = cloudNormal;
    const float3 n = dot(stored, disc) < 0.0 ? -stored : stored;
    const uint4 m = materialOf(i);
    const uint pbr = m.x;
    const float4 w0 = compactParams.row0;
    const float4 w1 = compactParams.row1;
    const float4 w2 = compactParams.row2;
    SplatSurface surface;
    surface.position = relightToWorld(w0, w1, w2, g.center);
    surface.normal = relightNormalToWorld(w0, w1, w2, n);
    surface.albedo = colour;
    surface.metallic = pbrMetallic(pbr);
    surface.roughness = pbrRoughness(pbr);
    surface.transmission = pbrTransmission(pbr);
    surface.thinWalled = pbrThinWalled(pbr);
    surface.schlickMetal = pbrSchlickMetal(pbr);
    surface.ior = compactParams.ior;
    surface.backFacing = splatFacing(r, g.scales, g.center, compactParams.eyeObject.xyz) < 0.0;
    surface.hasBehind = false;
    surface.behindColour = float3(0.0);
    surface.exitThrough = float3(1.0);
    surface.hasReflection = false;
    surface.reflectionColour = float3(0.0);
    surface.reflectionTraced = false;
    surface.hasExit = false;
    surface.exitDirection = float3(0.0, 0.0, 1.0);
    surface.lit = false;
    surface.emission = float3(0.0);
    surface.lobes = unpackLobes(m.yzw);
    if (surface.ior > 1.0 && surface.transmission > 0.0) {
        surface.ior = max(surface.lobes.specularIor, 1.0);
    }
    surface.transferCount = 112u;
    surface.transferBase = 0u;
    splatTransferFrame(surface, compactData, r, w0, w1, w2);
    surface.hasShadowBits = false;
    surface.shadowBits = uint2(0u);
    surface.cellWords = 8u;
    surface.cellBase = cellBaseOf(i);
    // The curvature: the lens of a solid glass, and the footprint prefilter.
    o.lensCurvature = 0.0;
    if (hasFlag(kCompactCurvature)) {
        const float3 shapeUV = curvatureOf(i);
        const float3 axisN = float3(r[0][2], r[1][2], r[2][2]);
        const float faces = dot(n, axisN) >= 0.0 ? 1.0 : -1.0;
        o.lensCurvature = faces * 0.5 * (shapeUV.x + shapeUV.z);
        if (compactParams.footprint > 0.0) {
            surface.roughness = footprintOf(surface.roughness, shapeUV, g.scales, compactParams.footprint,
                                            compactParams.footprintClamp);
            surface.lobes.coatRoughness = footprintOf(surface.lobes.coatRoughness, shapeUV, g.scales,
                                                      compactParams.footprint, compactParams.footprintClamp);
        }
    }
    o.surface = surface;
    o.cluster = fieldAt(compactParams.cluster, 1u, i);
    o.index = i;
    return o;
}

/// The alpha the draw starts from: the stored opacity, or the one the
/// relight gives a glass (shadeColour's, from the material alone).
float drawAlphaOf(uint i, CompactSplat g) {
    if (hasFlag(kCompactPlain)) {
        return g.rgba.a;
    }
    const uint pbr = materialOf(i).x;
    const float transmission = pbrTransmission(pbr);
    const bool thin = pbrThinWalled(pbr);
    if (thin && transmission > 0.5) {
        return max(max(min(g.rgba.a, 1.0) - 1.0 / 255.0, 0.0), 0.1);
    }
    if (!thin && transmission > 0.5 && compactParams.glassCover > 1.0) {
        return coveredGlass(g.rgba.a, compactParams.glassCover);
    }
    return g.rgba.a;
}

/// The eye's direction from a surface.
float3 eyeFrom(SplatSurface s) { return normalize(compactParams.eyeWorld.xyz - s.position); }

/// The reflected field along the mirror (polish, coat) and, for a
/// transmitting splat, along the way through: what relitCompact reads.
void reflectedTerms(CompactSurface c, uint2 wr, out float3 polish, out float3 coat, out float3 through) {
    const SplatSurface s = c.surface;
    // eyeField.w (checks): 1 every splat reads eyeField; 2 or 3 the splats
    // of the other parity than (w - 2) read it, the rest the eye.
    const float mode = compactParams.eyeField.w;
    const bool stale = mode < 1.5 || float(c.index & 1u) != mode - 2.0;
    const float3 wo = normalize((stale ? compactParams.eyeField.xyz : compactParams.eyeWorld.xyz) - s.position);
    const float3 mirror = reflect(-wo, s.normal);
    const bool coated = s.lobes.coatWeight > 0.0;
    reflectedAlong(c.cluster, wr, mirror, s.roughness, coated ? s.lobes.coatRoughness : s.roughness,
                   transmittedWeightOf(s) > 0.0, -wo, polish, coat, through);
}

/// Splat i's relit colour (linear) and alpha (-1: its own), as relight.slang's
/// relitColour<2, 2> (the joint CPCA under one dome, no slope).
public float4 shadeColour(uint i, CompactSplat g) {
    if (hasFlag(kCompactPlain)) {
        return float4(cloudLight(g.rgba.rgb, hasFlag(kCompactLinear)), -1.0);
    }
    const CompactSurface cs = surfaceOf(i, g);
    SplatSurface surface = cs.surface;
    const uint c = cs.cluster;
    // The kept terms, from the clusters' projections.
    const uint2 wd = weightsOf(i, 0u);
    const uint2 wi = weightsOf(i, 1u);
    const bool indirect = hasFlag(kCompactIndirect);
    float3 front;
    float3 coupling;
    float sunShare;
    compactViewless(surface, c, wd, wi, 0u, indirect, front, coupling, sunShare);
    float3 polish = float3(0.0);
    float3 coat = float3(0.0);
    float3 through = float3(0.0);
    if (indirect) {
        reflectedTerms(cs, weightsOf(i, 2u), polish, coat, through);
    }
    const float openness = opennessValue(c, wd);
    float3 light = relitCompact(surface, eyeFrom(surface), 0u, indirect, front, coupling, sunShare, polish, coat,
                                through, openness, cs.lensCurvature);
    float alpha = -1.0;
    if (surface.thinWalled && surface.transmission > 0.5) {
        const float alphaOwn = max(min(g.rgba.a, 1.0) - 1.0 / 255.0, 0.0);
        alpha = max(alphaOwn, 0.1);
        light *= alphaOwn / alpha;
    } else if (!surface.thinWalled && surface.transmission > 0.5 && compactParams.glassCover > 1.0) {
        alpha = coveredGlass(g.rgba.a, compactParams.glassCover);
    }
    return float4(light, alpha);
}

// ------------------------------------------------------------- the passes

/// A cluster's projections on the dome (cpca_joint.slang atheneaJointProject
/// over the f16 basis).
[shader("compute")]
[numthreads(64, 1, 1)]
void atheneaCompactProject(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c == 0u) {
        const uint at = compactParams.tableBase + compactParams.clusters * kCompactProjWords;
        [unroll] for (uint k = 0; k < 4u; ++k) {
            compactFrameOut[at + k] = asuint(envSun[0][k]);
            compactFrameOut[at + 4u + k] = asuint(envSun[1][k]);
        }
    }
    if (c >= compactParams.clusters) {
        return;
    }
    const uint sh = 0u;
    const float4 disc = envSun[0];
    const float3 sun = disc.w > 0.0 ? disc.xyz : float3(0.0, 0.0, 1.0);
    const uint out = compactParams.tableBase + c * kCompactProjWords;
    for (uint j = 0; j < 9u; ++j) {
        float3 body = float3(0.0);
        [unroll] for (uint k = 0; k < 16u; ++k) {
            body += envSh[sh + k].rgb * basisAt(c, j * 16u + k);
        }
        float3 gathered = float3(0.0);
        float3 along = float3(0.0);
        float3 white = float3(0.0);
        const uint in0 = 144u + j * 48u;
        for (uint k = 0; k < 16u; ++k) {
            const float3 rgb = float3(basisAt(c, in0 + 3u * k), basisAt(c, in0 + 3u * k + 1u),
                                      basisAt(c, in0 + 3u * k + 2u));
            gathered += envSh[sh + k].rgb * rgb;
            along += envBasisValue(k, sun) * rgb;
            if (k == 0u) {
                white = rgb;
            }
        }
        const uint w = out + j * 4u;
        compactFrameOut[w] = asuint(body.x);
        compactFrameOut[w + 1u] = asuint(body.y);
        compactFrameOut[w + 2u] = asuint(body.z);
        compactFrameOut[w + 36u] = asuint(gathered.x);
        compactFrameOut[w + 37u] = asuint(gathered.y);
        compactFrameOut[w + 38u] = asuint(gathered.z);
        compactFrameOut[w + 72u] = asuint(along.x);
        compactFrameOut[w + 73u] = asuint(along.y);
        compactFrameOut[w + 74u] = asuint(along.z);
        compactFrameOut[w + 108u] = asuint(white.x);
        compactFrameOut[w + 109u] = asuint(white.y);
        compactFrameOut[w + 110u] = asuint(white.z);
    }
}

float3 toView(float3 p) {
    return float3(dot(compactParams.view0, float4(p, 1.0)), dot(compactParams.view1, float4(p, 1.0)),
                  dot(compactParams.view2, float4(p, 1.0)));
}

/// Whether a view-space centre is drawn: ahead, within the planes, and
/// inside clipXY of the frame (generate's cull).
bool inFrustum(float3 v) {
    if (!(v.z > compactParams.near) || v.z >= compactParams.far) {
        return false;
    }
    const float2 px = float2(compactParams.focal.x * v.x / v.z + compactParams.focal.z,
                             compactParams.focal.y * v.y / v.z + compactParams.focal.w);
    const float2 mid = 0.5 * compactParams.viewport.xy;
    return all(abs(px - mid) <= compactParams.clipXY * mid);
}

uint gaussianIndex(uint3 tid) { return tid.x + tid.y * kCompactDispatchRow; }

/// A gaussian's sort key (24 bits, far first), or kCompactNone where it is
/// not drawn: transparent, outside the frustum, or a culled back.
uint cullKeyOf(uint i) {
    if (fieldAt(compactParams.opacity, 1u, i) == 0u) {
        return kCompactNone;
    }
    const float3 p = positionOf(i);
    const float3 v = toView(p);
    if (!inFrustum(v)) {
        return kCompactNone;
    }
    // relight.slang's culledBack: an opaque splat (no transmission, not a
    // thin sheet) whose stored face looks away from the eye is a closed
    // surface's back, hidden by its front.
    if (hasFlag(kCompactCullBacks)) {
        const uint pbr = materialOf(i).x;
        if (!(pbrTransmission(pbr) > 0.0) && !pbrThinWalled(pbr)) {
            const float3 toEye = compactParams.eyeObject.xyz - p;
            const float far = length(toEye);
            if (far > 0.0 && dot(normalOf(i), toEye / far) < compactParams.cullFacing) {
                return kCompactNone;
            }
        }
    }
    return uint((1u << kCompactKeyBits) - 1u - min(length(v) * compactParams.keyScale, 16777215.0));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaCompactCull(uint3 tid: SV_DispatchThreadID) {
    const uint t = gaussianIndex(tid);
    if (t >= compactParams.count) {
        return;
    }
    // With a draw list (a LoD cut chosen on the host), thread t culls its
    // t-th entry; else gaussian t.
    const uint i = hasFlag(kCompactDrawList) ? compactFrame[compactParams.drawList + t] : t;
    const bool stable = hasFlag(kCompactStableOrder);
    if (stable && t >= compactParams.capacity) {
        return;
    }
    const uint key = cullKeyOf(i);
    if (stable) {
        // The radix sort is stable: equal keys keep the list's order.
        sortKeys[t] = key == kCompactNone ? 0u : key;
        sortVals[t] = key == kCompactNone ? kCompactNone : i;
        return;
    }
    if (key == kCompactNone) {
        return;
    }
    const uint slot = sortArgs[1].add(1u);
    if (slot < compactParams.capacity) {
        sortKeys[slot] = key;
        sortVals[slot] = i;
    }
}

/// The drawn count held to the sort's slots (a view that draws more than
/// the slots sized for it drops the rest for this frame).
[shader("compute")]
[numthreads(1, 1, 1)]
void atheneaCompactClamp(uint3 tid: SV_DispatchThreadID) {
    sortArgs[1].min(compactParams.capacity);
}

/// Spark's projection of a 3D splat (splat_shape.slang projectSplat, its
/// default path: the anti-aliasing blur with its alpha kept, LoD opacity,
/// the faint path), from the view-space centre: false where not drawn.
/// sigma: the blurred covariance's two deviations, angle: the major axis'.
bool projectCompact(CompactSplat g, float3 v, uint index, inout float alpha, out float2 centre, out float2 sigma,
                    out float angle, out bool lod) {
    centre = float2(0.0);
    sigma = float2(0.0);
    angle = 0.0;
    lod = false;
    float a0 = alpha;
    if (a0 > 1.0) {
        a0 = min(a0 * 4.0 - 3.0, 5.0);
        lod = true;
    }
    const float3x3 r = splatRotation(g.quaternion);
    const float3x3 wv = float3x3(compactParams.view0.xyz, compactParams.view1.xyz, compactParams.view2.xyz);
    const float3x3 rv = mul(wv, r);
    // The covariance in view space: (R S)(R S)^T.
    const float3x3 m = float3x3(rv[0] * g.scales, rv[1] * g.scales, rv[2] * g.scales);
    const float fx = compactParams.focal.x;
    const float fy = compactParams.focal.y;
    const float invZ = 1.0 / v.z;
    centre = float2(fx * v.x * invZ + compactParams.focal.z, fy * v.y * invZ + compactParams.focal.w);
    const float3 jx = float3(fx * invZ, 0.0, -fx * v.x * invZ * invZ);
    const float3 jy = float3(0.0, fy * invZ, -fy * v.y * invZ * invZ);
    const float3 ax = mul(jx, m);
    const float3 ay = mul(jy, m);
    float a = dot(ax, ax);
    float d = dot(ay, ay);
    const float b = dot(ax, ay);
    const float blur = compactParams.blurAmount;
    const float detOrig = a * d - b * b;
    const float a00 = a;
    const float d00 = d;
    a += blur;
    d += blur;
    float det = a * d - b * b;
    const float blurred = a0 * sqrt(max(0.0, detOrig / det));
    const float minAlpha = compactParams.minAlpha;
    const float faintTarget = 4.0 * minAlpha;
    if (hasFlag(kCompactFaint) && a0 <= 1.0 && compactParams.falloff == 1.0 && detOrig > 0.0 &&
        blurred < faintTarget) {
        const float mass = a0 * sqrt(detOrig);
        const float tr = a00 + d00;
        const float want = (mass / faintTarget) * (mass / faintTarget);
        const float b2 = clamp(0.5 * (sqrt(max(tr * tr + 4.0 * (want - detOrig), 0.0)) - tr), 0.15, max(blur, 0.15));
        a = a00 + b2;
        d = d00 + b2;
        det = a * d - b * b;
        float fa = mass / sqrt(det);
        if (fa < faintTarget) {
            uint x = (index * 0x9e3779b9u) ^ 0x85ebca6bu;
            x ^= x >> 16u;
            x *= 0x7feb352du;
            x ^= x >> 15u;
            x *= 0x846ca68bu;
            x ^= x >> 16u;
            if (float(x >> 8u) / 16777216.0 >= fa / faintTarget) {
                return false;
            }
            fa = faintTarget;
        }
        alpha = fa + minAlpha;
        lod = false;
    } else {
        alpha = blurred;
    }
    if (alpha < minAlpha) {
        return false;
    }
    const float avg = 0.5 * (a + d);
    const float delta = sqrt(max(0.0, avg * avg - det));
    const float e1 = avg + delta;
    const float e2 = avg - delta;
    const float2 v1 = abs(b) > 0.001 ? normalize(float2(b, e1 - a)) : (a >= d ? float2(1.0, 0.0) : float2(0.0, 1.0));
    sigma = sqrt(max(float2(e1, e2), float2(0.0)));
    angle = atan2(v1.y, v1.x);
    if (angle < 0.0) {
        angle += 3.14159265358979;
    }
    return true;
}

/// The adjusted deviations a footprint's quad spans (projectSplat): LoD
/// opacity widens it, a faint one's edge comes in where splatAlpha falls
/// under minAlpha.
float adjustedStdDevOf(float alpha, bool lod, float2 sigma, float preBlurAlpha) {
    float adjusted = compactParams.maxStdDev;
    if (lod) {
        adjusted = compactParams.maxStdDev + 0.7 * (preBlurAlpha - 1.0);
    }
    const float f = compactParams.falloff;
    const float floorAlpha = (1.0 - f) * alpha;
    if (alpha <= 1.0 && f > 0.0 && floorAlpha < compactParams.minAlpha &&
        adjusted * sigma.x <= compactParams.maxPixelRadius) {
        const float cut = sqrt(max(0.0, 2.0 * log(f * alpha / (compactParams.minAlpha - floorAlpha))));
        adjusted = min(adjusted, cut);
    }
    return adjusted;
}

uint packCentre(float2 c) {
    const uint2 q = uint2(clamp(round((c + kCompactCentreOffset) * kCompactCentreScale), float2(0.0), float2(65535.0)));
    return q.x | (q.y << 16u);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaCompactShade(uint3 lid: SV_GroupThreadID, uint3 gid: SV_GroupID) {
    const uint count = min(sortCount[1], compactParams.capacity);
    for (uint pass = 0u; pass < kCompactRounds; ++pass) {
        uint slot = (gid.x * kCompactRounds + pass) * kCompactWorkgroup + lid.x;
        if (slot >= count) {
            return;
        }
        const uint t = slot;
        if (hasFlag(kCompactBinned)) {
            slot = binsIn[2u * kCompactBins + t];
        }
        const uint i = rec0[slot];
        if (i == kCompactNone) {
            // A culled entry of kCompactStableOrder: drawn as nothing.
            rec0[slot] = 0u;
            rec1[slot] = 0u;
            rec2[slot] = 0u;
            rec3[slot] = 0u;
            continue;
        }
        CompactSplat g = readCompact(i);
        const float3 v = toView(g.center);
        // The footprint first: a splat it drops (under minAlpha, a faint
        // one's draw) is not relit at all. Its alpha is the relight's
        // (drawAlphaOf), so the colour and footprint are as before.
        float alpha = drawAlphaOf(i, g);
        float2 centre, sigma;
        float angle;
        bool lod;
        const bool drawn = alpha > 0.0 && projectCompact(g, v, i, alpha, centre, sigma, angle, lod);
        if (!drawn || max(sigma.x, sigma.y) * compactParams.maxStdDev < compactParams.minPixelRadius) {
            rec0[slot] = 0u;
            rec1[slot] = 0u;
            rec2[slot] = 0u;
            rec3[slot] = 0u;
            continue;
        }
        const float4 lit = shadeColour(i, g);
        rec0[slot] = packCentre(centre);
        rec1[slot] = (f32tof16(sigma.x) & 0xFFFFu) | (uint(round(angle / 3.14159265358979 * 65535.0)) << 16u);
        rec2[slot] = (f32tof16(sigma.y) & 0xFFFFu) | ((f32tof16(lod ? -alpha : alpha) & 0xFFFFu) << 16u);
        rec3[slot] = packRgb9e5(max(lit.rgb, float3(0.0)));
    }
}

// The drawn slots by cluster (kCompactBinned): a counting sort of the
// sorted slots by their gaussian's cluster, so the shade's neighbouring
// threads read the same cluster's basis and projections (a warp of mixed
// clusters read hundreds of scattered words a thread). The order within a
// cluster does not matter: each slot is shaded into its own place.

uint drawnClusterOf(uint slot) {
    const uint i = rec0[slot];
    return i == kCompactNone ? 0u : fieldAt(compactParams.cluster, 1u, i);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaCompactBinCount(uint3 lid: SV_GroupThreadID, uint3 gid: SV_GroupID) {
    const uint count = min(sortCount[1], compactParams.capacity);
    for (uint pass = 0u; pass < kCompactRounds; ++pass) {
        const uint slot = (gid.x * kCompactRounds + pass) * kCompactWorkgroup + lid.x;
        if (slot >= count) {
            return;
        }
        bins[drawnClusterOf(slot)].add(1u);
    }
}

groupshared uint binScan[kCompactBins];

/// The clusters' starts (exclusive sums of the counts), the counts cleared
/// for the next frame.
[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaCompactBinScan(uint3 lid: SV_GroupThreadID) {
    const uint t = lid.x;
    const uint n = bins[t].load();
    binScan[t] = n;
    GroupMemoryBarrierWithGroupSync();
    for (uint step = 1u; step < kCompactBins; step <<= 1u) {
        const uint add = t >= step ? binScan[t - step] : 0u;
        GroupMemoryBarrierWithGroupSync();
        binScan[t] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    bins[kCompactBins + t].store(binScan[t] - n);
    bins[t].store(0u);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaCompactBinScatter(uint3 lid: SV_GroupThreadID, uint3 gid: SV_GroupID) {
    const uint count = min(sortCount[1], compactParams.capacity);
    for (uint pass = 0u; pass < kCompactRounds; ++pass) {
        const uint slot = (gid.x * kCompactRounds + pass) * kCompactWorkgroup + lid.x;
        if (slot >= count) {
            return;
        }
        const uint at = bins[kCompactBins + drawnClusterOf(slot)].add(1u);
        bins[2u * kCompactBins + at].store(slot);
    }
}

/// Every gaussian's relit colour and alpha (alpha as the draw starts from:
/// its own where the relight keeps it), for the accuracy checks.
[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaCompactColours(uint3 tid: SV_DispatchThreadID) {
    const uint i = gaussianIndex(tid);
    if (i >= compactParams.count) {
        return;
    }
    const CompactSplat g = readCompact(i);
    const float4 lit = shadeColour(i, g);
    debugOut[3u * i] = lit;
    // The footprint as the draw takes it: its two axes' lengths in pixels,
    // the alpha, the adjusted deviations (0s: not drawn).
    float alpha = lit.w >= 0.0 ? lit.w : g.rgba.a;
    const float3 v = toView(g.center);
    float2 centre, sigma;
    float angle;
    bool lod;
    float4 shape = float4(0.0);
    float4 place = float4(0.0);
    if (inFrustum(v) && alpha > 0.0) {
        float pre = alpha > 1.0 ? min(alpha * 4.0 - 3.0, 5.0) : alpha;
        if (projectCompact(g, v, i, alpha, centre, sigma, angle, lod)) {
            const float adjusted = adjustedStdDevOf(alpha, lod, sigma, pre);
            shape = float4(min(compactParams.maxPixelRadius, adjusted * sigma.x),
                           min(compactParams.maxPixelRadius, adjusted * sigma.y), alpha, adjusted);
            place = float4(centre, angle, 0.0);
        }
    }
    debugOut[3u * i + 1u] = shape;
    debugOut[3u * i + 2u] = place;
}

// ------------------------------------------------------------- the draw

struct CompactVertexOut {
    float4 position : SV_Position;
    nointerpolation float4 rgba : COLOR0;
    float2 uv : TEXCOORD0;
    nointerpolation float adjusted : TEXCOORD1;
};

static const float2 kCorners[4] = { float2(-1.0, -1.0), float2(1.0, -1.0), float2(-1.0, 1.0), float2(1.0, 1.0) };

[shader("vertex")]
CompactVertexOut atheneaCompactVertex(uint vertexId: SV_VertexID, uint instanceId: SV_InstanceID) {
    CompactVertexOut o;
    o.position = float4(0.0, 0.0, 2.0, 1.0);
    o.rgba = float4(0.0);
    o.uv = float2(0.0);
    o.adjusted = 0.0;
    const uint w1 = draw1[instanceId];
    const uint w2 = draw2[instanceId];
    if (w1 == 0u && w2 == 0u) {
        return o;
    }
    const uint w0 = draw0[instanceId];
    const float2 centre = float2(float(w0 & 0xFFFFu), float(w0 >> 16u)) / kCompactCentreScale - kCompactCentreOffset;
    const float2 sigma = float2(f16tof32(w1 & 0xFFFFu), f16tof32(w2 & 0xFFFFu));
    const float angle = float(w1 >> 16u) / 65535.0 * 3.14159265358979;
    const float marked = f16tof32(w2 >> 16u);
    const bool lod = marked < 0.0;
    const float alpha = abs(marked);
    // LoD opacity's own width: the opacity before the blur took its share.
    float pre = alpha;
    if (lod) {
        const float blur = compactParams.blurAmount;
        const float e1 = sigma.x * sigma.x;
        const float e2 = sigma.y * sigma.y;
        const float detOrig = max((e1 - blur) * (e2 - blur), 0.0);
        pre = alpha / max(sqrt(detOrig / max(e1 * e2, 1.0e-30)), 1.0e-30);
    }
    const float adjusted = adjustedStdDevOf(alpha, lod, sigma, pre);
    const float2 v1 = float2(cos(angle), sin(angle));
    const float2 v2 = float2(v1.y, -v1.x);
    const float s1 = min(compactParams.maxPixelRadius, adjusted * sigma.x);
    const float s2 = min(compactParams.maxPixelRadius, adjusted * sigma.y);
    const float2 corner = kCorners[vertexId & 3u];
    const float2 px = centre + corner.x * v1 * s1 + corner.y * v2 * s2;
    const float2 ndc = float2(px.x / compactParams.viewport.x * 2.0 - 1.0, 1.0 - px.y / compactParams.viewport.y * 2.0);
    o.position = float4(ndc, 0.5, 1.0);
    o.rgba = float4(unpackRgb9e5(draw3[instanceId]), alpha);
    o.uv = corner * adjusted;
    o.adjusted = adjusted;
    return o;
}

float splatAlphaOf(float a, float z2) {
    if (a <= 1.0) {
        return lerp(a, a * exp(-0.5 * z2), compactParams.falloff);
    }
    const float e = exp((a * a - 1.0) / 2.718281828459045);
    const float alpha = 1.0 - pow(1.0 - exp(-0.5 * z2), e);
    return lerp(1.0, alpha, compactParams.falloff);
}

[shader("fragment")]
float4 atheneaCompactFragment(CompactVertexOut v) : SV_Target {
    const float z2 = dot(v.uv, v.uv);
    if (z2 > v.adjusted * v.adjusted) {
        discard;
    }
    const float a = splatAlphaOf(v.rgba.a, z2);
    if (a < compactParams.minAlpha) {
        discard;
    }
    return float4(v.rgba.rgb * a, a);
}

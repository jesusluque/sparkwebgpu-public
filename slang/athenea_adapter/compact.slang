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
//   cluster            u8    athenea's joint CPCA cluster, its low 8 bits (C <= 512:
//                            the 9th is the shadow index's bit 15, thread CX)
//   weights            3 planes of 8 int8 (direct, indirect, reflected), or of
//                            8 int4 (weightBits 4, 097's mode 1)
//   shadow             u16   into its piece's codebook (the open-direction
//                            cells, 8 words; a piece is 16384 gaussians), bits
//                            0-14 (a codebook of at most 32768); bit 15 the
//                            cluster's 9th
// and tables: the chunks' ranges (12 words a chunk), the pieces' codebook
// offsets, the codebooks, the materials, and athenea's basis in f16 (504
// words a cluster: per block its mean and its 8 directions times scale/127).
//
// PAGED (slotNodes > 0; thread CN, src/webgpu/athenea/compactPages.ts): the
// gaussians are athenea's error-band pages, each in a pool slot of slotNodes
// (3072) at slotsBase + slot x slotWords: the slot's head (its page's box and
// ranges, kCompactChunkWords; word 9 its piece's codebook, word 10 its
// materials, in words; word 11 the faint dither's seed) then the planes at the plane uniforms' offsets within
// the slot, slotNodes elements each. Gaussian i is slot i / slotNodes,
// element i % slotNodes; the codebooks, materials and the basis are global
// to a piece (the pool's tables, always resident).
//
// NODE POOL (nodeWords > 0; thread CT, compactPages.ts CompactNodePool): only
// the drawn cut is on the GPU, a record of nodeWords words a node at
// slotsBase + i x nodeWords (the GPU's 40 bytes, the plane uniforms their
// byte offsets in it, then the origin word: the node's page in the head
// table, bits 0-15, its place in the page, 16-31). Every page's head is in
// the head table (kCompactHeadWords a page, the slot head's words).
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
import athenea.material.microfacet;
import athenea.scene.splat_encoding;
import core.bindings;

public static const uint kCompactChunk = 8192;
public static const uint kCompactChunkWords = 12;
public static const uint kCompactHeadWords = 16;      // a page's head in the node pool's head table
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
public static const uint kCompactBins = 2048u;     // clusters a cloud may have: its bases' together (thread CX; 2048 for v82b, thread DA)
public static const uint kCompactBinsPerThread = 8u; // the bin passes' 256 threads, 8 bins each (binLocal + binBase: 16 KiB of workgroup storage)
public static const uint kCompactStableOrder = 256u; // each listed gaussian its own slot (culled: ~0), for repeatable images
// athenea 35ae3cb, as relight.slang's kRelightCoatFold and kRelightPolishField:
// a coat's bake widening held to kCoatFoldRoughness (0.3) where the surface
// turns fast under the gaussian; the material's own roughness (pbr bits
// 26..31) read, and a polish at least kPolishFieldRoughness rough has the
// reflected field answer its closed part. (The compact path has no sharp
// per-pixel lobes, so athenea's unwidened lobes, b26d276, do not apply.)
public static const uint kCompactCoatFold = 512u;
public static const uint kCompactPolishField = 1024u;
// THE SHARP LOBES PER PIXEL (thread CW; relight.slang's kRelightSlope with
// the sharp coat and polish, athenea's SplatSlope / splat_blend kSharpMark):
// a coat under kSharpRoughness and a polish under kSharpPolishRoughness are
// not answered once at the centre. The shade takes the centre's reading of
// the sky out of the colour and hands the draw the lobe's weight, its
// roughness unwidened (athenea's `unwidened`, the bake's Toksvig spread taken
// back), and the mirror at the centre and its step a pixel right and down
// (from the curvature); the fragment reads the sky again along the mirror
// its pixel turns to. Five words a drawn slot (packSharp), after the bins (2 x
// kCompactBins + capacity words) in the shade's aux buffer (shadeAux: the
// shade's tenth storage buffer, a phone's limit, shared with the bins).
public static const uint kCompactSharp = 2048u;
// The env's texels read from the atlas (thread CU): with kCompactSharp the
// atlas holds every level's texels as the same f16 values, so the shade reads
// them there (compactEnvRadiance) and the env buffer is freed once it exists.
public static const uint kCompactEnvAtlas = 4096u;
// THE LOAD PIXELATE (thread DC): a draw list entry's top byte (the node
// pool's slots are under 2^24) marks a splat drawn coarse because the cut
// wanted it finer and its children are not here yet (pending), or one that
// just replaced such a splat (fading: its arrival's 6-bit stamp,
// kCompactLoadTicks a second, over kCompactLoadFadeTicks). The shade
// carries them into the record: pending in sigma2's sign bit, fading in
// sigma1's with the stamp in the angle's low 6 bits (a fading splat's angle
// to 10 bits, for half a second). The fragment of a marked splat evaluates
// its gaussian at its screen block's centre, not at the pixel: where every
// splat is marked the picture is constant a block (pixelated), and the
// block shrinks to a pixel as the splat's weight fades.
public static const uint kCompactLoadFx = 8192u;
public static const uint kCompactLoadIndex = 0x00FFFFFFu;
public static const uint kCompactLoadPending = 0x80000000u;
public static const uint kCompactLoadFading = 0x40000000u;
public static const uint kCompactLoadTicks = 32u;
public static const uint kCompactLoadFadeTicks = 16u;
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
    public float sheetFade;    ///< a thin sheet's floor after the AA blur, faded under A0 px^2 (0: unfaded); < 0: floored before the blur (CN's)
    public uint  drawList;     ///< kCompactDrawList: the list's start in compactFrame, words (count entries)
    public uint  nodeWords;    ///< node pool (thread CT): words a node record (0: planes); the plane uniforms are then bytes in the record
    public uint  headTable;    ///< node pool: the pages' heads, words (kCompactHeadWords a page)
    public float sharpMinPx;   ///< kCompactSharp: the smallest footprint (major deviation, px) shaded per pixel
    public float sharpFlatTexels; ///< kCompactSharp: a lobe whose mirror turns less than this many sky texels over the footprint is read once
    public float reflectMinPx; ///< a footprint under it (major deviation, px) reads its reflected field band-limited
    public uint  reflectSubTerms; ///< ...to this many of its 16 coefficients (1: the mean, 4: degree 1, 9: degree 2)
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
    public float4 viewport;    ///< width, height; checks: z coatFloor, w debugBits
    public float4 loadFx;      ///< kCompactLoadFx: the largest block (px), its scale (0..1), -, now (ticks, mod 64)
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
RWStructuredBuffer<uint>            shadeAux;  ///< the bins, read by the shade (kCompactBinned); then the sharp records it writes (kCompactSharp)
StructuredBuffer<uint>              sharpIn;   ///< the same buffer, read by the draw (kCompactSharp)
/// kCompactSharp: the dome's prefiltered levels (envTexels) as one filtered
/// rgba16float atlas, so the draw reads a lobe with the hardware's trilinear
/// in two samples instead of decoding eight texels (atheneaCompactEnvAtlas
/// fills it when the dome changes): level 0 at the origin, the others down a
/// column to its right.
[format("rgba16f")] WTexture2D<float4> envAtlasOut;
[Filterable] Texture2D<float4>         envAtlas;
SamplerState                           envAtlasSampler;

/// Where slot `slot`'s sharp records are in shadeAux / sharpIn (words).
uint sharpAt(uint slot) { return 2u * kCompactBins + compactParams.capacity + 5u * slot; }
void writeSharp(uint slot, uint4 a, uint4 b) {
    const uint at = sharpAt(slot);
    shadeAux[at] = a.x;
    shadeAux[at + 1u] = a.y;
    shadeAux[at + 2u] = a.z;
    shadeAux[at + 3u] = a.w;
    shadeAux[at + 4u] = b.x;
}
// Five words a slot (thread CU, athenea's pick: the same image to 60 dB):
// the mirror at the centre unorm16 x2, its steps a pixel right and down as
// f16 pairs, coat weight u10 | coat roughness u11 | polish roughness u11
// (the roughnesses as their square roots, any up to 1: a fold widens them,
// athenea f9c9a2b), and in the fifth the polish weights in RGB9E5 (b.x).
uint4 packSharp(float2 at, float2 sx, float2 sy, float coatW, float coatR, float3 polishW, float polishR, out uint polish) {
    const uint2 a = uint2(round(saturate(at) * 65535.0));
    const uint w = uint(round(saturate(coatW) * 1023.0)) | (uint(round(sqrt(saturate(coatR)) * 2047.0)) << 10u) |
                   (uint(round(sqrt(saturate(polishR)) * 2047.0)) << 21u);
    polish = packRgb9e5(max(polishW, float3(0.0)));
    return uint4(a.x | (a.y << 16u), packHalves(sx.x, sx.y), packHalves(sy.x, sy.y), w);
}
ConstantBuffer<CompactParams>       compactParams;

bool hasFlag(uint f) { return (compactParams.flags & f) != 0u; }

float2 unpackHalves(uint w) { return float2(lowHalf(w), highHalf(w)); }

/// Where level `level` starts in the atlas (texels).
uint2 atlasOrigin(uint level, uint baseSide) {
    if (level == 0u) {
        return uint2(0u);
    }
    uint y = 0u;
    for (uint l = 1u; l < level; ++l) {
        y += envSideOf(l, baseSide);
    }
    return uint2(envSideOf(0u, baseSide), y);
}

/// The atlas' size: level 0 and the column of the others.
uint2 atlasSize(uint baseSide) {
    uint h = 0u;
    for (uint l = 1u; l < kEnvLevels; ++l) {
        h += envSideOf(l, baseSide);
    }
    return uint2(envSideOf(0u, baseSide) + envSideOf(1u, baseSide), max(envSideOf(0u, baseSide), h));
}

/// One level of the atlas along `dir`: the hardware's bilinear over the same
/// texels environment.slang's envLevelAt reads (clamped inside the level).
float3 atlasLevel(uint level, float3 dir, uint baseSide) {
    const float side = float(envSideOf(level, baseSide));
    const float2 at = clamp(octEncode(dir) * side - 0.5, 0.0, side - 1.0);
    const float2 uv = (float2(atlasOrigin(level, baseSide)) + at + 0.5) / float2(atlasSize(baseSide));
    return envAtlas.SampleLevel(envAtlasSampler, uv, 0.0).rgb;
}

/// environment.slang's envRadiance from the atlas: the two levels bracketing
/// the roughness (its square root along the chain), blended.
float3 atlasRadiance(float3 dir, float roughness, uint baseSide) {
    const float level = sqrt(saturate(roughness)) * float(kEnvLevels - 1);
    const uint lo = uint(floor(level));
    const uint hi = min(lo + 1u, kEnvLevels - 1u);
    return lerp(atlasLevel(lo, dir, baseSide), atlasLevel(hi, dir, baseSide), level - float(lo));
}

/// environment.slang's envLevelAt and envRadiance with their texels read
/// from the atlas (kCompactEnvAtlas; the same values and arithmetic).
float3 atlasTexelAt(uint level, uint2 uv, uint baseSide) {
    return envAtlas.Load(int3(int2(atlasOrigin(level, baseSide) + uv), 0)).rgb;
}
float3 atlasLevelAt(uint level, float3 dir, uint baseSide) {
    const float side = float(envSideOf(level, baseSide));
    const float2 at = clamp(octEncode(dir) * side - 0.5, 0.0, side - 1.0);
    const uint2 lo = uint2(floor(at));
    const uint2 hi = min(lo + 1, uint2(uint(side) - 1, uint(side) - 1));
    const float2 f = at - float2(lo);
    const float3 v00 = atlasTexelAt(level, lo, baseSide);
    const float3 v10 = atlasTexelAt(level, uint2(hi.x, lo.y), baseSide);
    const float3 v01 = atlasTexelAt(level, uint2(lo.x, hi.y), baseSide);
    const float3 v11 = atlasTexelAt(level, hi, baseSide);
    return lerp(lerp(v00, v10, f.x), lerp(v01, v11, f.x), f.y);
}
float3 compactEnvRadiance(StructuredBuffer<uint> texels, uint dome, float3 dir, float roughness,
                          uint baseSide) {
    if (!hasFlag(kCompactEnvAtlas) || dome != 0u) {
        return envRadiance(texels, dome, dir, roughness, baseSide);
    }
    const float level = sqrt(saturate(roughness)) * float(kEnvLevels - 1);
    const uint lo = uint(floor(level));
    const uint hi = min(lo + 1, kEnvLevels - 1);
    const float f = level - float(lo);
    return lerp(atlasLevelAt(lo, dir, baseSide), atlasLevelAt(hi, dir, baseSide), f);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void atheneaCompactEnvAtlas(uint3 tid: SV_DispatchThreadID) {
    const uint baseSide = compactParams.envBaseSide;
    const uint2 size = atlasSize(baseSide);
    if (any(tid.xy >= size)) {
        return;
    }
    float4 texel = float4(0.0, 0.0, 0.0, 1.0);
    for (uint l = 0u; l < kEnvLevels; ++l) {
        const uint2 o = atlasOrigin(l, baseSide);
        const uint side = envSideOf(l, baseSide);
        if (all(tid.xy >= o) && all(tid.xy < o + side)) {
            texel = float4(envTexelAt(envTexels, 0u, l, tid.xy - o, baseSide), 1.0);
            break;
        }
    }
    envAtlasOut.Store(tid.xy, texel);
}



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

/// Node pool (nodeWords > 0): node i's record, words.
uint recordOf(uint i) {
    return compactParams.slotsBase + i * compactParams.nodeWords;
}

/// Node pool: node i's origin (its page in the head table, bits 0-15; its
/// place in the page, 16-31), the record's last word.
uint originOf(uint i) {
    return compactData[recordOf(i) + compactParams.nodeWords - 1u];
}

/// `width` bytes (1..4) of element i of the plane at word `plane` (node
/// pool: at byte `plane` of node i's record).
/// Gaussian i's CPCA cluster: its cluster byte, the 9th bit from the shadow
/// index's bit 15 (thread CX: 512 clusters with the record's size unchanged;
/// a codebook of at most 32768 entries, so files of 256 clusters read as before).
uint clusterAt(uint i) {
    const uint own = fieldAt(compactParams.cluster, 1u, i) | ((fieldAt(compactParams.shadow, 2u, i) >> 15u) << 8u);
    // Paged (thread CX): its piece's basis's first cluster among the pool's
    // bases, x 256, in the top byte of its head's word 10 (0 with one basis).
    if (compactParams.slotNodes == 0u && compactParams.nodeWords == 0u) {
        return own;
    }
    return own + (chunkWord(i, 10u) >> 24u) * 256u;
}

uint fieldAt(uint plane, uint width, uint i) {
    uint base;
    uint at;
    if (compactParams.nodeWords != 0u) {
        base = recordOf(i);
        at = plane;
    } else {
        uint e;
        base = planeOf(plane, i, e);
        at = e * width;
    }
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
    if (compactParams.nodeWords != 0u) {
        return compactData[compactParams.headTable + (originOf(i) & 0xFFFFu) * kCompactHeadWords + k];
    }
    const uint nodes = compactParams.slotNodes;
    if (nodes == 0u) {
        return compactData[compactParams.chunkTable + (i / kCompactChunk) * kCompactChunkWords + k];
    }
    return compactData[compactParams.slotsBase + (i / nodes) * compactParams.slotWords + k];
}

/// A gaussian's id for the faint splats' dither: its index, or, paged, its
/// node's place in its piece (slot head word 11, the page's base) -- the
/// same whatever slot the page is in.
uint stableIndexOf(uint i) {
    if (compactParams.nodeWords != 0u) {
        return chunkWord(i, 11u) + (originOf(i) >> 16u);
    }
    const uint nodes = compactParams.slotNodes;
    return nodes == 0u ? i : chunkWord(i, 11u) + (i % nodes);
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
    const uint table = compactParams.slotNodes == 0u && compactParams.nodeWords == 0u ? compactParams.materials
                                                                                      : chunkWord(i, 10u) & 0xFFFFFFu;
    const uint at = table + fieldAt(compactParams.material, 2u, i) * kCompactMaterialWords;
    return uint4(compactData[at], compactData[at + 1u], compactData[at + 2u], compactData[at + 3u]);
}

/// Where splat i's open-direction cells are in `compact` (words).
uint cellBaseOf(uint i) {
    const uint book = compactParams.slotNodes == 0u && compactParams.nodeWords == 0u
                          ? compactData[compactParams.pieceTable + i / kCompactPiece]
                          : chunkWord(i, 9u);
    return book + (fieldAt(compactParams.shadow, 2u, i) & 0x7fffu) * 8u;
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
    uint plane;
    if (compactParams.nodeWords != 0u) {
        // Node pool: the block's word(s) at byte weights + b x weightsStride.
        plane = recordOf(i) + (compactParams.weights + b * compactParams.weightsStride) / 4u;
        e = 0u;
    } else {
        plane = planeOf(compactParams.weights + b * compactParams.weightsStride, i, e);
    }
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
                    out float3 polish, out float3 coat, out float3 along, uint terms = 16u) {
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
        // Band-limited (a footprint under reflectMinPx): the first `terms`.
        if (2u * jp >= terms) {
            break;
        }
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
            if (j >= terms) {
                break;
            }
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
    const float3 seen = s.ior > 1.0 ? compactEnvRadiance(envTexels, dome, bentThrough(s, wo), s.roughness,
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
/// kCompactSharp: what relitCompact hands the draw (see kCompactSharp).
struct CompactSharp {
    bool want;          ///< in: the slope is known (curved, projected)
    float coatWeight;   ///< out: the coat's weight (luma, narrowed), 0 none
    float3 coatSeen;
    float3 polishWeight;
    float3 polishSeen;
};

float3 relitCompact(SplatSurface s, float3 wo, uint dome, bool withIndirect, float3 front, float3 coupling,
                    float sunShare, float3 fieldPolish, float3 fieldCoat, float3 fieldThrough, float openness,
                    float lensCurvature, inout CompactSharp sharp) {
    const uint side = compactParams.envBaseSide;
    const float nDotV = max(dot(s.normal, wo), 1.0e-4);
    const float3 mirror = reflect(-wo, s.normal);
    const SplatLayers layers = splatLayers(s, nDotV);
    const bool coated = s.lobes.coatWeight > 0.0;
    const float4 disc = sunDisc();
    const float3 irradiance = disc.w > 0.0 ? sunIrradiance() : float3(0.0);
    const bool cells = s.cellWords >= 8u;
    const bool hasField = cells && withIndirect;
    // What a solid glass lets through (the body's transmitted share): read
    // only where it is (transmittedWeightOf > 0), as it weighs nothing else.
    const bool transmits = transmittedWeightOf(s) > 0.0;
    float3 behind = transmits ? transmittedBehindOf(s, wo, dome) : float3(0.0);
    if (hasField && transmits && s.ior > 1.0) {
        const float3 white = fieldThrough;
        // athenea 798184d: the way through reads open against what the two
        // faces' Fresnel and the tint let through at all.
        const float3 lumW = float3(0.2126, 0.7152, 0.0722);
        const float fresnelIn = fresnelDielectric(saturate(nDotV), max(s.ior, 1.0));
        const float3 tintIn = lerp(float3(1.0), saturate(s.albedo), s.transmission);
        const float expected = max(dot((1.0 - fresnelIn) * (1.0 - fresnelIn) * tintIn * tintIn, lumW), 1.0e-3);
        const float open = smoothstep(0.5, 0.85, saturate(dot(white, lumW) / expected));
        const float3 sharp =
            lensCurvature > 0.0
                ? compactEnvRadiance(envTexels, dome, lensExit(s, wo, lensCurvature),
                              transmittedRoughness(s.roughness, s.ior), side) *
                      lensExitThrough(s, wo, lensCurvature)
                : transmittedBehindOf(s, wo, dome);
        behind = open * sharp + (1.0 - open) * coupling * white * s.exitThrough;
    } else if (hasField && transmits) {
        const float3 through = -wo;
        const float alpha = max(s.roughness * s.roughness, 1.0e-3);
        const float openBehind = splatConeOpen(s, compactData, through, min(1.4 * atan(alpha), 1.2), -s.normal);
        const float open = openBehind >= 0.0 ? openBehind : 0.0;
        behind = (open * compactEnvRadiance(envTexels, dome, through, s.roughness, side) +
                  (1.0 - open) * coupling * fieldThrough) *
                 s.exitThrough;
    }
    const float3 passes = transmits ? transmittedShareOf(s, nDotV) : float3(0.0);
    const float3 seen = compactEnvRadiance(envTexels, dome, mirror, s.roughness, side);
    // athenea a112076: a solid's far face reflects back out through the near
    // one (farFaceReflected), the light it sends read through the lens's third
    // point where the cloud keeps its curvature (lensFarReflection).
    float3 farBack = float3(0.0);
    if (transmits) {
        float3 farSeen = seen;
        if (lensCurvature > 0.0) {
            const float3 farOut = lensFarReflection(s, wo, lensCurvature);
            if (dot(farOut, farOut) > 0.5) {
                farSeen = compactEnvRadiance(envTexels, dome, farOut, transmittedRoughness(s.roughness, s.ior), side);
            }
        }
        farBack = farFaceReflected(s, nDotV, farSeen);
    }
    const float3 body2 = layers.under * s.albedo * (1.0 - s.metallic) *
                         ((1.0 - s.transmission) * front + transmittedWeightOf(s) * (passes * behind + farBack));
    const float openAll = saturate(openness / 0.28209479177387814);
    // athenea 35ae3cb: a polish as wide as the field asks its own lobe's
    // cells, and the field answers what they close.
    const bool polishField = hasField && s.materialRoughness >= kPolishFieldRoughness;
    const float narrowed = polishField ? splatLobeOpen(s, compactData, mirror, s.roughness, s.normal)
                                       : specularOpenness(nDotV, openAll, s.roughness);
    const float3 sunPolish = !(disc.w > 0.0) ? float3(0.0)
                                             : splatPolish(s, nDotV) * irradiance *
                                                   (sunShare * sunReflectPdf(mirror, disc.xyz, s.roughness, disc.w));
    if (sharp.want && dome == 0u && s.roughness < kSharpPolishRoughness &&
        (uint(compactParams.viewport.w) & 2u) == 0u) {
        const float3 polishWeight = layers.under * thinWallGainOf(s) * splatPolish(s, nDotV) * narrowed;
        if (any(polishWeight > float3(0.0))) {
            sharp.polishWeight = polishWeight;
            sharp.polishSeen = seen;
        }
    }
    float3 coat = float3(0.0);
    if (coated) {
        const float3 coatSeen = compactEnvRadiance(envTexels, dome, mirror, s.lobes.coatRoughness, side);
        const float coatNarrowed = cells ? splatLobeOpen(s, compactData, mirror, s.lobes.coatRoughness, s.normal)
                                         : specularOpenness(nDotV, openAll, s.lobes.coatRoughness);
        if (sharp.want && dome == 0u && s.lobes.coatRoughness < kSharpRoughness) {
            sharp.coatWeight = dot(layers.coat, float3(0.2126, 0.7152, 0.0722)) * coatNarrowed;
            sharp.coatSeen = coatSeen;
        }
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
                   (max(splatPolish(s, nDotV) *
                            (narrowed * seen + (polishField ? (1.0 - narrowed) * coupling * fieldPolish : float3(0.0))),
                        float3(0.0)) +
                    sunPolish) +
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
    /// kCompactSharp: the curvature and the roughnesses as stored (before
    /// the footprint prefilter), and the bake's spread.
    bool curved;
    float3 shapeUV;
    float storedCoat;
    float storedPolish;
    float spread;
    uint cluster;
    uint index;
};

/// The material's own roughness (athenea's pbrMaterialRoughness on the
/// material table's pbr word), -1 where the asset does not carry it.
float compactMaterialRoughness(uint i) { return pbrMaterialRoughness(materialOf(i).x); }

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
    surface.materialRoughness = hasFlag(kCompactPolishField) ? compactMaterialRoughness(i) : -1.0;
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
    o.curved = false;
    o.shapeUV = float3(0.0);
    o.storedCoat = surface.lobes.coatRoughness;
    o.storedPolish = surface.roughness;
    o.spread = 0.0;
    if (hasFlag(kCompactCurvature)) {
        const float3 shapeUV = curvatureOf(i);
        o.curved = true;
        o.shapeUV = shapeUV;
        {
            const float a = shapeUV.x, b = shapeUV.y, c = shapeUV.z;
            const float kMax = abs(0.5 * (a + c)) + sqrt(0.25 * (a - c) * (a - c) + b * b);
            const float sp = kMax * max(g.scales.x, g.scales.y);
            o.spread = isfinite(sp) ? sp : 0.0;
        }
        const float3 axisN = float3(r[0][2], r[1][2], r[2][2]);
        const float faces = dot(n, axisN) >= 0.0 ? 1.0 : -1.0;
        o.lensCurvature = faces * 0.5 * (shapeUV.x + shapeUV.z);
        // With the sharp lobes per pixel (kCompactSharp) the mirror turns
        // per pixel: no splat footprint term (relight.slang with kRelightSlope).
        if (compactParams.footprint > 0.0 && !hasFlag(kCompactSharp)) {
            surface.roughness = footprintOf(surface.roughness, shapeUV, g.scales, compactParams.footprint,
                                            compactParams.footprintClamp);
            surface.lobes.coatRoughness = footprintOf(surface.lobes.coatRoughness, shapeUV, g.scales,
                                                      compactParams.footprint, compactParams.footprintClamp);
        }
        // A fold's coat (relight.slang's kRelightCoatFold, after the footprint).
        // Only on a paint-rough lacquer (relight.slang's kRelightCoatFold).
        if (hasFlag(kCompactCoatFold) && surface.lobes.coatWeight > 0.0 &&
            compactMaterialRoughness(i) >= kPolishFieldRoughness) {
            const float a = shapeUV.x, b = shapeUV.y, c = shapeUV.z;
            const float kMax = abs(0.5 * (a + c)) + sqrt(0.25 * (a - c) * (a - c) + b * b);
            const float spread = kMax * max(g.scales.x, g.scales.y);
            const float coat = surface.lobes.coatRoughness;
            if (isfinite(spread)) {
                surface.lobes.coatRoughness = max(unwidened(coat, spread), min(coat, kCoatFoldRoughness));
            }
        }
    }
    o.surface = surface;
    // Check only (thread CW, CompactPathOptions.coatFloor): the roughnesses
    // held at least to viewport.z.
    if (compactParams.viewport.z > 0.0) {
        o.surface.roughness = max(o.surface.roughness, compactParams.viewport.z);
        o.surface.lobes.coatRoughness = max(o.surface.lobes.coatRoughness, compactParams.viewport.z);
    }
    o.cluster = clusterAt(i);
    o.index = i;
    return o;
}

/// athenea's kSheetAlpha (splat_project): a thin sheet's floor.
static const float kCompactSheetAlpha = 0.1;

/// A thin sheet floored AFTER the AA blur, as the normal path's draw
/// (splat_shape.slang sheetFloor, pieces.html's sheet=draw): CN's compact
/// path floored it before the blur, so a sheet under a pixel or two (the
/// headlight cover from a few metres) lost the blur's share of its floor,
/// down to minAlpha and gone (thread CS).
bool sheetAfterBlur() { return compactParams.sheetFade >= 0.0; }

/// splat_shape.slang sheetFadeOf, the gaussian AA blur only.
float compactSheetFade(float major2, float minor2, float blur) {
    const float a0 = compactParams.sheetFade;
    if (!(a0 > 0.0) || !(blur > 0.0)) {
        return 1.0;
    }
    const float ref2 = a0 / (2.0 * 3.14159265358979);
    const float e2 = sqrt(max(major2 * minor2, 0.0));
    if (!(e2 < ref2)) {
        return 1.0;
    }
    return min(1.0, e2 * (ref2 + blur) / (ref2 * (e2 + blur)));
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
        const float own = max(min(g.rgba.a, 1.0) - 1.0 / 255.0, 0.0);
        // After the blur (sheetAfterBlur): its own alpha, floored by
        // projectCompact; CN's: floored here, before the blur took its share.
        return sheetAfterBlur() ? own : max(own, kCompactSheetAlpha);
    }
    if (!thin && transmission > 0.5 && compactParams.glassCover > 1.0) {
        return coveredGlass(g.rgba.a, compactParams.glassCover);
    }
    return g.rgba.a;
}

/// Whether splat i is a thin sheet of glass (athenea's splat_project floor).
bool isSheet(uint i) {
    if (hasFlag(kCompactPlain)) {
        return false;
    }
    const uint pbr = materialOf(i).x;
    return pbrThinWalled(pbr) && pbrTransmission(pbr) > 0.5;
}

/// The eye's direction from a surface.
float3 eyeFrom(SplatSurface s) { return normalize(compactParams.eyeWorld.xyz - s.position); }

/// The reflected field along the mirror (polish, coat) and, for a
/// transmitting splat, along the way through: what relitCompact reads.
void reflectedTerms(CompactSurface c, uint2 wr, out float3 polish, out float3 coat, out float3 through,
                    uint terms = 16u) {
    const SplatSurface s = c.surface;
    // eyeField.w (checks): 1 every splat reads eyeField; 2 or 3 the splats
    // of the other parity than (w - 2) read it, the rest the eye.
    const float mode = compactParams.eyeField.w;
    const bool stale = mode < 1.5 || float(c.index & 1u) != mode - 2.0;
    const float3 wo = normalize((stale ? compactParams.eyeField.xyz : compactParams.eyeWorld.xyz) - s.position);
    const float3 mirror = reflect(-wo, s.normal);
    const bool coated = s.lobes.coatWeight > 0.0;
    reflectedAlong(c.cluster, wr, mirror, s.roughness, coated ? s.lobes.coatRoughness : s.roughness,
                   transmittedWeightOf(s) > 0.0, -wo, polish, coat, through, terms);
}

/// Splat i's relit colour (linear) and alpha (-1: its own), as relight.slang's
/// relitColour<2, 2> (the joint CPCA under one dome, no slope).
public float4 shadeColour(uint i, CompactSplat g) {
    uint4 a, b;
    return shadeColour(i, g, a, b);
}

/// kCompactSharp: how the normal turns a pixel right and down (relight.slang
/// projectForSlope + normalTurnPerPixel, in the cloud's space), from the
/// curvature; false where unknown or edge-on.
bool compactTurnPerPixel(CompactSplat g, float3x3 r, float3 shapeUV, float3 cloudNormal, out float3 dx,
                         out float3 dy) {
    dx = float3(0.0);
    dy = float3(0.0);
    const float3x3 wv = float3x3(compactParams.view0.xyz, compactParams.view1.xyz, compactParams.view2.xyz);
    const float3 t = toView(g.center);
    if (!(t.z > 1.0e-6)) {
        return false;
    }
    const float fx = compactParams.focal.x;
    const float fy = compactParams.focal.y;
    const float cx = compactParams.focal.z;
    const float cy = compactParams.focal.w;
    const float width = compactParams.viewport.x;
    const float height = compactParams.viewport.y;
    const float invZ = 1.0 / t.z;
    const float2 lo = (float2(-0.15 * width, -0.15 * height) - float2(cx, cy)) / float2(fx, fy);
    const float2 hi = (float2(1.15 * width, 1.15 * height) - float2(cx, cy)) / float2(fx, fy);
    const float tx = clamp(t.x * invZ, min(lo.x, hi.x), max(lo.x, hi.x)) * t.z;
    const float ty = clamp(t.y * invZ, min(lo.y, hi.y), max(lo.y, hi.y)) * t.z;
    const float3 jx = float3(fx * invZ, 0.0, -fx * tx * invZ * invZ);
    const float3 jy = float3(0.0, fy * invZ, -fy * ty * invZ * invZ);
    const float3 tx3 = mul(jx, wv);
    const float3 ty3 = mul(jy, wv);
    const float3 u = float3(r[0][0], r[1][0], r[2][0]);
    const float3 v = float3(r[0][1], r[1][1], r[2][1]);
    const float2x2 toPixels = float2x2(dot(tx3, u), dot(tx3, v), dot(ty3, u), dot(ty3, v));
    const float det = toPixels[0][0] * toPixels[1][1] - toPixels[0][1] * toPixels[1][0];
    if (!(abs(det) > 1.0e-12)) {
        return false;
    }
    const float2 stepX = float2(toPixels[1][1], -toPixels[1][0]) / det;
    const float2 stepY = float2(-toPixels[0][1], toPixels[0][0]) / det;
    const float2x2 shapeOf = float2x2(shapeUV.x, shapeUV.y, shapeUV.y, shapeUV.z);
    const float3 axis = float3(r[0][2], r[1][2], r[2][2]);
    const float side = dot(cloudNormal, axis) >= 0.0 ? 1.0 : -1.0;
    const float2 turnX = mul(shapeOf, stepX) * side;
    const float2 turnY = mul(shapeOf, stepY) * side;
    dx = u * turnX.x + v * turnX.y;
    dy = u * turnY.x + v * turnY.y;
    return true;
}

/// The same, and (kCompactSharp) the draw's two sharp records: the mirror at
/// the centre and its steps a pixel right and down (octahedral, f16), the
/// coat's weight and roughness; the polish's weight and roughness. The
/// centre's readings of both lobes are out of the colour (the draw reads
/// them per pixel). Zero records: nothing per pixel.
public float4 shadeColour(uint i, CompactSplat g, out uint4 sharpA, out uint4 sharpB, bool perPixel = true,
                          float footprintPx = 1.0e9) {
    sharpA = uint4(0u);
    sharpB = uint4(0u);
    if (hasFlag(kCompactPlain)) {
        return float4(cloudLight(g.rgba.rgb, hasFlag(kCompactLinear)), -1.0);
    }
    const CompactSurface cs = surfaceOf(i, g);
    CompactSharp sharp;
    sharp.want = false;
    sharp.coatWeight = 0.0;
    sharp.coatSeen = float3(0.0);
    sharp.polishWeight = float3(0.0);
    sharp.polishSeen = float3(0.0);
    float3 normalX = float3(0.0);
    float3 normalY = float3(0.0);
    float pixelTurn2 = 0.0;
    if (perPixel && hasFlag(kCompactSharp) && cs.curved && hasFlag(kCompactIndirect)) {
        const float3x3 r = splatRotation(g.quaternion);
        const float3 cloudNormal = splatShadingNormal(r, g.scales, g.center, compactParams.eyeObject.xyz, false, 0u);
        float3 dx, dy;
        if (compactTurnPerPixel(g, r, cs.shapeUV, cloudNormal, dx, dy)) {
            // splat_project's turn, held to half a radian a pixel; a tube
            // turns, a fold widens (athenea f9c9a2b): past kFoldTurnLow of
            // turn across +-2 sigma (twice the bake's spread) the pixels'
            // turn goes and the sharp lobes widen by it (foldWidened, below).
            const float held = 1.0 - smoothstep(kFoldTurnLow, kFoldTurnHigh, 2.0 * cs.spread);
            const float most = 0.5;
            const float3 hx = held * dx * min(1.0, most / max(held * length(dx), 1.0e-9));
            const float3 hy = held * dy * min(1.0, most / max(held * length(dy), 1.0e-9));
            // The pixel's own spread (athenea's splat_project): the normal
            // turns by these across a pixel.
            pixelTurn2 = held * held * (dot(dx, dx) + dot(dy, dy));
            const float3 stored = normalOf(i);
            const float3 n = dot(stored, cloudNormal) < 0.0 ? -stored : stored;
            const float4 w0 = compactParams.row0;
            const float4 w1 = compactParams.row1;
            const float4 w2 = compactParams.row2;
            normalX = relightNormalToWorld(w0, w1, w2, normalize(n + hx));
            normalY = relightNormalToWorld(w0, w1, w2, normalize(n + hy));
            sharp.want = true;
        }
    }
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
        // A footprint under reflectMinPx reads its reflected field
        // band-limited (reflectSubTerms of its 16 coefficients).
        const uint terms = footprintPx < compactParams.reflectMinPx ? compactParams.reflectSubTerms : 16u;
        reflectedTerms(cs, weightsOf(i, 2u), polish, coat, through, terms);
    }
    const float openness = opennessValue(c, wd);
    float3 light = relitCompact(surface, eyeFrom(surface), 0u, indirect, front, coupling, sunShare, polish, coat,
                                through, openness, cs.lensCurvature, sharp);
    if (sharp.coatWeight > 0.0 || any(sharp.polishWeight > float3(0.0))) {
        // The centre's readings out; the draw reads the sky per pixel.
        light = max(light - sharp.coatWeight * sharp.coatSeen - sharp.polishWeight * sharp.polishSeen, float3(0.0));
        const float3 wo = eyeFrom(surface);
        const float2 at = octEncode(reflect(-wo, surface.normal));
        float2 stepX = octEncode(reflect(-wo, normalX)) - at;
        float2 stepY = octEncode(reflect(-wo, normalY)) - at;
        stepX = all(abs(stepX) < 0.25) ? stepX : float2(0.0);
        stepY = all(abs(stepY) < 0.25) ? stepY : float2(0.0);
        const float turn = 2.0 * cs.spread;
        const float widen = smoothstep(kFoldTurnLow, kFoldTurnHigh, turn) * 0.25 * turn * turn;
        // THE PIXEL'S OWN SPREAD (athenea's splat_project): the read is at
        // the pixel's centre and the normal turns by dx, dy across it, a box
        // of slopes whose variance, twice for the mirror, widens the lobe:
        // alpha^2 += 2 (|dx|^2 + |dy|^2) / 12.
        const float coatA = foldWidened(unwidened(cs.storedCoat, cs.spread), widen);
        const float polishA = foldWidened(unwidened(cs.storedPolish, cs.spread), widen);
        const float coatR = sqrt(sqrt(min(coatA * coatA * coatA * coatA + pixelTurn2 / 6.0, 1.0)));
        const float polishR = sqrt(sqrt(min(polishA * polishA * polishA * polishA + pixelTurn2 / 6.0, 1.0)));
        // A LOBE WHOSE MIRROR STAYS IN ONE TEXEL across the footprint (its
        // turn over 3 sigma under sharpFlatTexels of the finer of the
        // sky's two levels it reads) reads the same sky at every pixel: read
        // once here, at the per-pixel roughness, and not drawn per pixel.
        float coatW = sharp.coatWeight;
        float3 polishW = sharp.polishWeight;
        const float across = max(length(stepX), length(stepY)) * 3.0 * footprintPx;
        const uint base = compactParams.envBaseSide;
        const float3 mirror0 = reflect(-wo, surface.normal);
        if (coatW > 0.0 &&
            across * float(envSideOf(uint(sqrt(saturate(coatR)) * float(kEnvLevels - 1)), base)) < compactParams.sharpFlatTexels) {
            light += coatW * compactEnvRadiance(envTexels, 0u, mirror0, coatR, base);
            coatW = 0.0;
        }
        if (any(polishW > float3(0.0)) &&
            across * float(envSideOf(uint(sqrt(saturate(polishR)) * float(kEnvLevels - 1)), base)) < compactParams.sharpFlatTexels) {
            light += polishW * compactEnvRadiance(envTexels, 0u, mirror0, polishR, base);
            polishW = float3(0.0);
        }
        if (coatW > 0.0 || any(polishW > float3(0.0))) {
            sharpA = uint4(packHalves(at.x, at.y), packHalves(stepX.x, stepX.y), packHalves(stepY.x, stepY.y),
                           packHalves(coatW, coatR));
            sharpB = uint4(packHalves(polishW.x, polishW.y), packHalves(polishW.z, polishR), 0u, 0u);
        }
    }
    float alpha = -1.0;
    if (surface.thinWalled && surface.transmission > 0.5) {
        const float alphaOwn = max(min(g.rgba.a, 1.0) - 1.0 / 255.0, 0.0);
        if (sheetAfterBlur()) {
            // Floored after the blur (projectCompact gives the colour's scale).
            alpha = alphaOwn;
        } else {
            alpha = max(alphaOwn, kCompactSheetAlpha);
            light *= alphaOwn / alpha;
        }
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

/// A draw list entry's (or a sorted value's) gaussian, its load marks
/// apart (kCompactLoadFx).
uint loadIndexOf(uint entry) { return hasFlag(kCompactLoadFx) ? entry & kCompactLoadIndex : entry; }
uint loadBitsOf(uint entry) { return hasFlag(kCompactLoadFx) ? entry & ~kCompactLoadIndex : 0u; }

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
    const bool stable = hasFlag(kCompactStableOrder);
    if (stable && t >= compactParams.capacity) {
        return;
    }
    if (t >= compactParams.count) {
        // kCompactStableOrder sorts every slot: the ones past the list sort
        // last (the nearest key), past the drawn count, and draw nothing.
        if (stable) {
            sortKeys[t] = (1u << kCompactKeyBits) - 1u;
            sortVals[t] = kCompactNone;
        }
        return;
    }
    // With a draw list (a LoD cut chosen on the host), thread t culls its
    // t-th entry; else gaussian t.
    const uint entry = hasFlag(kCompactDrawList) ? compactFrame[compactParams.drawList + t] : t;
    const uint i = loadIndexOf(entry);
    const uint key = cullKeyOf(i);
    if (stable) {
        // The radix sort is stable: equal keys keep the list's order.
        sortKeys[t] = key == kCompactNone ? 0u : key;
        sortVals[t] = key == kCompactNone ? kCompactNone : i | loadBitsOf(entry);
        return;
    }
    if (key == kCompactNone) {
        return;
    }
    const uint slot = sortArgs[1].add(1u);
    if (slot < compactParams.capacity) {
        sortKeys[slot] = key;
        sortVals[slot] = i | loadBitsOf(entry);
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
bool projectCompact(CompactSplat g, float3 v, uint index, bool sheet, inout float alpha, out float2 centre,
                    out float2 sigma, out float angle, out bool lod, out float sheetScale) {
    sheetScale = 1.0;
    centre = float2(0.0);
    sigma = float2(0.0);
    angle = 0.0;
    lod = false;
    float a0 = alpha;
    // Check only (thread CW, CompactPathOptions.debugBits & 1): a merged
    // node's LoD opacity held to 1.
    if ((uint(compactParams.viewport.w) & 1u) != 0u) {
        a0 = min(a0, 1.0);
    }
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
    if (sheet && sheetAfterBlur()) {
        // sheetFloor: the floor on the alpha the blur left (paid), faded by
        // the splat's area on screen; the colour scaled by paid / floor.
        const float spread = sqrt(max(0.0, 0.25 * (a00 - d00) * (a00 - d00) + b * b));
        const float major = 0.5 * (a00 + d00) + spread;
        const float minor = max(0.5 * (a00 + d00) - spread, 0.0);
        const float gFade = compactSheetFade(major, minor, blur);
        const float low = kCompactSheetAlpha * gFade;
        if (blurred >= low) {
            alpha = blurred;
        } else {
            float target = low;
            if (target < faintTarget) {
                uint x = (stableIndexOf(index) * 0x9e3779b9u) ^ 0x85ebca6bu;
                x ^= x >> 16u;
                x *= 0x7feb352du;
                x ^= x >> 15u;
                x *= 0x846ca68bu;
                x ^= x >> 16u;
                if (float(x >> 8u) / 16777216.0 >= target / faintTarget) {
                    return false;
                }
                target = faintTarget;
            }
            sheetScale = blurred / low;
            alpha = target;
        }
        lod = false;
    } else if (hasFlag(kCompactFaint) && a0 <= 1.0 && compactParams.falloff == 1.0 && detOrig > 0.0 &&
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
            uint x = (stableIndexOf(index) * 0x9e3779b9u) ^ 0x85ebca6bu;
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
            slot = shadeAux[2u * kCompactBins + t];
        }
        const uint entry = rec0[slot];
        if (entry == kCompactNone) {
            // A culled entry of kCompactStableOrder: drawn as nothing.
            rec0[slot] = 0u;
            rec1[slot] = 0u;
            rec2[slot] = 0u;
            rec3[slot] = 0u;
            continue;
        }
        const uint i = loadIndexOf(entry);
        CompactSplat g = readCompact(i);
        const float3 v = toView(g.center);
        // The footprint first: a splat it drops (under minAlpha, a faint
        // one's draw) is not relit at all. Its alpha is the relight's
        // (drawAlphaOf), so the colour and footprint are as before.
        float alpha = drawAlphaOf(i, g);
        float2 centre, sigma;
        float angle;
        bool lod;
        float sheetScale;
        const bool sheet = sheetAfterBlur() && isSheet(i);
        const bool drawn = (alpha > 0.0 || sheet) && projectCompact(g, v, i, sheet, alpha, centre, sigma, angle, lod, sheetScale);
        if (!drawn || max(sigma.x, sigma.y) * compactParams.maxStdDev < compactParams.minPixelRadius) {
            if (hasFlag(kCompactSharp)) {
                writeSharp(slot, uint4(0u), uint4(0u));
            }
            rec0[slot] = 0u;
            rec1[slot] = 0u;
            rec2[slot] = 0u;
            rec3[slot] = 0u;
            continue;
        }
        uint4 sharpA, sharpB;
        // A footprint under sharpMinPx: its centre's reading is the pixel's.
        const float4 lit = shadeColour(i, g, sharpA, sharpB, max(sigma.x, sigma.y) >= compactParams.sharpMinPx,
                                       max(sigma.x, sigma.y)) *
                           float4(float3(sheetScale), 1.0);
        if (hasFlag(kCompactSharp)) {
            // The weights scaled with the colour (a sheet's floor).
            const float2 cw = unpackHalves(sharpA.w);
            sharpA.w = packHalves(cw.x * sheetScale, cw.y);
            const float2 p0 = unpackHalves(sharpB.x);
            const float2 p1 = unpackHalves(sharpB.y);
            sharpB.x = packHalves(p0.x * sheetScale, p0.y * sheetScale);
            sharpB.y = packHalves(p1.x * sheetScale, p1.y);
            {
                const float2 a0 = unpackHalves(sharpA.x);
                const float2 sx = unpackHalves(sharpA.y);
                const float2 sy = unpackHalves(sharpA.z);
                const float2 c = unpackHalves(sharpA.w);
                const float2 q0 = unpackHalves(sharpB.x);
                const float2 q1 = unpackHalves(sharpB.y);
                const bool any0 = sharpA.w != 0u || sharpB.x != 0u || sharpB.y != 0u;
                uint pw = 0u;
                const uint4 r4 = any0 ? packSharp(a0, sx, sy, c.x, c.y, float3(q0.x, q0.y, q1.x), q1.y, pw) : uint4(0u);
                writeSharp(slot, r4, uint4(pw, 0u, 0u, 0u));
            }
        }
        rec0[slot] = packCentre(centre);
        uint w1 = (f32tof16(sigma.x) & 0xFFFFu) | (uint(round(angle / 3.14159265358979 * 65535.0)) << 16u);
        uint w2 = (f32tof16(sigma.y) & 0xFFFFu) | ((f32tof16(lod ? -alpha : alpha) & 0xFFFFu) << 16u);
        const uint marks = loadBitsOf(entry);
        if ((marks & kCompactLoadPending) != 0u) {
            w2 |= 0x8000u;
        } else if ((marks & kCompactLoadFading) != 0u) {
            w1 = (w1 & ~0x003F0000u) | (((entry >> 24u) & 63u) << 16u) | 0x8000u;
        }
        rec1[slot] = w1;
        rec2[slot] = w2;
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
    return i == kCompactNone ? 0u : clusterAt(loadIndexOf(i));
}

/// The workgroup's counts by cluster (the bin passes): one global atomic a
/// cluster a workgroup instead of one a slot (thread CW).
groupshared Atomic<uint> binLocal[kCompactBins];
groupshared uint binBase[kCompactBins];

[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaCompactBinCount(uint3 lid: SV_GroupThreadID, uint3 gid: SV_GroupID) {
    const uint count = min(sortCount[1], compactParams.capacity);
    for (uint j = 0u; j < kCompactBinsPerThread; ++j) {
        binLocal[lid.x + 256u * j].store(0u);
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint pass = 0u; pass < kCompactRounds; ++pass) {
        const uint slot = (gid.x * kCompactRounds + pass) * kCompactWorkgroup + lid.x;
        if (slot < count) {
            binLocal[drawnClusterOf(slot)].add(1u);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint j = 0u; j < kCompactBinsPerThread; ++j) {
        const uint b = lid.x + 256u * j;
        const uint n = binLocal[b].load();
        if (n != 0u) {
            bins[b].add(n);
        }
    }
}

groupshared uint binScan[256];

/// The clusters' starts (exclusive sums of the counts), the counts cleared
/// for the next frame: one workgroup, a thread kCompactBinsPerThread
/// consecutive bins (their sum scanned across the threads).
[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaCompactBinScan(uint3 lid: SV_GroupThreadID) {
    const uint t = lid.x;
    uint n[kCompactBinsPerThread];
    uint sum = 0u;
    for (uint j = 0u; j < kCompactBinsPerThread; ++j) {
        n[j] = bins[kCompactBinsPerThread * t + j].load();
        sum += n[j];
    }
    binScan[t] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint step = 1u; step < 256u; step <<= 1u) {
        const uint add = t >= step ? binScan[t - step] : 0u;
        GroupMemoryBarrierWithGroupSync();
        binScan[t] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    uint start = binScan[t] - sum;
    for (uint j = 0u; j < kCompactBinsPerThread; ++j) {
        const uint b = kCompactBinsPerThread * t + j;
        bins[kCompactBins + b].store(start);
        bins[b].store(0u);
        start += n[j];
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaCompactBinScatter(uint3 lid: SV_GroupThreadID, uint3 gid: SV_GroupID) {
    const uint count = min(sortCount[1], compactParams.capacity);
    // Each slot's place among its workgroup's of the same cluster, then one
    // range a cluster reserved for the workgroup, then the slots written.
    for (uint j = 0u; j < kCompactBinsPerThread; ++j) {
        binLocal[lid.x + 256u * j].store(0u);
    }
    GroupMemoryBarrierWithGroupSync();
    uint clusterOf[kCompactRounds];
    uint rank[kCompactRounds];
    [unroll] for (uint pass = 0u; pass < kCompactRounds; ++pass) {
        const uint slot = (gid.x * kCompactRounds + pass) * kCompactWorkgroup + lid.x;
        clusterOf[pass] = slot < count ? drawnClusterOf(slot) : kCompactBins;
        rank[pass] = clusterOf[pass] < kCompactBins ? binLocal[clusterOf[pass]].add(1u) : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint j = 0u; j < kCompactBinsPerThread; ++j) {
        const uint b = lid.x + 256u * j;
        const uint n = binLocal[b].load();
        binBase[b] = n != 0u ? bins[kCompactBins + b].add(n) : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint pass = 0u; pass < kCompactRounds; ++pass) {
        const uint slot = (gid.x * kCompactRounds + pass) * kCompactWorkgroup + lid.x;
        if (clusterOf[pass] < kCompactBins) {
            bins[2u * kCompactBins + binBase[clusterOf[pass]] + rank[pass]].store(slot);
        }
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
    float alpha = drawAlphaOf(i, g);
    const bool sheet = sheetAfterBlur() && isSheet(i);
    float sheetScale;
    const float3 v = toView(g.center);
    float2 centre, sigma;
    float angle;
    bool lod;
    float4 shape = float4(0.0);
    float4 place = float4(0.0);
    if (inFrustum(v) && (alpha > 0.0 || sheet)) {
        float pre = alpha > 1.0 ? min(alpha * 4.0 - 3.0, 5.0) : alpha;
        if (projectCompact(g, v, i, sheet, alpha, centre, sigma, angle, lod, sheetScale)) {
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
    /// kCompactSharp: the pixel's step from the centre (px, the shade's
    /// rows down) and the slot.
    float2 offset : TEXCOORD2;
    nointerpolation uint slot : TEXCOORD3;
    /// kCompactLoadFx: the load weight (1 pending, fading to 0).
    nointerpolation float load : TEXCOORD4;
};

static const float2 kCorners[4] = { float2(-1.0, -1.0), float2(1.0, -1.0), float2(-1.0, 1.0), float2(1.0, 1.0) };

[shader("vertex")]
CompactVertexOut atheneaCompactVertex(uint vertexId: SV_VertexID, uint instanceId: SV_InstanceID) {
    CompactVertexOut o;
    o.position = float4(0.0, 0.0, 2.0, 1.0);
    o.rgba = float4(0.0);
    o.uv = float2(0.0);
    o.adjusted = 0.0;
    o.offset = float2(0.0);
    o.slot = instanceId;
    o.load = 0.0;
    uint w1 = draw1[instanceId];
    uint w2 = draw2[instanceId];
    if (w1 == 0u && w2 == 0u) {
        return o;
    }
    if (hasFlag(kCompactLoadFx)) {
        if ((w2 & 0x8000u) != 0u) {
            o.load = 1.0;
        } else if ((w1 & 0x8000u) != 0u) {
            float age = compactParams.loadFx.w - float((w1 >> 16u) & 63u);
            if (age < 0.0) {
                age += 64.0;
            }
            o.load = saturate(1.0 - age / float(kCompactLoadFadeTicks));
        }
        w1 &= 0xFFFF7FFFu;
        w2 &= 0xFFFF7FFFu;
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
    o.offset = corner.x * v1 * s1 + corner.y * v2 * s2;
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
    float z2 = dot(v.uv, v.uv);
    float2 shift = float2(0.0);
    if (hasFlag(kCompactLoadFx)) {
        // The splat's own coordinates are affine on screen: at the centre
        // of the pixel's block they are its own plus the screen steps.
        const float2 ux = ddx(v.uv);
        const float2 uy = ddy(v.uv);
        const float block = floor(1.0 + (compactParams.loadFx.x - 1.0) * compactParams.loadFx.y * v.load);
        if (block >= 2.0) {
            const float2 p = v.position.xy;
            shift = (floor(p / block) + 0.5) * block - p;
            const float2 uvb = v.uv + ux * shift.x + uy * shift.y;
            z2 = dot(uvb, uvb);
        }
    }
    if (z2 > v.adjusted * v.adjusted) {
        discard;
    }
    const float a = splatAlphaOf(v.rgba.a, z2);
    if (a < compactParams.minAlpha) {
        discard;
    }
    float3 rgb = v.rgba.rgb;
    if (hasFlag(kCompactSharp)) {
        const uint at = sharpAt(v.slot);
        const uint4 r0 = uint4(sharpIn[at], sharpIn[at + 1u], sharpIn[at + 2u], sharpIn[at + 3u]);
        const uint4 r = uint4(r0.x, r0.y, r0.w, sharpIn[at + 4u]);
        if (r.z != 0u || r.w != 0u) {
            // (The block's centre too, pixelated: kCompactLoadFx.)
            const float2 d = v.offset + shift;
            const float2 a0 = float2(float(r.x & 0xFFFFu), float(r.x >> 16u)) / 65535.0;
            const float2 sx = unpackHalves(r0.y);
            const float2 sy = unpackHalves(r0.z);
            const float2 at = a0 + sx * d.x + sy * d.y;
            const float3 mirror = octDecode(saturate(at));
            const uint side = compactParams.envBaseSide;
            const float cr = float((r.z >> 10u) & 0x7FFu) / 2047.0;
            const float2 coat = float2(float(r.z & 0x3FFu) / 1023.0, cr * cr);
            if (coat.x > 0.0) {
                rgb += coat.x * atlasRadiance(mirror, coat.y, side);
            }
            const float3 polish = unpackRgb9e5(r.w);
            if (any(polish > float3(0.0))) {
                const float pr = float(r.z >> 21u) / 2047.0;
                rgb += polish * atlasRadiance(mirror, pr * pr, side);
            }
        }
    }
    return float4(rgb * a, a);
}

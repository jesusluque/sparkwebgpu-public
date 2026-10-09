// An experimental compute rasterizer for the sorted splats, after the
// reference 3DGS one (graphdeco-inria/diff-gaussian-rasterization): the
// screen is cut into 16 x 16 pixel tiles, each splat is listed in every tile
// its footprint touches, and one workgroup per tile blends its list front to
// back, stopping once every pixel's transmittance is under a threshold.
//
//   tilePreprocess  per sorted slot: the footprint (draw/splat_shape, as the
//                   quad draw), its tile rectangle and tile count
//   (scan)          sort_radix's exclusive scan of the counts: key offsets
//   tileTotal       the key count, into the tile sorter's draw args
//   tileDuplicate   a (tile, slot) pair per touched tile
//   (sort)          GpuSorter's stable radix sort on the tile bits only
//   tileRanges      each tile's [start, end) in the sorted pairs
//   tileBlend       per tile, front to back into an rgba16float texture
//
// The slots come from the depth sort, back to front, and the pairs are
// written in slot order, so a stable sort by tile alone leaves each tile's
// list back to front: no depth bits in the keys, and fewer radix passes.
// The blend walks the lists from the end. Colours accumulate premultiplied
// with alpha 1 - T, which composited with the quad draw's blend
// (one, one-minus-src-alpha) gives the back-to-front result.
//
// Default limits only: no subgroups, 256 invocations, 8.5 KiB of workgroup
// memory, at most 7 storage buffers a kernel.

import core.math;
import draw.splat_shape;
import plugins.blend_term;

// The blend plugins of this build (slang/plugins/blend_term.slang), as in
// the quad draw: a preset defines SPLAT_BLEND before including this file.
#ifndef SPLAT_BLEND
#define SPLAT_BLEND DefaultBlendTerm
#endif

static const uint TILE_SIZE = 16;
static const uint TILE_THREADS = TILE_SIZE * TILE_SIZE;
// Splats a tile's workgroup loads at a time: 128 keeps its workgroup memory
// at 8.5 KiB (6.5 before the surfels' rows), for more workgroups in flight.
static const uint TILE_BATCH = 128;
// tileRanges covers GpuSorter's dispatch blocks: 128 threads x 8 keys.
static const uint RANGE_WG = 128;
static const uint RANGE_ROUNDS = 8;

static const uint TILE_DEPTH_TEST = 1u; // test against depthTex (less-equal)
static const uint TILE_CLAMP = 2u;      // clamp each splat's colour, as unorm targets do

struct TileParams {
    uint slots;       // sort capacity: entries of tileCounts to scan
    uint tilesX;
    uint tilesY;
    uint keyCapacity; // pairs the key buffers hold

    uint width;
    uint height;
    uint flags;
    float minTransmittance; // a pixel stops blending under this
};

StructuredBuffer<uint> ordering;     // back-to-front accumulator indices
StructuredBuffer<uint> sortCount;    // the depth sort's draw args: [1] slots
// Per slot, TILE_SPLAT_WORDS: [0] center, u1  [1] u2, r, g  [2] unused; a
// surfel's (negative stdDev^2): [0] its centre in surfel pixels, rows.x.xy
// [1] rows.y.xy, r, g  [2] rows.w, 0.
static const uint TILE_SPLAT_WORDS = 3u;
RWStructuredBuffer<uint4> tileSplats;
RWStructuredBuffer<uint4> tileExtra;  // per slot: stdDev^2 (< 0: a surfel), alpha, depth, b
RWStructuredBuffer<uint2> tileRects;  // per slot: tile x0 | y0 << 16, x1 | y1 << 16
RWStructuredBuffer<uint> tileCounts;  // per slot tiles, scanned in place to offsets
RWStructuredBuffer<uint> keysOut;
RWStructuredBuffer<uint> valsOut;
RWStructuredBuffer<uint> keyArgs;     // the tile sorter's draw args: [1] pairs
RWStructuredBuffer<uint> keyTotal;    // [0] pairs wanted, for the host to grow

StructuredBuffer<uint> sortedKeys;
StructuredBuffer<uint> keyCount;      // keyArgs, read
// Flat: two threads write the two ends of a range, and a vector
// component store may rewrite the whole vector.
RWStructuredBuffer<uint> rangesOut;

StructuredBuffer<uint4> splatsIn;     // tileSplats, read
StructuredBuffer<uint4> extraIn;      // tileExtra, read
StructuredBuffer<uint> sortedVals;
StructuredBuffer<uint2> rangesIn;     // rangesOut, read
Texture2D<float> depthTex;
[format("rgba16f")] WTexture2D<float4> outTex;

ConstantBuffer<TileParams> tileParams;

bool hasTileFlag(uint f) { return (tileParams.flags & f) != 0u; }

// Splat-space squared radius of pixel offset d: |uv|^2 as the blend
// computes it.
float uvRadius2(float2 d, float2 u1, float2 u2) {
    return sqr(dot(d, u1)) + sqr(dot(d, u2));
}

// The least uvRadius2 on the segment p + t * e, t in [0, 1].
float segmentMin(float2 p, float2 e, float2 u1, float2 u2) {
    float a = dot(e, u1);
    float b = dot(e, u2);
    float c = dot(p, u1);
    float d = dot(p, u2);
    float ee = a * a + b * b;
    float t = ee > 0.0 ? clamp(-(c * a + d * b) / ee, 0.0, 1.0) : 0.0;
    return sqr(c + t * a) + sqr(d + t * b);
}

// Whether a tile has pixel centres inside slot i's footprint ellipse: the
// least uvRadius2 over the box of its centres (relative to the splat's
// centre) is within stdDev^2. A box missing the centre has it on an edge.
// Most of a large or thin splat's bounding rectangle can miss it.
// A surfel touches every tile of its rectangle (surfelQuad's box).
bool tileTouched(uint i, uint2 tile) {
    float stdDev2 = asfloat(tileExtra[i].x);
    if (stdDev2 < 0.0) {
        return true;
    }
    uint4 a = tileSplats[TILE_SPLAT_WORDS * i];
    uint4 b = tileSplats[TILE_SPLAT_WORDS * i + 1u];
    float2 center = asfloat(a.xy);
    float2 u1 = asfloat(a.zw);
    float2 u2 = asfloat(b.xy);
    if (hasFlag(DRAW_AA_BOX)) {
        // u1, u2 are the box's axes: every tile of the rectangle.
        return true;
    }
    float2 last = float2(float(tileParams.width), float(tileParams.height)) - 1.0;
    float2 lo = float2(tile * TILE_SIZE) + 0.5 - center;
    float2 hi = min(float2(tile * TILE_SIZE + TILE_SIZE - 1u), last) + 0.5 - center;
    if (all(lo <= 0.0) && all(hi >= 0.0)) {
        return true;
    }
    float m = min(
        min(segmentMin(lo, float2(hi.x - lo.x, 0.0), u1, u2), segmentMin(float2(lo.x, hi.y), float2(hi.x - lo.x, 0.0), u1, u2)),
        min(segmentMin(lo, float2(0.0, hi.y - lo.y), u1, u2), segmentMin(float2(hi.x, lo.y), float2(0.0, hi.y - lo.y), u1, u2)));
    return m <= stdDev2;
}

// Tiles of slot i's rectangle that its ellipse touches.
uint touchedTiles(uint i) {
    uint2 r = tileRects[i];
    uint x0 = r.x & 0xffffu;
    uint y0 = r.x >> 16u;
    uint x1 = r.y & 0xffffu;
    uint y1 = r.y >> 16u;
    uint n = 0u;
    if (x1 >= x0 && y1 >= y0) {
        for (uint y = y0; y <= y1; y++) {
            for (uint x = x0; x <= x1; x++) {
                if (tileTouched(i, uint2(x, y))) {
                    n++;
                }
            }
        }
    }
    return n;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void tilePreprocess(uint3 tid: SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= tileParams.slots) {
        return;
    }
    uint2 rect = uint2(0xffffffffu, 0u); // no tiles
    SplatShape s;
    uint splatIndex = i < sortCount[1] ? ordering[i] : 0xffffffffu;
    bool drawn = splatIndex != 0xffffffffu && projectSplat(splatIndex, s);
    if (drawn && s.surfel) {
        // The pixels whose centres the quad draw's surfelQuad covers
        // (edges included: a centre on the snapped edge is outside the
        // ellipse and the filter's disc, and draws nothing).
        float2 size = float2(float(tileParams.width), float(tileParams.height));
        float2 lo, hi;
        surfelQuad(s.rows, s.adjustedStdDev, lo, hi);
        float2 first = max(ceil(float2(lo.x, size.y - hi.y) - 0.5), float2(0.0));
        float2 last = min(floor(float2(hi.x, size.y - lo.y) - 0.5), size - 1.0);
        if (all(first <= last)) {
            uint2 t0 = uint2(first) / TILE_SIZE;
            uint2 t1 = uint2(last) / TILE_SIZE;
            rect = uint2(t0.x | (t0.y << 16u), t1.x | (t1.y << 16u));
            float3 rgb = s.rgba.rgb;
            if (hasFlag(DRAW_ENCODE_LINEAR)) {
                rgb = srgbToLinear(rgb);
            }
            const SurfelRows r = s.rows;
            const uint o = TILE_SPLAT_WORDS * i;
            tileSplats[o] = asuint(float4(r.x.z, r.y.z, r.x.xy));
            tileSplats[o + 1u] = asuint(float4(r.y.xy, rgb.rg));
            tileSplats[o + 2u] = asuint(float4(r.w, 0.0));
            tileExtra[i] = uint4(
                asuint(-s.adjustedStdDev * s.adjustedStdDev),
                asuint(markedRgba(s).a),
                asuint(s.clipCenter.z / s.clipCenter.w),
                asuint(rgb.b));
        }
    } else if (drawn) {
        float2 size = float2(float(tileParams.width), float(tileParams.height));
        float2 ndc = s.clipCenter.xy / s.clipCenter.w;
        float2 center = float2(ndc.x * 0.5 + 0.5, 0.5 - 0.5 * ndc.y) * size;
        // The quad's axes in framebuffer pixels (y down). They stay
        // orthogonal, so a pixel's corner coordinate along each is a
        // projection, and its splat-space coordinate (the quad's splatUv)
        // is that times adjustedStdDev.
        float2 a1 = float2(s.axis1.x, -s.axis1.y) / params.focalAdjustment;
        float2 a2 = float2(s.axis2.x, -s.axis2.y) / params.focalAdjustment;
        float l1 = dot(a1, a1);
        float l2 = dot(a2, a2);
        if (l1 > 0.0 && l2 > 0.0) {
            float2 u1 = a1 * (s.adjustedStdDev / l1);
            float2 u2 = a2 * (s.adjustedStdDev / l2);
            float3 box = float3(0.0);
            if (hasFlag(DRAW_AA_BOX)) {
                // The offset along the box's axes in scaled pixels
                // (boxFrame, boxRadius2): the axes turned by rot.
                float2 rot;
                box = boxFrame(normalize(s.axis1), s.box, rot);
                const float2 n1 = a1 * (params.focalAdjustment / sqrt(l1));
                const float2 n2 = a2 * (params.focalAdjustment / sqrt(l2));
                u1 = rot.x * n1 + rot.y * n2;
                u2 = rot.x * n2 - rot.y * n1;
            }
            // Pixels whose centres fall in the ellipse inscribed in the quad
            // (the fragment discards the rest).
            float2 extent = sqrt(float2(a1.x * a1.x + a2.x * a2.x, a1.y * a1.y + a2.y * a2.y));
            float2 lo = max(ceil(center - extent - 0.5), float2(0.0));
            float2 hi = min(floor(center + extent - 0.5), size - 1.0);
            if (all(lo <= hi)) {
                uint2 t0 = uint2(lo) / TILE_SIZE;
                uint2 t1 = uint2(hi) / TILE_SIZE;
                rect = uint2(t0.x | (t0.y << 16u), t1.x | (t1.y << 16u));
                float3 rgb = s.rgba.rgb;
                if (hasFlag(DRAW_ENCODE_LINEAR)) {
                    rgb = srgbToLinear(rgb);
                }
                tileSplats[TILE_SPLAT_WORDS * i] = uint4(asuint(center.x), asuint(center.y), asuint(u1.x), asuint(u1.y));
                tileSplats[TILE_SPLAT_WORDS * i + 1u] = uint4(asuint(u2.x), asuint(u2.y), asuint(rgb.r), asuint(rgb.g));
                tileSplats[TILE_SPLAT_WORDS * i + 2u] = asuint(float4(box, 0.0));
                tileExtra[i] = uint4(
                    asuint(s.adjustedStdDev * s.adjustedStdDev),
                    asuint(markedRgba(s).a),
                    asuint(s.clipCenter.z / s.clipCenter.w),
                    asuint(rgb.b));
            }
        }
    }
    tileRects[i] = rect;
    tileCounts[i] = touchedTiles(i);
}

// The pairs to write, once tileCounts is scanned: into keyArgs[1], clamped
// to the key buffers, and keyTotal[0] unclamped.
[shader("compute")]
[numthreads(1, 1, 1)]
void tileTotal() {
    uint n = min(sortCount[1], tileParams.slots);
    uint total = n > 0u ? tileCounts[n - 1u] + touchedTiles(n - 1u) : 0u;
    keyTotal[0] = total;
    keyArgs[1] = min(total, tileParams.keyCapacity);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void tileDuplicate(uint3 tid: SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= min(sortCount[1], tileParams.slots)) {
        return;
    }
    uint2 r = tileRects[i];
    uint x0 = r.x & 0xffffu;
    uint y0 = r.x >> 16u;
    uint x1 = r.y & 0xffffu;
    uint y1 = r.y >> 16u;
    uint k = tileCounts[i];
    for (uint y = y0; y <= y1; y++) {
        for (uint x = x0; x <= x1; x++) {
            if (!tileTouched(i, uint2(x, y))) {
                continue;
            }
            if (k >= tileParams.keyCapacity) {
                return;
            }
            keysOut[k] = y * tileParams.tilesX + x;
            valsOut[k] = i;
            k++;
        }
    }
}

// Each tile's [start, end) in the sorted pairs (rangesOut cleared first).
[shader("compute")]
[numthreads(128, 1, 1)]
void tileRanges(uint3 lid: SV_GroupThreadID, uint3 gid: SV_GroupID) {
    uint n = keyCount[1];
    uint base = gid.x * RANGE_WG * RANGE_ROUNDS;
    for (uint r = 0; r < RANGE_ROUNDS; r++) {
        uint k = base + r * RANGE_WG + lid.x;
        if (k >= n) {
            return;
        }
        uint t = sortedKeys[k];
        if (k == 0u || sortedKeys[k - 1u] != t) {
            rangesOut[2u * t] = k;
        }
        if (k + 1u == n || sortedKeys[k + 1u] != t) {
            rangesOut[2u * t + 1u] = k + 1u;
        }
    }
}

// A batch of a tile's splats, front first, shared by its pixels.
groupshared float4 sGeom0[TILE_BATCH]; // center, u1 (a surfel's: centre, rows.x.xy)
groupshared float4 sGeom1[TILE_BATCH]; // u2 (rows.y.xy), stdDev^2, alpha
groupshared float3 sGeom2[TILE_BATCH]; // a surfel's rows.w
groupshared float4 sColor[TILE_BATCH]; // rgb, ndc depth
groupshared float sViewDepth[TILE_BATCH];
groupshared Atomic<uint> sDone;
groupshared uint sDoneCount;

[shader("compute")]
[numthreads(16, 16, 1)]
void tileBlend(uint3 lid: SV_GroupThreadID, uint3 gid: SV_GroupID) {
    uint t = lid.y * TILE_SIZE + lid.x;
    if (t == 0u) {
        sDone.store(0u);
    }
    uint2 pix = gid.xy * TILE_SIZE + lid.xy;
    bool inside = pix.x < tileParams.width && pix.y < tileParams.height;
    float2 p = float2(pix) + 0.5;
    float sceneDepth = 1.0;
    if (inside && hasTileFlag(TILE_DEPTH_TEST)) {
        sceneDepth = depthTex.Load(int3(int2(pix), 0));
    }
    bool disk = hasFlag(DRAW_DISK_CLIP) && params.diskCenter.w != 0.0;
    float2 size = float2(float(tileParams.width), float(tileParams.height));
    float2 ndc = float2(p.x / size.x * 2.0 - 1.0, 1.0 - p.y / size.y * 2.0);
    float3 viewRay = float3(ndc.x / params.proj0.x, ndc.y / params.proj1.y, -1.0);
    bool clampColor = hasTileFlag(TILE_CLAMP);

    float3 color = float3(0.0);
    float T = 1.0;
    bool done = !inside;
    bool counted = false;
    uint2 range = rangesIn[gid.y * tileParams.tilesX + gid.x];
    uint end = range.y;
    GroupMemoryBarrierWithGroupSync();
    while (end > range.x) {
        uint batch = min(TILE_BATCH, end - range.x);
        if (t < batch) {
            uint slot = sortedVals[end - 1u - t];
            uint4 a = splatsIn[TILE_SPLAT_WORDS * slot];
            uint4 b = splatsIn[TILE_SPLAT_WORDS * slot + 1u];
            sGeom2[t] = asfloat(splatsIn[TILE_SPLAT_WORDS * slot + 2u].xyz);
            uint4 e = extraIn[slot];
            sGeom0[t] = asfloat(a);
            sGeom1[t] = float4(asfloat(b.xy), asfloat(e.xy));
            sColor[t] = float4(asfloat(b.zw), asfloat(e.w), asfloat(e.z));
            if (disk) {
                // As the quad draw's viewDepth: the clip w, the view depth.
                SplatShape s;
                projectSplat(ordering[slot], s);
                sViewDepth[t] = s.clipCenter.w;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (!done) {
            for (uint j = 0; j < batch; j++) {
                float4 g0 = sGeom0[j];
                float4 g1 = sGeom1[j];
                float2 uv;
                float z2;
                float fade = 0.0;
                float stdDev2 = g1.z;
                if (stdDev2 < 0.0) {
                    // A surfel, as the quad draw's fragment (y up).
                    stdDev2 = -stdDev2;
                    float2 d = surfelOffset(float2(p.x, size.y - p.y), g0.xy);
                    z2 = surfelRadius2(d, float3(g0.zw, 0.0), float3(g1.xy, 0.0), sGeom2[j], uv, fade);
                } else {
                    float2 d = p - g0.xy;
                    uv = float2(dot(d, g0.zw), dot(d, g1.xy));
                    if (hasFlag(DRAW_AA_BOX)) {
                        const float3 box = sGeom2[j];
                        z2 = boxRadius2(uv, box.xy);
                        uv /= max(box.xy, float2(1e-6));
                    } else {
                        z2 = dot(uv, uv);
                    }
                }
                if (z2 > stdDev2) {
                    continue;
                }
                // Additive (a negative alpha, markedRgba): adds its light,
                // leaves the transmittance.
                const bool additive = g1.w < 0.0;
                float alpha = splatAlpha(abs(g1.w), z2 + fade);
                if (alpha < params.minAlpha) {
                    continue;
                }
                float4 c = sColor[j];
                if (c.w > sceneDepth || (disk && diskClipped(viewRay, sViewDepth[j]))) {
                    continue;
                }
                // Colours here are already linear when DRAW_ENCODE_LINEAR.
                BlendInput blendIn = { ordering[sortedVals[end - 1u - j]], uv, float4(p, c.w, 1.0) };
                float4 blended = applyBlendTerm<SPLAT_BLEND>(float4(c.rgb, alpha), blendIn);
                alpha = blended.a;
                float3 rgb = blended.rgb * alpha;
                if (clampColor) {
                    rgb = min(rgb, float3(1.0));
                }
                color += T * rgb;
                if (!additive) {
                    T *= 1.0 - alpha;
                }
                if (T < tileParams.minTransmittance) {
                    done = true;
                    break;
                }
            }
        }
        end -= batch;
        if (done && !counted) {
            sDone.add(1u);
            counted = true;
        }
        GroupMemoryBarrierWithGroupSync();
        if (t == 0u) {
            sDoneCount = sDone.load();
        }
        if (workgroupUniformLoad(sDoneCount) == TILE_THREADS) {
            break;
        }
    }
    if (inside) {
        outTex.Store(pix, float4(color, 1.0 - T));
    }
}

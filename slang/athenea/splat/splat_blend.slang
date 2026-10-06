// Copyright (c) 2026 jesus luque.
//
// Per pixel: its tile's splats, nearest first, until the pixel is opaque.
module splat_blend;
import athenea.common.color;
import athenea.common.packing;
import frame;
import athenea.common.octahedral;
import athenea.light.environment;
import splat_relight;

StructuredBuffer<uint>      ranges;
StructuredBuffer<uint>      pairSplats;
StructuredBuffer<ProjRecord> proj;
StructuredBuffer<uint4>     slopes;   // a slot each, read where the record is marked (kSlopeMark, kSharpMark)
// The first dome's prefiltered sky, which a sharp lobe reads per pixel
// (kSharpMark; athenea/light/environment.slang). Bound whether or not one is.
StructuredBuffer<uint>      envTexels;
StructuredBuffer<uint4>     sharpPolish;   // a sharp record's base polish and its sun weights (kSharpMark)
// The first dome's sun, as env_sun hands it over: (direction, half angle), (irradiance, threshold).
StructuredBuffer<float4>    envSun;
// What is under the splats: a rasterised point layer (or nothing: bound to a
// placeholder with params.hasUnder == 0). Linear, premultiplied, bottom row first.
StructuredBuffer<float4>    underColour;
StructuredBuffer<float>     underDepth;
// CRYPTOMATTE. `cryptoIds` is the id a projected slot carries (splat_project
// wrote it), `underCrypto` the id of the opaque layer under the splats, and
// `crypto` the matte itself: kCryptoSlots ids and coverages a pixel, three
// float4 (id, cov, id, cov) in descending coverage. All bound whether the
// frame keeps a matte or not: a name a shader declares must be bound.
StructuredBuffer<uint>      cryptoIds;
StructuredBuffer<uint>      underCrypto;
RWStructuredBuffer<float4>  crypto;
RWStructuredBuffer<float4>  colour;   // width * height, premultiplied, linear
RWStructuredBuffer<float>   depth;    // width * height
ConstantBuffer<FrameParams> params;

/// How many ids a pixel can name. Six is the Cryptomatte specification's
/// three layers of two ranks: past that the lightest is dropped, which is
/// what every renderer that writes the format does.
static const uint kCryptoSlots = 6;

/// The matte a pixel builds as the walk goes: an id's coverage is the sum of
/// the weights (transmittance * alpha) of everything with that id, which is
/// exactly the alpha that id contributed to the pixel. Held in registers, so
/// the loop costs a search over six and no memory.
struct CryptoList {
    uint  ids[kCryptoSlots];
    float weights[kCryptoSlots];

    [[mutating]] void add(uint id, float weight) {
        // Id 0 is nothing named: a capture with no ancestry, a point, the
        // background. `cryptomatteId` never returns it (the fixup moves a
        // hash of zero off it), so it is free to mean absence, and what it
        // covers is simply missing from the matte rather than called
        // something it is not.
        if (weight <= 0.0 || id == 0) {
            return;
        }
        uint lightest = 0;
        for (uint k = 0; k < kCryptoSlots; ++k) {
            if (ids[k] == id && weights[k] > 0.0) {
                weights[k] += weight;
                return;
            }
            if (weights[k] < weights[lightest]) {
                lightest = k;
            }
        }
        // A free slot is one whose weight is still zero, so the same test
        // finds it; a full list gives up its lightest rank, and only if this
        // one outweighs it.
        if (weight > weights[lightest]) {
            ids[lightest] = id;
            weights[lightest] = weight;
        }
    }

    /// Ranks in descending coverage, written as (id, cov) pairs.
    void write(uint at, uint pixels) {
        uint order[kCryptoSlots];
        for (uint k = 0; k < kCryptoSlots; ++k) {
            order[k] = k;
        }
        for (uint i = 1; i < kCryptoSlots; ++i) {
            const uint pick = order[i];
            uint j = i;
            while (j > 0 && weights[order[j - 1]] < weights[pick]) {
                order[j] = order[j - 1];
                --j;
            }
            order[j] = pick;
        }
        for (uint layer = 0; layer < kCryptoSlots / 2; ++layer) {
            const uint a = order[layer * 2];
            const uint b = order[layer * 2 + 1];
            crypto[layer * pixels + at] =
                float4(weights[a] > 0.0 ? bit_cast<float>(ids[a]) : 0.0, max(weights[a], 0.0),
                       weights[b] > 0.0 ? bit_cast<float>(ids[b]) : 0.0, max(weights[b], 0.0));
        }
    }
};

/// The walk, with `composite` a constant at each call site: the splat-only
/// entry point folds the disc and under-layer branches away. Both cost the
/// blend -- the hottest loop in the engine -- half as much again when they
/// are live in a frame that has neither.
// A TILE'S SPLATS, READ ONCE FOR ITS 256 PIXELS. The group is the tile
// (16 x 16 threads): it loads the next 256 of the tile's records into group
// memory together, and every pixel walks them from there, rather than each of
// the 256 reading every record through `pairSplats` on its own. A pixel that
// is done still loads for the others; the group stops loading when all are.
static const uint kBatch = kTile * kTile;
groupshared ProjRecord sRecords[kBatch];
groupshared uint       sSplats[kBatch];
// The colour's slope across the footprint, where the record is marked so
// (frame.slang's kSlopeMark): x then y, rgb each, as halves.
groupshared uint4      sSlopes[kBatch];
groupshared uint       sDone;

void blendPixel(uint3 tid, uint3 groupThread, bool composite, bool matte) {
    const uint px = tid.x;
    const uint py = tid.y;
    const uint lane = groupThread.y * kTile + groupThread.x;
    // Outside the image: no pixel, but a pair of hands for the loads.
    const bool inside = px < params.width && py < params.height;
    const uint tile = (py / kTile) * params.tilesX + px / kTile;
    const uint start = ranges[tile * 2];
    const uint end = ranges[tile * 2 + 1];
    const float2 pixel = float2(float(px) + 0.5, float(py) + 0.5);

    const uint at = inside ? py * params.width + px : 0u;
    const bool under = inside && composite && params.hasUnder != 0;
    const float underZ = under ? underDepth[at] : 0.0;
    const bool underHit = under && underZ > 0.0;

    float3 rgb = float3(0.0);
    float transmittance = 1.0;
    float depthSum = 0.0;
    float depthAt = -1.0;
    // What ends the walk from behind: the raster under-layer, or the first
    // opaque disc. Either is a linear colour composited under the splats'
    // contribution, which is linear too -- the same arithmetic whichever route drew
    // the points, which is what lets the two routes agree.
    bool opaqueHit = false;
    float4 opaqueColour = float4(0.0);
    float opaqueZ = 0.0;
    uint opaqueId = 0;
    CryptoList list;
    if (matte) {
        for (uint k = 0; k < kCryptoSlots; ++k) {
            list.ids[k] = 0;
            list.weights[k] = 0.0;
        }
    }
    bool done = !inside;
    for (uint first = start; first < end; first += kBatch) {
        // Stop loading once every pixel of the tile has stopped walking.
        if (lane == 0) {
            sDone = 0;
        }
        GroupMemoryBarrierWithGroupSync();
        if (done) {
            InterlockedAdd(sDone, 1u);
        }
        GroupMemoryBarrierWithGroupSync();
        if (sDone == kBatch) {
            break;
        }
        const uint mine = first + lane;
        if (mine < end) {
            const uint splat = pairSplats[mine];
            sSplats[lane] = splat;
            sRecords[lane] = proj[splat];
            const float mark = sRecords[lane].colour.w;
            sSlopes[lane] = abs(mark - kSlopeMark) < 0.01 || abs(mark - kSharpMark) < 0.01 ? slopes[splat] : uint4(0u);
        }
        GroupMemoryBarrierWithGroupSync();
        const uint count = min(kBatch, end - first);
        for (uint j = 0; j < count && !done; ++j) {
            const ProjRecord record = sRecords[j];
            const float4 screen = record.screen;
            // A shadow catcher lies on the layer under it, so its centre's
            // depth reads behind it as often as not: it is drawn where it is
            // within a percent and a half of that layer's depth -- the ground
            // it lies on, seen at a slant -- and not where something stands
            // in front of it (a mesh car's body over its own shadow, which
            // went black when every catcher gaussian was drawn).
            const bool lying = abs(record.colour.w - kCatcherMark) < 0.01 && screen.z <= underZ * 1.015;
            if (underHit && screen.z >= underZ && !lying) {
                done = true;
                break;
            }
            const float2 d = pixel - screen.xy;
            const float4 conic = record.conic;
            const float power = conic.x * d.x * d.x + 2.0 * conic.y * d.x * d.y + conic.z * d.y * d.y;
            if (power > screen.w) {
                continue;
            }
            if (composite && record.colour.w > 0.5) {
                opaqueHit = true;
                // A point of the opaque layer: linear, encoded here for a
                // display blend as a splat is at its projection.
                opaqueColour = float4(params.displayBlend != 0 ? linearToSrgb(record.colour.rgb)
                                                               : record.colour.rgb, 1.0);
                opaqueZ = screen.z;
                if (matte) {
                    opaqueId = cryptoIds[sSplats[j]];
                }
                done = true;
                break;
            }
            const float alpha = min(0.99, conic.w * exp(-0.5 * power));
            if (alpha < 1.0 / 255.0) {
                continue;
            }
            const float weight = transmittance * alpha;
            if (matte) {
                list.add(cryptoIds[sSplats[j]], weight);
            }
            // A TX transfer's reflection turns across the footprint: the
            // colour at this pixel is the centre's plus its slope over the
            // step from the centre (splat_project's SplatSlope).
            float3 shade = record.colour.xyz;
            if (abs(record.colour.w - kSlopeMark) < 0.01) {
                const uint3 w = sSlopes[j].xyz;
                const float3 slopeX = float3(lowHalf(w.x), highHalf(w.x), lowHalf(w.y));
                const float3 slopeY = float3(highHalf(w.y), lowHalf(w.z), highHalf(w.z));
                shade = max(shade + slopeX * d.x + slopeY * d.y, float3(0.0));
            } else if (abs(record.colour.w - kSharpMark) < 0.01) {
                // A SHARP LOBE, READ AT THIS PIXEL: the sky along the mirror
                // this pixel turns to, a step from the centre's, at the
                // lobe's own roughness (splat_project's sharp lobe). A
                // facade or the sun mirrored in a curved coat is drawn where
                // it falls inside the gaussian, not as the centre's reading
                // spread over all of it.
                const uint4 w = sSlopes[j];
                const float2 at = float2(lowHalf(w.x), highHalf(w.x)) + float2(lowHalf(w.y), highHalf(w.y)) * d.x +
                                  float2(lowHalf(w.z), highHalf(w.z)) * d.y;
                const float3 mirror = octDecode(saturate(at));
                float3 sky = float3(0.0);
                if (lowHalf(w.w) > 0.0) {
                    sky += lowHalf(w.w) * envRadiance(envTexels, 0u, mirror, highHalf(w.w), params.envBaseSide);
                }
                // The base's polish along the same mirror, its weight and
                // roughness read from memory: one record's for every pixel
                // of the tile, so it stays in cache.
                const uint4 p = sharpPolish[sSplats[j]];
                const float3 polish = float3(lowHalf(p.x), highHalf(p.x), lowHalf(p.y));
                if (any(polish > float3(0.0))) {
                    sky += polish * envRadiance(envTexels, 0u, mirror, highHalf(p.y), params.envBaseSide);
                }
                // The sun, which the map does not hold: through each lobe at
                // this pixel's mirror (splat_relight's sunReflectPdf).
                const float4 disc = envSun[0];
                if (disc.w > 0.0) {
                    const float3 irradiance = envSun[1].rgb;
                    const float3 polishSun = float3(lowHalf(p.z), highHalf(p.z), lowHalf(p.w));
                    sky += irradiance * (polishSun * sunReflectPdf(mirror, disc.xyz, highHalf(p.y), disc.w) +
                                         highHalf(p.w) * sunReflectPdf(mirror, disc.xyz, highHalf(w.w), disc.w));
                }
                shade = max(shade + sky, float3(0.0));
            }
            rgb += weight * shade;
            depthSum += weight * screen.z;
            transmittance *= 1.0 - alpha;
            if (depthAt < 0.0 && 1.0 - transmittance >= params.depthThreshold) {
                depthAt = screen.z;
            }
            if (transmittance < 1e-4) {
                done = true;
                break;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (!inside) {
        return;
    }
    if (!opaqueHit && underHit) {
        opaqueHit = true;
        opaqueColour = underColour[at];
        // The layer under the splats is linear light; a display blend
        // composites over it encoded, as the splats are (FrameParams).
        if (params.displayBlend != 0) {
            opaqueColour.rgb = linearToSrgb(opaqueColour.rgb);
        }
        opaqueZ = underZ;
        if (matte && params.hasUnderCrypto != 0) {
            opaqueId = underCrypto[at];
        }
    }
    // What closed the walk takes the rest of the pixel: whatever light got
    // through the splats is its coverage, which is what makes the layer's
    // coverages sum to the pixel's alpha.
    if (matte) {
        list.add(opaqueId, opaqueHit ? transmittance : 0.0);
        list.write(at, params.width * params.height);
    }

    const float coverage = 1.0 - transmittance;
    // Blended in linear light: every splat's colour was made linear when it
    // was projected (frame.slang), so nothing is decoded here.
    const float3 linearRgb = rgb;
    float4 behind = float4(params.bgR, params.bgG, params.bgB, params.bgA);
    if (opaqueHit) {
        behind = opaqueColour;
        depthSum += transmittance * opaqueZ;
        if (depthAt < 0.0) {
            depthAt = opaqueZ;
        }
    }
    const float total = opaqueHit ? 1.0 : coverage;
    float4 pixelColour = float4(linearRgb + transmittance * behind.rgb, coverage + transmittance * behind.w);
    if (params.displayBlend != 0) {
        // Summed encoded (FrameParams): the background was given linear, and
        // the pixel goes back to linear light once, unpremultiplied.
        if (!opaqueHit) {
            pixelColour.rgb = linearRgb + transmittance * linearToSrgb(behind.rgb);
        }
        pixelColour.rgb = pixelColour.w > 1.0e-6 ? srgbToLinear(pixelColour.rgb / pixelColour.w) * pixelColour.w
                                                 : float3(0.0);
    }
    colour[at] = pixelColour;
    depth[at] = params.depthMode == 0 ? (total > 1e-6 ? depthSum / total : 0.0) : max(depthAt, 0.0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void splatBlend(uint3 tid: SV_DispatchThreadID, uint3 groupThread: SV_GroupThreadID) {
    blendPixel(tid, groupThread, false, false);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void splatBlendComposite(uint3 tid: SV_DispatchThreadID, uint3 groupThread: SV_GroupThreadID) {
    blendPixel(tid, groupThread, true, false);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void splatBlendCrypto(uint3 tid: SV_DispatchThreadID, uint3 groupThread: SV_GroupThreadID) {
    blendPixel(tid, groupThread, false, true);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void splatBlendCompositeCrypto(uint3 tid: SV_DispatchThreadID, uint3 groupThread: SV_GroupThreadID) {
    blendPixel(tid, groupThread, true, true);
}

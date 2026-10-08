// athenea's light groups (plan-athenea.md phase 6; proposals 062, 066, 076)
// as an ISplatColour plugin of generate, chained after the relight or the
// raster plugin (src/webgpu/athenea/lightsPlugin.ts):
//
//   colour = base + sum_k w_k * (direct_k * profile_k + layer_k)
//
//   base       the colour so far (the relit TX colour, or the raster's
//              light), in linear light: undone from Spark's transfer
//   w_k        group k's weight this frame, RGB (the .lights.usda half:
//              state, rules, curves, colour; LightRig on the host). Only
//              this changes when a lamp is switched.
//   direct_k   the group's emitter polygons, diffusely, at the splat: albedo
//              x the polygons' form factor (LTC with the cosine lobe,
//              Lambert's closed form, clipped to the horizon; 060 §2),
//              no shadow (060's bits quotient is not here yet)
//   profile_k  076: the lens's tint times its beam map, looked up in the
//              direction from the lamp to the splat in the lamp's frame
//   layer_k    the .athl's sparse layers at the splat: the light sent on
//              after bouncing (indirect), the emitter's own glow
//              (emission), and a view-dependent field (degree-2 harmonics
//              of the view direction, Spark's basis)
//
// The .athl lives in one buffer (`atheneaLightData`, laid out by
// src/webgpu/athenea/athl.ts AthlStore.pack): per group and kind a chunk
// table, per chunk a directory of 256-splat blocks, the blocks dense f16.
// A splat's index is mapped to the .athl's (the cloud's page-aligned
// virtual order) as the mesh holds it: a whole decoded .athc keeps its
// merged nodes then its splats back to back; a paged one maps pages to
// chunks through `pageChunk`.
module lights;

import core.attrib;
import core.splat_types;
import plugins.splat_colour;
import athenea.common.color;
import athenea.common.packing;

public static const uint kAtheneaLightsMaxGroups = 16;
public static const uint kAtheneaLightsMaxPages = 256;

// terms
public static const uint kLightsDirect = 1;
public static const uint kLightsIndirect = 2;
public static const uint kLightsEmission = 4;
public static const uint kLightsField = 8;

// layout
public static const uint kLightsWhole = 0;
public static const uint kLightsPaged = 1;

// AthlStore.pack
public static const uint kAthlHeaderWords = 16;
public static const uint kAthlGroupWords = 32;
public static const uint kAthlPolygonWords = 32;
public static const uint kAthlProfileHeadWords = 12;
public static const uint kAthlBlockSplats = 256;
public static const uint kAthlTwoSided = 1;
public static const uint kAthlNone = 0xFFFFFFFFu;

public struct AtheneaLightsParams {
    /// Groups with data in `atheneaLightData` (0: the plugin passes through).
    public uint groupCount;
    /// kLights* terms in use.
    public uint terms;
    /// The pool's normal stream (packing.slang octahedral), or ATTRIB_NONE.
    public uint normalId;
    /// 1: the splats' stored colours are linear light (else sRGB).
    public uint storedLinear;
    public uint layout;
    /// Whole layout: the cloud's merged nodes and its first splat's index
    /// in the .athl's virtual order (0, 0: the same order).
    public uint merged;
    public uint splatBase;
    public uint pad0;
    /// w_k: rgb, a unused.
    public float4 weights[kAtheneaLightsMaxGroups];
    /// Paged layout: pool page p's chunk is component p % 4 of row p / 4.
    public uint4 pageChunk[kAtheneaLightsMaxPages / 4];
};

public ConstantBuffer<AtheneaLightsParams> atheneaLights;
public StructuredBuffer<uint> atheneaLightData;
/// The mesh's attribute pool (core/attrib.slang), for its normals.
public StructuredBuffer<uint> atheneaLightsPool;

/// The .athl's index of the mesh's splat `index`, or kAthlNone.
public uint athlIndex(uint index) {
    if (atheneaLights.layout == kLightsPaged) {
        const uint page = index >> 16u;
        if (page >= kAtheneaLightsMaxPages) {
            return kAthlNone;
        }
        const uint chunk = atheneaLights.pageChunk[page / 4u][page % 4u];
        return chunk == kAthlNone ? kAthlNone : (chunk << 16u) | (index & 0xFFFFu);
    }
    const uint merged = atheneaLights.merged;
    return index < merged ? index : index - merged + atheneaLights.splatBase;
}

public float3 athlFloat3(uint at) {
    return asfloat(uint3(atheneaLightData[at], atheneaLightData[at + 1u], atheneaLightData[at + 2u]));
}

/// The f16 at half-index `h` of the data from word `base`.
public float athlHalf(uint base, uint h) {
    const uint w = atheneaLightData[base + (h >> 1u)];
    return f16tof32((h & 1u) != 0u ? (w >> 16u) : (w & 0xFFFFu));
}

/// Component c (of `components`) of group g's `kind` layer at element v.
public uint athlBlock(uint g, uint kind, uint v) {
    const uint table = atheneaLightData[kAthlHeaderWords + kAthlGroupWords * g + 24u + kind];
    if (table == kAthlNone) {
        return kAthlNone;
    }
    const uint chunk = v >> 16u;
    if (chunk >= atheneaLightData[1]) {
        return kAthlNone;
    }
    const uint directory = atheneaLightData[table + chunk];
    if (directory == kAthlNone) {
        return kAthlNone;
    }
    return atheneaLightData[directory + ((v & 0xFFFFu) / kAthlBlockSplats)];
}

public float3 athlRgb(uint g, uint kind, uint v) {
    const uint block = athlBlock(g, kind, v);
    if (block == kAthlNone) {
        return float3(0.0);
    }
    const uint h = 3u * (v % kAthlBlockSplats);
    return float3(athlHalf(block, h), athlHalf(block, h + 1u), athlHalf(block, h + 2u));
}

/// The field (27 halves a splat: 9 coefficients x rgb) towards `dir` (from
/// the eye to the splat), Spark's real harmonics of degree 2.
public float3 athlField(uint g, uint v, float3 dir) {
    const uint block = athlBlock(g, 2u, v);
    if (block == kAthlNone) {
        return float3(0.0);
    }
    const float x = dir.x;
    const float y = dir.y;
    const float z = dir.z;
    float basis[9];
    basis[0] = 0.28209479;
    basis[1] = -0.48860251 * y;
    basis[2] = 0.48860251 * z;
    basis[3] = -0.48860251 * x;
    basis[4] = 1.09254843 * x * y;
    basis[5] = -1.09254843 * y * z;
    basis[6] = 0.31539157 * (2.0 * z * z - x * x - y * y);
    basis[7] = -1.09254843 * x * z;
    basis[8] = 0.54627422 * (x * x - y * y);
    const uint at = 27u * (v % kAthlBlockSplats);
    float3 sum = float3(0.0);
    for (uint c = 0; c < 9u; ++c) {
        sum += basis[c] * float3(athlHalf(block, at + 3u * c), athlHalf(block, at + 3u * c + 1u),
                                 athlHalf(block, at + 3u * c + 2u));
    }
    return sum;
}

/// Lambert's form factor of polygon `p` (AthlStore.pack's polygon) seen
/// from x with normal n: (1/pi) x the integral of the cosine over the
/// polygon, the polygon clipped to the horizon (rust athl.rs
/// polygon_form_factor, the same arithmetic).
public float athlFormFactor(uint p, float3 x, float3 n, bool twoSided) {
    const uint count = min(atheneaLightData[p], 8u);
    if (count < 3u) {
        return 0.0;
    }
    float3 d[8];
    for (uint i = 0; i < 8u; ++i) {
        d[i] = i < count ? athlFloat3(p + 8u + 3u * i) - x : float3(0.0);
    }
    if (!twoSided) {
        const float3 face = cross(d[1] - d[0], d[2] - d[0]);
        if (dot(face, -d[0]) <= 0.0) {
            return 0.0;
        }
    }
    float3 c[9];
    uint m = 0;
    for (uint i = 0; i < count; ++i) {
        const float3 a = d[i];
        const float3 b = d[(i + 1u) % count];
        const float ha = dot(n, a);
        const float hb = dot(n, b);
        if (ha > 0.0) {
            c[m] = a;
            m += 1u;
        }
        if ((ha > 0.0) != (hb > 0.0)) {
            c[m] = a + (ha / (ha - hb)) * (b - a);
            m += 1u;
        }
    }
    if (m < 3u) {
        return 0.0;
    }
    float sum = 0.0;
    for (uint i = 0; i < m; ++i) {
        const float3 a = normalize(c[i]);
        const float3 b = normalize(c[(i + 1u) % m]);
        const float angle = acos(clamp(dot(a, b), -1.0, 1.0));
        const float3 g = cross(a, b);
        const float gl = length(g);
        if (gl > 1e-12) {
            sum += angle * dot(n, g) / gl;
        }
    }
    return abs(sum) / (2.0 * 3.14159265);
}

/// Group g's lens (076): its tint times its beam map towards x.
public float3 athlProfile(uint group, float3 x) {
    const float3 tint = athlFloat3(group + 4u);
    const uint p = atheneaLightData[group + 3u];
    if (p == kAthlNone) {
        return tint;
    }
    const float3 dir = x - athlFloat3(group + 8u);
    const float3 local = normalize(float3(dot(dir, athlFloat3(group + 12u)), dot(dir, athlFloat3(group + 16u)),
                                          dot(dir, athlFloat3(group + 20u))));
    const float lon = atan2(local.x, local.z);
    const float lat = asin(clamp(local.y, -1.0, 1.0));
    const uint w = atheneaLightData[p];
    const uint h = atheneaLightData[p + 1u];
    const float4 sector = asfloat(uint4(atheneaLightData[p + 2u], atheneaLightData[p + 3u],
                                        atheneaLightData[p + 4u], atheneaLightData[p + 5u]));
    const float u = (lon - sector.x) / (sector.y - sector.x);
    const float v = (lat - sector.z) / (sector.w - sector.z);
    if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0) {
        return athlFloat3(p + 6u) * tint;
    }
    const float fx = clamp(u * float(w) - 0.5, 0.0, float(w - 1u));
    const float fy = clamp(v * float(h) - 0.5, 0.0, float(h - 1u));
    const uint x0 = uint(floor(fx));
    const uint y0 = uint(floor(fy));
    const uint x1 = min(x0 + 1u, w - 1u);
    const uint y1 = min(y0 + 1u, h - 1u);
    const float tx = fx - float(x0);
    const float ty = fy - float(y0);
    const uint base = p + kAthlProfileHeadWords;
    const float3 t00 = athlTexel(base, y0 * w + x0);
    const float3 t10 = athlTexel(base, y0 * w + x1);
    const float3 t01 = athlTexel(base, y1 * w + x0);
    const float3 t11 = athlTexel(base, y1 * w + x1);
    return lerp(lerp(t00, t10, tx), lerp(t01, t11, tx), ty) * tint;
}

public float3 athlTexel(uint base, uint k) {
    const uint a = atheneaLightData[base + 2u * k];
    const uint b = atheneaLightData[base + 2u * k + 1u];
    return float3(f16tof32(a & 0xFFFFu), f16tof32(a >> 16u), f16tof32(b & 0xFFFFu));
}

/// Group g's direct light leaving a splat at x (normal n, albedo), per
/// unit of w_k: albedo x sum over its polygons of radiance x form factor,
/// through its lens.
public float3 athlDirect(uint g, float3 x, float3 n, float3 albedo) {
    const uint group = kAthlHeaderWords + kAthlGroupWords * g;
    const uint first = atheneaLightData[group + 1u];
    const uint count = atheneaLightData[group + 2u];
    if (count == 0u) {
        return float3(0.0);
    }
    const bool twoSided = (atheneaLightData[group] & kAthlTwoSided) != 0u;
    const uint polygons = atheneaLightData[2];
    float3 sum = float3(0.0);
    for (uint k = 0; k < count; ++k) {
        const uint p = polygons + kAthlPolygonWords * (first + k);
        float3 normal = n;
        if (dot(normal, normal) < 0.5) {
            // No normal: the splat faces the polygon (an upper bound).
            float3 centre = float3(0.0);
            const uint vc = min(atheneaLightData[p], 8u);
            for (uint i = 0; i < vc; ++i) {
                centre += athlFloat3(p + 8u + 3u * i);
            }
            normal = normalize(centre / float(max(vc, 1u)) - x);
        }
        sum += athlFloat3(p + 2u) * athlFormFactor(p, x, normal, twoSided);
    }
    if (all(sum == float3(0.0))) {
        return sum;
    }
    return albedo * sum * athlProfile(group, x);
}

public struct AtheneaLightsColour : ISplatColour {
    public static float4 colour(float4 rgba, SplatColourInput i) {
        const uint groups = min(atheneaLights.groupCount, kAtheneaLightsMaxGroups);
        if (groups == 0u) {
            return rgba;
        }
        const uint v = athlIndex(i.index);
        if (v == kAthlNone) {
            return rgba;
        }
        const uint terms = atheneaLights.terms;
        // The colour so far, back to linear light (raster.slang
        // atheneaEncodeForDraw's inverse).
        float3 light = pow(max(rgba.rgb, float3(0.0)), float3(2.2));
        const float3 albedo = cloudLight(i.splat.rgba.rgb, atheneaLights.storedLinear != 0u);
        const float3 x = i.splat.center;
        float3 n = float3(0.0);
        const uint normalId = atheneaLights.normalId;
        if (normalId != ATTRIB_NONE && attribResident(atheneaLightsPool, normalId, i.index)) {
            n = unpackNormal(readAttribU(atheneaLightsPool, normalId, i.index, 0u));
        }
        const float3 dir = normalize(x - i.viewObject);
        // A thin sheet of glass the relight drew at athenea's sheet alpha
        // (kSheetAlpha 0.1, its colour scaled by alphaOwn / 0.1 to keep
        // alpha x colour): athenea adds the lamps before that scale
        // (splat_project: colour += lampLight, then colour *= alphaOwn /
        // alpha), so the sheet's own lamp light is scaled the same way.
        const float own = max(min(i.splat.rgba.a, 1.0) - 1.0 / 255.0, 0.0);
        const float sheetScale = (rgba.a == 0.1 && i.splat.rgba.a != rgba.a && own < 0.1) ? own / 0.1 : 1.0;
        for (uint g = 0; g < groups; ++g) {
            const float3 w = atheneaLights.weights[g].rgb;
            if (all(w == float3(0.0))) {
                continue;
            }
            float3 add = float3(0.0);
            if ((terms & kLightsDirect) != 0u) {
                add += athlDirect(g, x, n, albedo);
            }
            if ((terms & kLightsIndirect) != 0u) {
                add += athlRgb(g, 0u, v);
            }
            if ((terms & kLightsEmission) != 0u) {
                add += athlRgb(g, 1u, v);
            }
            if ((terms & kLightsField) != 0u) {
                add += athlField(g, v, dir);
            }
            light += sheetScale * w * add;
        }
        return float4(pow(max(light, float3(0.0)), float3(1.0 / 2.2)), rgba.a);
    }
}

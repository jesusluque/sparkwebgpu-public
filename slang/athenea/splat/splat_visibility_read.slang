// Copyright (c) 2026 jesus luque.
//
// The read of a baked per-part visibility, a frame: one thread a gaussian,
// a factor a light, a product over the parts that are not its own, each
// asked in that part's own current frame. splat_visibility.slang bakes what
// this reads and says why it is baked that way.
module splat_visibility_read;
import athenea.common.packing;
import athenea.common.octahedral;
import athenea.light.lights;
import athenea.splat.splat_visibility_common;

StructuredBuffer<VisibilityPart> parts;
StructuredBuffer<uint>           texels;     // two f16 a word
StructuredBuffer<uint>           ambient;    // two f16 a word: a probe's mean over its directions

// ----------------------------------------------------------------------------
// 3. The read: one thread a gaussian, a factor a light, a product over parts.

struct FactorParams {
    uint  count;         // gaussians in this cloud
    uint  base;          // its first slot in `factors`
    uint  lights;        // lights a factor is measured for
    uint  partCount;
    uint  categoriesLo;  // this cloud's light linking
    uint  categoriesHi;
    uint  skinned;       // 1: parts move by `skinningXforms`; 0: the bind pose is the pose
    uint  pad0;
    // object -> world, rows of a 3x4.
    float w00; float w01; float w02; float w03;
    float w10; float w11; float w12; float w13;
    float w20; float w21; float w22; float w23;
    // The cloud's space into the skeleton's bind space, rows of a 3x4: what
    // the skinner applies before a joint's transform (`geomBindTransform`).
    // A posed position is A . X_k . B . rest, so rest is B^-1 . X_k^-1 . A^-1
    // . posed; A is the identity for a cloud drawn where its skeleton stands.
    float g00; float g01; float g02; float g03;
    float g10; float g11; float g12; float g13;
    float g20; float g21; float g22; float g23;
};

StructuredBuffer<float4>      positions;      // posed, what the frame draws
StructuredBuffer<uint>        ownPart;        // the part each gaussian is of
StructuredBuffer<float4>      skinningXforms; // 4 rows a joint, row-major, bind -> posed
StructuredBuffer<LightRecord> lights;
RWStructuredBuffer<float>     factors;
// WHERE EACH PART'S FIELD WAS BAKED, FROM WHERE IT STANDS NOW: three float4
// rows a part, the affine map from a posed position to the rest space its
// field is in (visPartFrames). The same twelve maps for every gaussian of a
// frame, so they are made once and read -- they were two 3x3 inverses a
// gaussian a part a light, some 260 million a frame on the sparrow.
RWStructuredBuffer<float4>    partFrames;
ConstantBuffer<FactorParams>  factorParams;

float3 toWorld(float3 p) {
    return float3(factorParams.w00 * p.x + factorParams.w01 * p.y + factorParams.w02 * p.z + factorParams.w03,
                  factorParams.w10 * p.x + factorParams.w11 * p.y + factorParams.w12 * p.z + factorParams.w13,
                  factorParams.w20 * p.x + factorParams.w21 * p.y + factorParams.w22 * p.z + factorParams.w23);
}

float3x3 inverse3(float3x3 m) {
    const float3 c0 = float3(m[0][0], m[1][0], m[2][0]);
    const float3 c1 = float3(m[0][1], m[1][1], m[2][1]);
    const float3 c2 = float3(m[0][2], m[1][2], m[2][2]);
    const float3 r0 = cross(c1, c2);
    const float3 r1 = cross(c2, c0);
    const float3 r2 = cross(c0, c1);
    const float det = dot(c0, r0);
    const float inv = abs(det) > 1.0e-20 ? 1.0 / det : 0.0;
    return float3x3(r0 * inv, r1 * inv, r2 * inv);
}

/// One texel of a part's field, as a value.
float texelAt(VisibilityPart part, uint3 cell, uint2 uv) {
    const uint probe = (cell.z * part.grid + cell.y) * part.grid + cell.x;
    const uint t = part.offset + probe * part.octave * part.octave + uv.y * part.octave + uv.x;
    const uint word = texels[t >> 1];
    return f16tof32((t & 1u) != 0u ? (word >> 16) : (word & 0xFFFFu));
}

/// A probe's mean over every direction, for a light that has none.
float ambientTexel(uint part, VisibilityPart field, uint3 cell) {
    const uint probes = field.grid * field.grid * field.grid;
    const uint at = part * probes + (cell.z * field.grid + cell.y) * field.grid + cell.x;
    const uint word = ambient[at >> 1];
    return f16tof32((at & 1u) != 0u ? (word >> 16) : (word & 0xFFFFu));
}

/// The ambient term at a point: trilinear over the probes. Outside the box
/// the part stops nothing.
float ambientAt(uint index, VisibilityPart part, float3 p) {
    const float3 unit = (p - part.boxMin.xyz) / max(part.boxMax.xyz - part.boxMin.xyz, 1.0e-20);
    if (any(unit < 0.0) || any(unit > 1.0)) {
        return 1.0;
    }
    const float gf = float(part.grid);
    const float3 at = clamp(unit * gf - 0.5, 0.0, gf - 1.0);
    const uint3 lo = uint3(floor(at));
    const uint3 hi = min(lo + 1, uint3(part.grid - 1));
    const float3 fr = at - float3(lo);
    float result = 0.0;
    [unroll] for (uint c = 0; c < 8; ++c) {
        const uint3 cell = uint3((c & 1u) != 0u ? hi.x : lo.x, (c & 2u) != 0u ? hi.y : lo.y, (c & 4u) != 0u ? hi.z : lo.z);
        const float wc = ((c & 1u) != 0u ? fr.x : 1.0 - fr.x) * ((c & 2u) != 0u ? fr.y : 1.0 - fr.y) *
                         ((c & 4u) != 0u ? fr.z : 1.0 - fr.z);
        result += wc * ambientTexel(index, part, cell);
    }
    return result;
}

/// The field at a point in a direction, trilinear over the probes and
/// bilinear over the map. Outside the box the part stops nothing.
float fieldAt(VisibilityPart part, float3 p, float3 w) {
    const float3 unit = (p - part.boxMin.xyz) / max(part.boxMax.xyz - part.boxMin.xyz, 1.0e-20);
    if (any(unit < 0.0) || any(unit > 1.0)) {
        return 1.0;
    }
    const float gf = float(part.grid);
    const float3 at = clamp(unit * gf - 0.5, 0.0, gf - 1.0);
    const uint3 lo = uint3(floor(at));
    const uint3 hi = min(lo + 1, uint3(part.grid - 1));
    const float3 fr = at - float3(lo);
    const float of = float(part.octave);
    const float2 ouv = clamp(octEncode(w) * of - 0.5, 0.0, of - 1.0);
    const uint2 ulo = uint2(floor(ouv));
    const uint2 uhi = min(ulo + 1, uint2(part.octave - 1));
    const float2 ufr = ouv - float2(ulo);
    float result = 0.0;
    [unroll] for (uint c = 0; c < 8; ++c) {
        const uint3 cell = uint3((c & 1u) != 0u ? hi.x : lo.x, (c & 2u) != 0u ? hi.y : lo.y, (c & 4u) != 0u ? hi.z : lo.z);
        const float wc = ((c & 1u) != 0u ? fr.x : 1.0 - fr.x) * ((c & 2u) != 0u ? fr.y : 1.0 - fr.y) *
                         ((c & 4u) != 0u ? fr.z : 1.0 - fr.z);
        const float v00 = texelAt(part, cell, ulo);
        const float v10 = texelAt(part, cell, uint2(uhi.x, ulo.y));
        const float v01 = texelAt(part, cell, uint2(ulo.x, uhi.y));
        const float v11 = texelAt(part, cell, uhi);
        result += wc * lerp(lerp(v00, v10, ufr.x), lerp(v01, v11, ufr.x), ufr.y);
    }
    return result;
}

/// The posed-to-rest map of each part: rest = U . X_k^-1 . (posed - t_k) - U . g,
/// with X_k the part's joint (bind -> posed) and U the inverse of the bind's
/// linear part; the identity where nothing moves.
[shader("compute")]
[numthreads(64, 1, 1)]
void visPartFrames(uint3 tid: SV_DispatchThreadID) {
    const uint part = tid.x;
    if (part >= factorParams.partCount) {
        return;
    }
    float3x3 m = float3x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    float3 c = float3(0.0);
    if (factorParams.skinned != 0) {
        const VisibilityPart field = parts[part];
        const uint joint = uint(field.boxMin.w);
        const float4 r0 = skinningXforms[joint * 4];
        const float4 r1 = skinningXforms[joint * 4 + 1];
        const float4 r2 = skinningXforms[joint * 4 + 2];
        const float3x3 back = inverse3(float3x3(r0.xyz, r1.xyz, r2.xyz));
        const float3x3 bindLinear = float3x3(factorParams.g00, factorParams.g01, factorParams.g02,
                                             factorParams.g10, factorParams.g11, factorParams.g12,
                                             factorParams.g20, factorParams.g21, factorParams.g22);
        const float3x3 unbind = inverse3(bindLinear);
        m = mul(unbind, back);
        c = -mul(m, float3(r0.w, r1.w, r2.w)) -
            mul(unbind, float3(factorParams.g03, factorParams.g13, factorParams.g23));
    }
    partFrames[part * 3] = float4(m[0], c.x);
    partFrames[part * 3 + 1] = float4(m[1], c.y);
    partFrames[part * 3 + 2] = float4(m[2], c.z);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void visFactors(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= factorParams.count) {
        return;
    }
    const uint slot = factorParams.base + i;
    const float3 posed = positions[i].xyz;
    const float3 world = toWorld(posed);
    // World directions back into the cloud's own space, once a gaussian.
    const float3x3 fromWorldLinear = inverse3(float3x3(factorParams.w00, factorParams.w01, factorParams.w02,
                                                       factorParams.w10, factorParams.w11, factorParams.w12,
                                                       factorParams.w20, factorParams.w21, factorParams.w22));
    for (uint k = 0; k < factorParams.lights; ++k) {
        float through = 1.0;
        const LightRecord light = lights[k];
        const bool linked = lightLinked(light.lightCategory, factorParams.categoriesLo, factorParams.categoriesHi) &&
                            (light.flags & kLightShadow) != 0;
        if (linked && light.kind == kLightDome) {
            // A dome comes from everywhere: what cuts it is each part's solid
            // angle from here, which the ambient term holds.
            const uint own = ownPart[i];
            for (uint part = 0; part < factorParams.partCount; ++part) {
                if (part == own) continue;
                const VisibilityPart field = parts[part];
                if (field.grid == 0u) continue;
                const float4 f0 = partFrames[part * 3];
                const float4 f1 = partFrames[part * 3 + 1];
                const float4 f2 = partFrames[part * 3 + 2];
                const float3 p = float3(dot(f0.xyz, posed) + f0.w, dot(f1.xyz, posed) + f1.w,
                                        dot(f2.xyz, posed) + f2.w);
                through *= ambientAt(part, field, p);
            }
        } else if (linked && light.kind != kLightDome) {
            const LightSample ls = sampleLight(light, world, float3(0.0, 0.0, 1.0), float2(0.5, 0.5));
            if (ls.valid) {
                // The direction in the cloud's own space, which is the space
                // the fields were baked in before any part moved.
                const float3 wCloud = normalize(mul(fromWorldLinear, ls.wi));
                // A PART DOES NOT SHADOW ITSELF. Its field, read at one of
                // its own gaussians, is the crowd that gaussian stands in --
                // the same thing that made a shadow ray from a splat black,
                // one level up -- and it would darken every part from every
                // light. What a part's field is for is the other parts: the
                // wing on the body. The part's own local shadowing is the
                // term this gives up (docs/decisions.md).
                const uint own = ownPart[i];
                for (uint part = 0; part < factorParams.partCount; ++part) {
                    if (part == own) {
                        continue;
                    }
                    const VisibilityPart field = parts[part];
                    if (field.grid == 0u) {
                        continue;   // nothing stands on this part: it casts nothing
                    }
                    // Back through this part's frame: where the gaussian
                    // and the light would be if the part had not moved, which
                    // is where the field was baked (visPartFrames).
                    const float4 f0 = partFrames[part * 3];
                    const float4 f1 = partFrames[part * 3 + 1];
                    const float4 f2 = partFrames[part * 3 + 2];
                    const float3 p = float3(dot(f0.xyz, posed) + f0.w, dot(f1.xyz, posed) + f1.w,
                                            dot(f2.xyz, posed) + f2.w);
                    const float3 w = factorParams.skinned != 0
                                         ? normalize(float3(dot(f0.xyz, wCloud), dot(f1.xyz, wCloud),
                                                            dot(f2.xyz, wCloud)))
                                         : wCloud;
                    through *= fieldAt(field, p, w);
                }
            }
        }
        factors[slot * factorParams.lights + k] = through;
    }
}


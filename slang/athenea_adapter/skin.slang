// athenea's skinner on Spark's splats (plan-athenea.md, phase 5): a cloud a
// skeleton carries, posed each frame before the relight and generate.
//
// The math is athenea's `scene/splat_skin.slang` `splatSkin` as vendored
// (txf 89a04d9: 7dff879 "A skinned gaussian deforms by the whole Jacobian of
// its blend", with athenea's own packing from common/packing.slang), line for
// line in `skinSplat` below. It is restated here rather than dispatched as it
// is because that kernel binds eleven storage buffers (its shutter's second
// pose and the motion it writes among them) and Chrome on a Mac gives a stage
// ten; test/gpu/atheneaSkin.test.ts runs athenea's kernel itself (on a device
// with the adapter's limits) on the same splats and checks this against it.
//
// What differs is only where the numbers live:
//   - the rest splat is Spark's (skinRest: a copy of the source as loaded,
//     ExtSplats or PackedSplats), and the posed one goes back to the source
//     (skinSrc) in the same layout, so the relight, generate, the sort and
//     the draw read the pose without knowing;
//   - the influences are athenea's packed words (common/influences.slang at
//     t22 390670e: joint u16 | weight unorm16), and the weights' gradients
//     its halves, as the .athc's SKIN section keeps them -- the attributes
//     skinInfluences and skinGradients of the mesh's AttribPool;
//   - the shading normal (attribute normalOct, athenea's packNormal word) is
//     turned into the pool from the rest normals (skinRestNormals);
//   - the cached skinning transforms make the skeleton's space the cloud's
//     (athenea's carryCloud leaves skelToWorld and worldToPrim identity), so
//     those two matrices are not taken; no shutter (motion) is written.
//
// flags: kSkinJacobian off takes the blend of the joints' linear parts alone
// (athenea before 7dff879, and Spark's linear-blend SplatSkinning), which is
// also what a cloud without gradients gets.
module skin;

import core.math;
import core.splat_types;
import core.packed;
import core.ext;
import core.attrib;
import athenea.common.packing;

public static const uint kSkinExt = 1u;
public static const uint kSkinJacobian = 2u;
public static const uint kSkinNormals = 4u;
/// Splats a dispatch row covers (dispatched as rows past 65535 workgroups).
public static const uint kSkinDispatchRow = 65535u * 256u;

struct AtheneaSkinParams {
    uint   count;          ///< splats in the source
    uint   flags;          ///< kSkin*
    uint   perSplat;       ///< influences a splat
    uint   gradientWords;  ///< 0, or perSplat - 1
    uint   influences;     ///< pool attribute ids (ATTRIB_NONE: none)
    uint   gradients;
    uint   normals;
    uint   joints;         ///< transforms in skinXforms
    float4 encoding;       ///< PackedSplats encoding (packed sources)
    float4 geomBind0;      ///< rows of a 4x4, vectors on the right (USD's, transposed)
    float4 geomBind1;
    float4 geomBind2;
    float4 geomBind3;
};

StructuredBuffer<uint4>             skinRest;
RWStructuredBuffer<uint4>           skinSrc;
RWStructuredBuffer<uint>            skinPool;
StructuredBuffer<uint>              skinRestNormals;
StructuredBuffer<float4>            skinXforms;      // 4 rows a joint, vectors on the right
ConstantBuffer<AtheneaSkinParams>   skinParams;

static const float kEps = 1.0e-5;

bool skinFlag(uint f) { return (skinParams.flags & f) != 0u; }

float3x3 linearOf(float4x4 m) {
    return float3x3(m[0].xyz, m[1].xyz, m[2].xyz);
}

uint poolWord(uint id, uint i, uint k) {
    return skinPool[attribBase(skinPool, attribDesc(skinPool, id), i) + k];
}

/// The posed splat: centre, the two in-plane sizes, its frame's
/// quaternion, and the turned shading normal (when asked for).
struct PosedSplat {
    float3 center;
    float2 sizes;
    float4 quaternion;
    float3 normal;
};

/// splat_skin.slang's splatSkin body, a splat: `center`, `quaternion` and
/// `scales` as Spark holds them at rest.
PosedSplat skinSplat(uint at, float3 center, float4 quaternion, float3 scale, float3 restNormal) {
    const float4x4 geomBind = float4x4(skinParams.geomBind0, skinParams.geomBind1, skinParams.geomBind2,
                                       skinParams.geomBind3);
    const float4 bound = mul(geomBind, float4(center, 1.0));
    float4   moved = float4(0.0, 0.0, 0.0, 0.0);
    float3x3 blended = float3x3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    float3   alongU = float3(0.0, 0.0, 0.0);
    float3   alongV = float3(0.0, 0.0, 0.0);
    float2   gradientSum = float2(0.0, 0.0);
    float2   lastGradient = float2(0.0, 0.0);
    float    total = 0.0;
    const bool gradientsIn = skinFlag(kSkinJacobian) && skinParams.gradientWords > 0u &&
                             skinParams.gradients != ATTRIB_NONE;
    const uint gradientWords = skinParams.gradientWords;
    for (uint k = 0; k < skinParams.perSplat; ++k) {
        // influenceAt, the packed layout.
        const uint word = poolWord(skinParams.influences, at, k);
        const uint joint = word & 0xFFFFu;
        const float weight = float(word >> 16) / 65535.0;
        float2 gradient = float2(0.0, 0.0);
        if (gradientsIn) {
            if (k < gradientWords) {
                const uint g = poolWord(skinParams.gradients, at, k);
                gradient = float2(lowHalf(g), highHalf(g));
                lastGradient -= gradient;
            } else {
                gradient = lastGradient;
            }
            if (any(isnan(gradient)) || any(isinf(gradient))) {
                gradient = float2(0.0, 0.0);
            }
        }
        const bool pulls = any(gradient != float2(0.0, 0.0));
        if ((!(weight > kEps) && !pulls) || joint >= skinParams.joints) {
            continue;
        }
        const float4x4 xform = float4x4(skinXforms[joint * 4], skinXforms[joint * 4 + 1], skinXforms[joint * 4 + 2],
                                        skinXforms[joint * 4 + 3]);
        const float4 carried = mul(xform, bound);
        if (pulls) {
            alongU += carried.xyz * gradient.x;
            alongV += carried.xyz * gradient.y;
            gradientSum += gradient;
        }
        if (!(weight > kEps)) {
            continue;
        }
        moved += carried * weight;
        blended += linearOf(xform) * weight;
        total += weight;
    }
    if (!(total > kEps)) {
        moved = bound;
        blended = float3x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        alongU = float3(0.0, 0.0, 0.0);
        alongV = float3(0.0, 0.0, 0.0);
        gradientSum = float2(0.0, 0.0);
    }
    alongU -= moved.xyz * gradientSum.x;
    alongV -= moved.xyz * gradientSum.y;
    PosedSplat out;
    out.center = moved.xyz;

    const float3x3 chain = mul(blended, linearOf(geomBind));
    const float3x3 axes = axesOfQuaternion(quaternion);
    const float3 restU = float3(axes[0][0], axes[1][0], axes[2][0]);
    const float3 restV = float3(axes[0][1], axes[1][1], axes[2][1]);
    const float3 restN = float3(axes[0][2], axes[1][2], axes[2][2]);
    const float3 jacobianU = mul(chain, restU) + alongU;
    const float3 jacobianV = mul(chain, restV) + alongV;
    const float3 movedU = jacobianU * scale.x;
    const float3 movedV = jacobianV * scale.y;

    const float lengthU = length(movedU);
    float3 e1 = lengthU > 1.0e-20 ? movedU / lengthU : restU;
    float3 e2 = movedV - e1 * dot(e1, movedV);
    const float across = length(e2);
    e2 = across > 1.0e-20 * max(length(movedV), 1.0e-20) && across > 1.0e-30
             ? e2 / across
             : normalize(cross(restN, e1) + float3(1.0e-6, 0.0, 0.0));
    const float k00 = lengthU;
    const float k01 = dot(e1, movedV);
    const float k11 = dot(e2, movedV);
    const float c00 = k00 * k00 + k01 * k01;
    const float c01 = k01 * k11;
    const float c11 = k11 * k11;
    const float mean = 0.5 * (c00 + c11);
    const float spread = 0.5 * (c00 - c11);
    const float radius = sqrt(spread * spread + c01 * c01);
    const float major = mean + radius;
    const float determinant = k00 * k11;
    const float minor = major > 0.0 ? determinant * determinant / major : 0.0;
    float angle = abs(c01) > 1.0e-6 * (c00 + c11) ? 0.5 * atan2(2.0 * c01, c00 - c11)
                                                   : (c00 >= c11 ? 0.0 : 1.5707963);
    float sizeU = sqrt(major);
    float sizeV = sqrt(minor);
    if (abs(angle) > 0.78539816) {
        angle -= angle > 0.0 ? 1.5707963 : -1.5707963;
        sizeU = sqrt(minor);
        sizeV = sqrt(major);
    }
    const float cosine = cos(angle);
    const float sine = sin(angle);
    const float3 u = normalize(e1 * cosine + e2 * sine);
    const float3 v = normalize(e2 * cosine - e1 * sine);
    const float3 n = cross(u, v);
    const float3x3 frame = float3x3(u.x, v.x, n.x, u.y, v.y, n.y, u.z, v.z, n.z);
    out.quaternion = quaternionOfAxes(frame);
    out.sizes = float2(sizeU > 1.0e-20 ? sizeU : scale.x, sizeV > 1.0e-20 ? sizeV : scale.y);

    out.normal = restNormal;
    if (skinFlag(kSkinNormals)) {
        const float3 helper = abs(restNormal.x) < 0.9 ? float3(1.0, 0.0, 0.0) : float3(0.0, 1.0, 0.0);
        const float3 a = normalize(cross(helper, restNormal));
        const float3 b = cross(restNormal, a);
        const float3 turnedA = mul(chain, a) + (alongU * dot(a, restU) + alongV * dot(a, restV));
        const float3 turnedB = mul(chain, b) + (alongU * dot(b, restU) + alongV * dot(b, restV));
        const float3 turned = cross(turnedA, turnedB);
        out.normal = dot(turned, turned) > 1.0e-30 ? turned : restNormal;
    }
    return out;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void atheneaSkin(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x + tid.y * kSkinDispatchRow;
    if (at >= skinParams.count) {
        return;
    }
    const bool ext = skinFlag(kSkinExt);
    float3 center;
    float3 scales;
    float4 quaternion;
    float4 rgba;
    uint4 restA = skinRest[ext ? 2u * at : at];
    uint4 restB = ext ? skinRest[2u * at + 1u] : uint4(0u);
    if (ext) {
        unpackSplatExt(restA, restB, center, scales, quaternion, rgba);
    } else {
        unpackSplat(restA, center, scales, quaternion, rgba, skinParams.encoding);
    }
    // An inactive splat (no scale) and a splat nothing carries stay as they are.
    if (all(scales == float3(0.0)) || skinParams.influences == ATTRIB_NONE) {
        if (ext) {
            skinSrc[2u * at] = restA;
            skinSrc[2u * at + 1u] = restB;
        } else {
            skinSrc[at] = restA;
        }
        return;
    }
    const bool normals = skinFlag(kSkinNormals) && skinParams.normals != ATTRIB_NONE;
    const float3 restNormal = normals ? unpackNormal(skinRestNormals[at]) : float3(0.0, 0.0, 1.0);
    const PosedSplat p = skinSplat(at, center, quaternion, scales, restNormal);
    // The third size and the colour are the conversion's and stay.
    const float3 posedScales = float3(p.sizes, scales.z);
    if (ext) {
        uint4 a;
        uint4 b;
        packSplatExt(a, b, p.center, posedScales, p.quaternion, rgba);
        skinSrc[2u * at] = a;
        skinSrc[2u * at + 1u] = b;
    } else {
        uint4 w = packSplat(p.center, posedScales, p.quaternion, rgba, skinParams.encoding);
        w.x = restA.x;
        skinSrc[at] = w;
    }
    if (normals) {
        skinPool[attribBase(skinPool, attribDesc(skinPool, skinParams.normals), at)] = packNormal(p.normal);
    }
}

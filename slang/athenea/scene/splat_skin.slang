// Copyright (c) 2026 jesus luque.
//
// A CLOUD CARRIED BY A SKELETON, one thread a gaussian.
//
// The cloud is built once, in the bind pose, and each gaussian keeps the four
// joints that move it (mesh2splat's conversion blends them from the corners of
// the triangle it stands on). Here those joints are applied: the position by
// exactly the linear blend `geom/skinning.slang` does for a mesh point, and
// the frame by the Jacobian of that same map -- which is what makes a
// gaussian stretch with its triangle instead of sliding along beside it.
//
// THE JACOBIAN, WHOLE. With `q = G p` the point in bind space, `X_j q = A_j q
// + t_j` a joint's transform and `M` the chain out to the prim, the map is
// `p' = M sum_j w_j(p) X_j q`, and its derivative is
//
//     J = L(M) [ sum_j w_j A_j L(G)  +  sum_j (X_j q) grad w_j^T ].
//
// The first term is the blend of the joints' linear parts, which is all this
// kernel took until the weights' gradients travelled with the cloud. The
// second is what a weight changing across the gaussian does: where two joints
// disagree about where the point goes, the gaussian is stretched (or
// squeezed) between them -- on the sparrow's wing hand by 15 to 50 percent at
// the 90th and 99th percentiles. Only the gradient's two components in the
// gaussian's own plane act (the third axis is a disc's), so a conversion
// keeps them, in the rest axes, for all but the last joint
// (`weightGradients`; the weights sum to one, so the gradients sum to zero).
// A cloud without them is carried by the first term alone, as before.
//
// What it is for: an animated cloud that is three megabytes rather than three
// hundred, exact at every instant of the range rather than at the frames
// somebody sampled, and whose buffers do not change identity from one frame to
// the next -- so a ray tracer refits rather than rebuilds.
module splat_skin;
import athenea.common.packing;

struct SplatSkinParams {
    uint   count;          ///< gaussians
    uint   perSplat;       ///< influences each carries
    uint   motionOut;      ///< 1: also write what the shutter moved this gaussian
    uint   normalsOut;     ///< 1: the cloud keeps shading normals, and they are turned too
    uint   gradientsIn;    ///< 1: `weightGradients` holds the weights' tangential gradients
    uint   pad0;
    uint   pad1;
    uint   pad2;
    float4 geomBind0;      ///< rows of a 4x4, as geom/skinning.slang takes them
    float4 geomBind1;
    float4 geomBind2;
    float4 geomBind3;
    float4 skelToWorld0;
    float4 skelToWorld1;
    float4 skelToWorld2;
    float4 skelToWorld3;
    float4 worldToPrim0;
    float4 worldToPrim1;
    float4 worldToPrim2;
    float4 worldToPrim3;
};

StructuredBuffer<float4>   restPositions;   // xyz, w opacity
StructuredBuffer<uint>     restShape;       // 4 words: rotation, log scales, base colour
StructuredBuffer<float2>   influences;      // (joint, weight), perSplat a gaussian
StructuredBuffer<float4>   skinningXforms;  // 4 rows a joint (row-major 4x4)
// The same joints at the shutter's other end. Bound to `skinningXforms` when
// nothing asks for motion, since a name a shader declares must be bound.
StructuredBuffer<float4>   skinningXformsEnd;
// What the shutter moved this gaussian, in the cloud's own space: two words
// of halves each, which splat_project turns into a screen displacement. The
// blend is what `shape` is packed with, so this is the house convention and
// not a new one.
RWStructuredBuffer<uint>   motion;
RWStructuredBuffer<float4> positions;
RWStructuredBuffer<uint>   shape;
// The shading normal a gaussian keeps apart from its frame (packing's
// `packNormal`), at rest and posed. Bound to `restShape` and `shape` when the
// cloud has none, since a name a shader declares must be bound.
StructuredBuffer<uint>     restNormals;
RWStructuredBuffer<uint>   normals;
// THE WEIGHTS' GRADIENTS, in the gaussian's rest axes: `perSplat - 1` words a
// gaussian, each two halves (d w_k / d restU, d w_k / d restV); the last
// joint's is minus the sum of the others'. Bound to `restShape` when the cloud
// has none (`gradientsIn` 0), since a name a shader declares must be bound.
StructuredBuffer<uint>     weightGradients;
ConstantBuffer<SplatSkinParams> params;

static const float kEps = 1.0e-5;

float4x4 rows4(float4 a, float4 b, float4 c, float4 d) {
    return float4x4(a, b, c, d);
}

float3x3 linearOf(float4x4 m) {
    return float3x3(m[0].xyz, m[1].xyz, m[2].xyz);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void splatSkin(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    if (at >= params.count) {
        return;
    }
    const uint row = at * 4;
    const float4 rest = restPositions[at];
    const float4x4 geomBind = rows4(params.geomBind0, params.geomBind1, params.geomBind2, params.geomBind3);
    const float4x4 skelToWorld =
        rows4(params.skelToWorld0, params.skelToWorld1, params.skelToWorld2, params.skelToWorld3);
    const float4x4 worldToPrim =
        rows4(params.worldToPrim0, params.worldToPrim1, params.worldToPrim2, params.worldToPrim3);

    // The blend, as a whole matrix: the position needs it applied to a point
    // and the frame needs its linear part applied to two axes, and taking one
    // sum for both is what keeps them from disagreeing.
    const float4 bound = mul(geomBind, float4(rest.xyz, 1.0));
    float4   moved = float4(0.0, 0.0, 0.0, 0.0);
    // The same blend under the shutter's other pose. Only the point is taken
    // there, never the frame: a blurred gaussian is smeared along its path,
    // and what its rotation did on the way is the arc a straight smear
    // straightens anyway.
    float4   movedEnd = float4(0.0, 0.0, 0.0, 0.0);
    float3x3 blended = float3x3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    // The elastic term, `sum_k (X_k q) g_k^T` taken on the two rest axes:
    // `alongU = sum_k (X_k q) g_k.x`, `alongV` likewise. `gradientSum` is what
    // the stored gradients add up to (zero, but for the halves' rounding),
    // so the term can be taken relative to the blended point below.
    float3   alongU = float3(0.0, 0.0, 0.0);
    float3   alongV = float3(0.0, 0.0, 0.0);
    float2   gradientSum = float2(0.0, 0.0);
    float2   lastGradient = float2(0.0, 0.0);
    float    total = 0.0;
    const uint offset = at * params.perSplat;
    const uint gradientWords = params.perSplat > 0 ? params.perSplat - 1 : 0;
    for (uint k = 0; k < params.perSplat; ++k) {
        const float2 influence = influences[offset + k];
        const float weight = influence.y;
        float2 gradient = float2(0.0, 0.0);
        if (params.gradientsIn != 0) {
            if (k < gradientWords) {
                const uint word = weightGradients[at * gradientWords + k];
                gradient = float2(lowHalf(word), highHalf(word));
                lastGradient -= gradient;
            } else {
                gradient = lastGradient;
            }
            if (any(isnan(gradient)) || any(isinf(gradient))) {
                gradient = float2(0.0, 0.0);
            }
        }
        // A joint of no weight here may still have a gradient -- a point on
        // the edge where its corner's share runs out -- and that pulls too.
        const bool pulls = any(gradient != float2(0.0, 0.0));
        if (!(weight > kEps) && !pulls) {
            continue;
        }
        const uint joint = uint(influence.x);
        const float4x4 xform = rows4(skinningXforms[joint * 4], skinningXforms[joint * 4 + 1],
                                     skinningXforms[joint * 4 + 2], skinningXforms[joint * 4 + 3]);
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
        if (params.motionOut != 0) {
            // The influence and the joint are already in hand, which is what
            // makes the second pose cost a third of a kernel and not another.
            const float4x4 xformEnd = rows4(skinningXformsEnd[joint * 4], skinningXformsEnd[joint * 4 + 1],
                                            skinningXformsEnd[joint * 4 + 2], skinningXformsEnd[joint * 4 + 3]);
            movedEnd += mul(xformEnd, bound) * weight;
        }
    }
    if (!(total > kEps)) {
        // Carried by nothing: it stays where the bind pose left it, taken
        // through the same two transforms so it lands in the same space.
        moved = bound;
        movedEnd = bound;
        blended = float3x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        alongU = float3(0.0, 0.0, 0.0);
        alongV = float3(0.0, 0.0, 0.0);
        gradientSum = float2(0.0, 0.0);
    }
    // Relative to the blended point: `sum_k (X_k q - y) g_k` is the same term
    // for any `y` when the gradients sum to zero, and taken about the point
    // itself it is a difference of nearby points rather than of two large
    // ones -- what keeps a cloud far from its origin exact in float.
    alongU -= moved.xyz * gradientSum.x;
    alongV -= moved.xyz * gradientSum.y;
    const float3 world = mul(skelToWorld, float4(moved.xyz, 1.0)).xyz;
    positions[at] = float4(mul(worldToPrim, float4(world, 1.0)).xyz, rest.w);
    if (params.motionOut != 0) {
        const float3 worldEnd = mul(skelToWorld, float4(movedEnd.xyz, 1.0)).xyz;
        const float3 step = mul(worldToPrim, float4(worldEnd, 1.0)).xyz - positions[at].xyz;
        motion[at * 2] = packHalves(step.x, step.y);
        motion[at * 2 + 1] = packHalves(step.z, 0.0);
    }

    // THE FRAME: the Jacobian carries the two axes the gaussian spreads
    // along, and the posed covariance is `J E S^2 E^T J^T` with `E` those
    // axes. Its two in-plane axes and sizes are that covariance's
    // eigenvectors and the square roots of its eigenvalues -- exactly, in the
    // plane the two carried axes span. Squaring the carried axes up instead
    // (Gram-Schmidt, which this was) keeps the first one's direction and
    // throws the shear away, and the elastic term above is mostly shear.
    const float3x3 toPrim = mul(linearOf(worldToPrim), linearOf(skelToWorld));
    const float3x3 chain = mul(toPrim, mul(blended, linearOf(geomBind)));
    const float3 scale = exp(float3(lowHalf(restShape[row + 1]), highHalf(restShape[row + 1]),
                                    lowHalf(restShape[row + 2])));
    const float3x3 axes = axesOfQuaternion(decodeQuaternion(restShape[row]));
    const float3 restU = float3(axes[0][0], axes[1][0], axes[2][0]);
    const float3 restV = float3(axes[0][1], axes[1][1], axes[2][1]);
    const float3 restN = float3(axes[0][2], axes[1][2], axes[2][2]);
    // J's columns on the rest frame: the blend's part, and the elastic one
    // (`d w / d restU` and `d w / d restV` are what was stored; the normal's
    // own component is zero, a disc's third axis does not see it).
    const float3 jacobianU = mul(chain, restU) + mul(toPrim, alongU);
    const float3 jacobianV = mul(chain, restV) + mul(toPrim, alongV);
    const float3 movedU = jacobianU * scale.x;
    const float3 movedV = jacobianV * scale.y;

    // An orthonormal basis of their plane, `e1` along the first, and the two
    // carried axes in it: `[a b] = [e1 e2] K`, K upper triangular.
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
    // C = K K^T, symmetric 2x2.
    const float c00 = k00 * k00 + k01 * k01;
    const float c01 = k01 * k11;
    const float c11 = k11 * k11;
    const float mean = 0.5 * (c00 + c11);
    const float spread = 0.5 * (c00 - c11);
    const float radius = sqrt(spread * spread + c01 * c01);
    const float major = mean + radius;
    // The smaller one from the determinant, `(k00 k11)^2`, not as `mean -
    // radius`: a thin gaussian's two would cancel to nothing in float.
    const float determinant = k00 * k11;
    const float minor = major > 0.0 ? determinant * determinant / major : 0.0;
    // The angle of the major axis from e1. Where the off-diagonal term is
    // float's rounding and nothing more -- a rigid joint, a gaussian at rest
    // -- the axes are e1 and e2 exactly, so a round sized one is not spun.
    float angle = abs(c01) > 1.0e-6 * (c00 + c11) ? 0.5 * atan2(2.0 * c01, c00 - c11) : (c00 >= c11 ? 0.0 : 1.5707963);
    // The first axis is the eigenvector nearer the first carried one, so a
    // gaussian whose second size is its larger keeps it second.
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

    shape[row] = encodeQuaternion(quaternionOfAxes(frame));
    shape[row + 1] = packHalves(log(max(sizeU > 1.0e-20 ? sizeU : scale.x, 1.0e-20)),
                                log(max(sizeV > 1.0e-20 ? sizeV : scale.y, 1.0e-20)));
    // The third size and the base colour are the conversion's and stay: only
    // the two halves of this word that hold the scale are rewritten.
    shape[row + 2] = restShape[row + 2];
    shape[row + 3] = restShape[row + 3];

    // THE SHADING NORMAL, turned by the same Jacobian as the frame and in the
    // same way: a normal goes as the cross product of two directions in its
    // surface, and `(J a) x (J b) = cof(J) (a x b)`, which is what the disc's
    // own axis above is too (`cross(u, v)`). So the two stay on the same side
    // of each other through a shear, and a limb's relief bends with the limb
    // rather than staying where the bind pose left the light. J is the whole
    // of it: the blend's part on any direction, and the elastic part on the
    // direction's components along the rest axes the gradients were kept in.
    if (params.normalsOut != 0) {
        const float3 rest = unpackNormal(restNormals[at]);
        const float3 helper = abs(rest.x) < 0.9 ? float3(1.0, 0.0, 0.0) : float3(0.0, 1.0, 0.0);
        const float3 a = normalize(cross(helper, rest));
        const float3 b = cross(rest, a);
        const float3 turnedA = mul(chain, a) + mul(toPrim, alongU * dot(a, restU) + alongV * dot(a, restV));
        const float3 turnedB = mul(chain, b) + mul(toPrim, alongU * dot(b, restU) + alongV * dot(b, restV));
        const float3 turned = cross(turnedA, turnedB);
        normals[at] = packNormal(dot(turned, turned) > 1.0e-30 ? turned : rest);
    }
}

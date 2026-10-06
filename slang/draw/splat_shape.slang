// A splat's screen footprint and its fragment alpha, shared by the quad
// draw (draw/splat_draw.slang) and the compute tile rasterizer
// (tiles/tile_raster.slang), so both render the same image: the projection
// and 2D covariance, the anti-alias blur, LOD opacity, falloff and minAlpha.
// Ported from src/shaders/splatVertex.glsl and splatFragment.glsl, for
// WebGPU clip space (depth in [0, 1]).
//
// Splats are read from the accumulator in its packed (1 x uint4) or ext
// (2 x uint4) layout, the one kernels/generate.slang writes. Matrices come
// as float4 columns: no layout rules to get wrong between Slang, WGSL and
// the host. mul(M, v) below means the GLSL M * v.

import core.math;
import core.splat_types;
import core.packed;
import core.ext;

static const uint DRAW_EXT = 1u;            // accumulator is ext (else packed)
static const uint DRAW_COV = 2u;            // covariance splats
static const uint DRAW_2DGS = 4u;           // flat splats when a scale is 0
static const uint DRAW_LOD_INFLATE = 8u;    // trade LOD opacity for size
static const uint DRAW_ORTHOGRAPHIC = 16u;
static const uint DRAW_ENCODE_LINEAR = 32u; // sRGB colours to linear output
static const uint DRAW_PREMULTIPLIED = 64u; // write premultiplied alpha
static const uint DRAW_DISK_CLIP = 128u;    // portal disk clip (diskCenter)

struct DrawParams {
    float4 proj0;   // projection matrix columns
    float4 proj1;
    float4 proj2;
    float4 proj3;
    float4 renderToViewQuat;
    float4 renderToViewPos;   // xyz
    float4 renderToViewBasis0; // columns, for covariance splats
    float4 renderToViewBasis1;
    float4 renderToViewBasis2;

    float renderWidth;
    float renderHeight;
    float maxStdDev;
    float minPixelRadius;

    float maxPixelRadius;
    float minAlpha;
    float blurAmount;
    float preBlurAmount;

    float focalDistance;
    float apertureAngle;
    float clipXY;
    float focalAdjustment;

    float falloff;
    uint flags;
    uint pad0;
    uint pad1;

    // DRAW_DISK_CLIP, as SparkPortals' DISK_PORTAL_FRAGMENT_SHADER: a disk in
    // view space, center xyz and radius w (> 0: draw only what is behind it,
    // seen through it; < 0: drop what is behind it), normal xyz, w two-sided.
    float4 diskCenter;
    float4 diskNormal;
};

StructuredBuffer<uint4> splats;
ConstantBuffer<DrawParams> params;

bool hasFlag(uint f) { return (params.flags & f) != 0u; }

float4 project(float3 v) {
    return params.proj0 * v.x + params.proj1 * v.y + params.proj2 * v.z + params.proj3;
}

float3x3 fromColumns(float3 c0, float3 c1, float3 c2) {
    return transpose(float3x3(c0, c1, c2));
}

// A splat on screen. Its quad spans clipCenter + corner.x * axis1 +
// corner.y * axis2 (corners in [-1, 1]^2, axes in scaled pixels: the render
// size times focalAdjustment), where the splat-space coordinate is
// corner * adjustedStdDev.
struct SplatShape {
    float4 rgba;
    float adjustedStdDev;
    float4 clipCenter;
    float3 viewCenter;
    float2 axis1;
    float2 axis2;
    // 2DGS (DRAW_2DGS with a zero scale): the quad is the flat splat in view
    // space, viewCenter + viewQuaternion * (offset by scales); axes unset.
    bool flat;
    float4 viewQuaternion;
    float3 scales;
    bool3 zeroScales;
};

// The screen footprint of accumulator splat `splatIndex`; false when it
// draws nothing.
bool projectSplat(uint splatIndex, out SplatShape s) {
    s.rgba = float4(0.0);
    s.adjustedStdDev = 0.0;
    s.clipCenter = float4(0.0);
    s.viewCenter = float3(0.0);
    s.axis1 = float2(0.0);
    s.axis2 = float2(0.0);
    s.flat = false;
    s.viewQuaternion = float4(0.0, 0.0, 0.0, 1.0);
    s.scales = float3(0.0);
    s.zeroScales = bool3(false);

    float3 center, scales, xxyyzz, xyxzyz;
    float4 quaternion, rgba;
    bool3 zeroScales = bool3(false);
    bool cov = hasFlag(DRAW_COV);

    if (hasFlag(DRAW_EXT)) {
        uint4 a = splats[2u * splatIndex];
        float alpha = unpackSplatExtCenterAlpha(a).w;
        if (alpha == 0.0 || alpha < params.minAlpha) {
            return false;
        }
        uint4 b = splats[2u * splatIndex + 1u];
        if (!cov) {
            unpackSplatExt(a, b, center, scales, quaternion, rgba);
            zeroScales = scales == float3(0.0);
            if (all(zeroScales)) {
                return false;
            }
        } else {
            unpackSplatExtCov(a, b, center, rgba, xxyyzz, xyxzyz);
            if (all(xxyyzz == float3(0.0)) && all(xyxzyz == float3(0.0))) {
                return false;
            }
        }
    } else {
        uint4 p = splats[splatIndex];
        if (!cov) {
            unpackSplat(p, center, scales, quaternion, rgba);
            zeroScales = scales == float3(0.0);
            if (all(zeroScales)) {
                return false;
            }
        } else {
            unpackSplatCov(p, center, rgba, xxyyzz, xyxzyz);
            if (all(xxyyzz == float3(0.0)) && all(xyxzyz == float3(0.0))) {
                return false;
            }
        }
        // The packed accumulator stores alpha / 2.
        rgba.a *= 2.0;
        if (rgba.a == 0.0 || rgba.a < params.minAlpha) {
            return false;
        }
    }

    // Like the reference 3DGS rasterizer, clamp the SH colour to positive.
    rgba.rgb = max(rgba.rgb, float3(0.0));

    float adjustedStdDev = params.maxStdDev;
    if (rgba.a > 1.0) {
        // LOD opacity: stretch 1..2 to 1..5.
        rgba.a = min(rgba.a * 4.0 - 3.0, 5.0);
        if (hasFlag(DRAW_LOD_INFLATE)) {
            float opacity = exp((rgba.a * rgba.a - 1.0) / 2.718281828459045);
            scales *= pow(opacity, 1.0 / 3.0);
            rgba.a = 1.0;
        }
        adjustedStdDev = params.maxStdDev + 0.7 * (rgba.a - 1.0);
    }

    float3x3 basis = fromColumns(params.renderToViewBasis0.xyz, params.renderToViewBasis1.xyz, params.renderToViewBasis2.xyz);
    float3 viewCenter = (cov ? mul(basis, center) : quatVec(params.renderToViewQuat, center)) + params.renderToViewPos.xyz;
    if (viewCenter.z >= 0.0) {
        return false;
    }
    float4 clipCenter = project(viewCenter);
    if (clipCenter.z < 0.0 || clipCenter.z >= clipCenter.w) {
        return false;
    }
    float clip = params.clipXY * clipCenter.w;
    if (abs(clipCenter.x) > clip || abs(clipCenter.y) > clip) {
        return false;
    }

    s.rgba = rgba;
    s.adjustedStdDev = adjustedStdDev;
    s.clipCenter = clipCenter;
    s.viewCenter = viewCenter;

    float3x3 cov3D;
    if (!cov) {
        float4 viewQuaternion = quatQuat(params.renderToViewQuat, quaternion);
        if (hasFlag(DRAW_2DGS) && any(zeroScales)) {
            s.flat = true;
            s.viewQuaternion = viewQuaternion;
            s.scales = scales;
            s.zeroScales = zeroScales;
            return true;
        }
        float3x3 RS = scaleQuaternionToMatrix(scales, viewQuaternion);
        cov3D = mul(RS, transpose(RS));
    } else {
        float3x3 c = float3x3(
            xxyyzz.x, xyxzyz.x, xyxzyz.y,
            xyxzyz.x, xxyyzz.y, xyxzyz.z,
            xyxzyz.y, xyxzyz.z, xxyyzz.z);
        cov3D = mul(basis, mul(c, transpose(basis)));
    }

    // Jacobian of the projection at the centre. The GLSL builds it from
    // columns (J1.x, 0, J2.x), (0, J1.y, J2.y), 0; this is the same matrix.
    float2 scaledRenderSize = float2(params.renderWidth, params.renderHeight) * params.focalAdjustment;
    float2 focal = 0.5 * scaledRenderSize * float2(params.proj0.x, params.proj1.y);
    float3x3 J;
    if (hasFlag(DRAW_ORTHOGRAPHIC)) {
        J = float3x3(focal.x, 0.0, 0.0, 0.0, focal.y, 0.0, 0.0, 0.0, 0.0);
    } else {
        float invZ = 1.0 / viewCenter.z;
        float2 J1 = focal * invZ;
        float2 J2 = -(J1 * viewCenter.xy) * invZ;
        J = float3x3(J1.x, 0.0, 0.0, 0.0, J1.y, 0.0, J2.x, J2.y, 0.0);
    }
    float3x3 cov2D = mul(transpose(J), mul(cov3D, J));
    float a = cov2D[0][0] + params.preBlurAmount;
    float d = cov2D[1][1] + params.preBlurAmount;
    float b = cov2D[1][0];

    float fullBlurAmount = params.blurAmount;
    if (params.focalDistance > 0.0 && params.apertureAngle > 0.0) {
        float focusBlur = abs((-viewCenter.z - params.focalDistance) / viewCenter.z);
        float focusRadius = focusBlur * focal.x * tan(0.5 * params.apertureAngle);
        fullBlurAmount = clamp(sqr(focusRadius), params.blurAmount, sqr(params.maxPixelRadius));
    }

    // Convolve with a ~0.5-pixel Gaussian for anti-aliasing, keeping energy.
    float detOrig = a * d - b * b;
    a += fullBlurAmount;
    d += fullBlurAmount;
    float det = a * d - b * b;
    rgba.a *= sqrt(max(0.0, detOrig / det));
    if (rgba.a < params.minAlpha) {
        return false;
    }
    s.rgba.a = rgba.a;

    float eigenAvg = 0.5 * (a + d);
    float eigenDelta = sqrt(max(0.0, eigenAvg * eigenAvg - det));
    float eigen1 = eigenAvg + eigenDelta;
    float eigen2 = eigenAvg - eigenDelta;
    float2 eigenVec1 = (abs(b) > 0.001) ? normalize(float2(b, eigen1 - a)) : ((a >= d) ? float2(1.0, 0.0) : float2(0.0, 1.0));
    float2 eigenVec2 = float2(eigenVec1.y, -eigenVec1.x);
    float scale1 = min(params.maxPixelRadius, adjustedStdDev * sqrt(eigen1));
    float scale2 = min(params.maxPixelRadius, adjustedStdDev * sqrt(eigen2));
    if (scale1 < params.minPixelRadius && scale2 < params.minPixelRadius) {
        return false;
    }
    // Past the radius where splatAlpha falls under minAlpha every fragment
    // is discarded: end the footprint there (fewer pixels and tiles, same
    // image). Faint splats shrink most. Not where maxPixelRadius clamps,
    // which ties the quad's edge to adjustedStdDev.
    float f = params.falloff;
    float floorAlpha = (1.0 - f) * rgba.a;
    if (rgba.a <= 1.0 && f > 0.0 && floorAlpha < params.minAlpha && adjustedStdDev * sqrt(eigen1) <= params.maxPixelRadius) {
        float cut = sqrt(max(0.0, 2.0 * log(f * rgba.a / (params.minAlpha - floorAlpha))));
        if (cut < adjustedStdDev) {
            scale1 *= cut / adjustedStdDev;
            scale2 *= cut / adjustedStdDev;
            adjustedStdDev = cut;
            s.adjustedStdDev = cut;
        }
    }
    s.axis1 = eigenVec1 * scale1;
    s.axis2 = eigenVec2 * scale2;
    return true;
}

// The fragment alpha of a splat of alpha `a` at squared splat-space radius
// z2 (inside adjustedStdDev^2): Gaussian falloff, or LOD opacity above 1.
float splatAlpha(float a, float z2) {
    if (a <= 1.0) {
        return lerp(a, a * exp(-0.5 * z2), params.falloff);
    }
    float e = exp((a * a - 1.0) / 2.718281828459045);
    float alpha = 1.0 - pow(1.0 - exp(-0.5 * z2), e);
    return lerp(1.0, alpha, params.falloff);
}

// Whether the portal disk hides a fragment on the view ray `viewRay` (z = -1)
// of a splat at axial depth `depth` (DRAW_DISK_CLIP).
bool diskClipped(float3 viewRay, float depth) {
    float3 viewDir = normalize(viewRay);
    float rayT = depth / max(1e-6, -viewDir.z);
    float radius = abs(params.diskCenter.w);
    bool renderBehind = params.diskCenter.w > 0.0;
    float3 diskN = normalize(params.diskNormal.xyz);
    float denom = dot(viewDir, diskN);
    bool allowPortal = params.diskNormal.w > 0.5 ? abs(denom) > 1e-6 : denom < -1e-6;
    bool hitsDisk = false;
    float t = 0.0;
    if (allowPortal) {
        t = dot(params.diskCenter.xyz, diskN) / denom;
        if (t > 0.0) {
            float3 q = t * viewDir - params.diskCenter.xyz;
            hitsDisk = dot(q, q) <= radius * radius;
        }
    }
    float eps = 1e-4 * max(1.0, abs(t));
    if (renderBehind) {
        return !hitsDisk || rayT <= t + eps;
    }
    return hitsDisk && rayT >= t - eps;
}

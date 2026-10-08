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
// Surfels (2DGS): a splat with exactly one scale 0, as decoded, is a disc,
// drawn by its exact ray-splat intersection with Huang et al.'s screen
// filter (projectSplat's surfel). The renderer sets it unless `surfels:
// "ewa"`, which draws discs as projected 3D Gaussians of rank 2.
static const uint DRAW_2DGS = 4u;
static const uint DRAW_LOD_INFLATE = 8u;    // trade LOD opacity for size
static const uint DRAW_ORTHOGRAPHIC = 16u;
static const uint DRAW_ENCODE_LINEAR = 32u; // sRGB colours to linear output
static const uint DRAW_PREMULTIPLIED = 64u; // write premultiplied alpha
static const uint DRAW_DISK_CLIP = 128u;    // portal disk clip (diskCenter)
// With additive splats (generate's GEN_ADDITIVE) drawn as a layer of their
// own: the pass that draws only them, blended among themselves as any
// splats (athenea's light catcher, `lampTransmittance`), into a layer added
// to the frame; and the pass that draws the rest over it.
static const uint DRAW_ADDITIVE_ONLY = 256u;
static const uint DRAW_NO_ADDITIVE = 512u;

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

// A surfel's ray-splat record (Huang et al. 2024, 2DGS; gsplat's 3x3 M):
// the rows of T = H [s_u t_u, s_v t_v, p] in "surfel pixels" (scaled
// pixels from the screen centre, y up: ndc * scaledRenderSize / 2), where H
// is the projection to homogeneous surfel pixels and t_u, t_v the disc's
// axes in view space. A point (u, v) of the disc in sigmas projects to
// T (u, v, 1). The rows are kept relative to the projected centre c
// (row_x - c.x row_w, row_y - c.y row_w), so x.z = y.z = 0 and those slots
// carry c itself; w.z is the centre's clip w.
struct SurfelRows {
    float3 x;
    float3 y;
    float3 w;
};

// The 2DGS screen filter's sigma (Huang et al., gsplat's
// FILTER_INV_SQUARE_2DGS = 2, athenea's raster): sqrt(2) / 2 pixels, so
// its squared splat-space radius at pixel offset d is 2 |d|^2.
static const float SURFEL_FILTER_INV_SQUARE = 2.0;

// The squared splat-space radius at offset `d` from a surfel's projected
// centre (surfel pixels; rows from SurfelRows, the centre slots ignored):
// the ray-splat intersection's |uv|^2, or the screen filter's where it is
// less: Huang's Ghat = max(G(u, v), G(d / sigma)). `uv` is the one taken.
float surfelRadius2(float2 d, float3 rx, float3 ry, float3 rw, out float2 uv) {
    const float3 x = float3(rx.xy, 0.0);
    const float3 y = float3(ry.xy, 0.0);
    const float3 hu = d.x * rw - x;
    const float3 hv = d.y * rw - y;
    const float3 s = cross(hu, hv);
    const float screen = SURFEL_FILTER_INV_SQUARE * dot(d, d);
    uv = d * sqrt(SURFEL_FILTER_INV_SQUARE);
    // Edge-on (s.z 0) the ray runs in the plane; and a plane met behind the
    // eye (clip w of the hit, dot(rw, s) / s.z, not positive) is not seen.
    if (s.z == 0.0 || !(dot(rw, s) * s.z > 0.0)) {
        return screen;
    }
    const float2 hit = s.xy / s.z;
    const float z2 = dot(hit, hit);
    if (!(z2 < screen)) {
        return screen;
    }
    uv = hit;
    return z2;
}

// A surfel's quad, relative to its projected centre (surfel pixels): the
// box of its ellipse of radius `stdDev` sigmas (the dual conic's tangents,
// gsplat's AABB with the radius in place of 3.33), joined with the
// filter's disc of the same radius around the centre, and held within
// maxPixelRadius of it. Where the ellipse reaches the eye's plane (no
// ellipse on screen) the whole maxPixelRadius square: the fragment's
// intersection decides.
void surfelBox(SurfelRows r, float stdDev, out float2 lo, out float2 hi) {
    const float2 rx = r.x.xy;
    const float2 ry = r.y.xy;
    const float3 w = r.w;
    const float k = 1.0 / (stdDev * stdDev);
    // Dual conic T diag(1, 1, -1/stdDev^2) T^T, relative rows (x.z = y.z = 0).
    const float cxx = dot(rx, rx);
    const float cyy = dot(ry, ry);
    const float cxw = dot(rx, w.xy);
    const float cyw = dot(ry, w.xy);
    const float cww = dot(w.xy, w.xy) - k * w.z * w.z;
    const float reach = stdDev / sqrt(SURFEL_FILTER_INV_SQUARE);
    const float most = params.maxPixelRadius;
    if (cww < 0.0) {
        const float2 mid = float2(cxw, cyw) / cww;
        const float2 ext = sqrt(max(mid * mid - float2(cxx, cyy) / cww, float2(0.0)));
        lo = max(min(mid - ext, float2(-reach)), float2(-most));
        hi = min(max(mid + ext, float2(reach)), float2(most));
    } else {
        lo = float2(-most);
        hi = float2(most);
    }
}

// The surfel pixels per unit of ndc: scaledRenderSize / 2.
float2 surfelPixelScale() {
    return 0.5 * float2(params.renderWidth, params.renderHeight) * params.focalAdjustment;
}

// A view-space vector `v` (w 0: a direction; 1: a point) in homogeneous
// surfel pixels.
float3 surfelHomogeneous(float3 v, float w) {
    const float4 c = params.proj0 * v.x + params.proj1 * v.y + params.proj2 * v.z + params.proj3 * w;
    return float3(c.xy * surfelPixelScale(), c.w);
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
    // A surfel (DRAW_2DGS, exactly one zero scale): its record, surfelRows
    // relative to its projected centre; axes unset. The quad is surfelBox.
    bool surfel;
    SurfelRows rows;
    // Adds its light and covers nothing (generate's GEN_ADDITIVE: an ext,
    // non-covariance splat whose alpha word has a non-zero high half).
    bool additive;
};

// rgba with the additive mark as alpha's sign, for the draw's varyings and
// the tile rasterizer's records (alpha is never 0 past projectSplat).
float4 markedRgba(SplatShape s) {
    return s.additive ? float4(s.rgba.rgb, -s.rgba.a) : s.rgba;
}

// projectSplat's surfel: the record and the cuts, with s's centre and
// colour set. Huang's filter adds no energy and takes none: rgba is the
// splat's own.
bool projectSurfel(inout SplatShape s, float4 viewQuaternion, float3 scales, bool3 zeroScales) {
    const float3 eu = zeroScales.x ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    const float3 ev = zeroScales.z ? float3(0.0, 1.0, 0.0) : float3(0.0, 0.0, 1.0);
    const float su = zeroScales.x ? scales.y : scales.x;
    const float sv = zeroScales.z ? scales.y : scales.z;
    const float3 hu = surfelHomogeneous(quatVec(viewQuaternion, eu * su), 0.0);
    const float3 hv = surfelHomogeneous(quatVec(viewQuaternion, ev * sv), 0.0);
    const float3 hp = surfelHomogeneous(s.viewCenter, 1.0);
    const float2 c = hp.xy / hp.z;
    s.surfel = true;
    s.rows.x = float3(hu.x - c.x * hu.z, hv.x - c.x * hv.z, c.x);
    s.rows.y = float3(hu.y - c.y * hu.z, hv.y - c.y * hv.z, c.y);
    s.rows.w = float3(hu.z, hv.z, hp.z);
    // As the 3D path: past the radius where the falloff's alpha is under
    // minAlpha every fragment is discarded.
    const float a = s.rgba.a;
    const float f = params.falloff;
    const float floorAlpha = (1.0 - f) * a;
    if (a <= 1.0 && f > 0.0 && floorAlpha < params.minAlpha) {
        const float cut = sqrt(max(0.0, 2.0 * log(f * a / (params.minAlpha - floorAlpha))));
        s.adjustedStdDev = min(s.adjustedStdDev, cut);
    }
    if (!(s.adjustedStdDev > 0.0)) {
        return false;
    }
    float2 lo, hi;
    surfelBox(s.rows, s.adjustedStdDev, lo, hi);
    // As the 3D path's minPixelRadius on its axes (the filter's disc keeps
    // an edge-on disc on screen).
    const float2 ext = 0.5 * (hi - lo);
    return !(ext.x < params.minPixelRadius && ext.y < params.minPixelRadius);
}

// The screen footprint of accumulator splat `splatIndex`; false when it
// draws nothing.
bool projectSplat(uint splatIndex, out SplatShape s) {
    s.rgba = float4(0.0);
    s.adjustedStdDev = 0.0;
    s.clipCenter = float4(0.0);
    s.viewCenter = float3(0.0);
    s.axis1 = float2(0.0);
    s.axis2 = float2(0.0);
    s.surfel = false;
    s.rows.x = float3(0.0);
    s.rows.y = float3(0.0);
    s.rows.w = float3(0.0);
    s.additive = false;

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
            s.additive = (a.w >> 16u) != 0u;
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
        // Exactly one zero scale: a disc (two: a line, drawn as a 3D splat).
        const uint zeros = uint(zeroScales.x) + uint(zeroScales.y) + uint(zeroScales.z);
        if (hasFlag(DRAW_2DGS) && zeros == 1u) {
            return projectSurfel(s, viewQuaternion, scales, zeroScales);
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

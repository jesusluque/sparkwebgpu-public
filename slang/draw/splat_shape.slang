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
// Faint splats keep their mass (projectSplat's faint path) instead of being
// dropped when the AA blur takes their alpha under minAlpha.
static const uint DRAW_FAINT = 1024u;
// The faint path: under FAINT_ALPHA x minAlpha after the blur, a splat
// takes less blur, down to FAINT_MIN_BLUR px^2.
static const float FAINT_ALPHA = 4.0;
static const float FAINT_MIN_BLUR = 0.15;
// Surfels keep their mass under a pixel (with DRAW_2DGS): Huang's
// max(G(u, v), G(d / sigma)) widens a small or edge-on disc to the filter's
// disc at the disc's own alpha, which adds coverage (a far sheet of
// sub-pixel surfels turns opaque; the Corvette's far view came out 7-28%
// darker with surfel cuts). With this flag the filter's term is the
// disc's own Gaussian, of its local affine projection, convolved with the
// 3D path's blur at the alpha that keeps its mass (surfelFiltered): the
// exact intersection where the disc is resolved, the 3D path's EWA where
// it is not. Without it, Huang's filter exactly (athenea's raster).
static const uint DRAW_SURFEL_MASS = 2048u;
// The anti-alias filter is the pixel's box (aaFilter "box"), in place of
// the Gaussian blur of blurAmount px^2 with its alpha compensation: a
// fragment's alpha is the mean of the splat's Gaussian over the pixel
// (separable: a difference of erfs per axis of the box's frame, boxFrame).
// Exact mass at any size and no ripple as a splat under a pixel moves (a
// narrow Gaussian sampled at pixel centres has one: 2 exp(-2 pi^2 v), 0.5%
// at v = 0.3, 28% at v = 0.1), while the edge is as sharp as a box
// filter's, which is what a supersampled reference shows: the Gaussian's
// 0.3 px^2 (variance 3.6x the box's 1/12) spreads a silhouette or a thin
// part over its neighbours. Past BOX_GAUSS_SIGMA pixels a splat takes the
// Gaussian of the same variance (s^2 + 1/12, compensated), within 1e-4.
// Surfels (DRAW_SURFEL_MASS): the disc's affine Gaussian J J^T box-filtered
// the same way. The focus blur of depth of field stays Gaussian.
static const uint DRAW_AA_BOX = 4096u;
static const float BOX_GAUSS_SIGMA = 2.0;

// erf(x), Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7).
float erfApprox(float x) {
    const float t = 1.0 / (1.0 + 0.3275911 * abs(x));
    const float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
    return x >= 0.0 ? y : -y;
}

// DRAW_AA_BOX along one axis: the mean of exp(-x^2 / 2 s^2) over the
// pixel's width 2h around offset x (h: half a pixel in the units of x and
// s). Its peak boxAxis(0, s, h) is the share of the alpha that keeps the
// mass (sqrt(2 pi) s / 2h under a pixel, 1 when large); past
// BOX_GAUSS_SIGMA pixels the Gaussian of variance s^2 + h^2 / 3, the box's.
float boxAxis(float x, float s, float h) {
    if (s >= 2.0 * BOX_GAUSS_SIGMA * h) {
        const float v = s * s + h * h / 3.0;
        return s * rsqrt(v) * exp(-0.5 * x * x / v);
    }
    if (!(s > 0.0)) {
        return 0.0;
    }
    const float k = 0.7071067811865476 / s;
    return 0.6266570686577501 * s / h * (erfApprox((x + h) * k) - erfApprox((x - h) * k));
}

// DRAW_AA_BOX: the peak of a splat's box-filtered Gaussian (axis sigmas
// `sigma` along the box's axes, scaled pixels), and the squared radius
// splatAlpha takes at offset `x` along those axes, relative to that peak:
// z2 such that G(z2) = boxed / peak, a sum over the axes (an axis past
// BOX_GAUSS_SIGMA pixels its Gaussian's x^2 / v: no erf, no log).
float boxPeak(float2 sigma) {
    const float h = 0.5 * params.focalAdjustment;
    return boxAxis(0.0, sigma.x, h) * boxAxis(0.0, sigma.y, h);
}
float boxAxisRadius2(float x, float s, float h) {
    if (s >= 2.0 * BOX_GAUSS_SIGMA * h) {
        return x * x / (s * s + h * h / 3.0);
    }
    const float b = boxAxis(x, s, h);
    const float p = boxAxis(0.0, s, h);
    return b > 0.0 && p > 0.0 ? -2.0 * log(min(b / p, 1.0)) : 1e30;
}
float boxRadius2(float2 x, float2 sigma) {
    const float h = 0.5 * params.focalAdjustment;
    return boxAxisRadius2(x.x, sigma.x, h) + boxAxisRadius2(x.y, sigma.y, h);
}

// DRAW_AA_BOX's frame: the box is the pixel's square, the integral is
// taken separable along the square's axes (the Gaussian's variances along
// them, its correlation dropped and its mass kept: boxMass), so a splat's
// coverage sums to its mass wherever it lies (the box's rows and columns
// tile the screen). A long splat (a wire's) would lose its direction that
// way: from 0.5 px along its major axis and a ratio of 1.5 the square
// turns towards the Gaussian's own axes (fully at 1 px and 3), where the
// box integral is exact for its shape and its mass still sums on average
// along it; so does a splat from 0.5 px along its minor axis (fully at 1
// px), which the box barely changes, so that its shape is kept. `e1`, `sigma`: the Gaussian's first axis and its sigmas
// (covSigmas, scaled pixels). Returns the sigmas along the box's axes and
// the box's peak (boxPeak of those); `rot` takes an offset along the
// Gaussian's axes to the box's: (cos, sin).
float3 boxFrame(float2 e1, float2 sigma, out float2 rot) {
    const float h = 0.5 * params.focalAdjustment;
    const float theta = atan2(e1.y, e1.x);
    const float quarter = 0.5 * PI;
    const float t90 = theta - quarter * round(theta / quarter);
    const float needle = saturate((sigma.x / sigma.y - 1.5) / 1.5) * saturate((sigma.x / (2.0 * h) - 0.5) / 0.5);
    const float w = max(needle, saturate((sigma.y / (2.0 * h) - 0.5) / 0.5));
    // The box's first axis cos e1 + sin e2 (e2 = e1 turned -90 degrees, as
    // projectSplat's axis2) is e1 turned by -delta: at w t90 (mod 90).
    const float delta = theta - (sigma.y > 0.0 ? w : 1.0) * t90;
    rot = float2(cos(delta), sin(delta));
    const float2 r2 = rot * rot;
    const float2 s2 = sigma * sigma;
    const float2 boxed = sqrt(float2(r2.x * s2.x + r2.y * s2.y, r2.y * s2.x + r2.x * s2.y));
    return float3(boxed, boxPeak(boxed));
}

// The share of a splat's alpha at its box's peak that keeps its mass:
// boxFrame's peak times the Gaussian's area over its box frame's
// (sigma.x sigma.y / boxed.x boxed.y, 1 on its own axes).
float boxMass(float2 sigma, float3 box) {
    return box.x * box.y > 0.0 ? box.z * min(sigma.x * sigma.y / (box.x * box.y), 1.0) : 0.0;
}

// An offset along a Gaussian's axes along its box's (boxFrame's rot).
float2 boxRotate(float2 uv, float2 rot) {
    return float2(rot.x * uv.x + rot.y * uv.y, rot.x * uv.y - rot.y * uv.x);
}

// The axis sigmas and the first axis of a 2x2 covariance (xx, xy, yy).
float2 covSigmas(float a, float b, float d, out float2 axis) {
    const float avg = 0.5 * (a + d);
    const float delta = sqrt(max(0.0, avg * avg - (a * d - b * b)));
    const float e1 = avg + delta;
    axis = (abs(b) > 1e-6 * max(e1, 1e-30)) ? normalize(float2(b, e1 - a)) : ((a >= d) ? float2(1.0, 0.0) : float2(0.0, 1.0));
    return sqrt(max(float2(e1, avg - delta), float2(0.0)));
}

// A hash of a splat index in [0, 1) (athc_measure's faint_hash).
float faintHash(uint i) {
    uint x = (i * 0x9e3779b9u) ^ 0x85ebca6bu;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return float(x >> 8u) / 16777216.0;
}

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

// DRAW_SURFEL_MASS's filter: the disc's Gaussian at its local affine
// projection (J, from the rows: surfel pixels per sigma at the centre),
// convolved with the 3D path's blur, as the inverse of its covariance
// C = J J^T + blur (xx, xy, yy) and w = det(J J^T) / det(C), the square of
// the share of the disc's peak alpha that keeps its mass (0 edge-on).
float4 surfelFiltered(float3 rx, float3 ry, float3 rw) {
    const float2 ju = float2(rx.x, ry.x) / rw.z;
    const float2 jv = float2(rx.y, ry.y) / rw.z;
    const float detJ = ju.x * jv.y - ju.y * jv.x;
    const float blur = hasFlag(DRAW_AA_BOX) ? 0.0 : params.blurAmount;
    const float cxx = ju.x * ju.x + jv.x * jv.x + blur;
    const float cxy = ju.x * ju.y + jv.x * jv.y;
    const float cyy = ju.y * ju.y + jv.y * jv.y + blur;
    const float det = cxx * cyy - cxy * cxy;
    if (!(det > 0.0)) {
        return float4(0.0);
    }
    return float4(cyy / det, -cxy / det, cxx / det, min(detJ * detJ / det, 1.0));
}

// The squared splat-space radius at offset `d` from a surfel's projected
// centre (surfel pixels; rows from SurfelRows, the centre slots ignored):
// the ray-splat intersection's |uv|^2, or the screen filter's where it is
// less: Huang's Ghat = max(G(u, v), G(d / sigma)); with DRAW_SURFEL_MASS
// the filtered disc's (surfelFiltered) in place of G(d / sigma). `uv` is
// the hit, or where the filter's term is taken d / sigma (with
// DRAW_SURFEL_MASS the filtered disc's mean point). The alpha is
// splatAlpha(a, z2 + fade): fade 0 but with DRAW_SURFEL_MASS.
float surfelRadius2(float2 d, float3 rx, float3 ry, float3 rw, out float2 uv, out float fade) {
    const float3 x = float3(rx.xy, 0.0);
    const float3 y = float3(ry.xy, 0.0);
    const float3 hu = d.x * rw - x;
    const float3 hv = d.y * rw - y;
    const float3 s = cross(hu, hv);
    const float screen = SURFEL_FILTER_INV_SQUARE * dot(d, d);
    uv = d * sqrt(SURFEL_FILTER_INV_SQUARE);
    fade = 0.0;
    // Edge-on (s.z 0) the ray runs in the plane; and a plane met behind the
    // eye (clip w of the hit, dot(rw, s) / s.z, not positive) is not seen.
    const bool met = s.z != 0.0 && dot(rw, s) * s.z > 0.0;
    const float2 hit = met ? s.xy / s.z : float2(0.0);
    const float z2 = met ? dot(hit, hit) : 1e30;
    if (hasFlag(DRAW_SURFEL_MASS)) {
        // max(G(u, v), G_C(d)) at the alpha that keeps the mass, the share
        // sqrt(f.w) as `fade` (share G(z) = G(z + fade)); the cut on the
        // radius itself, as the 3D path's. G_C covers G(u, v) where the
        // projection is affine, so the hit only adds the perspective's far
        // side. Edge-on, nothing. uv, the shading point: the hit where its
        // term is taken, else the filtered disc's mean point seen at d,
        // J^T C^-1 d (the hit where resolved, the centre under a pixel).
        const float4 f = surfelFiltered(rx, ry, rw);
        if (!(f.w > 0.0)) {
            return 1e30;
        }
        const float2 cd = float2(f.x * d.x + f.y * d.y, f.y * d.x + f.z * d.y);
        float filtered = dot(d, cd);
        fade = -log(f.w);
        if (hasFlag(DRAW_AA_BOX)) {
            // The disc's affine Gaussian J J^T box-filtered at d, at the
            // share of its peak (boxPeak) as the fade.
            const float2 ju = float2(rx.x, ry.x) / rw.z;
            const float2 jv = float2(rx.y, ry.y) / rw.z;
            float2 e;
            const float2 sigma = covSigmas(ju.x * ju.x + jv.x * jv.x, ju.x * ju.y + jv.x * jv.y, ju.y * ju.y + jv.y * jv.y, e);
            float2 rot;
            const float3 box = boxFrame(e, sigma, rot);
            if (!(boxMass(sigma, box) > 0.0)) {
                return 1e30;
            }
            fade = -2.0 * log(min(boxMass(sigma, box), 1.0));
            filtered = boxRadius2(boxRotate(float2(dot(d, e), dot(d, float2(e.y, -e.x))), rot), box.xy);
        }
        if (z2 <= filtered) {
            uv = hit;
            return z2;
        }
        uv = float2(dot(float2(rx.x, ry.x), cd), dot(float2(rx.y, ry.y), cd)) / rw.z;
        return filtered;
    }
    if (!(z2 < screen)) {
        return screen;
    }
    uv = hit;
    return z2;
}

// A surfel's quad, relative to its projected centre (surfel pixels): the
// box of its ellipse of radius `stdDev` sigmas (the dual conic's tangents,
// gsplat's AABB with the radius in place of 3.33), joined with the
// filter's disc of the same radius around the centre (DRAW_SURFEL_MASS: the
// filtered disc's ellipse), and held within
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
    float2 reach = float2(stdDev / sqrt(SURFEL_FILTER_INV_SQUARE));
    if (hasFlag(DRAW_SURFEL_MASS)) {
        // The filtered disc's box (its covariance's diagonal: J J^T + blur).
        const float4 f = surfelFiltered(r.x, r.y, w);
        reach = f.w > 0.0 ? stdDev * sqrt(float2(f.z, f.x) / (f.x * f.z - f.y * f.y)) : float2(0.0);
        if (hasFlag(DRAW_AA_BOX) && f.w > 0.0) {
            // The box's square, turned (boxFrame).
            reach += 0.7072 * params.focalAdjustment;
        }
    }
    const float most = params.maxPixelRadius;
    if (cww < 0.0) {
        const float2 mid = float2(cxw, cyw) / cww;
        const float2 ext = sqrt(max(mid * mid - float2(cxx, cyy) / cww, float2(0.0)));
        lo = max(min(mid - ext, -reach), float2(-most));
        hi = min(max(mid + ext, reach), float2(most));
    } else {
        lo = float2(-most);
        hi = float2(most);
    }
}

// A surfel's quad on screen, in pixels from the bottom-left corner (y up):
// surfelBox, its corners snapped outwards to a sixteenth of a pixel. The
// raster's snap of vertex positions (any 4+ subpixel bits) leaves those
// where they are, so the quad's interpolated offset is exact (unsnapped it
// is off by up to a 512th of a pixel, which a disc near edge-on turns into
// ~2e-3 of alpha); and the tile rasterizer covers the same pixels.
void surfelQuad(SurfelRows rows, float adjustedStdDev, out float2 lo, out float2 hi) {
    float2 boxLo, boxHi;
    surfelBox(rows, adjustedStdDev, boxLo, boxHi);
    const float2 c = float2(rows.x.z, rows.y.z);
    const float2 middle = 0.5 * float2(params.renderWidth, params.renderHeight);
    lo = floor((middle + (c + boxLo) / params.focalAdjustment) * 16.0) / 16.0;
    hi = ceil((middle + (c + boxHi) / params.focalAdjustment) * 16.0) / 16.0;
}

// The offset in surfel pixels from a surfel's centre `c` of the screen
// point `p` (pixels from the bottom-left, y up), as surfelRadius2 takes it.
float2 surfelOffset(float2 p, float2 c) {
    return (p - 0.5 * float2(params.renderWidth, params.renderHeight)) * params.focalAdjustment - c;
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
    // DRAW_AA_BOX (not surfels): the sigmas of the unfiltered Gaussian along
    // axis1 and axis2 (scaled pixels); the quad's axes then reach
    // adjustedStdDev sigmas plus the box's square (boxFrame).
    float2 box;
};

// rgba with the additive mark as alpha's sign, for the draw's varyings and
// the tile rasterizer's records (alpha is never 0 past projectSplat).
float4 markedRgba(SplatShape s) {
    return s.additive ? float4(s.rgba.rgb, -s.rgba.a) : s.rgba;
}

// projectSplat's surfel: the record and the cuts, with s's centre and
// colour set. rgba is the splat's own (with DRAW_SURFEL_MASS, under
// minAlpha at its filtered peak, the faint path's floor).
bool projectSurfel(inout SplatShape s, float4 viewQuaternion, float3 scales, bool3 zeroScales, uint splatIndex) {
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
    if (hasFlag(DRAW_SURFEL_MASS)) {
        // The faint path: under FAINT_ALPHA x minAlpha at its peak (alpha x
        // the filtered disc's peak share), alpha raised to that floor, with
        // the probability that keeps its mass; minAlpha added back. Without
        // it, dropped under minAlpha there.
        const float a = s.rgba.a;
        float peak = sqrt(surfelFiltered(s.rows.x, s.rows.y, s.rows.w).w);
        if (hasFlag(DRAW_AA_BOX) && peak > 0.0) {
            const float2 ju = float2(s.rows.x.x, s.rows.y.x) / s.rows.w.z;
            const float2 jv = float2(s.rows.x.y, s.rows.y.y) / s.rows.w.z;
            float2 e;
            const float2 sigma = covSigmas(ju.x * ju.x + jv.x * jv.x, ju.x * ju.y + jv.x * jv.y, ju.y * ju.y + jv.y * jv.y, e);
            float2 rot;
            peak = min(boxMass(sigma, boxFrame(e, sigma, rot)), 1.0);
        }
        const float faintTarget = FAINT_ALPHA * params.minAlpha;
        if (!(peak > 0.0)) {
            return false;
        }
        if (!(hasFlag(DRAW_FAINT) && params.falloff == 1.0 && a <= 1.0)) {
            if (a <= 1.0 && a * peak < params.minAlpha) {
                return false;
            }
        } else if (a * peak < faintTarget) {
            const float raised = min(faintTarget / peak, 1.0);
            if (!(faintHash(splatIndex) < a / raised)) {
                return false;
            }
            s.rgba.a = min(raised + params.minAlpha / peak, 1.0);
        }
    }
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
    s.box = float2(0.0);

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
            return projectSurfel(s, viewQuaternion, scales, zeroScales, splatIndex);
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

    if (hasFlag(DRAW_AA_BOX)) {
        return projectBoxed(s, splatIndex, a, b, d, fullBlurAmount - params.blurAmount, adjustedStdDev);
    }

    // Convolve with a ~0.5-pixel Gaussian for anti-aliasing, keeping energy.
    float detOrig = a * d - b * b;
    const float a0 = a;
    const float d0 = d;
    a += fullBlurAmount;
    d += fullBlurAmount;
    float det = a * d - b * b;
    const float blurred = rgba.a * sqrt(max(0.0, detOrig / det));
    // THE FAINT PATH (DRAW_FAINT): a faint splat under a pixel, which the
    // blur would take under minAlpha (glass, a far cloud without LoD), keeps
    // its mass alpha sqrt(det) instead of vanishing: less blur, for an alpha
    // of FAINT_ALPHA x minAlpha; past FAINT_MIN_BLUR drawn at that alpha with
    // the probability that keeps its mass on average (a hash of the splat);
    // and minAlpha added back, as the fragments' minAlpha cut keeps alpha -
    // minAlpha of a Gaussian's mass (falloff 1). athc_measure faint: the
    // pawn's glass head at 192 px, relMSE 0.41 -> 0.009.
    const float faintTarget = FAINT_ALPHA * params.minAlpha;
    if (hasFlag(DRAW_FAINT) && rgba.a <= 1.0 && params.falloff == 1.0 && detOrig > 0.0 && blurred < faintTarget) {
        const float mass = rgba.a * sqrt(detOrig);
        const float tr = a0 + d0;
        const float want = sqr(mass / faintTarget);
        const float blur = clamp(0.5 * (sqrt(max(tr * tr + 4.0 * (want - detOrig), 0.0)) - tr), FAINT_MIN_BLUR,
                                 max(fullBlurAmount, FAINT_MIN_BLUR));
        a = a0 + blur;
        d = d0 + blur;
        det = a * d - b * b;
        float alpha = mass / sqrt(det);
        if (alpha < faintTarget) {
            if (faintHash(splatIndex) >= alpha / faintTarget) {
                return false;
            }
            alpha = faintTarget;
        }
        rgba.a = alpha + params.minAlpha;
    } else {
        rgba.a = blurred;
    }
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

// projectSplat's 3D path with DRAW_AA_BOX: the 2D covariance (a, b, d)
// before any blur, `focus` the depth of field's blur past blurAmount (a
// Gaussian, compensated); the box's peak share as the compensation, the
// faint path as the Gaussian's (at its floor alpha with the probability
// that keeps the mass; the box keeps its width), the quad's axes adjustedStdDev
// sigmas plus half a pixel.
bool projectBoxed(inout SplatShape s, uint splatIndex, float a, float b, float d, float focus, float adjustedStdDev) {
    const float detOrig = a * d - b * b;
    float share = 1.0;
    if (focus > 0.0) {
        a += focus;
        d += focus;
        share = sqrt(max(0.0, detOrig / (a * d - b * b)));
    }
    float2 e1;
    const float2 sigma = covSigmas(a, b, d, e1);
    float2 rot;
    const float peak = boxMass(sigma, boxFrame(e1, sigma, rot)) * share;
    float alpha = s.rgba.a * peak;
    const float faintTarget = FAINT_ALPHA * params.minAlpha;
    if (hasFlag(DRAW_FAINT) && s.rgba.a <= 1.0 && params.falloff == 1.0 && alpha < faintTarget) {
        if (!(alpha > 0.0) || faintHash(splatIndex) >= alpha / faintTarget) {
            return false;
        }
        alpha = faintTarget + params.minAlpha;
    }
    if (alpha < params.minAlpha) {
        return false;
    }
    s.rgba.a = alpha;
    // As the Gaussian path: end the footprint where the falloff's alpha is
    // under minAlpha (on the sigmas; the half pixel is added after).
    const float f = params.falloff;
    const float floorAlpha = (1.0 - f) * alpha;
    if (alpha <= 1.0 && f > 0.0 && floorAlpha < params.minAlpha) {
        adjustedStdDev = min(adjustedStdDev, sqrt(max(0.0, 2.0 * log(f * alpha / (params.minAlpha - floorAlpha)))));
    }
    const float h = 0.5 * params.focalAdjustment;
    // The box's square, turned by rot, reaches h (|cos| + |sin|) on each axis.
    const float2 reach = min(float2(params.maxPixelRadius), adjustedStdDev * sigma + h * (abs(rot.x) + abs(rot.y)));
    if (reach.x < params.minPixelRadius && reach.y < params.minPixelRadius) {
        return false;
    }
    s.adjustedStdDev = adjustedStdDev;
    s.axis1 = e1 * reach.x;
    s.axis2 = float2(e1.y, -e1.x) * reach.y;
    s.box = sigma;
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

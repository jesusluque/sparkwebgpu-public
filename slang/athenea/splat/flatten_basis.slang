// Copyright (c) 2026 jesus luque.
//
// WHAT `athenea flatten` FITS WITH: the directions a gaussian is looked at
// from, the harmonics it is fitted to, the rotation of a fit into the frame
// a file is written in, and the record a flattened gaussian is kept as until
// a format packs it (docs/decisions.md, task TXF).
//
// THE FIT. A TX cloud's colour depends on the sky and on the eye; a standard
// file holds degree 3 harmonics a gaussian, as 3DGS evaluates them:
//
//   colour(v) = 0.5 + sum_k c_k Y_k(v)       v from the eye to the gaussian
//
// and a viewer clamps it at zero. What is fitted is what the viewer will
// show -- the sRGB-encoded colour by default -- along N directions of a
// Fibonacci sphere laid out in the gaussian's own frame (+z its facing). In
// that frame the weighted least squares
//
//   c = (Y^T W Y + lambda R)^-1 Y^T W g = M g,     R = diag(l^2 (l+1)^2)
//
// has the same matrix M for every gaussian of a class (a surface, weighted 1
// in front and `backWeight` behind; a thin wall, weighted 1 everywhere), so
// M is solved once on the device (flatten_fit.slang) and a gaussian's fit is
// one product of it with what it shows. The coefficients are then turned
// into the file's frame (`flatRotate`) by projecting the local fit, read at
// the 26 points of Lebedev's rule, onto the harmonics there: the product of
// two degree 3 harmonics is of degree 6 and that rule is exact to degree 7,
// so the turn is exact and needs no Wigner matrices.
module flatten_basis;
import athenea.common.sh;

/// Coefficients a colour channel: degree 3.
public static const uint kFlatCoefficients = 16;

/// THE FLATTENED RECORD, floats a gaussian: in the file's frame and unit.
public static const uint kFlatStride = 64;
public static const uint kFlatPosition = 0;   ///< x y z
public static const uint kFlatOpacity = 3;    ///< linear, 0..1
public static const uint kFlatLogScale = 4;   ///< ln sigma, three axes
public static const uint kFlatRotation = 7;   ///< w x y z
public static const uint kFlatSh = 11;        ///< 16 rgb triples, rgb per basis; [0] the 3DGS DC
public static const uint kFlatNormal = 59;    ///< the facing the fit was weighted about
public static const uint kFlatShrink = 62;    ///< what the higher bands were scaled by for positivity
public static const uint kFlatFlags = 63;     ///< bit 0: thin wall; bit 1: shadow catcher

/// The direction `k` of `n` on a Fibonacci sphere: equal areas, so one
/// weight 4 pi / n each.
public float3 fibonacciDirection(uint k, uint n) {
    const float golden = 2.39996322972865332;   // pi (3 - sqrt 5)
    const float z = 1.0 - (2.0 * float(k) + 1.0) / float(n);
    const float r = sqrt(max(0.0, 1.0 - z * z));
    const float phi = golden * float(k);
    return float3(r * cos(phi), r * sin(phi), z);
}

/// The sixteen harmonics at `v`, in the order and with the signs the
/// rasteriser evaluates (common/sh.slang's `shBasisValue`).
public void flatBasis(float3 v, out float y[16]) {
    [ForceUnroll]
    for (uint k = 0; k < 16; ++k) {
        y[k] = shBasisValue(k, v);
    }
}

/// The band of coefficient `k`.
public uint flatBand(uint k) {
    return k == 0 ? 0u : (k < 4 ? 1u : (k < 9 ? 2u : 3u));
}

/// LEBEDEV'S 26 POINTS ON THE SPHERE, exact for polynomials of degree 7:
/// the six axes, the twelve edge midpoints and the eight corners of a cube,
/// with weights that sum to 4 pi.
public void lebedevPoint(uint j, out float3 u, out float w) {
    const float fourPi = 12.566370614359172;
    if (j < 6) {
        const float s = (j & 1u) != 0u ? -1.0 : 1.0;
        u = j < 2 ? float3(s, 0.0, 0.0) : (j < 4 ? float3(0.0, s, 0.0) : float3(0.0, 0.0, s));
        w = fourPi / 21.0;
        return;
    }
    if (j < 18) {
        const uint e = j - 6;
        const uint pair = e / 4;   // 0: xy, 1: yz, 2: zx
        const float a = (e & 1u) != 0u ? -0.70710678118654752 : 0.70710678118654752;
        const float b = (e & 2u) != 0u ? -0.70710678118654752 : 0.70710678118654752;
        u = pair == 0 ? float3(a, b, 0.0) : (pair == 1 ? float3(0.0, a, b) : float3(b, 0.0, a));
        w = fourPi * 4.0 / 105.0;
        return;
    }
    const uint c = j - 18;
    const float r = 0.57735026918962576;
    u = float3((c & 1u) != 0u ? -r : r, (c & 2u) != 0u ? -r : r, (c & 4u) != 0u ? -r : r);
    w = fourPi * 9.0 / 280.0;
}

/// A fit made in a local frame, in the file's: `toLocal` takes a direction
/// of the file's frame to the local one (the transpose of the local frame's
/// axes written in the file's). Exact (module comment).
public void flatRotate(float3 local[16], float3x3 toLocal, out float3 turned[16]) {
    [ForceUnroll]
    for (uint m = 0; m < 16; ++m) {
        turned[m] = float3(0.0);
    }
    for (uint j = 0; j < 26; ++j) {
        float3 u;
        float w;
        lebedevPoint(j, u, w);
        float yl[16];
        flatBasis(mul(toLocal, u), yl);
        float3 h = float3(0.0);
        [ForceUnroll]
        for (uint k = 0; k < 16; ++k) {
            h += local[k] * yl[k];
        }
        float yu[16];
        flatBasis(u, yu);
        [ForceUnroll]
        for (uint m = 0; m < 16; ++m) {
            turned[m] += (w * yu[m]) * h;
        }
    }
}

/// Two axes perpendicular to `n`, so that (t, b, n) is a right-handed
/// orthonormal frame (Duff et al., "Building an Orthonormal Basis, Revisited").
public void flatFrame(float3 n, out float3 t, out float3 b) {
    const float s = n.z >= 0.0 ? 1.0 : -1.0;
    const float a = -1.0 / (s + n.z);
    const float c = n.x * n.y * a;
    t = float3(1.0 + s * n.x * n.x * a, s * c, -s * n.x);
    b = float3(c, s + n.y * n.y * a, -n.y);
}

/// THE ROUGHNESS FLOOR'S CANDIDATES: `kFloorCandidates` roughnesses, the
/// floor the first of them whose reflection lobe leaves less than the asked
/// share of its energy above band 3 (flatten_fit.slang's
/// `flattenRoughnessFloor`).
public static const uint kFloorCandidates = 64;

public float flatFloorCandidate(uint index) {
    return float(min(index, kFloorCandidates - 1) + 1) / float(kFloorCandidates);
}

/// A rotation matrix (columns the frame's axes) as a quaternion, w x y z.
public float4 flatQuaternion(float3x3 m) {
    const float trace = m[0][0] + m[1][1] + m[2][2];
    float4 q;
    if (trace > 0.0) {
        const float s = sqrt(trace + 1.0) * 2.0;
        q = float4(0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s);
    } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        const float s = sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0;
        q = float4((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s);
    } else if (m[1][1] > m[2][2]) {
        const float s = sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0;
        q = float4((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s);
    } else {
        const float s = sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0;
        q = float4((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s);
    }
    return normalize(q);
}

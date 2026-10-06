// Copyright (c) 2026 jesus luque.
//
// The fit a bake makes of its paths: the harmonics, fitted over the half of
// the sphere a surface faces, then bounded and made whole. Out of the path
// tracer's bake variant (technique/PathTracer.cpp), so that the kernels that
// resolve a bake kept as sums -- direct and indirect apart, added to over
// passes, the indirect filtered between neighbours -- fit exactly as the
// tracer does (`athenea/usd/bake_resolve`).
module bake_fit;
import athenea.common.sh;

/// The point seen from one direction of the hemisphere it faces.
///
/// A colour that is one number cannot be view-dependent, so what a bake stores
/// is the average of the radiance leaving the surface over that hemisphere,
/// weighted by the cosine -- which is exactly the radiance a diffuse surface
/// of the same radiosity has, and which spreads a highlight over the
/// gaussians that would show it instead of putting all of it on the ones whose
/// normal points at the light. Baking along the normal alone was tried first:
/// under a dome it makes every gaussian show the same reflection, and the
/// marble it was converted from came out as polished plastic.
///
/// One direction a sample, so the mean over the paths is the mean over the
/// hemisphere. The ray still starts where the caller put it and still ends on
/// the surface it named.
/// A point of the half of the sphere `n` faces, from a point of the unit
/// square: the elevation straight from `u.x`, so that the square's area maps
/// to the hemisphere's solid angle, and the azimuth from `u.y`.
public float3 bakeAim(float3 n, float2 u) {
    const float z = u.x;
    const float r = sqrt(max(1.0 - z * z, 0.0));
    const float phi = 2.0 * 3.14159265358979 * u.y;
    float3 tangent = abs(n.z) < 0.999 ? normalize(cross(float3(0.0, 0.0, 1.0), n))
                                      : float3(1.0, 0.0, 0.0);
    const float3 bitangent = cross(n, tangent);
    return normalize(tangent * (r * cos(phi)) + bitangent * (r * sin(phi)) + n * z);
}

/// The measure a uniform sample of the hemisphere stands for: `bakeAim`
/// draws directions uniformly over the half of the sphere the surface faces
/// (2 pi steradians), so the estimator of `integral(L * Y)` over that half
/// from N of them is `(2 pi / N) * sum(L * Y)`.
public static const float kBakeMeasure = 2.0 * 3.14159265358979;

/// Which band a coefficient belongs to: 0, 1, 1, 1, 2, 2, 2, 2, 2, 3...
public uint bakeBand(uint k) {
    return uint(sqrt(float(k)));
}

/// THE FIT IS OVER THE HALF OF THE SPHERE THE SURFACE FACES, and the basis is
/// not orthogonal there.
///
/// Projecting instead -- over the whole sphere, with the far half taken as
/// nothing -- is what the bake did, and it is a different question with a
/// visibly different answer. The function being projected jumps from the
/// radiance to zero at the equator, so the series takes the middle of that
/// jump where the two meet: **half**. And the equator is exactly where a
/// silhouette is looked at. Measured across the pawn's glass ball, the mesh
/// reads a flat 0.309 and the baked cloud read 0.152 at the edge, climbing
/// over a hundred pixels to 0.325 at the centre -- a dark rim around every
/// silhouette in the model. Degree 3 rang about it (0.224, 0.271, 0.193) and
/// came out further from the mesh than degree 2, which is Gibbs and not noise.
///
/// The fit is the normal equations, `G c = b`, with `b_k = integral(L Y_k)`
/// over that half -- what the samples already sum to -- and
/// `G_kj = integral(Y_k Y_j)` over it. What makes them small enough to solve
/// a gaussian at a time:
///
/// **Two bands of the same parity are orthogonal over any half of the
/// sphere.** `Y(-w) = (-1)^l Y(w)`, so the far half's integral is the near
/// half's times `(-1)^(l_k + l_j)`; when that is +1 the two halves are equal
/// and each is half the sphere's, which is `delta_kj / 2` -- whatever the
/// normal is. So the even-even and odd-odd blocks are exactly `I/2` and are
/// not accumulated at all. Only the even-odd block `E` depends on the
/// direction the surface faces, and only it is summed: at most six by ten.
///
/// That leaves `[[I/2, E], [E^T, I/2]]`, whose Schur complement on the even
/// side is `I/2 - 2 E E^T` -- **six by six at most**, symmetric, and positive
/// definite because `G` is. Cholesky solves it, and the odd half follows.
///
/// Degree 0 falls out of the same arithmetic: no odd terms, so `E` is empty,
/// the system is `x / 2 = b`, and the constant a gaussian stores is twice the
/// projection. Which is why the projection had it at a fifth of the mesh.
/// THE MATRIX COSTS NO RAYS. It is `integral(Y_k Y_j)` over the half of the
/// sphere the surface faces: it depends on the normal and on nothing else, not
/// on the light, not on the material, not on a single path. So it is not
/// estimated from the paths -- taken that way, at 256 of them, the whole fit
/// came apart: its worst mode is a hundred times smaller than its best, the
/// sampling error on the matrix is of the same size as the entries it is
/// estimating, and the pawn came back with pixels in the thousands.
///
/// It is integrated instead, deterministically, over a grid that costs no
/// rays: 32 elevations by 16 azimuths, which for a product of two basis
/// functions -- a trigonometric polynomial of degree six in the azimuth and,
/// once that sum has killed the terms with unequal order, a polynomial in the
/// elevation -- is worth about four figures. 512 directions of arithmetic
/// against 256 of ray tracing.
public void bakeGram(float3 n, out float e[6][10], uint evens[6], uint ne, uint odds[10], uint no) {
    for (uint i = 0; i < ne; ++i) {
        for (uint j = 0; j < no; ++j) {
            e[i][j] = 0.0;
        }
    }
    const uint kElevations = 32;
    const uint kAzimuths = 16;
    for (uint zi = 0; zi < kElevations; ++zi) {
        for (uint pi = 0; pi < kAzimuths; ++pi) {
            const float2 u = float2((float(zi) + 0.5) / float(kElevations),
                                    (float(pi) + 0.5) / float(kAzimuths));
            // The same direction the samples are read at, so the matrix and
            // the right hand side are over the same half of the sphere.
            const float3 towards = -bakeAim(n, u);
            float basis[16];
            for (uint c = 0; c < 16; ++c) {
                basis[c] = shBasisValue(c, towards);
            }
            for (uint i = 0; i < ne; ++i) {
                for (uint j = 0; j < no; ++j) {
                    e[i][j] += basis[evens[i]] * basis[odds[j]];
                }
            }
        }
    }
    const float measure = 2.0 * 3.14159265358979 / float(kElevations * kAzimuths);
    for (uint i = 0; i < ne; ++i) {
        for (uint j = 0; j < no; ++j) {
            e[i][j] *= measure;
        }
    }
}

/// WHERE THE FIT IS STOPPED FROM AMPLIFYING THE PATHS` OWN NOISE.
///
/// Half of the sphere does not determine sixteen harmonics equally: the
/// combinations that are nearly nothing on the half the surface faces, and
/// large on the half it does not, are what the data cannot see. Solved
/// exactly, the fit puts the noise of a few hundred paths into exactly those.
/// On the pawn a few gaussians came back with coefficients of **65344** --
/// fp16's ceiling, which is what a cloud stores them in -- and burned out as
/// white blobs.
///
/// A ridge was tried first, added to every diagonal. It works, and it charges
/// every gaussian for the few that need it: at 0.02 the fit shrinks by
/// `0.5 / 0.52`, and a Lambertian plane whose light is 0.4614 came back at
/// 0.4436 -- 3.9% low, exactly that ratio -- with degree 2 at 9%.
///
/// A floor under the pivot charges nobody. Cholesky reaches a small pivot
/// exactly where the matrix is near singular, which is the direction the data
/// could not see, and flooring it there bounds what that direction can
/// contribute while leaving every well determined gaussian solved exactly.
/// The diagonal being a half, a floor of 0.005 caps the amplification at
/// about fourteen.
public static const float kBakePivot = 0.005;

public void bakeFit(inout float3 c[16], float e[6][10], uint evens[6], uint ne, uint odds[10], uint no,
             float measure) {
    float3 be[6];
    float3 bo[10];
    for (uint i = 0; i < ne; ++i) {
        be[i] = c[evens[i]] * measure;
    }
    for (uint j = 0; j < no; ++j) {
        bo[j] = c[odds[j]] * measure;
    }
    // The diagonal: a half, from the two bands being of the same parity, which
    // holds whatever the normal is.
    const float d = 0.5;
    // The Schur complement, and the even side's right hand side with the odd
    // half eliminated.
    float  m[6][6];
    float3 r[6];
    for (uint i = 0; i < ne; ++i) {
        float3 sum = float3(0.0);
        for (uint j = 0; j < no; ++j) {
            sum += e[i][j] * bo[j];
        }
        r[i] = be[i] - sum / d;
        for (uint k = 0; k <= i; ++k) {
            float dot = 0.0;
            for (uint j = 0; j < no; ++j) {
                dot += e[i][j] * e[k][j];
            }
            m[i][k] = (i == k ? d : 0.0) - dot / d;
            m[k][i] = m[i][k];
        }
    }
    // Cholesky, in place: m = L L^T, then two triangular solves.
    for (uint i = 0; i < ne; ++i) {
        for (uint k = 0; k <= i; ++k) {
            float sum = m[i][k];
            for (uint j = 0; j < k; ++j) {
                sum -= m[i][j] * m[k][j];
            }
            m[i][k] = i == k ? sqrt(max(sum, kBakePivot)) : sum / m[k][k];
        }
    }
    float3 x[6];
    for (uint i = 0; i < ne; ++i) {
        float3 sum = r[i];
        for (uint j = 0; j < i; ++j) {
            sum -= m[i][j] * x[j];
        }
        x[i] = sum / m[i][i];
    }
    for (int i = int(ne) - 1; i >= 0; --i) {
        float3 sum = x[i];
        for (uint j = uint(i) + 1; j < ne; ++j) {
            sum -= m[j][i] * x[j];
        }
        x[i] = sum / m[i][i];
    }
    for (uint i = 0; i < ne; ++i) {
        c[evens[i]] = x[i];
    }
    for (uint j = 0; j < no; ++j) {
        float3 sum = bo[j];
        for (uint i = 0; i < ne; ++i) {
            sum -= e[i][j] * x[i];
        }
        c[odds[j]] = sum / d;
    }
}

/// THE FIT IS BOUNDED BY WHAT THE PATHS SAW AND DEFINED ON THE WHOLE SPHERE.
///
/// A cloud is blended in linear light and a bake writes light, so nothing is
/// encoded: the series stays in the space it was fitted in. (It used to be
/// taken to sRGB, when clouds were blended there -- first to first order,
/// with bands scaled by up to 12.92 at a dark mean, then direction by
/// direction -- and the bound below is what that taught.)
///
/// What is settled here is the series itself. The fitted radiance is read
/// back over the half of the sphere the fit is good on, held between nothing
/// and the brightest sample the paths returned -- a fit is not entitled to
/// light no path saw: the pawn's glass head baked at degree 3 read infinity
/// without it. And the fit only ever knew that half, while a frame reads the
/// series from every side: a disc seen edge on or from behind read the
/// extrapolation. So the far half is given the near half's mirror image
/// across the surface's plane -- continuous at the rim, bounded like the
/// rest -- and on the whole sphere the basis is orthonormal: the bounded
/// series is a plain projection, with no matrix to solve. At degree zero it
/// is the mean, clamped to the brightest sample.
public void bakeEncode(inout float3 c[16], uint count, float3 n, float3 brightest) {
    float3 b[16];
    for (uint k = 0; k < 16; ++k) {
        b[k] = float3(0.0);
    }
    const uint kElevations = 32;
    const uint kAzimuths = 16;
    for (uint zi = 0; zi < kElevations; ++zi) {
        for (uint pi = 0; pi < kAzimuths; ++pi) {
            const float2 u = float2((float(zi) + 0.5) / float(kElevations),
                                    (float(pi) + 0.5) / float(kAzimuths));
            const float3 towards = -bakeAim(n, u);
            const float3 mirrored = towards - 2.0 * dot(towards, n) * n;
            float3 radiance = float3(0.0);
            for (uint k = 0; k < count; ++k) {
                radiance += c[k] * shBasisValue(k, towards);
            }
            const float3 bounded = clamp(radiance, float3(0.0), brightest);
            for (uint k = 0; k < count; ++k) {
                b[k] += bounded * (shBasisValue(k, towards) + shBasisValue(k, mirrored));
            }
        }
    }
    const float measure = 2.0 * 3.14159265358979 / float(kElevations * kAzimuths);
    for (uint k = 0; k < 16; ++k) {
        c[k] = b[k] * measure;
    }
}

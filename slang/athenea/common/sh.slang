// Copyright (c) 2026 jesus luque.
//
// Real spherical harmonics up to degree 3, as 3DGS trains them, over the packed
// fp16 coefficients (common/packing.slang: rgb per basis, two halves a word).
module sh;
import athenea.common.packing;

static const float kC1 = 0.4886025119029199;
static const float kC2[5] = { 1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
                              -1.0925484305920792, 0.5462742152960396 };
static const float kC3[7] = { -0.5900435899266435, 2.890611442640554, -0.4570457994644658,
                              0.3731763325901154, -0.4570457994644658, 1.445305721320277,
                              -0.5900435899266435 };

float halfAt(StructuredBuffer<uint> sh, uint row, uint index) {
    const uint word = sh[row + index / 2];
    return (index & 1u) == 0u ? lowHalf(word) : highHalf(word);
}

float3 basis(StructuredBuffer<uint> sh, uint row, uint k) {
    return float3(halfAt(sh, row, k * 3), halfAt(sh, row, k * 3 + 1), halfAt(sh, row, k * 3 + 2));
}

/// The view-dependent part added to the base colour. `degree` <= what is stored.
public float3 evaluateRest(StructuredBuffer<uint> sh, uint row, uint degree, float3 dir) {
    if (degree == 0) {
        return float3(0.0);
    }
    const float x = dir.x, y = dir.y, z = dir.z;
    float3 result = -kC1 * y * basis(sh, row, 0) + kC1 * z * basis(sh, row, 1) -
                    kC1 * x * basis(sh, row, 2);
    if (degree < 2) {
        return result;
    }
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, yz = y * z, xz = x * z;
    result += kC2[0] * xy * basis(sh, row, 3) + kC2[1] * yz * basis(sh, row, 4) +
              kC2[2] * (2.0 * zz - xx - yy) * basis(sh, row, 5) +
              kC2[3] * xz * basis(sh, row, 6) + kC2[4] * (xx - yy) * basis(sh, row, 7);
    if (degree < 3) {
        return result;
    }
    result += kC3[0] * y * (3.0 * xx - yy) * basis(sh, row, 8) +
              kC3[1] * xy * z * basis(sh, row, 9) +
              kC3[2] * y * (4.0 * zz - xx - yy) * basis(sh, row, 10) +
              kC3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * basis(sh, row, 11) +
              kC3[4] * x * (4.0 * zz - xx - yy) * basis(sh, row, 12) +
              kC3[5] * z * (xx - yy) * basis(sh, row, 13) +
              kC3[6] * x * (xx - 3.0 * yy) * basis(sh, row, 14);
    return result;
}

/// One basis function of the same set, by index: 0 is the constant term and
/// 1..15 are the rest coefficients in the order `evaluateRest` reads them.
///
/// It exists for the other direction -- projecting a radiance somebody
/// measured onto these harmonics (`athenea mesh2splat --bake`) -- and it must
/// agree with `evaluateRest` term for term, which is why the two sit
/// together.
public float shBasisValue(uint k, float3 dir) {
    const float x = dir.x, y = dir.y, z = dir.z;
    const float xx = x * x, yy = y * y, zz = z * z;
    switch (k) {
    case 0: return kSH0;
    case 1: return -kC1 * y;
    case 2: return kC1 * z;
    case 3: return -kC1 * x;
    case 4: return kC2[0] * x * y;
    case 5: return kC2[1] * y * z;
    case 6: return kC2[2] * (2.0 * zz - xx - yy);
    case 7: return kC2[3] * x * z;
    case 8: return kC2[4] * (xx - yy);
    case 9: return kC3[0] * y * (3.0 * xx - yy);
    case 10: return kC3[1] * x * y * z;
    case 11: return kC3[2] * y * (4.0 * zz - xx - yy);
    case 12: return kC3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
    case 13: return kC3[4] * x * (4.0 * zz - xx - yy);
    case 14: return kC3[5] * z * (xx - yy);
    case 15: return kC3[6] * x * (xx - 3.0 * yy);
    default: return 0.0;
    }
}

/// What the constant term has to be shifted by, because a cloud stores its DC
/// the way 3DGS trains it: `colour = 0.5 + kSH0 * dc`, while a projection
/// gives `colour = c0 * kSH0`. So `dc = c0 - kShDcOffset`.
public static const float kShDcOffset = 0.5 / kSH0;

/// The constant harmonic itself, for a caller that has the coefficient and
/// wants the value: `colour = c0 * kShY0`. It is `packing.slang`'s `kSH0`,
/// republished because a module that imports this one does not see that one.
public static const float kShY0 = kSH0;

public uint degreeOf(uint restPerColour) {
    return restPerColour >= 15 ? 3 : restPerColour >= 8 ? 2 : restPerColour >= 3 ? 1 : 0;
}

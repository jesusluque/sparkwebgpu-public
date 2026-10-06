// Copyright (c) 2026 jesus luque.
//
// The ACES 2.0 output transform, ported function for function from the
// Academy's reference (aces-core, Lib.Academy.OutputTransform.ctl and
// Lib.Academy.Tonescale.ctl, Apache-2.0): scene-referred ACES2065-1 into a
// display's RGB through Hellwig 2022's colour appearance model (J, M, h), a
// tonescale on J, an in-gamut compression of M, and a gamut compression
// towards the display cube's boundary. The transform is analytic; what the
// reference precomputes at init -- the reach of the AP1 primaries at every
// hue, the display gamut's cusp at every hue, the gamma of its upper hull --
// are tables of a few hundred entries, built here by kernels
// (aces2_prepare.slang) into one float buffer, since a statistic of the
// display's gamut is arithmetic and arithmetic is a kernel's.
//
// Units: RGB in and out are "1.0 is 100 nits" (the reference's, with
// ref_luminance folded into the CAM matrices); a 1000-nit peak comes out
// as 10.0. Matrices are row-vector ones, `mul(v, M)`, as the reference's
// mult_f3_f33(v, M).
module aces2;

public struct Chromaticities {
    public float2 red;
    public float2 green;
    public float2 blue;
    public float2 white;
};

public static const Chromaticities kAP0 = {float2(0.73470, 0.26530), float2(0.00000, 1.00000),
                                           float2(0.00010, -0.07700), float2(0.32168, 0.33767)};
public static const Chromaticities kAP1 = {float2(0.713, 0.293), float2(0.165, 0.830), float2(0.128, 0.044),
                                           float2(0.32168, 0.33767)};
public static const Chromaticities kRec709 = {float2(0.64, 0.33), float2(0.30, 0.60), float2(0.15, 0.06),
                                              float2(0.3127, 0.3290)};
public static const Chromaticities kP3D65 = {float2(0.680, 0.320), float2(0.265, 0.690), float2(0.150, 0.060),
                                             float2(0.3127, 0.3290)};
static const Chromaticities kCAM16 = {float2(0.8336, 0.1735), float2(2.3854, -1.4659), float2(0.087, -0.125),
                                      float2(0.333, 0.333)};

// The tables: `kTableSize` hues plus one wrap entry each side.
public static const int kTableSize = 360;
public static const int kTotalTableSize = 362;
public static const int kBaseIndex = 1;
public static const int kReachOffset = 0;                       // reach M per uniform hue
public static const int kHuesOffset = kTotalTableSize;          // the cusp table's hues
public static const int kCuspJOffset = 2 * kTotalTableSize;
public static const int kCuspMOffset = 3 * kTotalTableSize;
public static const int kGammaOffset = 4 * kTotalTableSize;     // the upper hull's gamma per cusp entry
public static const int kTableWords = 5 * kTotalTableSize;

public static const float kHueLimit = 360.0;
public static const int kCuspCorners = 6;
public static const int kTotalCorners = 8;
public static const int kMaxSortedCorners = 12;
static const float kReachCuspTolerance = 1e-3;
static const float kDisplayCuspTolerance = 1e-7;
static const float kGammaMinimum = 0.0;
static const float kGammaMaximum = 5.0;
static const float kGammaSearchStep = 0.4;
static const float kGammaAccuracy = 1e-5;

// CAM parameters: 100 nit reference, dim surround.
public static const float kRefLuminance = 100.0;
static const float kL_A = 100.0;
static const float kY_b = 20.0;
static const float3 kSurround = float3(0.9, 0.59, 0.9);
static const float kJScale = 100.0;
static const float kCamNlYReference = 100.0;
static const float kCamNlOffset = 0.2713 * kCamNlYReference;
static const float kCamNlScale = 4.0 * kCamNlYReference;
static const float kModelGamma = 0.59 * (1.48 + 0.4472135955);   // surround[1] * (1.48 + sqrt(Y_b / ref))

// Chroma compression.
static const float kChromaCompress = 2.4;
static const float kChromaCompressFact = 3.3;
static const float kChromaExpand = 1.3;
static const float kChromaExpandFact = 0.69;
static const float kChromaExpandThr = 0.5;

// Gamut compression.
static const float kSmoothCusps = 0.12;
static const float kSmoothM = 0.27;
static const float kCuspMidBlend = 1.3;
static const float kFocusGainBlend = 0.3;
static const float kFocusAdjustGain = 0.55;
static const float kFocusDistance = 1.35;
static const float kFocusDistanceScaling = 1.75;
static const float kCompressionThreshold = 0.75;

public struct JMhParams {
    public float3x3 rgbToCam16c;
    public float3x3 cam16cToRgb;
    public float3x3 coneToAab;
    public float3x3 aabToCone;
    public float    F_L_n;
    public float    cz;
    public float    inv_cz;
    public float    A_w_J;
    public float    inv_A_w_J;
};

public struct TSParams {
    public float n;
    public float n_r;
    public float g;
    public float t_1;
    public float c_t;
    public float s_2;
    public float u_2;
    public float m_2;
    public float forward_limit;
    public float inverse_limit;
    public float log_peak;
};

public struct Aces2Params {
    public float     peakLuminance;
    public JMhParams input;    // ACES2065-1 in
    public JMhParams reach;    // AP1
    public JMhParams limit;    // the display's primaries
    public TSParams  ts;
    public float     limit_J_max;
    public float     model_gamma_inv;
    public float     sat;
    public float     sat_thr;
    public float     compr;
    public float     chroma_compress_scale;
    public float     mid_J;
    public float     focus_dist;
    public float     lower_hull_gamma_inv;
    public int       searchLo;   // hue_linearity_search_range
    public int       searchHi;
    public float3x3  sceneToAP0;   // the renderer's linear Rec.709 (D65) to ACES2065-1, Bradford
};

// --- Matrices ---------------------------------------------------------------

public float3x3 invert3(float3x3 m) {
    const float3 r0 = m[0];
    const float3 r1 = m[1];
    const float3 r2 = m[2];
    const float3 c0 = cross(r1, r2);
    const float3 c1 = cross(r2, r0);
    const float3 c2 = cross(r0, r1);
    const float det = dot(r0, c0);
    // The inverse's rows are the cofactor columns over the determinant.
    return float3x3(c0.x, c1.x, c2.x, c0.y, c1.y, c2.y, c0.z, c1.z, c2.z) / det;
}

/// The reference's RGBtoXYZ_f33: RGB (row vector) to XYZ for primaries C at white luminance Y.
public float3x3 rgbToXyz(Chromaticities C, float Y) {
    const float X = C.white.x * Y / C.white.y;
    const float Z = (1.0 - C.white.x - C.white.y) * Y / C.white.y;
    const float d = C.red.x * (C.blue.y - C.green.y) + C.blue.x * (C.green.y - C.red.y) +
                    C.green.x * (C.red.y - C.blue.y);
    const float Sr = (X * (C.blue.y - C.green.y) - C.green.x * (Y * (C.blue.y - 1.0) + C.blue.y * (X + Z)) +
                      C.blue.x * (Y * (C.green.y - 1.0) + C.green.y * (X + Z))) / d;
    const float Sg = (X * (C.red.y - C.blue.y) + C.red.x * (Y * (C.blue.y - 1.0) + C.blue.y * (X + Z)) -
                      C.blue.x * (Y * (C.red.y - 1.0) + C.red.y * (X + Z))) / d;
    const float Sb = (X * (C.green.y - C.red.y) - C.red.x * (Y * (C.green.y - 1.0) + C.green.y * (X + Z)) +
                      C.green.x * (Y * (C.red.y - 1.0) + C.red.y * (X + Z))) / d;
    return float3x3(Sr * C.red.x, Sr * C.red.y, Sr * (1.0 - C.red.x - C.red.y),
                    Sg * C.green.x, Sg * C.green.y, Sg * (1.0 - C.green.x - C.green.y),
                    Sb * C.blue.x, Sb * C.blue.y, Sb * (1.0 - C.blue.x - C.blue.y));
}

public float3x3 xyzToRgb(Chromaticities C, float Y) { return invert3(rgbToXyz(C, Y)); }

static const float3x3 kBradford = float3x3(0.89510, -0.75020, 0.03890,
                                           0.26640, 1.71350, -0.06850,
                                           -0.16140, 0.03670, 1.02960);

float3 xyToXyz(float2 xy) {
    return float3(xy.x / max(xy.y, 1e-10), 1.0, (1.0 - xy.x - xy.y) / max(xy.y, 1e-10));
}

/// The reference's calculate_cat_matrix with Bradford: a von Kries adaptation from one white to another.
public float3x3 catMatrix(float2 srcWhite, float2 dstWhite) {
    const float3 src = mul(xyToXyz(srcWhite), kBradford);
    const float3 dst = mul(xyToXyz(dstWhite), kBradford);
    const float3x3 vk = float3x3(dst.x / src.x, 0.0, 0.0, 0.0, dst.y / src.y, 0.0, 0.0, 0.0, dst.z / src.z);
    return mul(kBradford, mul(vk, invert3(kBradford)));
}

/// RGB in one set of primaries to RGB in another, whites reconciled (the reference's calculate_rgb_to_rgb_matrix).
public float3x3 rgbToRgb(Chromaticities from, Chromaticities to) {
    return mul(rgbToXyz(from, 1.0), mul(catMatrix(from.white, to.white), xyzToRgb(to, 1.0)));
}

// --- The colour appearance model (Hellwig 2022, as the reference specialises it) ---

float wrapTo360(float hue) {
    float y = fmod(hue, 360.0);
    if (y < 0.0) y += 360.0;
    return y;
}

int huePositionInUniformTable(float hue, int tableSize) {
    return int(wrapTo360(hue) / kHueLimit * float(tableSize));
}

float baseHueForPosition(int i_lo, int tableSize) { return float(i_lo) * kHueLimit / float(tableSize); }

float coneCompressionFwdAbs(float Rc) {
    const float F_L_Y = pow(Rc, 0.42);
    return F_L_Y / (kCamNlOffset + F_L_Y);
}

float coneCompressionInvAbs(float Ra) {
    const float Ra_lim = min(Ra, 0.99);
    const float F_L_Y = (kCamNlOffset * Ra_lim) / (1.0 - Ra_lim);
    return pow(F_L_Y, 1.0 / 0.42);
}

float copySign(float magnitude, float sign) { return sign < 0.0 ? -magnitude : magnitude; }

float coneCompressionFwd(float v) { return copySign(coneCompressionFwdAbs(abs(v)), v); }
float coneCompressionInv(float v) { return copySign(coneCompressionInvAbs(abs(v)), v); }

float achromaticToJ(float A, float cz) { return kJScale * pow(A, cz); }
float jToAchromatic(float J, float inv_cz) { return pow(J * (1.0 / kJScale), inv_cz); }

float aToY(float A, JMhParams p) {
    const float Ra = p.A_w_J * A;
    return coneCompressionInvAbs(Ra) / p.F_L_n;
}

public float jToY(float J, JMhParams p) { return aToY(jToAchromatic(abs(J), p.inv_cz), p); }

public float yToJ(float Y, JMhParams p) {
    const float Ra = coneCompressionFwdAbs(abs(Y) * p.F_L_n);
    return copySign(achromaticToJ(Ra * p.inv_A_w_J, p.cz), Y);
}

public float3 rgbToAab(float3 RGB, JMhParams p) {
    const float3 rgb_m = mul(RGB, p.rgbToCam16c);
    const float3 rgb_a = float3(coneCompressionFwd(rgb_m.x), coneCompressionFwd(rgb_m.y), coneCompressionFwd(rgb_m.z));
    return mul(rgb_a, p.coneToAab);
}

public float3 aabToJMh(float3 Aab, JMhParams p) {
    if (Aab.x <= 0.0) {
        return float3(0.0);
    }
    const float J = achromaticToJ(Aab.x, p.cz);
    const float M = sqrt(Aab.y * Aab.y + Aab.z * Aab.z);
    const float h = wrapTo360(degrees(atan2(Aab.z, Aab.y)));
    return float3(J, M, h);
}

public float3 rgbToJMh(float3 RGB, JMhParams p) { return aabToJMh(rgbToAab(RGB, p), p); }

public float3 jmhToAab(float3 JMh, JMhParams p) {
    const float h_rad = radians(JMh.z);
    const float A = jToAchromatic(JMh.x, p.inv_cz);
    return float3(A, JMh.y * cos(h_rad), JMh.y * sin(h_rad));
}

public float3 aabToRgb(float3 Aab, JMhParams p) {
    const float3 rgb_a = mul(Aab, p.aabToCone);
    const float3 rgb_m = float3(coneCompressionInv(rgb_a.x), coneCompressionInv(rgb_a.y), coneCompressionInv(rgb_a.z));
    return mul(rgb_m, p.cam16cToRgb);
}

public float3 jmhToRgb(float3 JMh, JMhParams p) { return aabToRgb(jmhToAab(JMh, p), p); }

/// The reference's init_JMhParams: the CAM's matrices for one set of primaries, adapted to its white.
public JMhParams initJMhParams(Chromaticities prims) {
    const float3x3 MATRIX_16 = xyzToRgb(kCAM16, 1.0);
    const float3x3 baseConeToAab = float3x3(2.0, 1.0, 1.0 / 9.0,
                                            1.0, -12.0 / 11.0, 1.0 / 9.0,
                                            1.0 / 20.0, 1.0 / 11.0, -2.0 / 9.0);
    const float3x3 RGB_TO_XYZ = rgbToXyz(prims, 1.0);
    const float3 XYZ_w = mul(float3(kRefLuminance), RGB_TO_XYZ);
    const float Y_w = XYZ_w.y;
    const float3 RGB_w = mul(XYZ_w, MATRIX_16);
    const float k = 1.0 / (5.0 * kL_A + 1.0);
    const float k4 = k * k * k * k;
    const float F_L = 0.2 * k4 * (5.0 * kL_A) + 0.1 * pow(1.0 - k4, 2.0) * pow(5.0 * kL_A, 1.0 / 3.0);
    const float F_L_n = F_L / kRefLuminance;
    const float3 D_RGB = F_L_n * Y_w / RGB_w;
    const float3 RGB_wc = D_RGB * RGB_w;
    const float3 RGB_Aw = float3(coneCompressionFwd(RGB_wc.x), coneCompressionFwd(RGB_wc.y), coneCompressionFwd(RGB_wc.z));
    const float3x3 coneToAab = kCamNlScale * baseConeToAab;
    const float A_w = coneToAab[0][0] * RGB_Aw.x + coneToAab[1][0] * RGB_Aw.y + coneToAab[2][0] * RGB_Aw.z;
    const float A_w_J = coneCompressionFwdAbs(F_L);
    const float3x3 rgbToCam16 = mul(mul(RGB_TO_XYZ, MATRIX_16), kRefLuminance * float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1));
    const float3x3 rgbToCam16c = mul(rgbToCam16, float3x3(D_RGB.x, 0, 0, 0, D_RGB.y, 0, 0, 0, D_RGB.z));
    const float s = 43.0 * kSurround.z;
    const float3x3 coneToAabScaled = float3x3(
        coneToAab[0][0] / A_w, coneToAab[0][1] * s, coneToAab[0][2] * s,
        coneToAab[1][0] / A_w, coneToAab[1][1] * s, coneToAab[1][2] * s,
        coneToAab[2][0] / A_w, coneToAab[2][1] * s, coneToAab[2][2] * s);
    JMhParams p;
    p.rgbToCam16c = rgbToCam16c;
    p.cam16cToRgb = invert3(rgbToCam16c);
    p.coneToAab = coneToAabScaled;
    p.aabToCone = invert3(coneToAabScaled);
    p.F_L_n = F_L_n;
    p.cz = kModelGamma;
    p.inv_cz = 1.0 / kModelGamma;
    p.A_w_J = A_w_J;
    p.inv_A_w_J = 1.0 / A_w_J;
    return p;
}

// --- The tonescale ---------------------------------------------------------

public TSParams initTSParams(float peakLuminance) {
    const float n = peakLuminance;
    const float n_r = 100.0;
    const float g = 1.15;
    const float c = 0.18;
    const float c_d = 10.013;
    const float w_g = 0.14;
    const float t_1 = 0.04;
    const float r_hit_min = 128.0;
    const float r_hit_max = 896.0;
    const float r_hit = r_hit_min + (r_hit_max - r_hit_min) * (log(n / n_r) / log(10000.0 / 100.0));
    const float m_0 = n / n_r;
    const float m_1 = 0.5 * (m_0 + sqrt(m_0 * (m_0 + 4.0 * t_1)));
    const float u = pow((r_hit / m_1) / ((r_hit / m_1) + 1.0), g);
    const float m = m_1 / u;
    const float w_i = log(n / 100.0) / log(2.0);
    const float c_t = c_d / n_r * (1.0 + w_i * w_g);
    const float g_ip = 0.5 * (c_t + sqrt(c_t * (c_t + 4.0 * t_1)));
    const float g_ipp2 = -(m_1 * pow(g_ip / m, 1.0 / g)) / (pow(g_ip / m, 1.0 / g) - 1.0);
    const float w_2 = c / g_ipp2;
    const float s_2 = w_2 * m_1;
    const float u_2 = pow((r_hit / m_1) / ((r_hit / m_1) + w_2), g);
    const float m_2 = m_1 / u_2;
    TSParams p;
    p.n = n;
    p.n_r = n_r;
    p.g = g;
    p.t_1 = t_1;
    p.c_t = c_t;
    p.s_2 = s_2;
    p.u_2 = u_2;
    p.m_2 = m_2;
    p.forward_limit = 8.0 * r_hit;
    p.inverse_limit = n / (u_2 * n_r);
    p.log_peak = log10(n / n_r);
    return p;
}

/// Scene-referred linear in, luminance in cd/m^2 out.
public float tonescaleFwd(float x, TSParams p) {
    const float f = p.m_2 * pow(max(0.0, x) / (x + p.s_2), p.g);
    const float h = max(0.0, f * f / (f + p.t_1));
    return h * p.n_r;
}

public float tonescaleInv(float Y, TSParams p) {
    const float Z = max(0.0, min(p.n / (p.u_2 * p.n_r), Y));
    const float h = (Z + sqrt(Z * (4.0 * p.t_1 + Z))) / 2.0;
    return p.s_2 / (pow(p.m_2 / h, 1.0 / p.g) - 1.0);
}

// --- Chroma compression -----------------------------------------------------

float reachMFromTable(float h, StructuredBuffer<float> tables) {
    const int base = huePositionInUniformTable(h, kTableSize);
    const float t = h - float(base);
    const int i_lo = base + kBaseIndex;
    return lerp(tables[kReachOffset + i_lo], tables[kReachOffset + i_lo + 1], t);
}

float reinhardRemap(float scale, float nd, bool invert) {
    if (invert) {
        return nd >= 1.0 ? scale : scale * -(nd / (nd - 1.0));
    }
    return scale * nd / (1.0 + nd);
}

float toe(float x, float limit, float k1_in, float k2_in, bool invert) {
    if (x > limit) return x;
    const float k2 = max(k2_in, 0.001);
    const float k1 = sqrt(k1_in * k1_in + k2 * k2);
    const float k3 = (limit + k1) / (limit + k2);
    if (invert) {
        return (x * x + k1 * x) / (k3 * (x + k2));
    }
    const float minus_b = k3 * x - k1;
    const float minus_c = k2 * k3 * x;
    return 0.5 * (minus_b + sqrt(minus_b * minus_b + 4.0 * minus_c));
}

float chromaCompressNorm(float h, float chroma_compress_scale) {
    const float hr = radians(h);
    const float a = cos(hr);
    const float b = sin(hr);
    const float cos_hr2 = a * a - b * b;
    const float sin_hr2 = 2.0 * a * b;
    const float cos_hr3 = 4.0 * a * a * a - 3.0 * a;
    const float sin_hr3 = 3.0 * b - 4.0 * b * b * b;
    const float M = 11.34072 * a + 16.46899 * cos_hr2 + 7.88380 * cos_hr3 + 14.66441 * b + -6.37224 * sin_hr2 +
                    9.19364 * sin_hr3 + 77.12896;
    return M * chroma_compress_scale;
}

float3 chromaCompressFwd(float3 JMh, float tonemapped_J, Aces2Params p, StructuredBuffer<float> tables) {
    const float J = JMh.x;
    const float M = JMh.y;
    const float h = JMh.z;
    float M_compr = M;
    if (M != 0.0) {
        const float nJ = tonemapped_J / p.limit_J_max;
        const float snJ = max(0.0, 1.0 - nJ);
        const float Mnorm = chromaCompressNorm(h, p.chroma_compress_scale);
        const float limit = pow(nJ, p.model_gamma_inv) * reachMFromTable(h, tables) / Mnorm;
        const float toe_limit = limit - 0.001;
        const float toe_snJ_sat = snJ * p.sat;
        const float toe_sqrt_nJ_sat_thr = sqrt(nJ * nJ + p.sat_thr);
        const float toe_nJ_compr = nJ * p.compr;
        M_compr = M * pow(tonemapped_J / J, p.model_gamma_inv);
        M_compr = M_compr / Mnorm;
        M_compr = limit - toe(limit - M_compr, toe_limit, toe_snJ_sat, toe_sqrt_nJ_sat_thr, false);
        M_compr = toe(M_compr, limit, toe_nJ_compr, snJ, false);
        M_compr = M_compr * Mnorm;
    }
    return float3(tonemapped_J, M_compr, h);
}

float3 chromaCompressInv(float3 JMh, float J, Aces2Params p, StructuredBuffer<float> tables) {
    const float tonemapped_J = JMh.x;
    const float M_compr = JMh.y;
    const float h = JMh.z;
    float M = M_compr;
    if (M_compr != 0.0) {
        const float nJ = tonemapped_J / p.limit_J_max;
        const float snJ = max(0.0, 1.0 - nJ);
        const float Mnorm = chromaCompressNorm(h, p.chroma_compress_scale);
        const float limit = pow(nJ, p.model_gamma_inv) * reachMFromTable(h, tables) / Mnorm;
        const float toe_limit = limit - 0.001;
        const float toe_snJ_sat = snJ * p.sat;
        const float toe_sqrt_nJ_sat_thr = sqrt(nJ * nJ + p.sat_thr);
        const float toe_nJ_compr = nJ * p.compr;
        M = M_compr / Mnorm;
        M = toe(M, limit, toe_nJ_compr, snJ, true);
        M = limit - toe(limit - M, toe_limit, toe_snJ_sat, toe_sqrt_nJ_sat_thr, true);
        M = M * Mnorm;
        M = M * pow(tonemapped_J / J, -p.model_gamma_inv);
    }
    return float3(J, M, h);
}

float3 tonemapAndCompressFwd(float3 JMh, Aces2Params p, StructuredBuffer<float> tables) {
    const float linear = jToY(JMh.x, p.input) / kRefLuminance;
    const float tonemapped_Y = tonescaleFwd(linear, p.ts);
    const float J_ts = yToJ(tonemapped_Y, p.input);
    return chromaCompressFwd(JMh, J_ts, p, tables);
}

float3 tonemapAndCompressInv(float3 JMh_tc, Aces2Params p, StructuredBuffer<float> tables) {
    const float luminance = jToY(JMh_tc.x, p.input);
    const float linear = tonescaleInv(luminance / kRefLuminance, p.ts);
    const float J = yToJ(linear * kRefLuminance, p.input);
    return chromaCompressInv(JMh_tc, J, p, tables);
}

// --- Gamut compression ------------------------------------------------------

float computeCompressionVectorSlope(float intersect_J, float focus_J, float limit_J_max, float slope_gain) {
    const float direction_scalar = intersect_J < focus_J ? intersect_J : limit_J_max - intersect_J;
    return direction_scalar * (intersect_J - focus_J) / (focus_J * slope_gain);
}

float solveJIntersect(float J, float M, float focusJ, float maxJ, float slope_gain) {
    const float M_scaled = M / slope_gain;
    const float a = M_scaled / focusJ;
    if (J < focusJ) {
        const float b = 1.0 - M_scaled;
        const float c = -J;
        const float det = b * b - 4.0 * a * c;
        const float root = sqrt(det);
        return -2.0 * c / (b + root);
    }
    const float b = -(1.0 + M_scaled + maxJ * a);
    const float c = maxJ * M_scaled + J;
    const float det = b * b - 4.0 * a * c;
    const float root = sqrt(det);
    return -2.0 * c / (b - root);
}

float sminScaled(float a, float b, float scale_reference) {
    const float s_scaled = kSmoothCusps * scale_reference;
    const float h = max(s_scaled - abs(a - b), 0.0) / s_scaled;
    return min(a, b) - h * h * h * s_scaled * (1.0 / 6.0);
}

float estimateLineAndBoundaryIntersectionM(float J_axis_intersect, float slope, float inv_gamma, float J_max,
                                           float M_max, float J_intersection_reference) {
    const float normalised_J = J_axis_intersect / J_intersection_reference;
    const float shifted_intersection = J_intersection_reference * pow(normalised_J, inv_gamma);
    return shifted_intersection * M_max / (J_max - slope * M_max);
}

float findGamutBoundaryIntersection(float2 JM_cusp, float J_max, float gamma_top_inv, float gamma_bottom_inv,
                                    float J_intersect_source, float slope, float J_intersect_cusp) {
    const float M_boundary_lower = estimateLineAndBoundaryIntersectionM(J_intersect_source, slope, gamma_bottom_inv,
                                                                        JM_cusp.x, JM_cusp.y, J_intersect_cusp);
    const float f_J_intersect_cusp = J_max - J_intersect_cusp;
    const float f_J_intersect_source = J_max - J_intersect_source;
    const float f_JM_cusp_J = J_max - JM_cusp.x;
    const float M_boundary_upper = estimateLineAndBoundaryIntersectionM(f_J_intersect_source, -slope, gamma_top_inv,
                                                                        f_JM_cusp_J, JM_cusp.y, f_J_intersect_cusp);
    return sminScaled(M_boundary_lower, M_boundary_upper, JM_cusp.y);
}

float getFocusGain(float J, float analytical_threshold, float limit_J_max, float focus_dist) {
    float gain = limit_J_max * focus_dist;
    if (J > analytical_threshold) {
        float gain_adjustment = log10((limit_J_max - analytical_threshold) / max(0.0001, limit_J_max - J));
        gain_adjustment = gain_adjustment * gain_adjustment + 1.0;
        gain = gain * gain_adjustment;
    }
    return gain;
}

float remapM(float M, float gamut_boundary_M, float reach_boundary_M, bool invert) {
    const float boundary_ratio = gamut_boundary_M / reach_boundary_M;
    const float proportion = max(boundary_ratio, kCompressionThreshold);
    const float threshold = proportion * gamut_boundary_M;
    if (M <= threshold || proportion >= 1.0) return M;
    const float m_offset = M - threshold;
    const float gamut_offset = gamut_boundary_M - threshold;
    const float reach_offset = reach_boundary_M - threshold;
    const float scale = reach_offset / ((reach_offset / gamut_offset) - 1.0);
    const float nd = m_offset / scale;
    return threshold + reinhardRemap(scale, nd, invert);
}

public struct HueDependentGamutParams {
    public float2 JMcusp;
    public float  gamma_bottom_inv;
    public float  gamma_top_inv;
    public float  focus_J;
    public float  analytical_threshold;
};

float3 compressGamut(float3 JMh, float Jx, Aces2Params p, HueDependentGamutParams hdp, bool invert,
                     StructuredBuffer<float> tables) {
    const float J = JMh.x;
    const float M = JMh.y;
    const float h = JMh.z;
    const float slope_gain = getFocusGain(Jx, hdp.analytical_threshold, p.limit_J_max, p.focus_dist);
    const float J_intersect_source = solveJIntersect(J, M, hdp.focus_J, p.limit_J_max, slope_gain);
    const float gamut_slope = computeCompressionVectorSlope(J_intersect_source, hdp.focus_J, p.limit_J_max, slope_gain);
    const float J_intersect_cusp = solveJIntersect(hdp.JMcusp.x, hdp.JMcusp.y, hdp.focus_J, p.limit_J_max, slope_gain);
    const float gamut_boundary_M = findGamutBoundaryIntersection(hdp.JMcusp, p.limit_J_max, hdp.gamma_top_inv,
                                                                 hdp.gamma_bottom_inv, J_intersect_source, gamut_slope,
                                                                 J_intersect_cusp);
    if (gamut_boundary_M <= 0.0) {
        return float3(J, 0.0, h);
    }
    const float reach_max_M = reachMFromTable(h, tables);
    const float reach_boundary_M = estimateLineAndBoundaryIntersectionM(J_intersect_source, gamut_slope,
                                                                        p.model_gamma_inv, p.limit_J_max, reach_max_M,
                                                                        p.limit_J_max);
    const float remapped_M = remapM(M, gamut_boundary_M, reach_boundary_M, invert);
    return float3(J_intersect_source + remapped_M * gamut_slope, remapped_M, h);
}

float2 cuspFromTable(float h, StructuredBuffer<float> tables) {
    int low_i = 0;
    int high_i = kBaseIndex + kTableSize;
    int i = huePositionInUniformTable(h, kTableSize) + kBaseIndex;
    while (low_i + 1 < high_i) {
        if (h > tables[kHuesOffset + i]) {
            low_i = i;
        } else {
            high_i = i;
        }
        i = (low_i + high_i) / 2;
    }
    const float lo_h = tables[kHuesOffset + high_i - 1];
    const float hi_h = tables[kHuesOffset + high_i];
    const float t = (h - lo_h) / (hi_h - lo_h);
    return float2(lerp(tables[kCuspJOffset + high_i - 1], tables[kCuspJOffset + high_i], t),
                  lerp(tables[kCuspMOffset + high_i - 1], tables[kCuspMOffset + high_i], t));
}

int lookupHueInterval(float h, int searchLo, int searchHi, StructuredBuffer<float> tables) {
    const int i0 = kBaseIndex + huePositionInUniformTable(h, kTotalTableSize);
    int i = i0;
    int i_lo = max(kBaseIndex, i0 + searchLo);
    int i_hi = min(kBaseIndex + kTableSize, i0 + searchHi);
    while (i_lo + 1 < i_hi) {
        if (h > tables[kHuesOffset + i]) {
            i_lo = i;
        } else {
            i_hi = i;
        }
        i = (i_lo + i_hi) / 2;
    }
    return max(1, i_hi);
}

public float computeFocusJ(float cusp_J, float mid_J, float limit_J_max) {
    return lerp(cusp_J, mid_J, min(1.0, kCuspMidBlend - (cusp_J / limit_J_max)));
}

HueDependentGamutParams initHueDependentGamutParams(float hue, Aces2Params p, StructuredBuffer<float> tables) {
    HueDependentGamutParams hdp;
    hdp.gamma_bottom_inv = p.lower_hull_gamma_inv;
    const int i_hi = lookupHueInterval(hue, p.searchLo, p.searchHi, tables);
    const float t = hue - tables[kHuesOffset + i_hi - 1];   // the reference's interpolation_weight, unnormalised
    hdp.JMcusp = cuspFromTable(hue, tables);
    hdp.gamma_top_inv = lerp(tables[kGammaOffset + i_hi - 1], tables[kGammaOffset + i_hi], t);
    hdp.focus_J = computeFocusJ(hdp.JMcusp.x, p.mid_J, p.limit_J_max);
    hdp.analytical_threshold = lerp(hdp.JMcusp.x, p.limit_J_max, kFocusGainBlend);
    return hdp;
}

float3 gamutCompressFwd(float3 JMh, Aces2Params p, StructuredBuffer<float> tables) {
    if (JMh.x <= 0.0) return float3(0.0, 0.0, JMh.z);
    if (JMh.y < 0.0 || JMh.x > p.limit_J_max) return float3(JMh.x, 0.0, JMh.z);
    const HueDependentGamutParams hdp = initHueDependentGamutParams(JMh.z, p, tables);
    return compressGamut(JMh, JMh.x, p, hdp, false, tables);
}

float3 gamutCompressInv(float3 JMh, Aces2Params p, StructuredBuffer<float> tables) {
    if (JMh.x <= 0.0) return float3(0.0, 0.0, JMh.z);
    if (JMh.y < 0.0 || JMh.x > p.limit_J_max) return float3(JMh.x, 0.0, JMh.z);
    const HueDependentGamutParams hdp = initHueDependentGamutParams(JMh.z, p, tables);
    float Jx = JMh.x;
    if (Jx > hdp.analytical_threshold) {
        Jx = compressGamut(JMh, Jx, p, hdp, true, tables).x;
    }
    return compressGamut(JMh, Jx, p, hdp, true, tables);
}

// --- The transform ----------------------------------------------------------

/// ACES2065-1 (1.0 is 100 nits) to the display's linear RGB in its limiting primaries (1.0 is 100 nits).
public float3 outputTransformFwd(float3 aces, Aces2Params p, StructuredBuffer<float> tables) {
    const float3x3 ap0ToAp1 = mul(rgbToXyz(kAP0, 1.0), xyzToRgb(kAP1, 1.0));
    const float3x3 ap1ToAp0 = mul(rgbToXyz(kAP1, 1.0), xyzToRgb(kAP0, 1.0));
    const float3 clampedAp0 = mul(clamp(mul(aces, ap0ToAp1), float3(0.0), float3(p.ts.forward_limit)), ap1ToAp0);
    const float3 JMh = rgbToJMh(clampedAp0, p.input);
    const float3 tonemapped = tonemapAndCompressFwd(JMh, p, tables);
    const float3 compressed = gamutCompressFwd(tonemapped, p, tables);
    return jmhToRgb(compressed, p.limit);
}

public float3 outputTransformInv(float3 rgbOut, Aces2Params p, StructuredBuffer<float> tables) {
    const float3 compressed = rgbToJMh(rgbOut, p.limit);
    const float3 tonemapped = gamutCompressInv(compressed, p, tables);
    const float3 JMh = tonemapAndCompressInv(tonemapped, p, tables);
    return jmhToRgb(JMh, p.input);
}

// --- Table building (aces2_prepare.slang's kernels call these) ---------------

/// The reference's generate_unit_cube_cusp_corners: R, Y, G, C, B, M.
public float3 unitCubeCuspCorner(int corner) {
    float3 result;
    result.x = ((corner + 1) % kCuspCorners) < 3 ? 1.0 : 0.0;
    result.y = ((corner + 5) % kCuspCorners) < 3 ? 1.0 : 0.0;
    result.z = ((corner + 3) % kCuspCorners) < 3 ? 1.0 : 0.0;
    return result;
}

/// The limiting gamut's cusp corners, in RGB and JMh, as a cycle with the lowest hue at [1].
public void buildLimitingCuspCorners(out float3 RGB_corners[kTotalCorners], out float3 JMh_corners[kTotalCorners],
                                     JMhParams params, float peakLuminance) {
    float3 temp_RGB[kCuspCorners];
    float3 temp_JMh[kCuspCorners];
    int min_index = 0;
    for (int i = 0; i < kCuspCorners; ++i) {
        temp_RGB[i] = (peakLuminance / kRefLuminance) * unitCubeCuspCorner(i);
        temp_JMh[i] = rgbToJMh(temp_RGB[i], params);
        if (temp_JMh[i].z < temp_JMh[min_index].z) min_index = i;
    }
    for (int i = 0; i < kCuspCorners; ++i) {
        RGB_corners[i + 1] = temp_RGB[(i + min_index) % kCuspCorners];
        JMh_corners[i + 1] = temp_JMh[(i + min_index) % kCuspCorners];
    }
    RGB_corners[0] = RGB_corners[kCuspCorners];
    RGB_corners[kCuspCorners + 1] = RGB_corners[1];
    JMh_corners[0] = JMh_corners[kCuspCorners];
    JMh_corners[kCuspCorners + 1] = JMh_corners[1];
    JMh_corners[0].z -= kHueLimit;
    JMh_corners[kCuspCorners + 1].z += kHueLimit;
}

/// The reach primaries' corners at limit J, as a cycle with the lowest hue at [1].
public void findReachCorners(out float3 JMh_corners[kTotalCorners], JMhParams params_reach, float limit_J_max,
                             float forward_limit) {
    float3 temp_JMh[kCuspCorners];
    const float limitA = jToAchromatic(limit_J_max, params_reach.inv_cz);
    int min_index = 0;
    for (int i = 0; i < kCuspCorners; ++i) {
        const float3 rgb_vector = unitCubeCuspCorner(i);
        float lower = 0.0;
        float upper = forward_limit;
        while ((upper - lower) > kReachCuspTolerance) {
            const float test = (lower + upper) / 2.0;
            const float A = rgbToAab(test * rgb_vector, params_reach).x;
            if (A < limitA) {
                lower = test;
            } else {
                upper = test;
            }
        }
        temp_JMh[i] = rgbToJMh(upper * rgb_vector, params_reach);
        if (temp_JMh[i].z < temp_JMh[min_index].z) min_index = i;
    }
    for (int i = 0; i < kCuspCorners; ++i) {
        JMh_corners[i + 1] = temp_JMh[(i + min_index) % kCuspCorners];
    }
    JMh_corners[0] = JMh_corners[kCuspCorners];
    JMh_corners[kCuspCorners + 1] = JMh_corners[1];
    JMh_corners[0].z -= kHueLimit;
    JMh_corners[kCuspCorners + 1].z += kHueLimit;
}

/// The two corner cycles' hues merged in order (the reference's extract_sorted_cube_hues).
public void extractSortedCubeHues(out float sorted_hues[kMaxSortedCorners], float3 reach_JMh[kTotalCorners],
                                  float3 limit_JMh[kTotalCorners]) {
    for (int k = 0; k < kMaxSortedCorners; ++k) sorted_hues[k] = 0.0;
    int idx = 0;
    int reach_idx = 1;
    int limit_idx = 1;
    while ((reach_idx < (kCuspCorners + 1)) || (limit_idx < (kCuspCorners + 1))) {
        const float reach_hue = reach_JMh[reach_idx].z;
        const float limit_hue = limit_JMh[limit_idx].z;
        if (reach_hue == limit_hue) {
            sorted_hues[idx] = reach_hue;
            reach_idx += 1;
            limit_idx += 1;
        } else if (reach_hue < limit_hue) {
            sorted_hues[idx] = reach_hue;
            reach_idx += 1;
        } else {
            sorted_hues[idx] = limit_hue;
            limit_idx += 1;
        }
        idx += 1;
        if (idx >= kMaxSortedCorners) break;
    }
}

float roundHalfUp(float x) { return x < 0.0 ? float(int(x - 0.5)) : float(int(x + 0.5)); }

/// The reference's build_hue_table: hues sampled as uniformly as the corners allow, into
/// `hues[kTotalTableSize]` (the wrap entries included).
public void buildHueTable(out float hues[kTotalTableSize], float sorted_hues[kMaxSortedCorners]) {
    for (int k = 0; k < kTotalTableSize; ++k) hues[k] = 0.0;
    const float ideal_spacing = float(kTableSize) / kHueLimit;
    int samples_count[kMaxSortedCorners];
    for (int k = 0; k < kMaxSortedCorners; ++k) samples_count[k] = 0;
    int last_idx = 0;
    int min_index = sorted_hues[0] == 0.0 ? 0 : 1;
    for (int hue_idx = 0; hue_idx < kMaxSortedCorners; ++hue_idx) {
        int nominal_idx = min(max(int(roundHalfUp(sorted_hues[hue_idx] * ideal_spacing)), min_index), kTableSize - 1);
        if (last_idx == nominal_idx) {
            if (hue_idx > 1 && samples_count[hue_idx - 2] != (samples_count[hue_idx - 1] - 1)) {
                samples_count[hue_idx - 1] = samples_count[hue_idx - 1] - 1;
            } else {
                nominal_idx = nominal_idx + 1;
            }
        }
        samples_count[hue_idx] = min(nominal_idx, kTableSize - 1);
        min_index = nominal_idx;
        last_idx = min_index;
    }
    int total_samples = 0;
    // An interval of `samples` hues from `lower` towards `upper`, from table entry `base`.
    {
        const int samples = samples_count[0];
        const float delta = (sorted_hues[0] - 0.0) / float(samples);
        for (int i = 0; i < samples; ++i) hues[total_samples + 1 + i] = 0.0 + float(i) * delta;
        total_samples += samples;
    }
    int i = 1;
    for (; i < kMaxSortedCorners; ++i) {
        const int samples = samples_count[i] - samples_count[i - 1];
        const float delta = (sorted_hues[i] - sorted_hues[i - 1]) / float(samples);
        for (int k = 0; k < samples; ++k) hues[total_samples + 1 + k] = sorted_hues[i - 1] + float(k) * delta;
        total_samples += samples;
    }
    {
        const int samples = kTableSize - total_samples;
        const float delta = (kHueLimit - sorted_hues[i - 1]) / float(samples);
        for (int k = 0; k < samples; ++k) hues[total_samples + 1 + k] = sorted_hues[i - 1] + float(k) * delta;
    }
    hues[0] = hues[kBaseIndex + kTableSize - 1] - kHueLimit;
    hues[kBaseIndex + kTableSize] = hues[kBaseIndex] + kHueLimit;
}

/// The display gamut's cusp (J, M) at a hue, by bisection along the cube edge between two corners.
public float2 findDisplayCuspForHue(float hue, float3 RGB_corners[kTotalCorners], float3 JMh_corners[kTotalCorners],
                                    JMhParams params) {
    int upper_corner = 1;
    bool found = false;
    for (int i = 1; i < kTotalCorners && !found; ++i) {
        if (JMh_corners[i].z > hue) {
            upper_corner = i;
            found = true;
        }
    }
    const int lower_corner = upper_corner - 1;
    if (JMh_corners[lower_corner].z == hue) {
        return float2(JMh_corners[lower_corner].x, JMh_corners[lower_corner].y);
    }
    const float3 cusp_lower = RGB_corners[lower_corner];
    const float3 cusp_upper = RGB_corners[upper_corner];
    float lower_t = 0.0;
    float upper_t = 1.0;
    float sample_t;
    float3 JMh;
    while ((upper_t - lower_t) > kDisplayCuspTolerance) {
        sample_t = (lower_t + upper_t) / 2.0;
        JMh = rgbToJMh(lerp(cusp_lower, cusp_upper, sample_t), params);
        if (JMh.z < JMh_corners[lower_corner].z) {
            upper_t = sample_t;
        } else if (JMh.z >= JMh_corners[upper_corner].z) {
            lower_t = sample_t;
        } else if (JMh.z > hue) {
            upper_t = sample_t;
        } else {
            lower_t = sample_t;
        }
    }
    sample_t = (lower_t + upper_t) / 2.0;
    JMh = rgbToJMh(lerp(cusp_lower, cusp_upper, sample_t), params);
    return float2(JMh.x, JMh.y);
}

/// The reach gamut's M at limit J for one hue (the reference's make_reach_m_table, one entry).
public float reachMForHue(float hue, JMhParams params, float limitJmax) {
    const float search_range = 50.0;
    const float search_maximum = 1300.0;
    float low = 0.0;
    float high = low + search_range;
    bool outside = false;
    while ((outside != true) && (high < search_maximum)) {
        const float3 rgb = jmhToRgb(float3(limitJmax, high, hue), params);
        outside = any(rgb < float3(0.0));
        if (!outside) {
            low = high;
            high = high + search_range;
        }
    }
    while (high - low > 1e-2) {
        const float sampleM = (high + low) / 2.0;
        const float3 rgb = jmhToRgb(float3(limitJmax, sampleM, hue), params);
        outside = any(rgb < float3(0.0));
        if (outside) {
            high = sampleM;
        } else {
            low = sampleM;
        }
    }
    return high;
}

static const int kGammaTests = 5;
static const float kTestPositions[kGammaTests] = {0.01, 0.1, 0.5, 0.8, 0.99};

bool evaluateGammaFit(float2 JMcusp, float hue, float top_gamma_inv, Aces2Params p) {
    const float luminance_limit = p.peakLuminance / kRefLuminance;
    const float analytical_threshold = lerp(JMcusp.x, p.limit_J_max, kFocusGainBlend);
    const float focus_J = computeFocusJ(JMcusp.x, p.mid_J, p.limit_J_max);
    for (int testIndex = 0; testIndex < kGammaTests; ++testIndex) {
        const float test_J = lerp(JMcusp.x, p.limit_J_max, kTestPositions[testIndex]);
        const float slope_gain = getFocusGain(test_J, analytical_threshold, p.limit_J_max, p.focus_dist);
        const float J_intersect = solveJIntersect(test_J, JMcusp.y, focus_J, p.limit_J_max, slope_gain);
        const float slope = computeCompressionVectorSlope(J_intersect, focus_J, p.limit_J_max, slope_gain);
        const float J_cusp = solveJIntersect(JMcusp.x, JMcusp.y, focus_J, p.limit_J_max, slope_gain);
        const float approxLimit_M = findGamutBoundaryIntersection(JMcusp, p.limit_J_max, top_gamma_inv,
                                                                  p.lower_hull_gamma_inv, J_intersect, slope, J_cusp);
        const float approxLimit_J = J_intersect + slope * approxLimit_M;
        const float3 newLimitRGB = jmhToRgb(float3(approxLimit_J, approxLimit_M, hue), p.limit);
        if (!any(newLimitRGB > float3(luminance_limit))) return false;
    }
    return true;
}

/// The upper hull's gamma at one cusp entry (the reference's make_upper_hull_gamma_table, one entry).
public float upperHullGammaForCusp(float2 JMcusp, float hue, Aces2Params p) {
    float low = kGammaMinimum;
    float high = low + kGammaSearchStep;
    bool outside = false;
    while (!outside && (high < kGammaMaximum)) {
        if (!evaluateGammaFit(JMcusp, hue, 1.0 / high, p)) {
            low = high;
            high = high + kGammaSearchStep;
        } else {
            outside = true;
        }
    }
    while ((high - low) > kGammaAccuracy) {
        const float testGamma = (high + low) / 2.0;
        if (evaluateGammaFit(JMcusp, hue, 1.0 / testGamma, p)) {
            high = testGamma;
        } else {
            low = testGamma;
        }
    }
    return 1.0 / high;
}

/// The scalar parameters the reference's init_ODTParams computes, the tables aside.
public Aces2Params initAces2Params(float peakLuminance, Chromaticities limiting, Chromaticities scene) {
    Aces2Params p;
    p.peakLuminance = peakLuminance;
    p.input = initJMhParams(kAP0);
    p.reach = initJMhParams(kAP1);
    p.limit = initJMhParams(limiting);
    p.ts = initTSParams(peakLuminance);
    p.limit_J_max = yToJ(peakLuminance, p.input);
    p.model_gamma_inv = 1.0 / kModelGamma;
    p.sat = max(0.2, kChromaExpand - (kChromaExpand * kChromaExpandFact) * p.ts.log_peak);
    p.sat_thr = kChromaExpandThr / peakLuminance;
    p.compr = kChromaCompress + (kChromaCompress * kChromaCompressFact) * p.ts.log_peak;
    p.chroma_compress_scale = pow(0.03379 * peakLuminance, 0.30596) - 0.45135;
    p.mid_J = yToJ(p.ts.c_t * kRefLuminance, p.input);
    p.focus_dist = kFocusDistance + kFocusDistance * kFocusDistanceScaling * p.ts.log_peak;
    const float lower_hull_gamma = 1.14 + 0.07 * p.ts.log_peak;
    p.lower_hull_gamma_inv = 1.0 / lower_hull_gamma;
    p.searchLo = 0;
    p.searchHi = 1;
    p.sceneToAP0 = rgbToRgb(scene, kAP0);
    return p;
}

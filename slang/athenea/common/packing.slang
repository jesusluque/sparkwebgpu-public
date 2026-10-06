// Copyright (c) 2026 jesus luque.
//
// How a Gaussian is stored on the GPU, in one place.
//
//   positions  float4   xyz, opacity (linear, 0..1)            16 bytes
//   shape      uint * 4                                         16 bytes
//     [0] rotation, smallest-three: 10+10+10 bits and the dropped index
//     [1] f16 log(scale.x) | f16 log(scale.y) << 16
//     [2] f16 log(scale.z) | f16 base colour r << 16
//     [3] f16 base colour g | f16 base colour b << 16
//   sh         uint * words   f16 rest coefficients, rgb per basis
//   normals    uint           optional: the shading normal, octahedral
//                             2 x 16 bits (`packNormal`), apart from the frame
//   emission   uint           optional: the radiance the surface gives off by
//                             itself, linear RGB9E5 (`packRgb9e5`)
//   lobes      uint * 3       optional: the layers over the base a material
//                             names -- specular weight and colour, coat,
//                             sheen -- a byte a value (`packLobes`)
//
// 32 bytes before harmonics, against openFXplayer's 64 float32. Why each
// field is what it is:
//   - position stays float32: fp16 would step two pixels at 4K close up
//     (openFXplayer measured it), and it is what a camera flies into.
//   - scale is stored as a logarithm in fp16: 0.1% relative precision at every
//     size, where fp16 of the scale itself loses thin splats below 1e-4.
//   - rotation in 10-bit smallest-three is within a tenth of a degree.
//   - colours are fp16: base colour can leave 0..1, and harmonics are signed.
module packing;

public static const float kSqrtHalf = 0.70710678;

/// Smallest-three quaternion (x, y, z, w) -> uint.
public uint encodeQuaternion(float4 q) {
    q = normalize(q);
    uint largest = 0;
    float best = abs(q.x);
    for (uint k = 1; k < 4; ++k) {
        if (abs(q[k]) > best) {
            best = abs(q[k]);
            largest = k;
        }
    }
    if (q[largest] < 0.0) {
        q = -q;
    }
    uint word = largest;
    uint shift = 22;
    for (uint k = 0; k < 4; ++k) {
        if (k == largest) {
            continue;
        }
        const float unit = clamp(q[k] / kSqrtHalf * 0.5 + 0.5, 0.0, 1.0);
        word |= uint(round(unit * 1023.0)) << shift;
        shift -= 10;
    }
    return word;
}

public float4 decodeQuaternion(uint word) {
    const uint largest = word & 3u;
    float3 small = float3(float((word >> 22) & 1023u), float((word >> 12) & 1023u),
                          float((word >> 2) & 1023u));
    small = (small / 1023.0 * 2.0 - 1.0) * kSqrtHalf;
    const float rest = sqrt(max(0.0, 1.0 - dot(small, small)));
    float4 q = float4(0.0);
    uint next = 0;
    for (uint k = 0; k < 4; ++k) {
        if (k == largest) {
            q[k] = rest;
        } else {
            q[k] = small[next];
            next += 1;
        }
    }
    return q;
}

/// A rotation matrix whose columns are the axes, as (x, y, z, w).
///
/// It lives here beside the quaternion's own packing because two things now
/// need it: the LOD merge, which takes the eigenframe of a merged covariance,
/// and the splat skinner, which takes the frame a joint left a gaussian in.
public float4 quaternionOfAxes(float3x3 m) {
    const float trace = m[0][0] + m[1][1] + m[2][2];
    float4 q;
    if (trace > 0.0) {
        const float s = 0.5 / sqrt(trace + 1.0);
        q = float4((m[2][1] - m[1][2]) * s, (m[0][2] - m[2][0]) * s, (m[1][0] - m[0][1]) * s, 0.25 / s);
    } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        const float s = 2.0 * sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]);
        q = float4(0.25 * s, (m[1][0] + m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s);
    } else if (m[1][1] > m[2][2]) {
        const float s = 2.0 * sqrt(1.0 - m[0][0] + m[1][1] - m[2][2]);
        q = float4((m[1][0] + m[0][1]) / s, 0.25 * s, (m[2][1] + m[1][2]) / s, (m[0][2] - m[2][0]) / s);
    } else {
        const float s = 2.0 * sqrt(1.0 - m[0][0] - m[1][1] + m[2][2]);
        q = float4((m[0][2] + m[2][0]) / s, (m[2][1] + m[1][2]) / s, 0.25 * s, (m[1][0] - m[0][1]) / s);
    }
    return normalize(q);
}

/// The other way: the three axes a quaternion turns the basis into, as the
/// columns of a matrix.
public float3x3 axesOfQuaternion(float4 q) {
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    return float3x3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w),
                    2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
                    2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y));
}

/// A SHADING NORMAL IN ONE WORD (`GpuSplats::normals`): the octahedral map
/// (common/octahedral.slang's, written out here so packing imports nothing),
/// x in the low sixteen bits and y in the high, each a unorm. A step is
/// 2^-15 of the square, a few thousandths of a degree -- finer than the
/// 10-bit rotation the frame keeps. A zero vector packs as +z.
public uint packNormal(float3 n) {
    const float l1 = abs(n.x) + abs(n.y) + abs(n.z);
    n = l1 > 1.0e-20 ? n / l1 : float3(0.0, 0.0, 1.0);
    float2 p = n.xy;
    if (n.z < 0.0) {
        p = (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    }
    const uint2 q = uint2(round(saturate(p * 0.5 + 0.5) * 65535.0));
    return q.x | (q.y << 16);
}

/// The other way: a unit vector.
public float3 unpackNormal(uint word) {
    const float2 f = float2(float(word & 0xFFFFu), float(word >> 16)) / 65535.0 * 2.0 - 1.0;
    float3 n = float3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    const float t = saturate(-n.z);
    n.x += n.x >= 0.0 ? -t : t;
    n.y += n.y >= 0.0 ? -t : t;
    return normalize(n);
}

/// LIGHT IN ONE WORD (`GpuSplats::emission`): RGB9E5, three 9-bit mantissas
/// under one 5-bit exponent, as the shared-exponent texture format has it.
/// Unsigned and HDR -- up to 65408, down to 2^-24 -- with every channel
/// stepped by 1/512 of the brightest, which is finer than f16 on the channel
/// that is seen. Negative and not-a-number are nothing; past the top is the
/// top.
public uint packRgb9e5(float3 rgb) {
    const float kTop = 65408.0;   // (511 / 512) * 2^16
    const float3 c = float3(rgb.x > 0.0 ? min(rgb.x, kTop) : 0.0, rgb.y > 0.0 ? min(rgb.y, kTop) : 0.0,
                            rgb.z > 0.0 ? min(rgb.z, kTop) : 0.0);
    const float largest = max(c.x, max(c.y, c.z));
    if (!(largest >= 5.9604645e-8)) {   // half the smallest step there is
        return 0u;
    }
    // The exponent that puts the largest channel's mantissa below 512, and
    // one more where rounding takes it to 512.
    int e = max(-16, int(floor(log2(largest)))) + 16;
    float step = exp2(float(e - 24));
    if (uint(floor(largest / step + 0.5)) >= 512u) {
        e += 1;
        step *= 2.0;
    }
    const uint3 m = min(uint3(floor(c / step + 0.5)), uint3(511u));
    return m.x | (m.y << 9) | (m.z << 18) | (uint(e) << 27);
}

/// The other way: linear RGB.
public float3 unpackRgb9e5(uint word) {
    const float step = exp2(float(int(word >> 27) - 24));
    return float3(float(word & 511u), float((word >> 9) & 511u), float((word >> 18) & 511u)) * step;
}

/// WHAT A MATERIAL LAYERS OVER ITS BASE, PER GAUSSIAN (`GpuSplats::lobes`).
///
/// The base -- colour, metallic, roughness, transmission -- is `pbr`'s word.
/// What OpenPBR, standard_surface and UsdPreviewSurface put over it is here,
/// in the units their own inputs take, for the lobes `splat_relight` already
/// evaluates: no lobe is new, these are the weights and parameters of the
/// dielectric, conductor and sheen it has.
///
///   specularWeight  0..1   scales the dielectric reflection
///   specularColour  0..1   tints the dielectric reflection; a metal's edge
///   specularIor     1..3   the dielectric's index (0.04 head on at 1.5)
///   coatWeight      0..1   a clear dielectric layer over all of it
///   coatRoughness   0..1
///   coatIor         1..3
///   coatDarkening   0..1   OpenPBR's `coat_darkening`: the base under the
///                          coat darkened by the light the coat's inside
///                          sends back to it (on at 0.5 and above)
///   sheenColour     0..1   the sheen's colour times its weight
///   sheenRoughness  0..1
public struct SplatLobes {
    public float  specularWeight;
    public float3 specularColour;
    public float  specularIor;
    public float  coatWeight;
    public float  coatRoughness;
    public float  coatIor;
    public float  coatDarkening;
    public float3 sheenColour;
    public float  sheenRoughness;
};

/// What a gaussian with no lobes reflects with: the specular of every
/// cloud before these existed -- weight one, white, an index of 1.5 -- and
/// no coat and no sheen. Packed, it unpacks to exactly this (1.5 is a whole
/// step of the index's byte), so a cloud that carries it and one that does
/// not are the same picture.
public SplatLobes plainLobes() {
    SplatLobes l;
    l.specularWeight = 1.0;
    l.specularColour = float3(1.0);
    l.specularIor = 1.5;
    l.coatWeight = 0.0;
    l.coatRoughness = 0.0;
    l.coatIor = 1.5;
    l.coatDarkening = 0.0;
    l.sheenColour = float3(0.0);
    l.sheenRoughness = 0.0;
    return l;
}

uint lobeByte(float v) {
    return uint(saturate(v) * 255.0 + 0.5);
}

/// An index of refraction in a byte: `1 + byte / 128`, 1 to 2.99 in steps of
/// 0.008 -- 1.5 and 1.25 exactly, every glass and lacquer there is.
uint iorByte(float ior) {
    return uint(clamp((ior - 1.0) * 128.0 + 0.5, 0.0, 255.0));
}

float byteIor(uint b) {
    return 1.0 + float(b & 0xFFu) / 128.0;
}

/// The coat's index in seven bits, `1 + bits / 64` (1 to 2.98, 1.5 and 1.25
/// exactly), and whether the coat darkens what is under it in the eighth.
uint coatByte(float ior, float darkening) {
    return uint(clamp((ior - 1.0) * 64.0 + 0.5, 0.0, 127.0)) | (darkening >= 0.5 ? 128u : 0u);
}

/// Three words: (specular colour, weight), (coat weight, roughness, index,
/// the specular's index), (sheen colour, roughness).
public uint3 packLobes(SplatLobes l) {
    const uint w0 = lobeByte(l.specularColour.x) | (lobeByte(l.specularColour.y) << 8) |
                    (lobeByte(l.specularColour.z) << 16) | (lobeByte(l.specularWeight) << 24);
    const uint w1 = lobeByte(l.coatWeight) | (lobeByte(l.coatRoughness) << 8) |
                    (coatByte(l.coatIor, l.coatDarkening) << 16) |
                    (iorByte(l.specularIor) << 24);
    const uint w2 = lobeByte(l.sheenColour.x) | (lobeByte(l.sheenColour.y) << 8) |
                    (lobeByte(l.sheenColour.z) << 16) | (lobeByte(l.sheenRoughness) << 24);
    return uint3(w0, w1, w2);
}

float byteOf(uint word, uint at) {
    return float((word >> (at * 8)) & 0xFFu) / 255.0;
}

public SplatLobes unpackLobes(uint3 w) {
    SplatLobes l;
    l.specularColour = float3(byteOf(w.x, 0), byteOf(w.x, 1), byteOf(w.x, 2));
    l.specularWeight = byteOf(w.x, 3);
    l.coatWeight = byteOf(w.y, 0);
    l.coatRoughness = byteOf(w.y, 1);
    l.coatIor = 1.0 + float((w.y >> 16) & 0x7Fu) / 64.0;
    l.coatDarkening = ((w.y >> 23) & 1u) != 0u ? 1.0 : 0.0;
    l.specularIor = byteIor(w.y >> 24);
    l.sheenColour = float3(byteOf(w.z, 0), byteOf(w.z, 1), byteOf(w.z, 2));
    l.sheenRoughness = byteOf(w.z, 3);
    return l;
}

/// The thirteen floats a record keeps them in (`io::SplatEncoding::lobes`),
/// in this order: specular weight, colour (3), index; coat weight, roughness,
/// index; sheen colour (3), roughness; coat darkening.
public static const uint kLobeFields = 13;

public uint packHalves(float a, float b) {
    return (f32tof16(a) & 0xFFFFu) | ((f32tof16(b) & 0xFFFFu) << 16);
}

public float lowHalf(uint word) { return f16tof32(word & 0xFFFFu); }
public float highHalf(uint word) { return f16tof32(word >> 16); }

/// f16 can hold +-65504; anything past that becomes infinity, which a
/// covariance turns into NaN. Clamped, not trusted.
public float halfSafe(float v) { return clamp(v, -65000.0, 65000.0); }

public static const float kSH0 = 0.28209479177387814;

/// Harmonic words per splat for `restPerColour` basis functions (3, 8, 15).
public uint shWordsFor(uint restPerColour) { return (restPerColour * 3 + 1) / 2; }

/// A texel written as the texture's own bytes, for a device whose store into
/// a texture does not convert (CUDA: surf2Dwrite writes the float4's bytes).
/// `kind`: 1 four 8-bit unorms in one word, 2 four halves in two, 3 four
/// floats in four. The words are the texture's rows packed, row 0 first,
/// which is what copyBufferToTexture takes.
public void atheneaPackTexel(RWStructuredBuffer<uint> packed, uint kind, uint index, float4 texel) {
    if (kind == 1) {
        const uint4 bytes = uint4(round(saturate(texel) * 255.0));
        packed[index] = bytes.x | (bytes.y << 8) | (bytes.z << 16) | (bytes.w << 24);
    } else if (kind == 2) {
        packed[index * 2] = (f32tof16(texel.x) & 0xFFFFu) | ((f32tof16(texel.y) & 0xFFFFu) << 16);
        packed[index * 2 + 1] = (f32tof16(texel.z) & 0xFFFFu) | ((f32tof16(texel.w) & 0xFFFFu) << 16);
    } else {
        packed[index * 4] = asuint(texel.x);
        packed[index * 4 + 1] = asuint(texel.y);
        packed[index * 4 + 2] = asuint(texel.z);
        packed[index * 4 + 3] = asuint(texel.w);
    }
}

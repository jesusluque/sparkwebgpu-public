// Copyright (c) 2026 jesus luque.
//
// A direction as a point of the unit square, and back.
//
// The octahedral map folds the sphere onto a square with no pole and one seam
// that runs along the square's edges, which is why a baked field of
// directions is stored in it rather than in a cube map: fewer seams at
// grazing angles and one texture, not six (Deep Gaussian Shadow Maps compared
// the two and kept this one). Cigolle et al., "A Survey of Efficient
// Representations for Independent Unit Vectors", JCGT 2014.
module octahedral;

float2 octWrap(float2 v) {
    return (1.0 - abs(v.yx)) * float2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}

/// A unit direction to the square [0, 1]^2.
public float2 octEncode(float3 n) {
    n /= max(abs(n.x) + abs(n.y) + abs(n.z), 1.0e-20);
    float2 p = n.z >= 0.0 ? n.xy : octWrap(n.xy);
    return p * 0.5 + 0.5;
}

/// The square [0, 1]^2 back to a unit direction.
public float3 octDecode(float2 f) {
    f = f * 2.0 - 1.0;
    float3 n = float3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    const float t = saturate(-n.z);
    n.x += n.x >= 0.0 ? -t : t;
    n.y += n.y >= 0.0 ? -t : t;
    return normalize(n);
}

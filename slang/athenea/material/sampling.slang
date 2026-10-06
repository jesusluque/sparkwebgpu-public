// Copyright (c) 2026 jesus luque.
//
// Random numbers and the hemisphere and frame helpers lobes sample with.
module sampling;

public static const float kPi = 3.14159265358979;
public static const float kInvPi = 0.318309886183791;

/// PCG (O'Neill), one 32-bit state per path or sample stream.
public struct Rng {
    public uint state;

    public __init(uint seed, uint stream) {
        state = 0;
        next();
        state += seed * 747796405u + stream * 2891336453u;
        next();
    }

    [mutating]
    public uint nextUint() {
        const uint old = state;
        state = old * 747796405u + 2891336453u;
        uint word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
        return (word >> 22u) ^ word;
    }

    /// [0, 1)
    [mutating]
    public float next() {
        return float(nextUint() >> 8) * (1.0 / 16777216.0);
    }
}

public float3 sampleCosineHemisphere(float2 u) {
    const float phi = 2.0 * kPi * u.x;
    const float cosTheta = sqrt(u.y);
    const float sinTheta = sqrt(max(0.0, 1.0 - u.y));
    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

public float3 sampleUniformSphere(float2 u) {
    const float z = 1.0 - 2.0 * u.y;
    const float r = sqrt(max(0.0, 1.0 - z * z));
    const float phi = 2.0 * kPi * u.x;
    return float3(r * cos(phi), r * sin(phi), z);
}

/// An orthonormal frame about a unit normal, its tangent along `tangent`
/// where that is not parallel to it.
public struct Frame {
    public float3 x;
    public float3 y;
    public float3 z;

    public __init(float3 normal, float3 tangent) {
        z = normal;
        float3 t = tangent - dot(tangent, normal) * normal;
        const float length2 = dot(t, t);
        if (length2 < 1e-12) {
            // Duff et al. 2017, "Building an Orthonormal Basis, Revisited".
            const float s = normal.z >= 0.0 ? 1.0 : -1.0;
            const float a = -1.0 / (s + normal.z);
            const float b = normal.x * normal.y * a;
            t = float3(1.0 + s * normal.x * normal.x * a, s * b, -s * normal.x);
        } else {
            t /= sqrt(length2);
        }
        x = t;
        y = cross(normal, t);
    }

    public float3 toLocal(float3 w) {
        return float3(dot(w, x), dot(w, y), dot(w, z));
    }

    public float3 toWorld(float3 w) {
        return x * w.x + y * w.y + z * w.z;
    }
}

// Copyright (c) 2026 jesus luque.
//
// The scene's lights as the device has them, in world space, and how a
// shading point samples one: a direction, the distance to shadow along, the
// radiance arriving and the solid-angle density it was drawn with. A light's
// colour is derived here, not on the host -- exposure, and the blackbody of a
// colour temperature -- so a record carries what USD authored.
module lights;
// Nothing here reads a texture: a dome is its colour. The image, and the warp
// that samples by it, live in athenea/light/lights_image.slang, which the kernels
// that have a texture table import instead. That way a kernel without one --
// the splat projection, which lives below material in the module order -- can
// still light a surface.

public static const uint kLightDistant = 0;
public static const uint kLightSphere  = 1;
public static const uint kLightDisk    = 2;
public static const uint kLightRect    = 3;
public static const uint kLightCylinder = 5;   // along its x axis; the lateral surface, emitting outward
public static const uint kLightDome    = 4;

/// A light with no collection reaches every prim.
public static const uint kLightUnlinked = 0xFFFFFFFF;

public static const uint kLightShadow      = 1;   // casts one
public static const uint kLightNormalize   = 2;   // divide the radiance by the area (a sun: its disc's projected solid angle)
public static const uint kLightTemperature = 4;   // tint by the blackbody of `temperature`
public static const uint kLightIesNormalize = 8;   // divide the IES intensity by the profile's power

public struct LightRecord {
    public uint   kind;
    public uint   flags;
    public float  sizeX;         // sphere and disk radius, rect width, distant angular diameter (radians)
    public float  sizeY;         // rect height
    public float3 colour;        // colour * intensity
    public float  exposure;
    public float  temperature;   // kelvin
    public float  coneCos;       // cosine of the shaping cone's half angle; -1: none
    public float  coneSoftness;
    public uint   texture;       // a dome's lat-long image; 0xFFFFFFFF: its colour alone
    public uint   sampler;
    public uint   lightCategory;    // the bit a prim needs for this light to light it
    public uint   shadowCategory;   // and for it to cast this light's shadow
    public float  cumulative;       // its share of the frame's power, accumulated
    public uint   ies;              // its IES profile's row; 0xFFFFFFFF: none
    public float  iesAngleScale;    // UsdLux: > 0 divides theta, < 0 scales from 180, 0 none
    public uint   pad0;          // the light BVH: its node's index
    public uint   group;         // 0: in no light group; else 1 + its index among the frame's groups
    public float4 row0;          // light to world, rows of a 3x4
    public float4 row1;
    public float4 row2;
    // UsdLux ShadowAPI: the colour a shadow is tinted (non-physical, and
    // the spec says so), how far from the light it reaches (-1: no limit),
    // the zone over which it fades away (-1: none) and a gamma on that fade.
    public float3 shadowColour;
    public float  shadowDistance;
    public float  shadowFalloff;
    public float  shadowFalloffGamma;
    public uint   pad1;
    public uint   pad2;
};

/// WHAT A SHADOW DOES TO THE LIGHT, given how much of it got through.
///
/// `through` is the transmittance a receiver measured (a ray, a baked field,
/// the map from the light): 1 is no shadow. ShadowAPI then has its say: the
/// shadow may be tinted (`shadow:color`: what is stopped is replaced by that
/// colour rather than by black), and beyond `shadow:distance` from the light
/// it fades out over `shadow:falloff`, by `falloffGamma`. A distant light is
/// at infinity, so the distance controls do nothing there.
public float3 shadowTint(LightRecord l, float through, float distance) {
    float shadowing = 1.0 - saturate(through);
    if (l.shadowDistance >= 0.0 && distance > l.shadowDistance) {
        if (l.shadowFalloff > 0.0) {
            const float t = saturate((distance - l.shadowDistance) / l.shadowFalloff);
            shadowing *= 1.0 - pow(t, max(l.shadowFalloffGamma, 1.0e-3));
        } else {
            shadowing = 0.0;
        }
    }
    return lerp(float3(1.0), l.shadowColour, shadowing);
}

/// One IES profile: where its angle lists and candela table sit in
/// `iesValues`, and its power, for normalize.
public struct IesRecord {
    public uint  verticalOffset;
    public uint  verticalCount;
    public uint  horizontalOffset;
    public uint  horizontalCount;
    public uint  candelaOffset;
    public uint  photometricType;
    public float multiplier;
    public float power;   // mean intensity over the sphere
};

public RWStructuredBuffer<IesRecord> iesRecords;
public StructuredBuffer<float>     iesValues;

/// The profile's intensity at a vertical angle and a horizontal one, in
/// degrees, bilinear between the nodes it authored; the horizontal range is
/// folded by the symmetry its last angle declares, as LM-63 has it.
public float iesLookup(IesRecord r, float thetaDeg, float phiDeg) {
    const uint nv = r.verticalCount;
    const uint nh = r.horizontalCount;
    float phi = phiDeg;
    if (nh > 1) {
        const float last = iesValues[r.horizontalOffset + nh - 1];
        phi = phi - 360.0 * floor(phi / 360.0);
        if (last <= 90.0 + 1.0e-3) {
            phi = phi - 180.0 * floor(phi / 180.0);
            if (phi > 90.0) phi = 180.0 - phi;
        } else if (last <= 180.0 + 1.0e-3) {
            if (phi > 180.0) phi = 360.0 - phi;
        }
    }
    uint i = 0;
    while (i + 1 < nv && iesValues[r.verticalOffset + i + 1] < thetaDeg) {
        ++i;
    }
    const float t0 = iesValues[r.verticalOffset + i];
    const float t1 = i + 1 < nv ? iesValues[r.verticalOffset + i + 1] : t0;
    const float ft = t1 > t0 ? clamp((thetaDeg - t0) / (t1 - t0), 0.0, 1.0) : 0.0;
    uint j = 0;
    float fp = 0.0;
    if (nh > 1) {
        while (j + 1 < nh && iesValues[r.horizontalOffset + j + 1] < phi) {
            ++j;
        }
        const float p0 = iesValues[r.horizontalOffset + j];
        const float p1 = j + 1 < nh ? iesValues[r.horizontalOffset + j + 1] : p0;
        fp = p1 > p0 ? clamp((phi - p0) / (p1 - p0), 0.0, 1.0) : 0.0;
    }
    const uint i1 = min(i + 1, nv - 1);
    const uint j1 = min(j + 1, nh - 1);
    const float c00 = iesValues[r.candelaOffset + j * nv + i];
    const float c01 = iesValues[r.candelaOffset + j * nv + i1];
    const float c10 = iesValues[r.candelaOffset + j1 * nv + i];
    const float c11 = iesValues[r.candelaOffset + j1 * nv + i1];
    const float value = lerp(lerp(c00, c01, ft), lerp(c10, c11, ft), fp);
    return value * r.multiplier;
}

/// The IES factor on what `l` emits towards a point that sees it along
/// `wi`: the emission direction in the light's own axes, straight down its
/// -Z at theta 0, as UsdLux orients a profile. One where there is no
/// profile. Radiance only -- never the density, so every pdf stands.
public float iesIntensity(LightRecord l, float3 wi) {
    if (l.ies == 0xFFFFFFFF) {
        return 1.0;
    }
    const float3 e = -wi;
    const float ex = dot(e, lightAxisX(l));
    const float ey = dot(e, lightAxisY(l));
    const float ez = dot(e, lightAxisZ(l));
    float theta = acos(clamp(-ez, -1.0, 1.0));
    const float s = l.iesAngleScale;
    if (s > 0.0) {
        theta = theta / s;
    } else if (s < 0.0) {
        theta = 3.14159265358979 - (3.14159265358979 - theta) / (-s);
    }
    theta = clamp(theta, 0.0, 3.14159265358979);
    float phi = degrees(atan2(ey, ex));
    if (phi < 0.0) phi += 360.0;
    const IesRecord r = iesRecords[l.ies];
    const float value = iesLookup(r, degrees(theta), phi);
    return (l.flags & kLightIesNormalize) != 0 ? value / max(r.power, 1.0e-9) : value;
}

/// Whether `l` reaches a surface in `categories`: USD resolves a light's
/// collection into a category the geometry it lights carries, so the test is
/// one bit -- and a light with no collection carries none and lights all.
public bool lightLinked(uint category, uint categoriesLo, uint categoriesHi) {
    if (category == kLightUnlinked) {
        return true;
    }
    return category < 32 ? (categoriesLo & (1u << category)) != 0 : (categoriesHi & (1u << (category - 32))) != 0;
}

/// One light out of many, in proportion to its power: a binary search over
/// the accumulated shares, so a pixel's cost stops growing with the number of
/// lights. The density it was chosen with comes back, to be divided out.
public struct LightChoice {
    public uint  index;
    public float probability;
    public bool  valid;
};

public LightChoice chooseLight(StructuredBuffer<LightRecord> lights, uint count, float u) {
    LightChoice choice;
    choice.index = 0;
    choice.probability = 0.0;
    choice.valid = false;
    if (count == 0) {
        return choice;
    }
    // The frame's total is the last cumulative share, accumulated on the
    // device by light_prefix.
    const float power = lights[count - 1].cumulative;
    if (!(power > 0.0)) {
        return choice;
    }
    const float target = clamp(u, 0.0, 0.999999) * power;
    uint lo = 0;
    uint hi = count - 1;
    while (lo < hi) {
        const uint mid = (lo + hi) / 2;
        if (lights[mid].cumulative <= target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    const float before = lo == 0 ? 0.0 : lights[lo - 1].cumulative;
    choice.index = lo;
    choice.probability = max(lights[lo].cumulative - before, 1.0e-9) / power;
    choice.valid = true;
    return choice;
}

public struct LightSample {
    public float3 wi;         // from the shading point towards the light, unit
    public float  distance;   // how far to shadow along wi
    public float3 radiance;   // arriving at the point
    public float  pdf;        // per solid angle; 0 for a delta light
    public bool   delta;
    public bool   valid;
    /// The emitter's cosine at the sampled point, for an area light; 1 where
    /// there is none. A pdf divides by it, so near an area light's silhouette
    /// the pdf is 1/cos of whatever error the point carries -- and the sample
    /// contributes cos of nothing.
    public float  emitterCosine = 1.0;
};

static const float kLightInfinity = 1.0e30;

float3 lightApply(LightRecord l, float3 v, float w) {
    return float3(dot(l.row0.xyz, v) + l.row0.w * w, dot(l.row1.xyz, v) + l.row1.w * w,
                  dot(l.row2.xyz, v) + l.row2.w * w);
}

/// The light's axes in world space. UsdLux shines down -Z and its shapes lie
/// in the xy plane.
public float3 lightAxisX(LightRecord l) { return normalize(float3(l.row0.x, l.row1.x, l.row2.x)); }
public float3 lightAxisY(LightRecord l) { return normalize(float3(l.row0.y, l.row1.y, l.row2.y)); }
public float3 lightAxisZ(LightRecord l) { return normalize(float3(l.row0.z, l.row1.z, l.row2.z)); }
public float3 lightOrigin(LightRecord l) { return float3(l.row0.w, l.row1.w, l.row2.w); }

/// Planck's law through the CIE curves, as a colour normalised to luminance
/// 1: the analytic fit USD's own renderers use (Krystek's uv, then sRGB).
public float3 blackbody(float kelvin) {
    const float t = clamp(kelvin, 1000.0, 15000.0);
    const float t2 = t * t;
    // Krystek's rational fit of the Planckian locus in CIE 1960 uv.
    const float u = (0.860117757 + 1.54118254e-4 * t + 1.28641212e-7 * t2) /
                    (1.0 + 8.42420235e-4 * t + 7.08145163e-7 * t2);
    const float v = (0.317398726 + 4.22806245e-5 * t + 4.20481691e-8 * t2) /
                    (1.0 - 2.89741816e-5 * t + 1.61456053e-7 * t2);
    const float denominator = 2.0 * u - 8.0 * v + 4.0;
    const float x = 3.0 * u / denominator;
    const float y = 2.0 * v / denominator;
    const float z = 1.0 - x - y;
    const float3 xyz = float3(x / max(y, 1e-6), 1.0, z / max(y, 1e-6));
    // XYZ to linear sRGB.
    const float3 rgb = float3(dot(float3(3.2404542, -1.5371385, -0.4985314), xyz),
                              dot(float3(-0.9692660, 1.8760108, 0.0415560), xyz),
                              dot(float3(0.0556434, -0.2040259, 1.0572252), xyz));
    const float3 positive = max(rgb, float3(0.0));
    const float luminance = dot(float3(0.2126, 0.7152, 0.0722), positive);
    return luminance > 0.0 ? positive / luminance : float3(1.0);
}

/// Where a direction lands in a lat-long image: u around the light's own
/// -Z, v from its +Y down, as UsdLux maps a dome.
/// A dome's image is a lat-long as OpenEXR defines one, which is what the
/// UsdLux specification cites for DomeLight_1: the first row is latitude
/// +pi/2 (the pole, +y of the dome's frame), the first column longitude +pi,
/// the middle column longitude 0 pointing +z and a quarter of the way in
/// longitude +pi/2 pointing +x -- Storm's `ProjectToLatLong`. The rows of a
/// texture here start at the bottom (texture_decode.slang, v = 0 there), so
/// the pole is at v = 1: a sky read with v = 0 at the pole hung upside down.
public float2 domeUv(LightRecord l, float3 wi) {
    const float3 d = normalize(float3(dot(lightAxisX(l), wi), dot(lightAxisY(l), wi), dot(lightAxisZ(l), wi)));
    const float u = frac((atan2(d.z, d.x) + 0.5 * 3.14159265358979) * (0.5 / 3.14159265358979));
    const float v = 1.0 - acos(clamp(d.y, -1.0, 1.0)) * (1.0 / 3.14159265358979);
    return float2(u, v);
}


/// A distant light's half angle, as UsdLux clips it: `angle` is the disc's
/// angular diameter, from 0 to 360 degrees.
float distantHalfAngle(LightRecord l) {
    return clamp(0.5 * l.sizeX, 0.0, 3.14159265358979);
}

/// 1 - cos of a distant light's half angle, without the cancellation of
/// 1 - cos(a) in float: for UsdLux's 0.53 degree sun that is 1.07e-5, and
/// as a difference of two numbers near 1 it carried 0.6% of rounding.
float distantOneMinusCos(LightRecord l) {
    const float s = sin(0.5 * distantHalfAngle(l));
    return 2.0 * s * s;
}

/// True where a distant light is a direction rather than a disc: an angle
/// of 0, and only that. A disc too small for a cosine to tell from 1 is
/// still a disc, sampled through `distantOneMinusCos`.
public bool distantIsDelta(LightRecord l) {
    return !(l.sizeX > 0.0);
}

/// What UsdLux's `normalize` divides a distant light's radiance by
/// (usdLux/schema.usda, LightAPI inputs:normalize): the disc's projected
/// solid angle, pi sin^2(theta) for a half angle theta up to 90 degrees,
/// (2 - sin^2(theta)) pi beyond -- the integral of |cos| over the cap -- and
/// 1 for a parallel light. A surface facing the light takes radiance times
/// this as its irradiance, so with normalize on the intensity is that
/// irradiance, in lux.
public float distantSizeFactor(LightRecord l) {
    if (distantIsDelta(l)) {
        return 1.0;
    }
    const float theta = distantHalfAngle(l);
    const float s = sin(theta);
    return 3.14159265358979 * (theta <= 0.5 * 3.14159265358979 ? s * s : 2.0 - s * s);
}

/// The area a light's `normalize` divides by, 0 where it has none. For a
/// distant light, the projected solid angle of its disc.
float lightArea(LightRecord l) {
    switch (l.kind) {
    case kLightDistant: return distantSizeFactor(l);
    case kLightSphere: return 4.0 * 3.14159265358979 * l.sizeX * l.sizeX;
    case kLightDisk:   return 3.14159265358979 * l.sizeX * l.sizeX;
    case kLightRect:   return l.sizeX * l.sizeY;
    case kLightCylinder: return 2.0 * 3.14159265358979 * l.sizeX * l.sizeY;   // the lateral surface
    default:           return 0.0;
    }
}

/// Where a ray from `p` along `w` meets a cylinder light's lateral surface
/// from outside: the nearer root of the infinite cylinder about its x axis,
/// within its length, with its outward normal facing the ray. False for a
/// miss, a hit past the ends, or the far wall seen from inside -- a cylinder
/// emits outward and has no inside to light from.
public bool cylinderHit(LightRecord l, float3 p, float3 w, out float t, out float3 normal, out float along) {
    t = 0.0;
    normal = float3(0.0);
    along = 0.0;
    const float3 axis = lightAxisX(l);
    const float3 ay = lightAxisY(l);
    const float3 az = lightAxisZ(l);
    const float3 o = p - lightOrigin(l);
    const float oy = dot(o, ay);
    const float oz = dot(o, az);
    const float dy = dot(w, ay);
    const float dz = dot(w, az);
    const float r = max(l.sizeX, 0.0);
    const float a = dy * dy + dz * dz;
    if (a <= 1.0e-12 || r <= 0.0) {
        return false;
    }
    // By the ray's closest approach to the axis rather than the quadratic's
    // b^2 - 4ac, which cancels catastrophically near a tangent ray -- where a
    // pdf that divides by the cosine is most sensitive to it. Measured with
    // the quadratic: 7826 of a million samples disagreed with their own pdf
    // beyond 1e-3, worst 29x, all beside the silhouette.
    const float tClosest = -(oy * dy + oz * dz) / a;
    const float cy = oy + dy * tClosest;
    const float cz = oz + dz * tClosest;
    const float h2 = r * r - (cy * cy + cz * cz);
    if (h2 < 0.0) {
        return false;
    }
    t = tClosest - sqrt(h2 / a);
    if (!(t > 0.0)) {
        return false;   // inside, or the light is behind
    }
    const float3 hit = o + w * t;
    along = dot(hit, axis);
    if (abs(along) > 0.5 * l.sizeY) {
        return false;
    }
    normal = normalize(ay * dot(hit, ay) + az * dot(hit, az));
    return dot(normal, -w) > 0.0;
}

/// What the light emits, before any shaping: colour, intensity, exposure,
/// colour temperature, and `normalize` where it applies.
public float3 lightEmission(LightRecord l) {
    float3 emission = l.colour * exp2(l.exposure);
    if ((l.flags & kLightTemperature) != 0) {
        emission *= blackbody(l.temperature);
    }
    const float area = lightArea(l);
    if ((l.flags & kLightNormalize) != 0 && area > 0.0) {
        emission /= area;
    }
    return emission;
}

/// The shaping cone's falloff along `wi` (the direction the light travels is
/// -wi), 1 where the light has no cone.
float lightCone(LightRecord l, float3 wi) {
    if (l.coneCos <= -1.0) {
        return 1.0;
    }
    const float cosine = dot(-wi, lightAxisZ(l) * -1.0);   // angle from the light's -Z
    if (cosine < l.coneCos) {
        return 0.0;
    }
    if (l.coneSoftness <= 0.0) {
        return 1.0;
    }
    // Softness eats into the cone from its edge, as UsdLux describes it.
    const float inner = lerp(1.0, l.coneCos, 1.0 - saturate(l.coneSoftness));
    return inner > l.coneCos ? smoothstep(l.coneCos, inner, cosine) : 1.0;
}

float2 sampleDisk(float2 u) {
    const float r = sqrt(u.x);
    const float phi = 2.0 * 3.14159265358979 * u.y;
    return float2(r * cos(phi), r * sin(phi));
}

/// A direction in the cone of half angle acos(cosMax) around +z.
float3 sampleCone(float cosMax, float2 u) {
    const float cosTheta = lerp(cosMax, 1.0, u.x);
    const float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    const float phi = 2.0 * 3.14159265358979 * u.y;
    return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

/// An orthonormal frame around `z`.
public void lightFrame(float3 z, out float3 x, out float3 y) {
    const float sign = z.z >= 0.0 ? 1.0 : -1.0;
    const float a = -1.0 / (sign + z.z);
    const float b = z.x * z.y * a;
    x = float3(1.0 + sign * z.x * z.x * a, sign * b, -sign * z.x);
    y = float3(b, sign + z.y * z.y * a, -z.y);
}

/// A dome's image, importance sampled.


/// Where a direction lands in the image, and back again.
public float3 domeDirection(LightRecord l, float2 uv) {
    // The inverse of domeUv: longitude from the column, latitude from the
    // row counted up from the bottom, in the light's own axes.
    const float phi = uv.x * 2.0 * 3.14159265358979 - 0.5 * 3.14159265358979;
    const float theta = (1.0 - uv.y) * 3.14159265358979;
    const float sinTheta = sin(theta);
    const float3 local = float3(sinTheta * cos(phi), cos(theta), sinTheta * sin(phi));
    return normalize(lightAxisX(l) * local.x + lightAxisY(l) * local.y + lightAxisZ(l) * local.z);
}



/// The density `sampleLight` draws `wi` with, as a function of direction:
/// what a chi-square asks for, and what MIS will weigh a hit by. Zero where
/// the light cannot be reached that way, and zero for a delta light, which
/// has no density at all.
public float lightPdf(LightRecord l, float3 p, float3 n, float3 wi) {
    switch (l.kind) {
    case kLightDistant: {
        if (distantIsDelta(l)) {
            return 0.0;   // a delta direction
        }
        const float oneMinus = distantOneMinusCos(l);
        return dot(wi, lightAxisZ(l)) >= 1.0 - oneMinus ? 1.0 / (2.0 * 3.14159265358979 * oneMinus) : 0.0;
    }
    case kLightDome:
        // Cosine weighted around the surface. Sampling by an image is
        // lights_image.slang's, and it reports its own density.
        return max(dot(n, wi), 0.0) / 3.14159265358979;
    case kLightSphere: {
        const float3 offset = lightOrigin(l) - p;
        const float distance2 = dot(offset, offset);
        const float radius = max(l.sizeX, 0.0);
        if (radius <= 0.0 || distance2 <= radius * radius) {
            return 0.0;   // a point, or inside the light
        }
        const float cosMax = sqrt(max(0.0, 1.0 - radius * radius / distance2));
        return dot(wi, normalize(offset)) >= cosMax ? 1.0 / (2.0 * 3.14159265358979 * (1.0 - cosMax)) : 0.0;
    }
    case kLightDisk:
    case kLightRect: {
        // Where the ray meets the light's plane, if it does so in front and
        // inside the shape: the area density turned into a solid angle one.
        const float3 normal = -lightAxisZ(l);
        const float facing = dot(normal, -wi);
        if (facing <= 0.0) {
            return 0.0;
        }
        const float3 centre = lightOrigin(l);
        const float denominator = dot(wi, -normal);
        if (denominator <= 0.0) {
            return 0.0;
        }
        const float t = dot(centre - p, -normal) / denominator;
        if (!(t > 0.0)) {
            return 0.0;
        }
        const float3 at = p + wi * t - centre;
        const float x = dot(at, lightAxisX(l));
        const float y = dot(at, lightAxisY(l));
        float area;
        if (l.kind == kLightDisk) {
            if (x * x + y * y > l.sizeX * l.sizeX) {
                return 0.0;
            }
            area = 3.14159265358979 * l.sizeX * l.sizeX;
        } else {
            if (abs(x) > 0.5 * l.sizeX || abs(y) > 0.5 * l.sizeY) {
                return 0.0;
            }
            area = l.sizeX * l.sizeY;
        }
        return area > 0.0 ? (t * t) / (facing * area) : 0.0;
    }
    case kLightCylinder: {
        float t;
        float3 normal;
        float along;
        if (!cylinderHit(l, p, wi, t, normal, along)) {
            return 0.0;
        }
        const float area = lightArea(l);
        const float facing = dot(normal, -wi);
        return area > 0.0 && facing > 0.0 ? (t * t) / (facing * area) : 0.0;
    }
    default:
        return 0.0;
    }
}

/// Where the ray from `p` along `wi` meets `l`, if it does: the distance
/// (kLightInfinity for a dome or a distant light, which only an escaping ray
/// reaches) and the radiance arriving -- the same radiance `sampleLight`
/// reports for that direction. What a material's sampled direction adds when
/// the two strategies are weighed. A delta light is never met. A dome's
/// image is lights_image.slang's to apply.
public struct LightHit {
    public float  t;
    public float3 radiance;
    public bool   valid;
};

public LightHit lightHit(LightRecord l, float3 p, float3 wi) {
    LightHit h;
    h.t = kLightInfinity;
    h.radiance = float3(0.0);
    h.valid = false;
    const float3 emission = lightEmission(l);
    switch (l.kind) {
    case kLightDistant: {
        // The disc's radiance is the emission itself, as `sampleLight` hands
        // it out: UsdLux's intensity is a radiance, and `normalize` is
        // already in `lightEmission`.
        if (distantIsDelta(l) || dot(wi, lightAxisZ(l)) < 1.0 - distantOneMinusCos(l)) {
            return h;
        }
        h.radiance = emission;
        break;
    }
    case kLightDome:
        h.radiance = emission;
        break;
    case kLightSphere: {
        const float radius = max(l.sizeX, 0.0);
        const float3 offset = lightOrigin(l) - p;
        const float distance2 = dot(offset, offset);
        if (radius <= 0.0 || distance2 <= radius * radius) {
            return h;
        }
        // Inside the cone the sphere subtends, as lightPdf and sampleLight
        // decide it, and the near root with the discriminant clamped as the
        // sampler clamps it: a ray drawn at the cone's edge is tangent, and
        // rounding (FMA contraction on NVIDIA's compiler) left its
        // discriminant a hair below zero -- 4 samples in a million missed.
        const float distance = sqrt(distance2);
        const float cosMax = sqrt(max(0.0, 1.0 - radius * radius / distance2));
        if (dot(wi, offset / distance) < cosMax) {
            return h;
        }
        const float b = dot(wi, offset);
        const float discriminant = max(0.0, b * b - (distance2 - radius * radius));
        const float t = b - sqrt(discriminant);
        if (!(t > 0.0)) {
            return h;
        }
        h.t = t;
        h.radiance = emission;
        break;
    }
    case kLightDisk:
    case kLightRect: {
        const float3 normal = -lightAxisZ(l);
        const float denominator = dot(wi, -normal);
        if (denominator <= 0.0) {
            return h;   // the back of a one-sided light, or along its plane
        }
        const float3 centre = lightOrigin(l);
        const float t = dot(centre - p, -normal) / denominator;
        if (!(t > 0.0)) {
            return h;
        }
        const float3 at = p + wi * t - centre;
        const float x = dot(at, lightAxisX(l));
        const float y = dot(at, lightAxisY(l));
        const bool inside = l.kind == kLightDisk ? x * x + y * y <= l.sizeX * l.sizeX
                                                 : abs(x) <= 0.5 * l.sizeX && abs(y) <= 0.5 * l.sizeY;
        if (!inside) {
            return h;
        }
        h.t = t;
        h.radiance = emission;
        break;
    }
    case kLightCylinder: {
        float t;
        float3 normal;
        float along;
        if (!cylinderHit(l, p, wi, t, normal, along)) {
            return h;
        }
        h.t = t;
        h.radiance = emission;
        break;
    }
    default:
        return h;
    }
    h.radiance *= lightCone(l, wi) * iesIntensity(l, wi);
    h.valid = any(h.radiance > float3(0.0));
    return h;
}

/// One sample of `l` as seen from `p`, whose surface faces `n`. Only a dome
/// reads the normal: the rest are somewhere rather than everywhere.
public LightSample sampleLight(LightRecord l, float3 p, float3 n, float2 u) {
    LightSample s;
    s.wi = float3(0.0, 0.0, 1.0);
    s.distance = kLightInfinity;
    s.radiance = float3(0.0);
    s.pdf = 0.0;
    s.delta = false;
    s.valid = false;
    const float3 emission = lightEmission(l);
    switch (l.kind) {
    case kLightDistant: {
        // UsdLux shines down -Z, so the light is towards +Z, spread over a
        // disc of angular diameter sizeX.
        const float3 towards = lightAxisZ(l);
        if (distantIsDelta(l)) {
            s.wi = towards;
            s.delta = true;
            s.pdf = 1.0;
        } else {
            // Uniform over the cap by 1 - cos, which keeps its precision
            // where a cosine near 1 would not.
            const float oneMinusMax = distantOneMinusCos(l);
            const float oneMinus = u.x * oneMinusMax;
            const float sinTheta = sqrt(max(0.0, oneMinus * (2.0 - oneMinus)));
            const float phi = 2.0 * 3.14159265358979 * u.y;
            float3 x, y;
            lightFrame(towards, x, y);
            s.wi = normalize(x * (sinTheta * cos(phi)) + y * (sinTheta * sin(phi)) + towards * (1.0 - oneMinus));
            s.pdf = 1.0 / (2.0 * 3.14159265358979 * oneMinusMax);
        }
        // UsdLux's intensity is the radiance of the disc (LightAPI
        // inputs:intensity), and `normalize` divides it by the disc's
        // projected solid angle (lightEmission, through lightArea), so that
        // the intensity is then the illuminance on a surface facing the
        // light. A parallel light has no disc: its intensity is that
        // irradiance whether normalize is on or not (sizeFactor 1).
        //
        // Until this the intensity was always the irradiance, divided by
        // the disc's solid angle 2 pi (1 - cos) whatever normalize said: a
        // sun authored without normalize -- UsdLux's default -- lit a plane
        // 1 / (pi sin^2(theta)) times what the schema says: 14900 times for
        // the 0.53 degree default.
        s.radiance = emission;
        break;
    }
    case kLightDome: {
        // Cosine weighted around the surface, which is where a dome's light
        // reaches it from: uniform over the sphere throws half its samples
        // below the horizon, and a plane under a constant dome came out with
        // 7% of noise at 4096 of them. The warp that follows the image's own
        // brightness (the plan's mip warp) is still to come, and is what a
        // dome with a sun in it needs.
        float3 x, y;
        lightFrame(n, x, y);
        const float2 d = sampleDisk(u);
        const float z = sqrt(max(0.0, 1.0 - dot(d, d)));
        s.wi = normalize(x * d.x + y * d.y + n * z);
        // The density lightPdf reports for this direction, to the letter: a
        // sample along the horizon carries no light and is not drawn at all,
        // rather than being given a floor its pdf does not have.
        s.pdf = max(dot(n, s.wi), 0.0) / 3.14159265358979;
        s.radiance = emission;
        break;
    }
    case kLightSphere: {
        const float3 centre = lightOrigin(l);
        const float3 offset = centre - p;
        const float distance2 = dot(offset, offset);
        const float radius = max(l.sizeX, 0.0);
        if (distance2 <= radius * radius) {
            return s;   // inside the light
        }
        const float distance = sqrt(distance2);
        if (radius <= 0.0) {
            s.wi = offset / distance;
            s.distance = distance;
            s.radiance = emission / distance2;
            s.delta = true;
            s.pdf = 1.0;
            break;
        }
        // The cone the sphere subtends: uniform in solid angle, which is the
        // density a plane's closed-form irradiance is written against.
        const float cosMax = sqrt(max(0.0, 1.0 - radius * radius / distance2));
        const float3 towards = offset / distance;
        float3 x, y;
        lightFrame(towards, x, y);
        const float3 local = sampleCone(cosMax, u);
        s.wi = normalize(x * local.x + y * local.y + towards * local.z);
        // Where the cone's ray meets the sphere.
        const float b = dot(s.wi, offset);
        const float c = distance2 - radius * radius;
        const float discriminant = max(0.0, b * b - c);
        s.distance = b - sqrt(discriminant);
        s.radiance = emission;
        s.pdf = 1.0 / (2.0 * 3.14159265358979 * (1.0 - cosMax));
        break;
    }
    case kLightDisk:
    case kLightRect: {
        const float3 normal = -lightAxisZ(l);   // it shines down -Z
        float3 at;
        float area;
        if (l.kind == kLightDisk) {
            const float2 d = sampleDisk(u) * max(l.sizeX, 0.0);
            at = lightOrigin(l) + lightAxisX(l) * d.x + lightAxisY(l) * d.y;
            area = 3.14159265358979 * l.sizeX * l.sizeX;
        } else {
            at = lightOrigin(l) + lightAxisX(l) * (u.x - 0.5) * l.sizeX +
                 lightAxisY(l) * (u.y - 0.5) * l.sizeY;
            area = l.sizeX * l.sizeY;
        }
        const float3 offset = at - p;
        const float distance2 = dot(offset, offset);
        if (distance2 <= 0.0 || area <= 0.0) {
            return s;
        }
        const float distance = sqrt(distance2);
        s.wi = offset / distance;
        const float cosine = dot(normal, -s.wi);
        if (cosine <= 0.0) {
            return s;   // the back of a one-sided light
        }
        s.distance = distance;
        s.radiance = emission;
        s.pdf = distance2 / (cosine * area);
        s.emitterCosine = cosine;
        break;
    }
    case kLightCylinder: {
        // Uniform on the lateral surface: an angle about the axis, a height
        // along it. Outward normal, one-sided.
        const float phi = 2.0 * 3.14159265358979 * u.x;
        const float along = (u.y - 0.5) * l.sizeY;
        const float3 radial = lightAxisY(l) * cos(phi) + lightAxisZ(l) * sin(phi);
        const float3 at = lightOrigin(l) + lightAxisX(l) * along + radial * max(l.sizeX, 0.0);
        const float area = lightArea(l);
        const float3 offset = at - p;
        const float distance2 = dot(offset, offset);
        if (distance2 <= 0.0 || area <= 0.0) {
            return s;
        }
        const float distance = sqrt(distance2);
        s.wi = offset / distance;
        const float cosine = dot(radial, -s.wi);
        if (cosine <= 0.0) {
            return s;   // the far side of the surface
        }
        s.distance = distance;
        s.radiance = emission;
        s.pdf = distance2 / (cosine * area);
        s.emitterCosine = cosine;
        break;
    }
    default:
        return s;
    }
    s.radiance *= lightCone(l, s.wi) * iesIntensity(l, s.wi);
    s.valid = s.pdf > 0.0 && any(s.radiance > float3(0.0));
    return s;
}

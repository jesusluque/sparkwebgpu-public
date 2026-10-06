// Copyright (c) 2026 jesus luque.
//
// WHAT A PRIM'S GAUSSIANS ARE MADE OF, SAID AGAIN FOR THE FRAME.
//
// A converted cloud carries the prim each gaussian came from -- the same id
// the Cryptomatte writes -- so the id a pixel hands back is also a selection:
// every gaussian of that prim, which is every gaussian that was standing on
// one material when `athenea mesh2splat` walked it. A table keyed on that id is
// therefore a material override, and picking is how you address it.
//
// It is read, never baked: the cloud's own `pbr` and colours stay as the file
// wrote them, and a row here is an opinion the frame holds over them. Clearing
// the row puts the file back, and nothing on disk is touched.
//
// Two float4 a row, so the host writes it without a struct layout to agree on:
//
//   row*2     (asfloat(id), metallic, roughness, transmission)
//   row*2 + 1 (tintR, tintG, tintB, replace)
//
// A negative metallic, roughness or transmission means "leave what the
// gaussian carries", which is what an untouched slider sends: a prim whose
// roughness came out of a map is not flattened by moving its tint.
//
// `replace` of 0 multiplies the gaussian's colour by the tint; 1 makes the
// tint the colour, which is what saying a feathered bird is clear glass needs:
// a texture multiplied by white is still the texture.
module splat_override;

public struct SplatOverride {
    public float  metallic;       ///< < 0: the gaussian's own
    public float  roughness;
    public float  transmission;
    public float3 tint;           ///< multiplies the colour, or is it (`replace`)
    public bool   replace;
    public bool   found;
};

public SplatOverride splatOverrideNone() {
    SplatOverride o;
    o.metallic = -1.0;
    o.roughness = -1.0;
    o.transmission = -1.0;
    o.tint = float3(1.0);
    o.replace = false;
    o.found = false;
    return o;
}

/// The row for `id`, or none. A linear walk: a frame holds a handful of these
/// -- one per prim somebody has touched -- and a table nobody touched costs
/// the `count == 0` test alone.
public SplatOverride splatOverrideFor(StructuredBuffer<float4> rows, uint count, uint id) {
    SplatOverride o = splatOverrideNone();
    for (uint k = 0; k < count; ++k) {
        const float4 head = rows[k * 2];
        if (asuint(head.x) != id) {
            continue;
        }
        const float4 grade = rows[k * 2 + 1];
        o.metallic = head.y;
        o.roughness = head.z;
        o.transmission = head.w;
        o.tint = grade.xyz;
        o.replace = grade.w > 0.5;
        o.found = true;
        return o;
    }
    return o;
}

/// The colour the row leaves a gaussian of colour `mine`.
public float3 splatOverrideColour(float3 mine, SplatOverride said) {
    return said.replace ? said.tint : mine * said.tint;
}

/// What the row says, over what the gaussian carries.
public float splatOverrideOf(float mine, float said) {
    return said < 0.0 ? mine : said;
}

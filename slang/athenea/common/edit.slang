// Copyright (c) 2026 jesus luque.
//
// A SplatEdit: a selection volume in a cloud's own space, what happens to the
// splats it selects, and a grade. openFXplayer's SplatEdit, rule for rule
// (plugins/splat/splat.slang), so a scene moved between the two renders the
// same. One definition, read by every renderer that draws splats: the tile
// rasteriser, both GPU references and the ray tracer -- as the Gaussian's own
// cut (frame.slang's cutoffPower) is one definition.
//
//   the filters, cheapest first, wherever the volume is:
//     opacity below minOpacity (when > 0)          -> gone
//     longest linear scale above maxScale (when > 0) -> gone
//   the volume: inside a box of half extents `size`, or a sphere of radius
//   size.x, around `centre`; `invert` turns the test round
//     mode 0 keep:   outside -> gone
//     mode 1 remove: inside  -> gone
//     mode 2 grade:  nothing goes
//   inside and still there: opacity *= opacity (gone below 1/255), and the
//   colour, in linear light (whatever space the cloud keeps it in), *= tint *
//   brightness, then lerped from its Rec.709 luminance by saturation.
module edit;

public struct EditParams {
    public uint  active;
    public uint  shape;        // 0 box, 1 sphere
    public uint  mode;         // 0 keep inside, 1 remove inside, 2 grade inside
    public uint  invert;
    public float centreX; public float centreY; public float centreZ;
    public float sizeX; public float sizeY; public float sizeZ;
    public float tintR; public float tintG; public float tintB;
    public float saturation;
    public float brightness;
    public float opacity;
    public float minOpacity;
    public float maxScale;
};

public struct EditFate {
    public bool  removed;
    public bool  graded;
    public float opacity;   // the splat's opacity after the edit
};

/// What the edit does to a splat at `position` (cloud space) with linear
/// `opacity` and linear `scale`.
public EditFate editFate(EditParams e, float3 position, float opacity, float3 scale) {
    EditFate fate;
    fate.removed = false;
    fate.graded = false;
    fate.opacity = opacity;
    if (e.active == 0) {
        return fate;
    }
    if (e.minOpacity > 0.0 && opacity < e.minOpacity) {
        fate.removed = true;
        return fate;
    }
    if (e.maxScale > 0.0 && max(max(abs(scale.x), abs(scale.y)), abs(scale.z)) > e.maxScale) {
        fate.removed = true;
        return fate;
    }
    const float3 rel = position - float3(e.centreX, e.centreY, e.centreZ);
    bool inside = e.shape == 1 ? dot(rel, rel) <= e.sizeX * e.sizeX
                               : (abs(rel.x) <= e.sizeX && abs(rel.y) <= e.sizeY && abs(rel.z) <= e.sizeZ);
    if (e.invert != 0) {
        inside = !inside;
    }
    if ((e.mode == 0 && !inside) || (e.mode == 1 && inside)) {
        fate.removed = true;
        return fate;
    }
    if (inside) {
        fate.graded = true;
        fate.opacity = opacity * e.opacity;
        fate.removed = !(fate.opacity >= 1.0 / 255.0);
    }
    return fate;
}

/// The grade, on a splat's colour in linear light: every splat is made
/// linear before it is graded and blended (common/color.slang, `cloudLight`),
/// so a brightness of 2 is twice the light, and the Rec.709 weights below are
/// a luminance and not a luma of encoded values. (It used to run on a
/// capture's sRGB; the same numbers now grade a little differently, and
/// openFXplayer, which shares the edit, grades encoded until it follows.)
public float3 editGrade(EditParams e, float3 colour) {
    float3 c = colour * float3(e.tintR, e.tintG, e.tintB) * e.brightness;
    const float luma = dot(c, float3(0.2126, 0.7152, 0.0722));
    return max(lerp(float3(luma), c, e.saturation), 0.0);
}

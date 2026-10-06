// Copyright (c) 2026 jesus luque.
//
// What a part's baked visibility field is, shared by the bake that writes it
// (splat_visibility.slang) and the read that multiplies it a frame
// (splat_visibility_read.slang). Two modules because Metal binds at most 31
// buffers to a kernel and a module's globals are laid out for every entry
// point it holds: with the read's light tables beside the bake's shadow
// tables the count went one over.
module splat_visibility_common;
import athenea.common.octahedral;

/// One part's field: where it lives in `texels` and the box it covers, in
/// the space the cloud was bound in.
public struct VisibilityPart {
    public float4 boxMin;      // xyz; w: the joint whose transform carries this part (as float)
    public float4 boxMax;      // xyz; w: unused
    public uint   grid;        // probes along each axis
    public uint   octave;      // directions along each side of the octahedral map
    public uint   offset;      // this part's first texel
    public uint   pad;
};


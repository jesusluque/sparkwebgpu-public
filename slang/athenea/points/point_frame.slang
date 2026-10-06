// Copyright (c) 2026 jesus luque.
//
// What the point passes agree on: the camera (same fields as the splat
// FrameParams, so one C++ setter serves both) and how big a point is.
module point_frame;

public struct PointParams {
    public uint  width;
    public uint  height;
    public float focalX;
    public float focalY;
    public float centreX;
    public float centreY;
    public float nearZ;
    public float farZ;
    public uint  orthographic;
    public uint  sizeMode;      // 0 world units, 1 pixels
    public float size;          // diameter
    public uint  count;
    public float m00; public float m01; public float m02; public float m03;
    public float m10; public float m11; public float m12; public float m13;
    public float m20; public float m21; public float m22; public float m23;
    public uint  colourMode;    // 0 the file's, 1 constant
    public float colourR; public float colourG; public float colourB;
    public float edlStrength;
    public float edlRadius;
    public uint  surface;       // 1: the fragment weights for surface splatting
    public float depthOffset;   // surface splatting's visibility slack, view units
};

public float3 pointToView(PointParams p, float3 x) {
    return float3(p.m00 * x.x + p.m01 * x.y + p.m02 * x.z + p.m03,
                  p.m10 * x.x + p.m11 * x.y + p.m12 * x.z + p.m13,
                  p.m20 * x.x + p.m21 * x.y + p.m22 * x.z + p.m23);
}

/// Radius in pixels, never below half a pixel: a point that projects smaller
/// still covers the pixel it lands in, or it vanishes and aliases.
public float pointRadius(PointParams p, float viewZ) {
    const float r = p.sizeMode == 1 ? p.size * 0.5
                                    : (p.orthographic != 0 ? p.size * 0.5 * p.focalX
                                                           : p.size * 0.5 * p.focalX / viewZ);
    return max(r, 0.5);
}

public float2 pointScreen(PointParams p, float3 v) {
    const float invZ = p.orthographic != 0 ? 1.0 : 1.0 / v.z;
    return float2(p.focalX * v.x * invZ + p.centreX, p.focalY * v.y * invZ + p.centreY);
}

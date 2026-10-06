// IBlendTerm: the plugin extension point of the splat draw
// (draw/splat_draw.slang, draw/splat_attrib_draw.slang) and the tile
// rasterizer (tiles/tile_raster.slang), per pixel a splat covers: what
// depends on the pixel rather than the splat (slope, sharp coat, the sun,
// glass, crypto ids for the athenea plugins).
//
// Runs on the fragment's colour once its alpha (falloff, LOD opacity) is
// known and it passed minAlpha, before the linear encode and premultiply,
// with colours as the accumulator holds them (sRGB-encoded, as Spark's).
// The default preset uses DefaultBlendTerm, the identity, so its kernels
// are the ones Spark always had.
//
// The tile rasterizer has two storage buffers to spare for plugin data
// (WebGPU guarantees eight a stage); the quad draw's fragment stage more.

module blend_term;

public struct BlendInput {
    /** Accumulator index of the splat. */
    public uint splatIndex;
    /** Splat-space coordinate of the pixel (|uv| = 1 at one std dev). */
    public float2 splatUv;
    /** Framebuffer position: pixel centre xy, NDC depth z of the splat, w 1. */
    public float4 fragCoord;
};

public interface IBlendTerm {
    /** The fragment colour so far (straight alpha) to the one after this term. */
    static float4 blend(float4 rgba, BlendInput i);
}

public struct DefaultBlendTerm : IBlendTerm {
    [ForceInline]
    public static float4 blend(float4 rgba, BlendInput i) {
        return rgba;
    }
}

/** A then B. Presets nest these for longer chains. */
public struct BlendChain<A : IBlendTerm, B : IBlendTerm> : IBlendTerm {
    [ForceInline]
    public static float4 blend(float4 rgba, BlendInput i) {
        return B.blend(A.blend(rgba, i), i);
    }
}

[ForceInline]
public float4 applyBlendTerm<B : IBlendTerm>(float4 rgba, BlendInput i) {
    return B.blend(rgba, i);
}

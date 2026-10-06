// Example blend plugin (IBlendTerm): per fragment, mixes the splat colour
// towards a fog colour by the splat's NDC depth and scales its opacity, set
// by src/webgpu/plugins/builtin.ts fadePlugin. It shows a pixel-stage plugin
// with its own uniform block in the quad draw and the tile rasterizer.

module fade;

import plugins.blend_term;

public struct FadeParams {
    public float4 color;  // rgb fog colour, a: fog at NDC depth 1
    public float opacity; // alpha multiplier
    public float power;   // fog = color.a * depth^power
    public float pad0;
    public float pad1;
};

public ConstantBuffer<FadeParams> fadeParams;

public struct FadeBlend : IBlendTerm {
    public static float4 blend(float4 rgba, BlendInput i) {
        float fog = fadeParams.color.a * pow(saturate(i.fragCoord.z), fadeParams.power);
        return float4(lerp(rgba.rgb, fadeParams.color.rgb, fog), rgba.a * fadeParams.opacity);
    }
}

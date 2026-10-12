// Binding attributes slang-build reads from the reflection (thread CW).
module bindings;

/// `[Filterable] Texture2D<float4> t;`: a texture sampled through a
/// SamplerState (its layout's sampleType "float", not "unfilterable-float").
[__AttributeUsage(_AttributeTargets.Var)]
public struct FilterableAttribute {};

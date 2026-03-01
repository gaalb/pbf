struct VSOutput
{
    float4 position : SV_Position; // clip space position (not used in the PS, consumed by therasterizer)
    float3 rayDir : RAYDIR; // world space ray direction, interpolated across the quad from the VS
};

// the cubemap texture — a special texture type that maps 3D directions to colors
TextureCube env : register(t0);
// sampler state — tells the GPU how to filter/interpolate texture samples
SamplerState sampl : register(s0);

float4 main(VSOutput input) : SV_Target
{
    // sample the cubemap in the view direction — this gives us the sky color for this pixel
    return env.Sample(sampl, normalize(input.rayDir));
}
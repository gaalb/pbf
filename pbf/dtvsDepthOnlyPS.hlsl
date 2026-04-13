// Depth-only pixel shader for the DTVS (Distance to Visible Surface) particle render pass.
// Reuses particleVS.cso + particleGS.cso to get circular billboard coverage identical to
// the visual render, ensuring correct occlusion classification in the compute pass.
// Only output is the hardware depth written by the rasterizer.

struct PSInput
{
    float4 position : SV_Position;
    float2 uv       : TEXCOORD; // (-1,-1)..(1,1) across the billboard quad
};

void main(PSInput input)
{
    // Discard fragments outside the unit circle so only the spherical disc writes depth,
    // matching the shape of visually rendered particles.
    if (dot(input.uv, input.uv) > 1.0f) discard;
    // No color output: only the rasterizer-written depth value is kept.
}

// root signature: defines what resources the shaders can access
// ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT: we're sending vertex data from CPU
// CBV(b0): one constant buffer at register b0 (our per-frame camera matrices)
#define RootSig "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), CBV(b0)"

// input from the vertex buffer — each particle is just a position
struct IAOutput
{
    float3 position : POSITION;
};

// output to the pixel shader
struct VSOutput
{
    float4 position : SV_Position; // clip space pos system-value semantic for rasterization
    float4 color : COLOR; // we'll pass a color to the pixel shader
};

// per-frame data uploaded from CPU every frame
cbuffer PerFrameCb : register(b0)
{
    float4x4 viewProjMat; // combined view * projection matrix
    float4 cameraPos; // camera position in world space
};

[RootSignature(RootSig)]
VSOutput main(IAOutput input)
{
    VSOutput output;
    // transform the particle's world-space position to clip space
    output.position = mul(viewProjMat, float4(input.position, 1.0f));
    // simple red color for now
    output.color = float4(1.0f, 0.0f, 0.0f, 1.0f);
    return output;
}
// liquidVS.hlsl
// Vertex shader for the liquid surface fullscreen pass.
// Functionally identical to bgVS.hlsl: receives fullscreen quad vertices in NDC
// space, computes the world-space ray direction for each screen position, and
// passes it to liquidPS for ray marching.
// Uses LiquidRootSig instead of BgRootSig so D3D12 PSO creation succeeds when
// paired with liquidPS.hlsl (both must embed the same root signature).

#define LiquidRootSig \
    "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), " \
    "CBV(b0), " \
    "DescriptorTable(SRV(t0, numDescriptors = 4)), " \
    "StaticSampler(s0, " \
        "filter = FILTER_MIN_MAG_MIP_LINEAR, " \
        "addressU = TEXTURE_ADDRESS_CLAMP, " \
        "addressV = TEXTURE_ADDRESS_CLAMP, " \
        "addressW = TEXTURE_ADDRESS_CLAMP)"

// FullScreenQuad uses PNT_Vertex — accept all fields even though only position is needed
struct IAOutput
{
    float3 position : POSITION;
    float3 normal   : NORMAL;   // unused
    float2 tex      : TEXCOORD; // unused
};

struct VSOutput
{
    float4 position : SV_Position; // placed at far plane so geometry draws in front
    float3 rayDir   : RAYDIR;      // world-space ray direction for this screen pixel
};

#include "SharedConfig.hlsli"

struct LightData { float4 direction; float4 color; };

cbuffer PerFrameCb : register(b0)
{
    float4x4 viewProjMat;
    float4x4 rayDirMat;   // clip-space -> world-space direction (inverse view-rotation * proj)
    float4   cameraPos;
    LightData lights[NUM_LIGHTS]; // must be here to match C++ layout
    float4   particleParams;
};

[RootSignature(LiquidRootSig)]
VSOutput main(IAOutput input)
{
    VSOutput output;
    output.position = float4(input.position, 1.0f);
    output.position.z = 0.999999f; // behind all scene geometry
    output.rayDir = mul(rayDirMat, float4(input.position, 1.0f)).xyz;
    return output;
}

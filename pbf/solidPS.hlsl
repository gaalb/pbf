// Pixel shader for solid obstacle rendering.
// Blinn-Phong shading, summing contributions from all NUM_LIGHTS directional lights.

#include "SharedConfig.hlsli"

#define SolidRootSig \
    "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), " \
    "CBV(b0), " \
    "CBV(b1)"

cbuffer SolidCb : register(b0) {
    float4x4 modelMat; // unused in PS, declared so reflection finds the cbuffer
}

struct LightData { float4 direction; float4 color; };

cbuffer PerFrameCb : register(b1) {
    float4x4 viewProjTransform;
    float4x4 rayDirTransform;
    float4 cameraPos;
    LightData lights[NUM_LIGHTS];
    float4 particleParams;
}

struct PSInput {
    float4 position : SV_Position;
    float3 worldPos : WORLDPOS;
    float3 normal : NORMAL;
};

static const float3 kMaterialColor  = float3(0.2f, 0.2f, 0.2f); // dark grey
static const float kAmbient = 0.3f;
static const float kSpecularStrength = 0.4f;
static const float kShininess = 48.0f;

[RootSignature(SolidRootSig)]
float4 main(PSInput input) : SV_Target
{
    float3 N = normalize(input.normal);
    float3 V = normalize(cameraPos.xyz - input.worldPos);

    float3 ambient = kAmbient * kMaterialColor;
    float3 litAccum = float3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < NUM_LIGHTS; i++)
    {
        float3 L    = normalize(lights[i].direction.xyz);
        float3 Hvec = normalize(L + V);  // renamed: H is a macro in SharedConfig.hlsli
        float3 lc   = lights[i].color.xyz;
        float3 diffuse  = max(dot(N, L), 0.0f) * kMaterialColor * lc;
        float  spec     = pow(max(dot(N, Hvec), 0.0f), kShininess);
        float3 specular = kSpecularStrength * spec * lc;
        litAccum += diffuse + specular;
    }

    return float4(ambient + litAccum, 1.0f);
}

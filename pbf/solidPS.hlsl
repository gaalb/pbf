// Pixel shader for solid obstacle rendering.
// Simple Blinn-Phong shading with a flat grey material colour.
//
// In:  WORLDPOS, NORMAL (interpolated from VS)
// Out: SV_Target (RGBA colour)

#define SolidRootSig \
    "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), " \
    "CBV(b0), " \
    "CBV(b1)"

cbuffer SolidCb : register(b0) {
    float4x4 modelMat; // unused in PS, declared so reflection finds the cbuffer
}

cbuffer PerFrameCb : register(b1) {
    float4x4 viewProjTransform; // unused in PS
    float4x4 rayDirTransform;   // unused in PS
    float4 cameraPos;
    float4 lightDir;
    float4 particleParams; // unused in PS
}

struct PSInput {
    float4 position : SV_Position;
    float3 worldPos : WORLDPOS;
    float3 normal : NORMAL;
};

static const float3 kMaterialColor  = float3(0.72f, 0.72f, 0.72f); // neutral grey
static const float kAmbient = 0.25f;
static const float kSpecularStrength = 0.4f;
static const float kShininess = 48.0f;

[RootSignature(SolidRootSig)]
float4 main(PSInput input) : SV_Target
{
    float3 N = normalize(input.normal);
    float3 L = normalize(lightDir.xyz);
    float3 V = normalize(cameraPos.xyz - input.worldPos);
    float3 H = normalize(L + V); // halfway vector for Blinn-Phong

    float3 ambient  = kAmbient * kMaterialColor;
    float3 diffuse  = max(dot(N, L), 0.0f) * kMaterialColor;
    float  spec     = pow(max(dot(N, H), 0.0f), kShininess);
    float3 specular = kSpecularStrength * spec * float3(1.0f, 1.0f, 1.0f);

    return float4(ambient + diffuse + specular, 1.0f);
}

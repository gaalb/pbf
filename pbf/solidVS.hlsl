// Vertex shader for solid obstacle rendering.
// Transforms mesh vertices from object space to clip space using
// the per-object model matrix and the shared per-frame view-projection.
//
// Input layout matches PNT_Vertex (position, normal, texcoord) produced by
// Egg::Importer::ImportSimpleObj. Texcoord is present in the vertex stream
// but not forwarded (we use flat Phong in the PS, no textures needed).

#define SolidRootSig \
    "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), " \
    "CBV(b0), " \
    "CBV(b1)"

cbuffer SolidCb : register(b0) {
    float4x4 modelMat; // object-to-world transform, updated each frame
}

#include "SharedConfig.hlsli"

struct LightData { float4 direction; float4 color; };

cbuffer PerFrameCb : register(b1) {
    float4x4 viewProjTransform;
    float4x4 rayDirTransform;
    float4 cameraPos;
    LightData lights[NUM_LIGHTS]; // must be here to match C++ layout
    float4 particleParams;
}

// match PNT_Vertex layout in Egg::Importer::ImportSimpleObj
struct VSInput {
    float3 position : POSITION; // object-space vertex position
    float3 normal : NORMAL; // object-space vertex normal
    float2 texcoord : TEXCOORD; // uv for texturing
};

struct VSOutput {
    float4 position : SV_Position; // system value semantic, clip-space vertex position
    float3 worldPos : WORLDPOS; // world-space vertex position, for lighting calculations in the PS
    float3 normal   : NORMAL; // world-space vertex normal, for lighting calculations in the PS
};

[RootSignature(SolidRootSig)]
VSOutput main(VSInput input)
{
    VSOutput output;

    float4 worldPos4 = mul(modelMat, float4(input.position, 1.0f)); // object-space position to world-space position 
    output.position = mul(viewProjTransform, worldPos4); // clip-space position for rasterization
    output.worldPos = worldPos4.xyz; // world-space position for lighting calculations in the PS

    // Normal transforms by the inverse-transpose of the model matrix.
    // For a rigid transform (rotation + translation, no scale) this equals
    // mul(v, M3x3), which produces the same rotation as the vertices.
    /// TODO: double check this
    float3x3 rotMat = (float3x3)modelMat;
    output.normal = normalize(mul(input.normal, rotMat));

    return output;
}

// Vertex shader for solid obstacle rendering.
// Transforms mesh vertices from object space to clip space using
// the per-object model matrix and the shared per-frame view-projection.
//
// Input layout matches PNT_Vertex (position, normal, texcoord) produced by
// Egg::Importer::ImportSimpleObj. Texcoord is present in the vertex stream
// but not forwarded (we use flat Phong in the PS, no textures needed).
//
// In:  POSITION, NORMAL, TEXCOORD (from vertex buffer)
// Out: SV_Position (clip space), WORLDPOS (world space), NORMAL (world space)

#define SolidRootSig \
    "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), " \
    "CBV(b0), " \
    "CBV(b1)"

cbuffer SolidCb : register(b0) {
    float4x4 modelMat; // object-to-world transform, updated each frame
}

cbuffer PerFrameCb : register(b1) {
    float4x4 viewProjTransform;
    float4x4 rayDirTransform; // unused here
    float4 cameraPos;
    float4 lightDir;
    float4 particleParams; // unused here
}

struct VSInput {
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float2 texcoord : TEXCOORD; // present in stream, not used
};

struct VSOutput {
    float4 position : SV_Position;
    float3 worldPos : WORLDPOS;
    float3 normal   : NORMAL;
};

[RootSignature(SolidRootSig)]
VSOutput main(VSInput input)
{
    VSOutput output;

    float4 worldPos4 = mul(modelMat, float4(input.position, 1.0f));
    output.position  = mul(viewProjTransform, worldPos4);
    output.worldPos  = worldPos4.xyz;

    // Normal transforms by the inverse-transpose of the model matrix.
    // For a rigid transform (rotation + translation, no scale) this equals
    // mul(v, M3x3), which produces the same rotation as the vertices.
    float3x3 rotMat = (float3x3)modelMat;
    output.normal = normalize(mul(input.normal, rotMat));

    return output;
}

// Background vertex shader.
// Receives the fullscreen quad vertices, which are already in NDC space (x,y in [-1,1]).
// For each vertex, computes the world-space ray direction the camera is looking through that
// screen position, by multiplying the NDC position with rayDirMat (the inverse of the
// view-rotation * projection matrix). The depth is hardcoded to 0.999999 (just inside the
// far plane) so that geometry drawn afterwards will always pass the depth test in front of it.
// The interpolated rayDir is passed to the pixel shader to sample the cubemap.

// root signature: defines what resources the shaders can access
// CBV(b0): per-frame constant buffer (same as particles, different root sig because we also need the cubemap)
// DescriptorTable(SRV(t0)): the cubemap texture
// StaticSampler(s0): a default texture sampler
#define BgRootSig "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), CBV(b0), DescriptorTable(SRV(t0, numDescriptors=1)), StaticSampler(s0)"

// FullScreenQuad uses PNT_Vertex, so we accept all its fields even if we only use position
struct IAOutput
{
    float3 position : POSITION; // In this specific case, NDC position (x,y in [-1,1], z=0)
    float3 normal : NORMAL; // unused
    float2 tex : TEXCOORD; // unused
};

struct VSOutput
{
    float4 position : SV_Position; // clip space position (w=1 so identical to NDC); z hardcoded to 0.999999 (far plane)
    float3 rayDir : RAYDIR; // world space ray direction from the camera through this screen pixel
};

cbuffer PerFrameCb : register(b0)
{
    float4x4 viewProjMat; // world space -> clip space  (view * projection)
    float4x4 rayDirMat; // NDC -> world space direction  (inverse of view-rotation * projection)
    float4 cameraPos; // camera position in world space
};

[RootSignature(BgRootSig)]
VSOutput main(IAOutput input)
{
    VSOutput output;
    // place the quad at the far plane so everything else draws in front of it
    output.position = float4(input.position, 1.0f);
    output.position.z = 0.999999;
    // transform the screen-space vertex position into a world-space ray direction
    // rayDirMat converts from clip space back to world space (without translation, just rotation)
    output.rayDir = mul(rayDirMat, float4(input.position, 1.0f)).xyz;
    return output;
}
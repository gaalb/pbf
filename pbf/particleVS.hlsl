// Root signature shared by VS, GS, and PS
// CBV(b0): per-frame constant buffer (camera matrices, light, particle params)
// DescriptorTable(SRV(t0)): particle structured buffer, read-only for the vertex shader
// ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT is intentionally omitted: we no longer use a vertex buffer,
// the VS fetches particle positions directly from the structured buffer using SV_VertexID
#define ParticleRootSig "CBV(b0), DescriptorTable(SRV(t0))"

#include "Particle.hlsli" // Particle struct: float3 position, float3 velocity

StructuredBuffer<Particle> particles : register(t0); // read-only view of the particle buffer, indexed by SV_VertexID

cbuffer PerFrameCb : register(b0)
{
    float4x4 viewProjMat;
    float4x4 rayDirMat;
    float4 cameraPos;
    float4 lightDir;
    float4 particleParams;
};

// VS just passes the world-space position and density to the geometry shader
struct VSOutput
{
    float3 worldPos : WORLDPOS; // particle center in world space, passed to GS for billboard generation
    float density : DENSITY; // SPH density estimate, passed through for coloring in PS
};

[RootSignature(ParticleRootSig)]
VSOutput main(uint vertexID : SV_VertexID) // SV_VertexID: auto-increments 0..N-1, one per DrawInstanced vertex, replaces IA vertex input
{
    VSOutput output;
    output.worldPos = particles[vertexID].position; // fetch this particle's world-space position from the structured buffer
    output.density = particles[vertexID].density; // pass density through for visualization
    return output;
}

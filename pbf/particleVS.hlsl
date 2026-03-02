// Root signature shared by VS, GS, and PS
// CBV(b0): per-frame constant buffer
#define ParticleRootSig "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), CBV(b0)"

struct IAOutput
{
    float3 position : POSITION; // the world position of the vertex
};

// VS just passes the world-space position to the geometry shader
struct VSOutput
{
    float3 worldPos : WORLDPOS; // the world position of the vertex, passed to GS for billboard generation
};

[RootSignature(ParticleRootSig)]
VSOutput main(IAOutput input)
{
    VSOutput output;
    output.worldPos = input.position;
    return output;
}
// Reduces the per-frame maximum DTVS (Distance To Visible Surface) into lodReduction[0].
// Each thread projects its particle into clip space, samples the depth texture written by the
// graphics queue's depth-only pass, and accumulates the maximum DTVS via atomic max.
// DTVS = max(0, particleNdcZ - sceneNdcZ): 0 for surface particles, positive for buried ones.

#define DtvsReductionRootSig \
    "CBV(b0), " \
    "DescriptorTable(UAV(u0, numDescriptors = 7)), " \
    "DescriptorTable(UAV(u7, numDescriptors = 1)), " \
    "DescriptorTable(SRV(t0, numDescriptors = 1))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"

RWStructuredBuffer<float3>          predictedPosition   : register(u2);
globallycoherent RWStructuredBuffer<uint> lodReduction  : register(u7);
Texture2D<float>                    particleDepthTex    : register(t0);

[RootSignature(DtvsReductionRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles) return;

    float4 clip = mul(viewProjTransform, float4(predictedPosition[i], 1.0f));

    // Skip particles behind the camera or outside the depth range
    if (clip.w <= 0.0f) return;
    float3 ndc = clip.xyz / clip.w;
    if (abs(ndc.x) > 1.0f || abs(ndc.y) > 1.0f || ndc.z < 0.0f || ndc.z > 1.0f) return;

    // Convert NDC to integer pixel coordinates
    float2 uv = float2(ndc.x * 0.5f + 0.5f, -ndc.y * 0.5f + 0.5f);
    int2 px = clamp(int2(uv * float2(viewportWidth, viewportHeight)),
                    int2(0, 0),
                    int2((int)viewportWidth - 1, (int)viewportHeight - 1));

    // Scene depth at this pixel (NDC Z of the closest particle billboard)
    float sceneDepth = particleDepthTex.Load(int3(px, 0));

    // DTVS: how much farther this particle is from the front visible surface
    float dtvs = max(0.0f, ndc.z - sceneDepth);

    // For positive floats, asuint preserves ordering, so InterlockedMax on bit patterns is correct
    InterlockedMax(lodReduction[0], asuint(dtvs));
}

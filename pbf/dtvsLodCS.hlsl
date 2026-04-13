// Assigns per-particle LOD countdowns based on DTVS (Distance To Visible Surface).
// Recomputes each particle's DTVS and normalises it against the frame-wide maximum
// accumulated by dtvsReductionCS, then interpolates between maxLOD (surface) and
// minLOD (most buried). Off-screen/behind-camera particles receive maxLOD (conservative).

#define DtvsLodRootSig \
    "CBV(b0), " \
    "DescriptorTable(UAV(u0, numDescriptors = 7)), " \
    "DescriptorTable(UAV(u7, numDescriptors = 1)), " \
    "DescriptorTable(UAV(u8, numDescriptors = 1)), " \
    "DescriptorTable(SRV(t0, numDescriptors = 1))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"

RWStructuredBuffer<float3> predictedPosition : register(u2);
RWStructuredBuffer<uint>   lod               : register(u7);
RWStructuredBuffer<uint>   lodReduction      : register(u8);
Texture2D<float>           particleDepthTex  : register(t0);

[RootSignature(DtvsLodRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles) return;

    float maxDtvs = asfloat(lodReduction[0]);

    // Default: max LOD (surface particles, or off-screen/behind-camera — conservative)
    uint lodVal = maxLOD;

    float4 clip = mul(viewProjTransform, float4(predictedPosition[i], 1.0f));
    if (clip.w > 0.0f)
    {
        float3 ndc = clip.xyz / clip.w;
        if (abs(ndc.x) <= 1.0f && abs(ndc.y) <= 1.0f && ndc.z >= 0.0f && ndc.z <= 1.0f)
        {
            float2 uv = float2(ndc.x * 0.5f + 0.5f, -ndc.y * 0.5f + 0.5f);
            int2 px = clamp(int2(uv * float2(viewportWidth, viewportHeight)),
                            int2(0, 0),
                            int2((int)viewportWidth - 1, (int)viewportHeight - 1));

            float sceneDepth = particleDepthTex.Load(int3(px, 0));
            float dtvs = max(0.0f, ndc.z - sceneDepth);

            float t = (maxDtvs > 0.0f) ? saturate(dtvs / maxDtvs) : 0.0f;
            lodVal = (uint)round(lerp((float)maxLOD, (float)minLOD, t));
        }
    }

    lod[i] = lodVal;
}

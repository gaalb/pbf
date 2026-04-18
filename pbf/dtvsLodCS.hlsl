// Assigns per-particle LOD countdowns based on DTVS (Distance To Visible Surface).
// Recomputes each particle's DTVS and normalises it against the frame-wide maximum
// accumulated by dtvsReductionCS, then interpolates between maxLOD (surface) and
// minLOD (most buried). Off-screen/behind-camera particles receive maxLOD (conservative).
//
// In: predictedPosition, lodReduction
// Out: lod

#define DtvsLodRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 7), UAV(u7, numDescriptors = 1), UAV(u8, numDescriptors = 1), SRV(t0, numDescriptors = 1))"

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

    // Default: avg
    uint lodVal = round(0.5*(float)maxLOD + 0.5*(float)minLOD);

    // Project world position into clip space. Result is a homogenous 4-component
    // vector, where clip.w is the view-space depth 
    float4 clip = mul(viewProjTransform, float4(predictedPosition[i], 1.0f));
    
    if (clip.w > 0.0f) { // Skip particles behind the camera
        float3 ndc = clip.xyz / clip.w; // normalize, z is now depth in [0,1] range, like in depth buffer
        if (abs(ndc.x) <= 1.0f && abs(ndc.y) <= 1.0f && ndc.z >= 0.0f && ndc.z <= 1.0f) { // skip outside of viewport
            float2 uv = float2(ndc.x * 0.5f + 0.5f, -ndc.y * 0.5f + 0.5f); // texture uv
            // integer pixel coordinate for Load call, with clamp to ensure we don't accidentally read outside the texture
            int2 px = clamp(int2(uv * float2(viewportWidth, viewportHeight)),
                            int2(0, 0),
                            int2((int)viewportWidth - 1, (int)viewportHeight - 1));
            // Scene depth at this pixel (NDC Z of the closest particle billboard)
            // We use Load, cause it's a direct texel fetch, as opposed to Sample which would do filtering and potentially 
            // read neighboring pixels, which would be bad for accuracy here. Mip level 0 is the full-res depth buffer.
            float sceneDepth = particleDepthTex.Load(int3(px, 0));
            // DTVS: how much farther this particle is from the front visible surface
            float dtvs = max(0.0f, ndc.z - sceneDepth);
            // linearly interpolate between maxLOD and minLOD based on dtvs, where dtvs=0 maps to maxLOD and dtvs=maxDtvs 
            // maps to minLOD, with a guard against the maxDtvs=0 edge case (which would cause divide-by-zero)
            float t = (maxDtvs > 0.0f) ? saturate(dtvs / maxDtvs) : 0.0f;
            lodVal = (uint)round(lerp((float)maxLOD, (float)minLOD, t));
        }
    }

    lod[i] = lodVal;
}

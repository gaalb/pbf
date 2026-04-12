#define LodRootSig \
    "CBV(b0), " \
    "DescriptorTable(UAV(u0, numDescriptors = 7)), " \
    "DescriptorTable(UAV(u7, numDescriptors = 1)), " \
    "DescriptorTable(UAV(u8, numDescriptors = 1)), " 

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"

RWStructuredBuffer<float3> predictedPosition : register(u2);
RWStructuredBuffer<uint>   lod               : register(u7);
RWStructuredBuffer<uint>   lodReduction      : register(u8);

[RootSignature(LodRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles) return;

    float dtcMin = asfloat(lodReduction[0]);
    float dtcMax = asfloat(lodReduction[1]);
    float dtc    = length(predictedPosition[i] - cameraPos);
    float t = (dtcMax > dtcMin) ? saturate((dtc - dtcMin) / (dtcMax - dtcMin)) : 0.0;
    uint lodVal = (uint)round(lerp((float)maxLOD, (float)minLOD, t));
    lod[i]      = lodVal;
}

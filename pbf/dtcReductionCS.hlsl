#define DtcReductionRootSig \
    "CBV(b0), " \
    "DescriptorTable(UAV(u0, numDescriptors = 7)), " \
    "DescriptorTable(UAV(u7, numDescriptors = 1))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"

RWStructuredBuffer<float3> predictedPosition : register(u2);
globallycoherent RWStructuredBuffer<uint> lodReduction : register(u7);

[RootSignature(DtcReductionRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles) return;

    float dtc = length(predictedPosition[i] - cameraPos);
    uint dtcBits = asuint(dtc);
    InterlockedMin(lodReduction[0], dtcBits);
    InterlockedMax(lodReduction[1], dtcBits);
}

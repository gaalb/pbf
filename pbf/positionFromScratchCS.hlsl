// deltaCS writes its corrected position to the scratch field to avoid a Gauss-Seidel race.
// This shader copies scratch back to predictedPosition so the next solver iteration
// (or updateVelocityCS) sees a consistent snapshot.
//
// In: scratch
// Out: predictedPosition


#define PositionFromScratchRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 7))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"

RWStructuredBuffer<float3> predictedPosition : register(u2);
RWStructuredBuffer<float3> scratch : register(u6);

[RootSignature(PositionFromScratchRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    predictedPosition[i] = scratch[i];
}

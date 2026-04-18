// viscosityCS writes its corrected velocity to the scratch field to avoid a Gauss-Seidel
// race. This shader copies scratch back to velocity.
//
// In: scratch
// Out: velocity

#define VelocityFromScratchRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 2))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"

RWStructuredBuffer<float3> velocity : register(u0);
RWStructuredBuffer<float3> scratch  : register(u1);

[RootSignature(VelocityFromScratchRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    velocity[i] = scratch[i];
}

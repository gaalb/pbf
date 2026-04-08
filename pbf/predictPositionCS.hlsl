// compute predicted position p* based on current position and velocity
//
// In:  position, velocity 
// Out: predictedPosition

#define PredictPositionRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 7))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"

RWStructuredBuffer<float3> position          : register(u0);
RWStructuredBuffer<float3> velocity          : register(u1);
RWStructuredBuffer<float3> predictedPosition : register(u2);

[RootSignature(PredictPositionRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    // p* is our best guess of where the particle will end up if no constraints
    // are applied. The constraint solver will nudge p* to satisfy density better.
    predictedPosition[i] = position[i] + velocity[i] * dt;
}

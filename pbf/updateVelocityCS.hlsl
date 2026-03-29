// After the constraint solver has converged, derive velocity from displacement:
//   v_i = (predictedPosition_i - position_i) / dt
//
// Position is NOT updated here. Per the paper, vorticity confinement and viscosity
// run next and need the old positions. updatePositionCS updates position at the very end.
//
// In: predictedPosition, position
// Out: velocity

#define UpdateVelocityRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 7))"

#include "ComputeCb.hlsli"

RWStructuredBuffer<float3> position : register(u0);
RWStructuredBuffer<float3> velocity : register(u1);
RWStructuredBuffer<float3> predictedPosition : register(u2);

[RootSignature(UpdateVelocityRootSig)]
[numthreads(256, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    // velocity = displacement / dt (implicit velocity update from PBD)
    velocity[i] = (predictedPosition[i] - position[i]) / dt;
}

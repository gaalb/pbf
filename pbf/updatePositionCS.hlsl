// Updates the current position from the solver's final predictedPosition:
//   position_i = predictedPosition_i
//
// This runs last in the frame, after vorticity confinement and viscosity have used the
// old positions. This matches the paper's ordering where position is updated after
// all velocity post-processing.
//
// In: predictedPosition
// Out: position

#define UpdatePositionRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 7))"

#include "ComputeCb.hlsli"

RWStructuredBuffer<float3> position : register(u0);
RWStructuredBuffer<float3> predictedPosition : register(u2);

[RootSignature(UpdatePositionRootSig)]
[numthreads(256, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    position[i] = predictedPosition[i];
}

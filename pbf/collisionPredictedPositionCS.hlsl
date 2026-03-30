// Clamps predictedPosition to the simulation box.
// Called every solver iteration (same cadence as the old collisionCS), but
// deliberately does NOT touch velocity. Velocity wall response is handled
// once per simulation step by collisionVelocityCS, which runs before
// predictPositionCS so that the correction actually survives into p*.
//
// In:  predictedPosition
// Out: predictedPosition

#define CollisionPositionRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 7))"

#include "ComputeCb.hlsli"

RWStructuredBuffer<float3> predictedPosition : register(u2);

[RootSignature(CollisionPositionRootSig)]
[numthreads(256, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    predictedPosition[i] = clamp(predictedPosition[i], boxMin, boxMax);
}

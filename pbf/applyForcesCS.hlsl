// Apply external forces to velocity.

// Separated from predictPositionCS so that collisionVelocityCS can run
// between them and zero wall-directed velocity components before the
// position prediction commits them into p*.
//
// In:  velocity, position  (read-only, needed for fountain region check)
// Out: velocity

#define ApplyForcesRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 7))"

#include "ComputeCb.hlsli"

RWStructuredBuffer<float3> position : register(u0);
RWStructuredBuffer<float3> velocity : register(u1);

[RootSignature(ApplyForcesRootSig)]
[numthreads(256, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    float3 force = float3(0.0, -9.8, 0.0) + externalForce;

    // fountain: upward jet in a corner of the box
    if (fountainEnabled) {
        float3 extent = boxMax - boxMin;
        float3 pos = position[i];
        if (pos.x > boxMax.x - extent.x * 0.05 &&
            pos.z > boxMax.z - extent.z * 0.05 &&
            pos.y < boxMin.y + extent.y * 0.3)
            force.y += 150.0;
    }

    velocity[i] += force * dt;
}

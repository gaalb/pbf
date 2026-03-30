// Applies velocity-level wall response. Because updateVelocityCS later 
// overwrites velocity with (p* - position) / dt, any velocity edits made 
// after the position prediction (using the external forces) is
// lost. By running here (after applyForcesCS, before predictPositionCS) the
// corrected velocity is baked into p* and survives.
//
// Two effects are applied when a particle is headed into a wall:
//
// Normal zeroing: zero the velocity component pointing into the wall,
// preventing the particle from accelerating further into it.
//
// Adhesion (tangential damping): reduce velocity components parallel to
// the wall surface slightly to mimic the adhesion/friction.
//
// Wall contact is detected by checking the current (committed) position against
// the box boundaries.

// In:  position, velocity  (force-applied by applyForcesCS)
// Out: velocity

#define CollisionVelocityRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 7))"

#include "ComputeCb.hlsli"

RWStructuredBuffer<float3> position : register(u0);
RWStructuredBuffer<float3> velocity : register(u1);

[RootSignature(CollisionVelocityRootSig)]
[numthreads(256, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    float3 pos = position[i];
    float3 v   = velocity[i];

    // Detect wall contact from the current (committed) position.
    // Positions are always clamped to [boxMin, boxMax] by collisionPositionCS,
    // so equality with a boundary means the particle is at that wall.
    if (pos.x <= boxMin.x) { v.x = max(v.x, 0.0); v.y *= (1.0 - adhesion); v.z *= (1.0 - adhesion); }
    if (pos.x >= boxMax.x) { v.x = min(v.x, 0.0); v.y *= (1.0 - adhesion); v.z *= (1.0 - adhesion); }
    if (pos.y <= boxMin.y) { v.y = max(v.y, 0.0); v.x *= (1.0 - adhesion); v.z *= (1.0 - adhesion); }
    if (pos.y >= boxMax.y) { v.y = min(v.y, 0.0); v.x *= (1.0 - adhesion); v.z *= (1.0 - adhesion); }
    if (pos.z <= boxMin.z) { v.z = max(v.z, 0.0); v.x *= (1.0 - adhesion); v.y *= (1.0 - adhesion); }
    if (pos.z >= boxMax.z) { v.z = min(v.z, 0.0); v.x *= (1.0 - adhesion); v.y *= (1.0 - adhesion); }

    velocity[i] = v;
}

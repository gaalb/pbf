// Clamps predictedPosition to the simulation box and zeroes velocity components
// that point into walls. 
// TODO: collisions should also influence the velocity component parallel to the
// wall, and slow them down somewhat, as there is some adhesion between the surface
// and the liquid.
//
// In: velocity, predictedPosition
// Out: velocity, predictedPosition

#define CollisionRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 7))"

#include "ComputeCb.hlsli"

RWStructuredBuffer<float3> velocity : register(u1);
RWStructuredBuffer<float3> predictedPosition : register(u2);

[RootSignature(CollisionRootSig)]
[numthreads(256, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    float3 pp = predictedPosition[i];
    float3 v  = velocity[i];

    // Zero velocity components pointing into walls
    if (pp.x < boxMin.x) v.x = max(v.x, 0.0);
    if (pp.y < boxMin.y) v.y = max(v.y, 0.0);
    if (pp.z < boxMin.z) v.z = max(v.z, 0.0);
    if (pp.x > boxMax.x) v.x = min(v.x, 0.0);
    if (pp.y > boxMax.y) v.y = min(v.y, 0.0);
    if (pp.z > boxMax.z) v.z = min(v.z, 0.0);

    // Clamp position to simulation box.
    pp = clamp(pp, boxMin, boxMax);

    predictedPosition[i] = pp;
    velocity[i] = v;
}

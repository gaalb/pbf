// Each thread handles one particle:
//   1. Apply external forces (gravity) to velocity.
//   2. Compute a predicted position:  p* = position + velocity * dt
//
// The committed 'position' field is intentionally not updated here.
// It stays frozen at its value from the previous frame until finalizeCS,
// which computes the new velocity as (p* - position) / dt and commits p*.
//
// In: position, velocity
// Out: predictedPosition, velocity

#define PredictRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 7))"

#include "ComputeCb.hlsli"

RWStructuredBuffer<float3> position : register(u0);
RWStructuredBuffer<float3> velocity : register(u1);
RWStructuredBuffer<float3> predictedPosition : register(u2);

[RootSignature(PredictRootSig)]
[numthreads(256, 1, 1)] // 256 threads per group; each thread = one particle
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x; // global thread index = particle index

    // Guard: we may dispatch more threads than particles (rounded up to 256).
    // Extra threads must do nothing.
    if (i >= numParticles)
        return;

    // apply external forces (gravity + arrow-key acceleration): update velocity first,
    // then use the updated velocity for the position prediction (semi-implicit Euler)
    float3 force = float3(0.0, -9.8, 0.0) + externalForce;

    // fountain: upward jet in a corner of the box
    if (fountainEnabled) {
        float3 extent = boxMax - boxMin;
        float3 pos = position[i];
        if (pos.x > boxMax.x - extent.x * 0.05 &&
            pos.z > boxMax.z - extent.z * 0.05 &&
            pos.y < boxMin.y + extent.y * 0.3)
            force.y += 400.0;
    }

    velocity[i] += force * dt;

    // p* is our best guess of where the particle will end up if no constraints
    // are applied. The constraint solver will nudge p* to satisfy density better
    predictedPosition[i] = position[i] + velocity[i] * dt;
}

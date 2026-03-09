// PBF finalize step (Macklin & Muller 2013):
//
// After the constraint solver has (hopefully) converged on a corrected predictedPosition,
// this pass commits the result:
//
//   velocity_i = (predictedPosition_i - position_i) / dt
//   position_i = predictedPosition_i
//
// Velocity is derived from the displacement rather than integrated directly.
// This means any corrections applied by deltaCS (pressure, boundary clamping)
// are automatically reflected in the new velocity -- no explicit collision
// response or velocity reflection is needed.
//
// Root signature:
//   CBV(b0)                  -- ComputeCb
//   DescriptorTable(UAV(u0)) -- particle buffer (read predictedPosition, write position + velocity)

#define FinalizeRootSig "CBV(b0), DescriptorTable(UAV(u0))"

#include "Particle.hlsli" // Particle struct

cbuffer ComputeCb : register(b0)
{
    float dt; // simulation timestep -- needed to compute velocity from displacement
    uint numParticles; // bounds check
    float h; // unused in this pass
    float rho0; // unused in this pass
    float epsilon; // unused in this pass
    float pad[3]; // padding to 32 bytes
};

RWStructuredBuffer<Particle> particles : register(u0);

[RootSignature(FinalizeRootSig)]
[numthreads(256, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    float3 oldPos = particles[i].position;
    float3 newPos = particles[i].predictedPosition;

    // velocity = displacement / dt (implicit velocity update from PBF)
    particles[i].velocity = (newPos - oldPos) / dt;

    // commit the predicted position as the new current position
    particles[i].position = newPos;
}

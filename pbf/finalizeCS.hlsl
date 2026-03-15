// PBF finalize step (Macklin & Muller 2013):
//
// After the constraint solver has (hopefully) converged on a corrected predictedPosition,
// this pass commits the result:
//
//   velocity_i = (predictedPosition_i - position_i) / dt
//   position_i = predictedPosition_i
//
// Velocity is derived from the displacement rather than integrated directly.
//
// Root signature:
//   CBV(b0)                  -- ComputeCb
//   DescriptorTable(UAV(u0)) -- particle buffer (read predictedPosition, write position + velocity)

#define FinalizeRootSig "CBV(b0), DescriptorTable(UAV(u0))"

#include "Particle.hlsli" // Particle struct

cbuffer ComputeCb : register(b0)
{
    float dt; // offset 0 (4 bytes): simulation timestep in seconds
    uint numParticles; // offset 4 (4 bytes): total particle count
    float h; // offset 8 (4 bytes): SPH smoothing radius
    float rho0; // offset 12 (4 bytes): rest density
    float3 boxMin; // offset 16 (12 bytes): simulation box minimum corner (world space)
    float epsilon; // offset 28 (4 bytes): constraint force mixing relaxation
    float3 boxMax; // offset 32 (12 bytes): simulation box maximum corner (world space)
    float viscosity; // offset 44 (4 bytes): XSPH viscosity coefficient c
    float sCorrK; // offset 48 (4 bytes): artificial pressure k
    float sCorrDeltaQ; // offset 52 (4 bytes): artificial pressure deltaq
    float sCorrN; // offset 56 (4 bytes): artificial pressure n
    float vorticityEpsilon; // offset 60 (4 bytes): vorticity confinement strength coefficient
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

    // velocity = displacement / dt (implicit velocity update from PBF)0
    particles[i].velocity = (newPos - oldPos) / dt;

    // commit the predicted position as the new current position
    particles[i].position = newPos;
}

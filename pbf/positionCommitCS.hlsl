// Position commit pass (Macklin & Muller 2013):
//
// Commits the solver's final predictedPosition as the new current position:
//   position_i = predictedPosition_i
//
// This runs last in the frame, after vorticity confinement and viscosity have used the
// old positions. This matches the paper's ordering where position is committed after
// all velocity post-processing.
//
// Root signature:
//   CBV(b0)                  -- ComputeCb
//   DescriptorTable(UAV(u0)) -- particle buffer (read predictedPosition, write position)

#define PositionCommitRootSig "CBV(b0), DescriptorTable(UAV(u0))"

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

[RootSignature(PositionCommitRootSig)]
[numthreads(256, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    particles[i].position = particles[i].predictedPosition;
}

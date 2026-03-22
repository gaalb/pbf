// Collision detection and response pass:
//
// Clamps predictedPosition to the simulation box and zeroes velocity components
// that point into walls. Dispatched at two points in the pipeline:
//
// 1. Pre-stabilization (after predictCS, before the grid build)
//
// 2. Post-delta (after positionFromScratchCS, inside the solver loop)
//
// Root signature:
//   CBV(b0)                                    -- ComputeCb
//   DescriptorTable(UAV(u0, numDescriptors=5)) -- u0: particles, u1-u4: grid + sort buffers (unused here)

#define CollisionRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 5))"

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
    float3 externalForce; // offset 64 (12 bytes): horizontal force from arrow keys (acceleration, m/s^2)
    uint maxPerCell; // offset 76 (4 bytes): max particle indices stored per grid cell
};

RWStructuredBuffer<Particle> particles : register(u0);

[RootSignature(CollisionRootSig)]
[numthreads(256, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    float3 pp = particles[i].predictedPosition;
    float3 v  = particles[i].velocity;
    
    // Zero velocity components pointing into walls
    if (pp.x < boxMin.x) v.x = max(v.x, 0.0);
    if (pp.y < boxMin.y) v.y = max(v.y, 0.0);
    if (pp.z < boxMin.z) v.z = max(v.z, 0.0);
    if (pp.x > boxMax.x) v.x = min(v.x, 0.0);
    if (pp.y > boxMax.y) v.y = min(v.y, 0.0);
    if (pp.z > boxMax.z) v.z = min(v.z, 0.0);

    // Clamp position to simulation box.
    pp = clamp(pp, boxMin, boxMax);

    particles[i].predictedPosition = pp;
    particles[i].velocity = v;
}

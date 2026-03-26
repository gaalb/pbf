// PBF prediction phase (Macklin & Muller 2013):
// Each thread handles one particle:
//   1. Apply external forces (gravity) to velocity.
//   2. Compute a predicted position:  p* = position + velocity * dt
//
// The committed 'position' field is intentionally not updated here.
// It stays frozen at its value from the previous frame until finalizeCS,
// which computes the new velocity as (p* - position) / dt and commits p*.
//
// Root signature:
//   CBV(b0)               — ComputeCb: dt, numParticles, h, rho0, epsilon
//   DescriptorTable(UAV(u0)) — particle buffer (read + write)

#define PredictRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 4))"

#include "Particle.hlsli" // Particle struct: position, velocity, predictedPosition, lambda

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
};

// RWStructuredBuffer: read-write access to the particle array.
// The VS reads this same resource as an SRV; the compute pipeline can
// write it because it was created with D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.
RWStructuredBuffer<Particle> particles : register(u0);

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
    float3 extent = boxMax - boxMin;
    float3 pos = particles[i].position;
    if (pos.x > boxMax.x - extent.x * 0.05 &&
        pos.z > boxMax.z - extent.z * 0.05 &&
        pos.y < boxMin.y + extent.y * 0.3)
        force.y += 400.0;

    particles[i].velocity += force * dt;

    // p* is our best guess of where the particle will end up if no constraints
    // are applied. The constraint solver will nudge p* to satisfy density better
    particles[i].predictedPosition = particles[i].position + particles[i].velocity * dt;
}

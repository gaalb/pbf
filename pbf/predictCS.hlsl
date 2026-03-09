// PBF prediction phase
// Each thread handles one particle:
//   1. Apply external forces (gravity) to velocity.
//   2. Compute a predicted position:  p* = position + velocity * dt
//
// The committed 'position' field is intentionally NOT updated here.
// It stays frozen at its value from the previous frame until finalizeCS,
// which computes the new velocity as (p* - position) / dt and commits p*.
//
// Root signature:
//   CBV(b0)               — ComputeCb: dt, numParticles, h, rho0, epsilon
//   DescriptorTable(UAV(u0)) — particle buffer (read + write)

#define PredictRootSig "CBV(b0), DescriptorTable(UAV(u0))"

#include "Particle.hlsli" // Particle struct: position, velocity, predictedPosition, lambda

cbuffer ComputeCb : register(b0)
{
    // Must match ComputeCb in ConstantBufferTypes.h exactly — fields in the same order
    // so their byte offsets agree between CPU and GPU.
    float dt;  // simulation timestep in seconds  
    uint  numParticles; // total particle count, for bounds check
    float h; // SPH smoothing radius (unused in this pass)
    float rho0; // rest density (unused in this pass)
    float epsilon; // lambda relaxation (unused in this pass)
    float pad[3]; // padding to 32 bytes 
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

    // --- Step 1a: apply gravity ---
    // Semi-implicit Euler: update velocity first, then use the updated velocity
    // for the position prediction. This gives slightly better energy conservation
    // than explicit Euler (using the old velocity for position).
    particles[i].velocity += float3(0.0, -9.8, 0.0) * dt;

    // --- Step 1b: predict position ---
    // p* is our best guess of where the particle will end up if no constraints
    // are applied. The constraint solver will nudge p* until density is satisfied.
    particles[i].predictedPosition = particles[i].position + particles[i].velocity * dt;
}

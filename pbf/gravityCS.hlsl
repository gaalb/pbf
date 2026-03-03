// Root signature for the gravity compute shader.
// CBV(b0): ComputeCb — simulation parameters (dt, numParticles)
// DescriptorTable(UAV(u0)): the particle buffer, read-write so we can update positions and velocities.
#define GravityRootSig "CBV(b0), DescriptorTable(UAV(u0))"

#include "Particle.hlsli" // Particle struct: float3 position, float3 velocity

cbuffer ComputeCb : register(b0)
{
    float dt; // simulation timestep in seconds
    uint numParticles; // total particle count, used for the bounds check below
    float2 pad; // padding to 16 bytes — must match the C++ ComputeCb struct layout exactly
};

// RWStructuredBuffer: read-write view of the particle buffer.
// This is the same GPU resource that particleVS reads as an SRV (read-only),
// but here we bind it as a UAV so the compute shader can write back updated values.
RWStructuredBuffer<Particle> particles : register(u0);

[RootSignature(GravityRootSig)]
[numthreads(256, 1, 1)] // 256 threads per group along X; Y and Z are 1 (1D dispatch)
void main(uint3 dispatchID : SV_DispatchThreadID) // SV_DispatchThreadID = groupID * groupSize + threadID
{
    uint i = dispatchID.x; // each thread handles one particle, indexed by the global thread ID

    // for example if we launch ceil(1000 / 256) = 4 groups = 1024 threads, but only indices 0–999 correspond to real particles
    // threads 1000–1023 must do nothing
    if (i >= numParticles)
        return;

    // F = ma, with m = 1 and a = (0, -9.8, 0) (gravitational acceleration, world-space Y is up), dv = a * dt
    particles[i].velocity += float3(0.0, -9.8, 0.0) * dt;

    // x_new = x_old + v_new * dt
    particles[i].position += particles[i].velocity * dt;
}

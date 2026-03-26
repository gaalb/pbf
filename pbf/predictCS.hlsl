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
//   CBV(b0)                        — ComputeCb: dt, numParticles, h, rho0, epsilon
//   DescriptorTable(UAV(u0..u6))   — particle field buffers
//   DescriptorTable(UAV(u7..u8))   — grid buffers (unused here)
//   DescriptorTable(UAV(u9..u15))  — sorted particle field buffers (unused here)

#define PredictRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 7)), DescriptorTable(UAV(u7, numDescriptors = 2)), DescriptorTable(UAV(u9, numDescriptors = 7))"

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
    uint fountainEnabled; // offset 76 (4 bytes): 1 = fountain jet active, 0 = off
};

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

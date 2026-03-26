// Vorticity estimation pass (Macklin & Muller 2013):
//
// For each particle i, estimates the local velocity field curl / vorticity via SPH.
//   omega_i = sum_j (v_j - v_i) x grad_W_spiky( p_i - p_j, h)
//
// Result is stored in particles[i].omega, and consumed by confinementCS in the next pass.
// Per the paper's ordering, this pass uses updated velocity (from updateVelocityCS) but
// the OLD positions (updatePositionCS has not run yet).
//
// Root signature:
//   CBV(b0)                  -- ComputeCb
//   DescriptorTable(UAV(u0)) -- particle buffer (read position + velocity, write omega)


#define VorticityRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 4))"

#include "Particle.hlsli"   // Particle struct
#include "SphKernels.hlsli" // SpikyGrad

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

#include "GridUtils.hlsli" // posToCell(), cellIndex(), gridDims()

RWStructuredBuffer<Particle> particles : register(u0);
RWStructuredBuffer<uint> cellCount : register(u1);
RWStructuredBuffer<uint> cellPrefixSum : register(u3);

[RootSignature(VorticityRootSig)]
[numthreads(256, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    float3 pi = particles[i].position; // committed position from finalizeCS
    float3 vi = particles[i].velocity; // post-constraint velocity from finalizeCS

    float3 omega = float3(0, 0, 0); // accumulates the vorticity vector for particle i

    // The grid was built from predictedPositions, but this pass uses old positions.
    // The displacement between old and predicted is small relative to h, so the
    // grid is still a valid acceleration structure for finding neighbors here.
    int3 myCell = posToCell(pi);
    int3 dims = gridDims();
    for (int dz = -1; dz <= 1; dz++)
    for (int dy = -1; dy <= 1; dy++)
    for (int dx = -1; dx <= 1; dx++)
    {
        int3 nc = myCell + int3(dx, dy, dz);
        if (nc.x < 0 || nc.x >= dims.x ||
            nc.y < 0 || nc.y >= dims.y ||
            nc.z < 0 || nc.z >= dims.z)
            continue;

        uint ci = cellIndex(nc);
        uint count = cellCount[ci];

        for (uint s = 0; s < count; s++)
        {
            uint j = cellPrefixSum[ci] + s;
            if (j == i)
                continue;

            float3 r = pi - particles[j].position;

            // The curl estimator uses the gradient of W with respect to p_j, not p_i.
            // grad_{p_j} W(p_i - p_j, h) = -grad_{p_i} W(p_i - p_j, h) = -SpikyGrad(r, h)
            // so the sign is negated compared to using SpikyGrad directly.
            omega += cross(particles[j].velocity - vi, -SpikyGrad(r, h));
        }
    }

    // store omega, confinementCS will read it from there
    particles[i].omega = omega;
}

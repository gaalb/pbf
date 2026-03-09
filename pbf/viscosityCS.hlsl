// XSPH viscosity pass (Macklin & Muller 2013):
//
// After finalizeCS has committed positions and computed velocities from displacement,
// this pass applies XSPH viscosity to smooth the velocity field:
// v_i = v_i + c * sum_{j != i} (v_j - v_i) * W_poly6(p_i - p_j, h)
//
// Each particle's velocity is nudged toward the weighted average of its neighbors'
// velocities. c controls how strongly -- 0 means no viscosity, 1 means full averaging.
//
// Position reads are race-free: finalizeCS has finished and a UAV barrier was issued.
// Velocity reads have the same Gauss-Seidel race as deltaCS: thread i reads velocity[j]
// while thread j may have already written its corrected velocity[j] in this dispatch.
//
// Root signature:
//   CBV(b0)                  -- ComputeCb
//   DescriptorTable(UAV(u0)) -- particle buffer (read position + velocity, write velocity)

#define ViscosityRootSig "CBV(b0), DescriptorTable(UAV(u0))"

#include "Particle.hlsli"   // Particle struct
#include "SphKernels.hlsli" // Poly6

cbuffer ComputeCb : register(b0)
{
    float dt;           // unused in this pass
    uint numParticles;  // loop bound and bounds check
    float h;            // SPH smoothing radius
    float rho0;         // unused in this pass
    float epsilon;      // unused in this pass
    float viscosity;    // XSPH viscosity coefficient c
    float pad[2];       // padding to 32 bytes
};

RWStructuredBuffer<Particle> particles : register(u0);

[RootSignature(ViscosityRootSig)]
[numthreads(256, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    float3 pi = particles[i].position; // committed position after finalizeCS
    float3 vi = particles[i].velocity; // velocity after finalizeCS: (p* - p_old) / dt

    float3 xsphSum = float3(0, 0, 0);

    for (uint j = 0; j < numParticles; j++)
    {
        if (j != i)
        {
            float3 r = pi - particles[j].position;
            float3 vj = particles[j].velocity;

            // (v_j - v_i) * W: neighbor's velocity contribution weighted by proximity
            xsphSum += (vj - vi) * Poly6(r, h);
        }
    }

    particles[i].velocity = vi + viscosity * xsphSum;
}

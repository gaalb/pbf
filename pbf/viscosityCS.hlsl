// XSPH viscosity pass (Macklin & Muller 2013):
//
// After finalizeCS has committed positions and computed velocities from displacement,
// this pass applies XSPH viscosity to smooth the velocity field:
// v_i = v_i + c * sum_{j != i} (v_j - v_i) * W_poly6(p_i - p_j, h)
//
// Each particle's velocity is nudged toward the weighted average of its neighbors'
// velocities. c controls how strongly: 0 means no viscosity, 1 means full averaging.
//
// Position reads are race-free: finalizeCS has finished and a UAV barrier was issued.
// Velocity reads have the same Gauss-Seidel race as deltaCS: thread i reads velocity[j]
// while thread j may have already written its corrected velocity[j] in this dispatch.
//
// Root signature:
//   CBV(b0)                  -- ComputeCb
//   DescriptorTable(UAV(u0)) -- particle buffer (read position + velocity, write velocity)
//
// Note that there is a slight deviationn from the paper at this point: in the paper, they
// finalize vi with respect to constraints and collision contributions, THEN apply vorticity  
// constraints and viscosity, THEN commit the position. This means that the vorticity and
// viscosity corrections see updated velocity but the old positions, as opposed to my implementation
// where they see the updated velocity and position. TODO: fix

#define ViscosityRootSig "CBV(b0), DescriptorTable(UAV(u0))"

#include "Particle.hlsli"   // Particle struct
#include "SphKernels.hlsli" // Poly6

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

            // (v_j - v_i) * W: neighbor's velocity contribution weighted by proximity.
            // The full XSPH formula (Schechter & Bridson 2012) is:
            //   sum_j (m_j / rho_j) * (v_j - v_i) * W(r_ij, h)
            // m_j = 1 is dropped as a uniform constant; it only scales the sum and is
            // absorbed into the viscosity coefficient c.
            // 1/rho_j is also dropped. Unlike m_j, rho_j varies per particle, so omitting it
            // changes the relative weighting of neighbors and is not trivially justified.
            // The assumption is that PBF keeps rho_j ≈ rho0 for all j (incompressibility),
            // making 1/rho_j approximately uniform. Under that assumption it too is absorbed
            // into c, and the formula reduces to what we compute here.
            xsphSum += (vj - vi) * Poly6(r, h);
        }
    }

    particles[i].velocity = vi + viscosity * xsphSum;
}

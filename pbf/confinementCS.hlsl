// Vorticity confinement pass (Macklin & Muller 2013):
//
// Reads the vorticity vectors omega_i stored in the omega field by vorticityCS,
// then computes a corrective force that pushes each particle toward regions of
// higher vorticity magnitude, counteracting numerical dissipation of swirling motion.
//
// Steps per particle i:
//   1. Compute eta_i, the SPH gradient of the vorticity magnitude field:
//        eta_i = sum_{j != i} |omega_j| * grad_W_spiky(r_ij, h)
//      eta_i points in the direction of increasing |omega| in the neighborhood of i.
//   2. Normalize to get the location vector N_i = eta_i / |eta_i|.
//   3. Compute confinement force: f_i = vorticityEpsilon * (N_i x omega_i)
//   4. Apply as a velocity correction: v_i += dt * f_i
//
// Per the paper's ordering, this pass uses the OLD positions (updatePositionCS has not run yet).
//
// Root signature:
//   CBV(b0)                  -- ComputeCb
//   DescriptorTable(UAV(u0)) -- particle buffer (read position + omega, write velocity)


#define ConfinementRootSig "CBV(b0), DescriptorTable(UAV(u0))"

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
};

RWStructuredBuffer<Particle> particles : register(u0);

[RootSignature(ConfinementRootSig)]
[numthreads(256, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    float3 pi = particles[i].position; // committed position from finalizeCS
    float3 omegaI = particles[i].omega; // vorticity vector written by vorticityCS

    float3 eta = float3(0, 0, 0); // accumulates the gradient of the vorticity magnitude field

    for (uint j = 0; j < numParticles; j++)
    {
        if (j != i)
        {
            // r points from neighbor j toward particle i
            float3 r = pi - particles[j].position;

            // |omega_j|: scalar vorticity magnitude of neighbor j, written by vorticityCS
            float omegaJLen = length(particles[j].omega);

            // gradient of the spiky kernel at r_ij, with respect to p_i
            float3 gradW = SpikyGrad(r, h);

            // Accumulate eta_i = sum_j |omega_j| * grad_{p_i} W(r_ij, h).
            // This is an SPH estimate of grad(|omega|) at p_i -- but note what is being omitted:
            // the full Monaghan 1992 SPH gradient formula is sum_j (m_j / rho_j) * f_j * grad W.
            // m_j = 1 is a uniform constant across all particles; omitting it scales eta uniformly,
            // and since we normalize eta to get N anyway, it cancels exactly. Fine.
            // rho_j is NOT a uniform constant -- it varies per particle. Omitting it changes
            // the direction of eta, not just its magnitude, so normalization does NOT save us
            // in the general case. The justification is that PBF actively drives rho_j toward rho0
            // (the incompressibility constraint). If the solver has converged, rho_j ≈ rho0 for all j,
            // making 1/rho_j approximately uniform, meaning we can drop it due to the coming normalization.
            eta += omegaJLen * gradW;
        }
    }

    float etaLen = length(eta);

    // if the vorticity gradient is negligible there is no meaningful direction
    // to apply the confinement force, so skip this particle
    if (etaLen < 1e-6)
        return;

    // N_i: unit vector pointing toward increasing vorticity magnitude
    float3 N = eta / etaLen;

    // confinement force: f_i = vorticityEpsilon * (N_i x omega_i)
    // the cross product produces a force perpendicular to both the vorticity axis
    // and the direction of increasing vorticity, which tightens the vortex
    float3 f = vorticityEpsilon * cross(N, omegaI);

    // apply as an impulse: v_i += dt * f_i
    particles[i].velocity += dt * f;
}

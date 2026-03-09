// PBF lambda calculation step (Macklin & Muller 2013):
//
// For each particle i, this shader:
// Estimates local density rho_i by summing Poly6 kernel contributions from all j:
// rho_i = sum_j(W_poly6(p_i - p_j, h))  (m = 1)
//
// Evaluates the density constraint: C_i = rho_i / rho0 - 1
//
// Computes lambda_i used by deltaCS for position corrections:
// lambda_i = -C_i / ( sum_k |grad_pk(C_i)|^2 + eps )
//
// The denominator sums the squared gradient of C_i with respect to every particle k.
// two cases depending on whether k == i or k == j:
//   k == i: grad_pi(C_i) = (1/rho0) * sum_{j != i}( grad_W_spiky(r_ij, h) )
//   k == j: grad_pj(C_i) = -(1/rho0) * grad_W_spiky(r_ij, h)
// eps prevents division by zero when a particle has no neighbors.
//
// Root signature:
//   CBV(b0)                  -- ComputeCb
//   DescriptorTable(UAV(u0)) -- particle buffer (read predictedPosition, write lambda)

#define LambdaRootSig "CBV(b0), DescriptorTable(UAV(u0))"

#include "Particle.hlsli" // Particle struct
#include "SphKernels.hlsli" // Poly6, SpikyGrad

cbuffer ComputeCb : register(b0)
{
    // Must match ComputeCb in ConstantBufferTypes.h -- same order, same offsets.
    float dt; // unused in this pass
    uint  numParticles; // loop bound and bounds check
    float h; // SPH smoothing radius
    float rho0; // rest density -- target of the constraint
    float epsilon; // relaxation factor for the lambda denominator
    float pad[3]; // padding to 32 bytes
};

RWStructuredBuffer<Particle> particles : register(u0);

[RootSignature(LambdaRootSig)]
[numthreads(256, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    // The lambda calculation involves three "indexes", i, j, and k, which aren't clearly explained
    // in the original research article:
    // i is the index of the particle we're computing lambda for (the "self" particle)
    // j is the index of a neighbor particle that contributes to i's density and constraint gradient
    // k is the index over a set of particles that includes the self particle and all the neighbors,
    // therefore it has two cases: k=i and k=j
    
    float3 pi = particles[i].predictedPosition; // cache to avoid repeated UAV reads

    float rho = 0.0; // density estimate rho_i
    float3 gradI = float3(0,0,0); // accumulates sum_{j != i}( grad_W(r_ij) ) for the k=i case
    float gradSqSum = 0.0; // accumulates sum_k( |grad_pk(C_i)|^2 )

    for (uint j = 0; j < numParticles; j++)
    {
        // r points from neighbor j toward particle i (r_ij = p_i - p_j)
        float3 r = pi - particles[j].predictedPosition;

        // Density: every particle j including i itself contributes.
        // Poly6 uses |r|^2 with no division, so r=0 (j==i) is safe and gives a nonzero value.
        rho += Poly6(r, h);

        if (j != i)
        {
            // k=j case: grad_pj(C_i) = -(1/rho0) * grad_W_spiky(r_ij, h)
            float3 gradW = SpikyGrad(r, h);
            float3 gradJ = -(1.0 / rho0) * gradW;
            gradSqSum += dot(gradJ, gradJ); // add |grad_pj(C_i)|^2 to denominator

            // Also accumulate gradW into gradI -- needed for the k=i term after the loop
            gradI += gradW;
        }
    }

    // k=i case
    // grad_pi(C_i) = (1/rho0) * sum_{j != i}( grad_W_spiky(r_ij, h) )
    // gradI now holds the raw sum; apply the 1/rho0 factor and add its squared magnitude.
    gradI /= rho0;
    gradSqSum += dot(gradI, gradI); // add |grad_pi(C_i)|^2 to denominator

    // Density constraint value
    float C = rho / rho0 - 1.0; // 0 at rest density, > 0 if compressed, < 0 if sparse

    // lambda_i = -C_i / ( sum_k(|grad_pk(C_i)|^2) + eps )
    particles[i].lambda = -C / (gradSqSum + epsilon);
}

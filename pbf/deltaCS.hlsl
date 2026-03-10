// PBF delta_p calculation step (Macklin & Muller 2013):
//
// For each particle i, compute a position correction:
// delta_p_i = (1/rho0) * sum_{j != i}( (lambda_i + lambda_j) * grad_W_spiky(r_ij, h) )
//
// Apply the correction to the predicted position:
// p*_i += delta_p_i
//
// Then clamp p*_i to the simulation box so particles don't escape the domain.
//
// Note on ordering: the algorithm as written in the paper implies two separate
// steps -- lambdaCS reads predictedPosition (written by predictCS or the previous deltaCS),
// then deltaCS reads that same predictedPosition to compute delta_p and writes a new
// predictedPosition. Within a single deltaCS dispatch however, thread i reads
// predictedPosition[j] while thread j may already have overwritten predictedPosition[j]
// in the same dispatch. A strictly correct Jacobi step would require double-buffering:
// deltaCS reads from one buffer and writes to a separate buffer, with no overlap.
//
// Root signature:
//   CBV(b0)                  -- ComputeCb
//   DescriptorTable(UAV(u0)) -- particle buffer (read predictedPosition + lambda, write predictedPosition)

#define DeltaRootSig "CBV(b0), DescriptorTable(UAV(u0))"

#include "Particle.hlsli" // Particle struct
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
};

RWStructuredBuffer<Particle> particles : register(u0);

[RootSignature(DeltaRootSig)]
[numthreads(256, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    float3 pi = particles[i].predictedPosition; // cache to avoid repeated UAV reads
    float lambdaI = particles[i].lambda;

    float3 deltaP = float3(0, 0, 0);

    for (uint j = 0; j < numParticles; j++)
    {
        if (j != i)
        {
            // r_ij points from neighbor j toward particle i
            float3 r = pi - particles[j].predictedPosition;

            // Eq. 12: each neighbor contributes (lambda_i + lambda_j) * grad_W_spiky(r_ij, h)
            // lambda_i + lambda_j: positive when both particles are compressed, producing a repulsive correction
            // grad_W_spiky points away from j toward i, so the correction pushes i away from compressed neighbors
            deltaP += (lambdaI + particles[j].lambda) * SpikyGrad(r, h);
        }
    }
    deltaP /= rho0;

    // Update the predicted position
    float3 newPos = pi + deltaP;

    // Clamp to the simulation box.
    // clamp() applies component-wise: newPos.x is clamped to [boxMin.x, boxMax.x], etc.
    newPos = clamp(newPos, boxMin, boxMax);

    particles[i].predictedPosition = newPos;
}

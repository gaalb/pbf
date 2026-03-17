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
// To avoid a Gauss-Seidel race (thread i reading predictedPosition[j] while thread j
// overwrites it in the same dispatch), this shader writes its result to the scratch field
// newPredictedPosition. A separate commitCS pass then copies the scratch field back to
// predictedPosition before the next solver iteration, ensuring all threads in the next
// iteration see a consistent snapshot.
//
// Root signature:
//   CBV(b0)                  -- ComputeCb
//   DescriptorTable(UAV(u0)) -- particle buffer (read predictedPosition + lambda, write newPredictedPosition)

#define DeltaRootSig "CBV(b0), DescriptorTable(UAV(u0))"

#include "Particle.hlsli" // Particle struct
#include "SphKernels.hlsli" // SpikyGrad, Poly6

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

[RootSignature(DeltaRootSig)]
[numthreads(256, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    // cache to avoid repeated UAV reads
    float3 pi = particles[i].predictedPosition; 
    float lambdaI = particles[i].lambda;

    // Precompute the s_corr denominator, same for every (i,j) pair.
    // Poly6 only uses the squared magnitude so the direction does not matter.
    float poly6AtDeltaQ = Poly6(float3(sCorrDeltaQ, 0, 0), h);

    float3 deltaP = float3(0, 0, 0);

    for (uint j = 0; j < numParticles; j++)
    {
        if (j != i)
        {
            // r_ij points from neighbor j toward particle i
            float3 r = pi - particles[j].predictedPosition;

            // Eq. 13: artificial pressure term s_corr to suppress tensile instability.
            // When lambda > 0 (sparse region), the standard Eq. 12 correction becomes attractive,
            // pulling surface particles into tight clumps. s_corr adds a small repulsive bias
            // that counteracts this without disturbing the bulk behavior.
            // s_corr = -k * (W(r, h) / W(delta_q, h))^n
            float wRatio = Poly6(r, h) / poly6AtDeltaQ;
            float sCorr = -sCorrK * pow(wRatio, sCorrN); // sCorrK > 0, pow >= 0, so sCorr <= 0 always

            // Eq. 12 + 13: position correction with artificial pressure included..
            // SpikyGrad with r = p_i - p_j points from i toward j.
            // A negative coefficient times that direction pushes i away from j -- repulsive.
            // So sCorr is a repulsive contribution. The "surface tension-like effect" is because
            // sCorr keeps the bulk density slightly below rho0, so even bulk particles
            // end up with a weakly positive lambda (attractive). At the surface, where
            // particle counts are low and density is even lower, this mild attraction
            // creates a coherent surface instead of the violent clumping (tensile instability)
            // that occurs without s_corr. Raising sCorrK increases this surface cohesion.
            deltaP += (lambdaI + particles[j].lambda + sCorr) * SpikyGrad(r, h);
        }
    }
    deltaP /= rho0;

    // Update the predicted position
    float3 newPos = pi + deltaP;

    // Perform collision check, for now this is just a simple clamp to keep particles inside the simulation box.
    // clamp() applies component-wise: newPos.x is clamped to [boxMin.x, boxMax.x], etc.
    newPos = clamp(newPos, boxMin, boxMax);

    particles[i].newPredictedPosition = newPos;
}

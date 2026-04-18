// Combines vorticity confinement and XSPH viscosity into a single neighbor pass.
//
// Both passes iterate over the same neighbor set, so merging them halves the number
// of neighbor traversals (and therefore cache misses) compared to running them separately.
//
// Confinement is applied first (per-particle, after the loop), then the combined
// velocity correction is written out.
//
// Confinement steps per particle i:
//   1. eta_i = sum_{j != i} |omega_j| * grad_W_spiky(r_ij, h)   (gradient of vorticity magnitude)
//   2. N_i   = eta_i / |eta_i|                                    (unit direction toward peak vorticity)
//   3. f_i   = vorticityEpsilon * (N_i x omega_i)                 (confinement force)
//   4. v_i  += dt * f_i
//
// Viscosity (XSPH, Schechter & Bridson 2012) per particle i:
//   v_new_i = v_i + c * sum_{j != i} (v_j - v_i) * W_poly6(r_ij, h)
//
// The viscosity term uses v_i as captured before the confinement correction (since we
// cannot know f_i until after the loop). The confinement delta is added on top before
// the final write, so the output encodes both corrections.
//
// In: position, velocity, omega, cellCount, cellPrefixSum
// Out: scratch (new velocity, Jacobi mode) or velocity (Gauss-Seidel mode)

#define ConfinementViscosityRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 7), UAV(u7, numDescriptors = 2))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"
#include "SphKernels.hlsli" // SpikyGrad, Poly6
#include "GridUtils.hlsli"  // posToCell(), cellIndex(), NeighborCellIndices()

RWStructuredBuffer<float3> position     : register(u0);
RWStructuredBuffer<float3> velocity     : register(u1);
RWStructuredBuffer<float3> omega        : register(u5);
RWStructuredBuffer<float3> scratch      : register(u6);
RWStructuredBuffer<uint>   cellCount    : register(u7);
RWStructuredBuffer<uint>   cellPrefixSum: register(u8);

[RootSignature(ConfinementViscosityRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    float3 pi     = position[i];
    float3 vi     = velocity[i];
    float3 omegaI = omega[i];

    float3 eta     = float3(0, 0, 0); // accumulates grad(|omega|) for confinement
    float3 xsphSum = float3(0, 0, 0); // accumulates weighted velocity differences for viscosity

    // The grid was built from predictedPositions, but this pass uses old positions.
    // See vorticityCS for the justification of this approximation.
    NeighborCells nCells = NeighborCellIndices(pi);
    for (uint c = 0; c < nCells.count; c++)
    {
        uint ci    = nCells.indices[c];
        uint count = cellCount[ci];

        for (uint s = 0; s < count; s++)
        {
            uint j = cellPrefixSum[ci] + s;
            if (j == i)
                continue;

            float3 r  = pi - position[j];
            float  r2 = dot(r, r);

            // Confinement: accumulate eta_i = sum_j |omega_j| * grad_W_spiky(r_ij)
            eta += length(omega[j]) * SpikyGrad(r, r2);

            // Viscosity: accumulate XSPH sum = sum_j (v_j - v_i) * W_poly6(r_ij)
            xsphSum += (velocity[j] - vi) * Poly6(r, r2);
        }
    }

    // --- Confinement (applied first) ---
    float etaLen = length(eta);
    if (etaLen >= 1e-6)
    {
        float3 N = eta / etaLen; // unit direction toward increasing vorticity magnitude
        vi += dt * vorticityEpsilon * cross(N, omegaI);
    }

    // --- Viscosity: write combined result ---
    // vi now includes the confinement impulse, so the output encodes both corrections.
    scratch[i] = vi + viscosity * xsphSum;
}

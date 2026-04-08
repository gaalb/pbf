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
// In: position, omega, cellCount, cellPrefixSum, velocity
// Out: velocity

#define ConfinementRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 7)), DescriptorTable(UAV(u7, numDescriptors = 2))"

#include "ComputeCb.hlsli"

#include "SphKernels.hlsli" // SpikyGrad

#include "GridUtils.hlsli" // posToCell(), cellIndex(), gridDim()

RWStructuredBuffer<float3> position : register(u0);
RWStructuredBuffer<float3> velocity : register(u1);
RWStructuredBuffer<float3> omega : register(u5);
RWStructuredBuffer<uint> cellCount : register(u7);
RWStructuredBuffer<uint> cellPrefixSum : register(u8);

[RootSignature(ConfinementRootSig)]
[numthreads(256, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    float3 pi = position[i]; // committed position from finalizeCS
    float3 omegaI = omega[i]; // vorticity vector written by vorticityCS

    float3 eta = float3(0, 0, 0); // accumulates the gradient of the vorticity magnitude field

    // The grid was built from predictedPositions, but this pass uses old positions.
    // See vorticityCS for the justification of this approximation.
    NeighborCells nCells = NeighborCellIndices(pi);
    for (uint c = 0; c < nCells.count; c++)
    {
        uint ci = nCells.indices[c];
        uint count = cellCount[ci];

        for (uint s = 0; s < count; s++)
        {
            uint j = cellPrefixSum[ci] + s;
            if (j == i)
                continue;

            // r points from neighbor j toward particle i
            float3 r = pi - position[j];

            // |omega_j|: scalar vorticity magnitude of neighbor j, written by vorticityCS
            float omegaJLen = length(omega[j]);

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
    velocity[i] += dt * f;
}

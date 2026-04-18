// Two-phase broadcast-GSM variant of lambdaCS.
// Applies the same broadcast pattern as GSM_confinementViscosityCS:
//
// Phase 1 — GSM broadcast sweep (k uniform across all threads → broadcast, no bank conflicts):
//   Iterate k = 0..THREAD_GROUP_SIZE-1. All threads read gs_predPos[k] at the same k each step.
//   The self slot (k == localIdx) contributes to rho (self-density) but is skipped for the gradient.
//   OOB slots are initialised to float3(1e10) so r2 >> H*H and both kernels return 0.
//
// Phase 2 — residual VRAM pass:
//   Full 27-cell neighbour traversal, skipping j in [groupStart, groupStart+THREAD_GROUP_SIZE).
//   i itself lies in that range, so the j==i self-skip is implicit.
//
// In: predictedPosition, cellCount, cellPrefixSum, lod
// Out: lambda, density

#define LambdaRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 7), UAV(u7, numDescriptors = 2), UAV(u9, numDescriptors = 1))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"
#include "SphKernels.hlsli"
#include "GridUtils.hlsli"

RWStructuredBuffer<float3> predictedPosition : register(u2);
RWStructuredBuffer<float>  lambda            : register(u3);
RWStructuredBuffer<float>  density           : register(u4);
RWStructuredBuffer<uint>   cellCount         : register(u7);
RWStructuredBuffer<uint>   cellPrefixSum     : register(u8);
RWStructuredBuffer<uint>   lod               : register(u9);

groupshared float3 gs_predPos[THREAD_GROUP_SIZE];

[RootSignature(LambdaRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(
    uint3 dispatchID : SV_DispatchThreadID,
    uint  localIdx   : SV_GroupIndex,
    uint3 groupID    : SV_GroupID)
{
    uint i          = dispatchID.x;
    uint groupStart = groupID.x * THREAD_GROUP_SIZE;

    // Preload: OOB slots get a sentinel far outside the grid so kernels return 0 for them.
    gs_predPos[localIdx] = (i < numParticles) ? predictedPosition[i] : float3(1e10f, 1e10f, 1e10f);
    GroupMemoryBarrierWithGroupSync();

    if (i >= numParticles)
        return;
    if (lod[i] == 0)
        return;

    float3 pi = gs_predPos[localIdx];

    float  rho        = 0.0;
    float3 gradI      = float3(0, 0, 0);
    float  gradSqSum  = 0.0;

    // ---- Phase 1: broadcast sweep over GSM ----
    for (uint k = 0; k < THREAD_GROUP_SIZE; k++)
    {
        float3 pj = gs_predPos[k]; // broadcast: same k for all threads
        float3 r  = pi - pj;
        float  r2 = dot(r, r);

        rho += Poly6(r, r2); // includes self (k == localIdx, r == 0 -> max Poly6 value)

        if (k == localIdx)   // skip gradient contribution for self
            continue;

        float3 gradW = SpikyGrad(r, r2);
        float3 gradJ = -(1.0 / RHO0) * gradW;
        gradSqSum += dot(gradJ, gradJ);
        gradI     += gradW;
    }

    // ---- Phase 2: residual VRAM pass ----
    // Skip j in [groupStart, groupStart+THREAD_GROUP_SIZE): covered by Phase 1.
    // i is in that range, so j==i is implicitly excluded.
    NeighborCells nCells = NeighborCellIndices(pi);
    for (uint c = 0; c < nCells.count; c++)
    {
        uint ci    = nCells.indices[c];
        uint count = cellCount[ci];

        for (uint s = 0; s < count; s++)
        {
            uint j = cellPrefixSum[ci] + s;

            if (j >= groupStart && j < groupStart + THREAD_GROUP_SIZE)
                continue;

            float3 pj = predictedPosition[j];
            float3 r  = pi - pj;
            float  r2 = dot(r, r);

            rho += Poly6(r, r2); // j != i guaranteed

            float3 gradW = SpikyGrad(r, r2);
            float3 gradJ = -(1.0 / RHO0) * gradW;
            gradSqSum += dot(gradJ, gradJ);
            gradI     += gradW;
        }
    }

    // k=i gradient term
    gradI    /= RHO0;
    gradSqSum += dot(gradI, gradI);

    float C     = rho / RHO0 - 1.0;
    density[i]  = rho;
    lambda[i]   = -C / (gradSqSum + epsilon);
}

// Two-phase broadcast-GSM variant of deltaCS.
// Applies the same broadcast pattern as GSM_confinementViscosityCS:
//
// Phase 1 — GSM broadcast sweep:
//   All threads read gs_predPos[k] and gs_lambda[k] at the same k each step -> broadcast.
//   Self (k == localIdx) is skipped. OOB slots get float3(1e10) so r2 >> H*H and
//   SpikyGrad returns 0; overlap-jitter is not triggered because r2 >> EPSILON^2.
//
// Phase 2 — residual VRAM pass:
//   Full 27-cell traversal, skipping j in [groupStart, groupStart+THREAD_GROUP_SIZE).
//   i lies in that range, so j==i is implicitly excluded.
//
// In: predictedPosition, lambda, cellCount, cellPrefixSum, lod
// Out: scratch (new predicted position, Jacobi)

#define DeltaRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 6))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"
#include "SphKernels.hlsli"
#include "GridUtils.hlsli"

RWStructuredBuffer<float3> predictedPosition : register(u0);
RWStructuredBuffer<float>  lambda            : register(u1);
RWStructuredBuffer<float3> scratch           : register(u2);
RWStructuredBuffer<uint>   cellCount         : register(u3);
RWStructuredBuffer<uint>   cellPrefixSum     : register(u4);
RWStructuredBuffer<uint>   lod               : register(u5);

groupshared float3 gs_predPos[THREAD_GROUP_SIZE];
groupshared float  gs_lambda[THREAD_GROUP_SIZE];

[RootSignature(DeltaRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(
    uint3 dispatchID : SV_DispatchThreadID,
    uint  localIdx   : SV_GroupIndex,
    uint3 groupID    : SV_GroupID)
{
    uint i          = dispatchID.x;
    uint groupStart = groupID.x * THREAD_GROUP_SIZE;

    // OOB slots get a sentinel position so Phase 1 produces r2 >> H*H -> SpikyGrad = 0.
    gs_predPos[localIdx] = (i < numParticles) ? predictedPosition[i] : float3(1e10f, 1e10f, 1e10f);
    gs_lambda[localIdx]  = (i < numParticles) ? lambda[i]            : 0.0;
    GroupMemoryBarrierWithGroupSync();

    if (i >= numParticles)
        return;
    if (lod[i] == 0)
        return;

    float3 pi      = gs_predPos[localIdx];
    float  lambdaI = gs_lambda[localIdx];

    float poly6AtDeltaQ = Poly6(float3(SCORR_DELTA_Q, 0, 0), SCORR_DELTA_Q * SCORR_DELTA_Q);
    float3 deltaP = float3(0, 0, 0);

    // Phase 1: broadcast sweep over GSM
    for (uint k = 0; k < THREAD_GROUP_SIZE; k++)
    {
        if (k == localIdx) // skip self
            continue;

        float3 pj      = gs_predPos[k]; // broadcast
        float3 r       = pi - pj;
        float  r2      = dot(r, r);

        // Overlapping particles: nudge apart. j = groupStart + k for the global index.
        if (r2 < EPSILON * EPSILON)
        {
            deltaP += overlapJitter(i, groupStart + k) * (H * 0.001);
            continue;
        }

        float  lambdaJ = gs_lambda[k]; // broadcast
        float  wRatio  = Poly6(r, r2) / poly6AtDeltaQ;
        float  sCorr   = -sCorrK * pow(wRatio, SCORR_N);
        deltaP += (lambdaI + lambdaJ + sCorr) * SpikyGrad(r, r2);
    }

    // Phase 2: residual VRAM pass
    NeighborCells nCells = NeighborCellIndices(pi);
    for (uint c = 0; c < nCells.count; c++)
    {
        uint ci    = nCells.indices[c];
        uint count = cellCount[ci];

        for (uint s = 0; s < count; s++)
        {
            uint j = cellPrefixSum[ci] + s;

            if (j >= groupStart && j < groupStart + THREAD_GROUP_SIZE)
                continue; // handled in Phase 1

            float3 pj = predictedPosition[j];
            float3 r  = pi - pj;
            float  r2 = dot(r, r);

            if (r2 < EPSILON * EPSILON)
            {
                deltaP += overlapJitter(i, j) * (H * 0.001);
                continue;
            }

            float  wRatio  = Poly6(r, r2) / poly6AtDeltaQ;
            float  sCorr   = -sCorrK * pow(wRatio, SCORR_N);
            float  lambdaJ = lambda[j];
            deltaP += (lambdaI + lambdaJ + sCorr) * SpikyGrad(r, r2);
        }
    }

    deltaP /= RHO0;
    scratch[i] = pi + deltaP;
}

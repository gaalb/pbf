// Two-phase broadcast-GSM variant of vorticityCS.
// Applies the same broadcast pattern as GSM_confinementViscosityCS:
//
// Phase 1 — GSM broadcast sweep:
//   All threads step through k = 0..THREAD_GROUP_SIZE-1 in lockstep.
//   gs_position[k] and gs_velocity[k] are served as broadcasts (zero bank conflicts).
//   When k == localIdx (self), r = 0 and SpikyGrad returns 0, so the cross product
//   is zero — no explicit skip needed, Phase 1 remains fully uniform.
//   OOB slots are initialised to float3(1e10) so r2 >> H*H and SpikyGrad returns 0.
//
// Phase 2 — residual VRAM pass:
//   Full 27-cell traversal, skipping j in [groupStart, groupStart+THREAD_GROUP_SIZE).
//   i lies in that range, so j==i is implicitly excluded.
//
// In: position, velocity, cellCount, cellPrefixSum
// Out: omega

#define VorticityRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 5))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"
#include "SphKernels.hlsli"
#include "GridUtils.hlsli"

RWStructuredBuffer<float3> position      : register(u0);
RWStructuredBuffer<float3> velocity      : register(u1);
RWStructuredBuffer<float3> omega         : register(u2);
RWStructuredBuffer<uint>   cellCount     : register(u3);
RWStructuredBuffer<uint>   cellPrefixSum : register(u4);

groupshared float3 gs_position[THREAD_GROUP_SIZE];
groupshared float3 gs_velocity[THREAD_GROUP_SIZE];

[RootSignature(VorticityRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(
    uint3 dispatchID : SV_DispatchThreadID,
    uint  localIdx   : SV_GroupIndex,
    uint3 groupID    : SV_GroupID)
{
    uint i          = dispatchID.x;
    uint groupStart = groupID.x * THREAD_GROUP_SIZE;

    gs_position[localIdx] = (i < numParticles) ? position[i] : float3(1e10f, 1e10f, 1e10f);
    gs_velocity[localIdx] = (i < numParticles) ? velocity[i] : (float3)0;
    GroupMemoryBarrierWithGroupSync();

    if (i >= numParticles)
        return;

    float3 pi = gs_position[localIdx];
    float3 vi = gs_velocity[localIdx];

    float3 omegaAccum = float3(0, 0, 0);

    // Phase 1: broadcast sweep over GSM
    // k is uniform across all threads -> gs_X[k] served as broadcast.
    // Self slot (k == localIdx): r = 0, SpikyGrad = 0, contribution is zero. No skip needed.
    for (uint k = 0; k < THREAD_GROUP_SIZE; k++)
    {
        float3 pj = gs_position[k]; // broadcast
        float3 r  = pi - pj;
        float  r2 = dot(r, r);
        omegaAccum += cross(gs_velocity[k] - vi, -SpikyGrad(r, r2)); // broadcast
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

            float3 r  = pi - position[j];
            float  r2 = dot(r, r);
            omegaAccum += cross(velocity[j] - vi, -SpikyGrad(r, r2));
        }
    }

    omega[i] = omegaAccum;
}

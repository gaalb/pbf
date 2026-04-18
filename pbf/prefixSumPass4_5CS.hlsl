// Parallel exclusive prefix sum — pass 4 of 5 (and pass 5 of 5).
//
// Propagates higher-level exclusive offsets into the lower-level local prefix sums produced
// by a pass-1-style local scan. Dispatched twice per frame with different descriptor bindings:
//
//   Pass 4 (M groups): adds superGroupSums offsets into groupPrefixSum
//     u0 = groupPrefixSum, u1 = superGroupSums
//
//   Pass 5 (N groups): adds groupPrefixSum offsets into cellPrefixSum
//     u0 = cellPrefixSum,  u1 = groupPrefixSum
//
// Each thread adds the global exclusive offset for its group to two consecutive elements.
//
// Dispatch: M or N groups of THREAD_GROUP_SIZE threads.

#define PrefixSumPass4RootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 2))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"

#define ELEMENTS_PER_GROUP (THREAD_GROUP_SIZE * 2)

RWStructuredBuffer<uint> localSums    : register(u0);
RWStructuredBuffer<uint> groupOffsets : register(u1);

[RootSignature(PrefixSumPass4RootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 groupID  : SV_GroupID,
          uint3 threadID : SV_GroupThreadID)
{
    uint tid     = threadID.x;
    uint baseIdx = groupID.x * ELEMENTS_PER_GROUP;
    uint offset  = groupOffsets[groupID.x];

    localSums[baseIdx + 2 * tid]     += offset;
    localSums[baseIdx + 2 * tid + 1] += offset;
}

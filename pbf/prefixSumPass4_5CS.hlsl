// Pass 4 and 5 of the 5-pass parallel prefix sum are very simple: each thread 
// just adds the global offset for its group, which was calculated by the previous pass, 
// into its local prefix sums. 
// 
// Pass 4 (M groups): adds superGroupSums offsets into groupPrefixSum
//  u0 = groupPrefixSum, u1 = superGroupSums
//
// Pass 5 (N groups): adds groupPrefixSum offsets into cellPrefixSum
//  u0 = cellPrefixSum,  u1 = groupPrefixSum
//

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

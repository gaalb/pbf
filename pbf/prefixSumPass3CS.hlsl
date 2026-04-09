// Parallel exclusive prefix sum (Blelloch scan) — pass 3 of 3.
//
// Adds the global cross-group offset to every cellPrefixSum entry in a group.
// After pass 1, cellPrefixSum holds intra-group exclusive prefix sums (correct
// relative offsets within each block of 512 cells, but no knowledge of how many
// particles came before that block globally).
// After pass 2, groupSums[g] holds the exclusive global offset for group g.
// This pass simply adds groupSums[groupID.x] to every element in the group,
// producing the final correct global exclusive prefix sum in cellPrefixSum.
//
// In:     groupSums[u9]     — global exclusive offsets per group (from pass 2)
// In/Out: cellPrefixSum[u8] — updated in-place to global exclusive prefix sums
//
// Dispatch: same as pass 1 (numCells / ELEMENTS_PER_GROUP groups).
// For gridDim=32: 64 groups of 256 threads.

#define PrefixSumPass3RootSig \
    "CBV(b0), " \
    "DescriptorTable(UAV(u8, numDescriptors = 1)), " \
    "DescriptorTable(UAV(u9, numDescriptors = 1))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"

#define ELEMENTS_PER_GROUP (THREAD_GROUP_SIZE * 2)

RWStructuredBuffer<uint> cellPrefixSum : register(u8);
RWStructuredBuffer<uint> groupSums     : register(u9);

[RootSignature(PrefixSumPass3RootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 groupID  : SV_GroupID,
          uint3 threadID : SV_GroupThreadID)
{
    uint tid     = threadID.x;
    uint baseIdx = groupID.x * ELEMENTS_PER_GROUP;
    uint offset  = groupSums[groupID.x]; // global exclusive offset for this group (0 for group 0)

    cellPrefixSum[baseIdx + 2 * tid]     += offset;
    cellPrefixSum[baseIdx + 2 * tid + 1] += offset;
}

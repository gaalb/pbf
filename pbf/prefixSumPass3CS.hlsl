// Parallel exclusive prefix sum — pass 3 of 5: single-group scan of superGroupSums.
//
// After pass 2, superGroupSums[g] holds the total particle count for pass-2 group g.
// This pass does an in-place exclusive Blelloch scan so that superGroupSums[g] becomes
// the exclusive global offset for pass-2 group g, ready for pass 4 to propagate down.
//
// PASS3_NUM_ELEMENTS = max(PASS2_GROUPS, 2), so that PASS3_THREAD_COUNT >= 1 always holds:
//   GRID_DIM=64:  PASS2_GROUPS=1 -> PASS3_NUM_ELEMENTS=2, PASS3_THREAD_COUNT=1
//   GRID_DIM=128: PASS2_GROUPS=8 -> PASS3_NUM_ELEMENTS=8, PASS3_THREAD_COUNT=4
//
// When PASS2_GROUPS=1 the second element (index 1) of superGroupSums is allocated but
// never written by pass 2 and never read by pass 4, so its value does not affect correctness.
//
// In/Out: superGroupSums[u0]
// Dispatch: 1 group of PASS3_THREAD_COUNT threads.

#define PrefixSumPass3RootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 1))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"

#define PASS2_GROUPS (GRID_DIM * GRID_DIM * GRID_DIM / (THREAD_GROUP_SIZE * 2) / (THREAD_GROUP_SIZE * 2))
#if PASS2_GROUPS < 2
#define PASS3_NUM_ELEMENTS 2
#else
#define PASS3_NUM_ELEMENTS PASS2_GROUPS
#endif
#define PASS3_THREAD_COUNT (PASS3_NUM_ELEMENTS / 2)

RWStructuredBuffer<uint> superGroupSums : register(u0);

groupshared uint s[PASS3_NUM_ELEMENTS];

[RootSignature(PrefixSumPass3RootSig)]
[numthreads(PASS3_THREAD_COUNT, 1, 1)]
void main(uint3 threadID : SV_GroupThreadID)
{
    uint tid = threadID.x;

    s[2 * tid]     = superGroupSums[2 * tid];
    s[2 * tid + 1] = superGroupSums[2 * tid + 1];
    GroupMemoryBarrierWithGroupSync();

    // Up-sweep
    for (uint stride = 1; stride < PASS3_NUM_ELEMENTS; stride <<= 1)
    {
        uint i = (tid + 1) * (stride << 1) - 1;
        if (i < PASS3_NUM_ELEMENTS)
            s[i] += s[i - stride];
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0)
        s[PASS3_NUM_ELEMENTS - 1] = 0;
    GroupMemoryBarrierWithGroupSync();

    // Down-sweep
    for (uint stride = PASS3_NUM_ELEMENTS >> 1; stride >= 1; stride >>= 1)
    {
        uint i = (tid + 1) * (stride << 1) - 1;
        if (i < PASS3_NUM_ELEMENTS)
        {
            uint left     = s[i - stride];
            s[i - stride] = s[i];
            s[i]         += left;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    superGroupSums[2 * tid]     = s[2 * tid];
    superGroupSums[2 * tid + 1] = s[2 * tid + 1];
}

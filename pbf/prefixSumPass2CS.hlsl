// SUPERSEDED. The 5-pass uniform design (pass1-5CS) replaces the old 3-pass design.
// Pass 2 is now served by prefixSumPass1CS.cso with groupSums/groupPrefixSum/superGroupSums bindings.
// Pass 3 (middle single-group scan) is now prefixSumPass3CS.hlsl.
// This file is kept for reference only but not compiled
//
// Parallel exclusive prefix sum (Blelloch scan) — pass 2 of 3 (old 3-pass design).
//
// Performs an in-place exclusive Blelloch scan on the groupSums[] array that pass 1
// produced.  After this pass, groupSums[g] holds the exclusive global offset for group g:
//   groupSums[0] = 0
//   groupSums[1] = total of group 0
//   groupSums[2] = total of group 0 + total of group 1
//   ...
// Pass 3 will add groupSums[g] to every cellPrefixSum entry belonging to group g,
// turning the intra-group sums into global exclusive prefix sums.
//
// Configuration (must match pass 1 and PbfApp.h):
//   PASS2_NUM_ELEMENTS = numCells / (THREAD_GROUP_SIZE * 2) = 32768 / 512 = 64
//   PASS2_THREAD_COUNT = PASS2_NUM_ELEMENTS / 2             = 32  (2 elements per thread)
//
// In/Out: groupSums[u9]
//
// Dispatch: 1 group of PASS2_THREAD_COUNT threads.

#define PrefixSumPass2RootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 1))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"

// Number of groups produced by pass 1: numCells / (2 * THREAD_GROUP_SIZE).
// For GRID_DIM > 64, numthreads would exceed 1024 — use prefixSumMiddleCS instead.
// This shader stubs out for GRID_DIM > 64 so the project still compiles; it is never dispatched.
#if GRID_DIM <= 64
#define PASS2_NUM_ELEMENTS (GRID_DIM * GRID_DIM * GRID_DIM / (THREAD_GROUP_SIZE * 2))
#define PASS2_THREAD_COUNT (PASS2_NUM_ELEMENTS / 2)
#else
#define PASS2_NUM_ELEMENTS 2
#define PASS2_THREAD_COUNT 1
#endif

RWStructuredBuffer<uint> groupSums : register(u0);

groupshared uint s[PASS2_NUM_ELEMENTS];

[RootSignature(PrefixSumPass2RootSig)]
[numthreads(PASS2_THREAD_COUNT, 1, 1)]
void main(uint3 threadID : SV_GroupThreadID)
{
    uint tid = threadID.x;   // 0 .. PASS2_THREAD_COUNT-1

    // Load the group totals from pass 1.
    s[2 * tid]     = groupSums[2 * tid];
    s[2 * tid + 1] = groupSums[2 * tid + 1];
    GroupMemoryBarrierWithGroupSync();

    // Up-sweep
    for (uint stride = 1; stride < PASS2_NUM_ELEMENTS; stride <<= 1)
    {
        uint i = (tid + 1) * (stride << 1) - 1;
        if (i < PASS2_NUM_ELEMENTS)
            s[i] += s[i - stride];
        GroupMemoryBarrierWithGroupSync();
    }

    // Exclusive scan: zero the last slot before the down-sweep.
    if (tid == 0)
        s[PASS2_NUM_ELEMENTS - 1] = 0;
    GroupMemoryBarrierWithGroupSync();

    // Down-sweep
    for (uint stride = PASS2_NUM_ELEMENTS >> 1; stride >= 1; stride >>= 1)
    {
        uint i = (tid + 1) * (stride << 1) - 1;
        if (i < PASS2_NUM_ELEMENTS)
        {
            uint left     = s[i - stride];
            s[i - stride] = s[i];
            s[i]         += left;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Write the global exclusive offsets back to groupSums[].
    groupSums[2 * tid] = s[2 * tid];
    groupSums[2 * tid + 1] = s[2 * tid + 1];
}

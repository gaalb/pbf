// Parallel exclusive prefix sum (Blelloch scan) — pass 1 of 3.
//
// Replaces the single-threaded prefixSumCS.hlsl with a GPU-parallel implementation.
// The Blelloch (work-efficient) algorithm works in two phases on a shared-memory array:
//
//   Up-sweep (reduce): builds partial sums bottom-up in log2(N) steps.
//     After the last step, s[N-1] holds the total sum of the entire block.
//
//   Down-sweep: propagates prefix sums top-down in log2(N) steps.
//     Before starting, s[N-1] is set to 0 (this is what makes it exclusive).
//
// Each thread group of THREAD_GROUP_SIZE (256) threads processes ELEMENTS_PER_GROUP (512)
// consecutive cellCount entries: each thread is responsible for two adjacent elements.
// The local exclusive prefix sum is written to cellPrefixSum[].  Cross-group offsets
// are NOT added here; that is done in pass 3 after pass 2 has scanned the group totals.
//
// The per-group total (s[ELEMENTS_PER_GROUP-1] before zeroing) is stored in groupSums[groupID.x]
// so that pass 2 can scan those 64 totals into global exclusive offsets.
//
// Algorithm correctness is verified by the worked example in GridUtils.hlsli comments and the
// standard Blelloch parallel prefix sum proof (Blelloch 1990, "Prefix Sums and Their Applications").
//
// In:  cellCount[u7]      — particle count per cell, written by countGridCS
// Out: cellPrefixSum[u8]  — intra-group exclusive prefix sum (global offsets added in pass 3)
//      groupSums[u9]      — total particle count per group, passed to pass 2
//
// Dispatch: numCells / ELEMENTS_PER_GROUP groups.
// For gridDim=32: 32768 / 512 = 64 groups of 256 threads.

#define PrefixSumPass1RootSig \
    "CBV(b0), " \
    "DescriptorTable(UAV(u7, numDescriptors = 2)), " \
    "DescriptorTable(UAV(u9, numDescriptors = 1))"

#include "SharedConfig.hlsli"   // THREAD_GROUP_SIZE = 256
#include "ComputeCb.hlsli"

// Each group processes 2 * THREAD_GROUP_SIZE = 512 elements (2 per thread).
#define ELEMENTS_PER_GROUP (THREAD_GROUP_SIZE * 2)

RWStructuredBuffer<uint> cellCount     : register(u7);
RWStructuredBuffer<uint> cellPrefixSum : register(u8);
RWStructuredBuffer<uint> groupSums     : register(u9);

groupshared uint s[ELEMENTS_PER_GROUP];

[RootSignature(PrefixSumPass1RootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 groupID  : SV_GroupID,
          uint3 threadID : SV_GroupThreadID)
{
    uint tid     = threadID.x;                          // 0 .. THREAD_GROUP_SIZE-1
    uint baseIdx = groupID.x * ELEMENTS_PER_GROUP;      // first global element index for this group

    // Load two consecutive cellCount entries per thread into shared memory.
    s[2 * tid]     = cellCount[baseIdx + 2 * tid];
    s[2 * tid + 1] = cellCount[baseIdx + 2 * tid + 1];
    GroupMemoryBarrierWithGroupSync();

    // -------------------------------------------------------------------------
    // Up-sweep (reduce) phase.
    // Each step doubles the active stride. Thread tid updates element i = (tid+1)*2*stride - 1
    // by adding the element stride positions to its left.  As stride grows, fewer threads
    // participate (those with i >= ELEMENTS_PER_GROUP become idle via the branch below).
    // After log2(ELEMENTS_PER_GROUP) = 9 steps, s[ELEMENTS_PER_GROUP-1] == total sum.
    // -------------------------------------------------------------------------
    for (uint stride = 1; stride < ELEMENTS_PER_GROUP; stride <<= 1)
    {
        uint i = (tid + 1) * (stride << 1) - 1;
        if (i < ELEMENTS_PER_GROUP)
            s[i] += s[i - stride];
        GroupMemoryBarrierWithGroupSync();
    }

    // Thread 0 records the group total, then zeroes the last slot so the
    // down-sweep produces an exclusive (not inclusive) prefix sum.
    if (tid == 0)
    {
        groupSums[groupID.x]      = s[ELEMENTS_PER_GROUP - 1];
        s[ELEMENTS_PER_GROUP - 1] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    // -------------------------------------------------------------------------
    // Down-sweep phase.
    // Each step halves the active stride.  Thread tid performs a "butterfly":
    //   left  = s[i - stride]    (save left child)
    //   s[i - stride] = s[i]     (left child becomes parent's current value)
    //   s[i]          = s[i] + left  (right child becomes parent + old left)
    // After log2(ELEMENTS_PER_GROUP) = 9 steps the array holds the exclusive
    // prefix sum: s[k] = sum of all input elements before index k.
    // -------------------------------------------------------------------------
    for (uint stride = ELEMENTS_PER_GROUP >> 1; stride >= 1; stride >>= 1)
    {
        uint i = (tid + 1) * (stride << 1) - 1;
        if (i < ELEMENTS_PER_GROUP)
        {
            uint left     = s[i - stride];
            s[i - stride] = s[i];
            s[i]         += left;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Write the intra-group exclusive prefix sums.
    // Global offsets (from pass 2) are added in pass 3.
    cellPrefixSum[baseIdx + 2 * tid]     = s[2 * tid];
    cellPrefixSum[baseIdx + 2 * tid + 1] = s[2 * tid + 1];
}

// Parallel exclusive prefix sum (Blelloch scan) — pass 1 of 3.
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
// In:  cellCount[u7]      — particle count per cell, written by countGridCS
// Out: cellPrefixSum[u8]  — intra-group exclusive prefix sum (global offsets added in pass 3)
//      groupSums[u9]      — total particle count per group, passed to pass 2
//
// Dispatch: numCells / ELEMENTS_PER_GROUP groups.
// For gridDim=32: 32768 / 512 = 64 groups of 256 threads.

#define PrefixSumPass1RootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 3))"

#include "SharedConfig.hlsli"   // THREAD_GROUP_SIZE = 256
#include "ComputeCb.hlsli"

// Each group processes 2 * THREAD_GROUP_SIZE = 512 elements (2 per thread).
#define ELEMENTS_PER_GROUP (THREAD_GROUP_SIZE * 2)

RWStructuredBuffer<uint> cellCount     : register(u0);
RWStructuredBuffer<uint> cellPrefixSum : register(u1);
RWStructuredBuffer<uint> groupSums     : register(u2);

groupshared uint s[ELEMENTS_PER_GROUP];

[RootSignature(PrefixSumPass1RootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 groupID  : SV_GroupID,
          uint3 threadID : SV_GroupThreadID)
{
    uint tid = threadID.x; // 0 .. THREAD_GROUP_SIZE-1
    uint baseIdx = groupID.x * ELEMENTS_PER_GROUP; // first global element index for this group

    // Load two consecutive cellCount entries per thread into shared memory.
    s[2 * tid]     = cellCount[baseIdx + 2 * tid];
    s[2 * tid + 1] = cellCount[baseIdx + 2 * tid + 1];
    GroupMemoryBarrierWithGroupSync();

    // Up-sweep (reduce) phase.    
    // build a binary reduction tree where each level doubles the span of partial
    // sums, ending with the total sum in the last element of s
    for (uint stride = 1; stride < ELEMENTS_PER_GROUP; stride <<= 1)
    {
        // Stride is the distance between the two children being combined at the
        // current tree level. Starts at 1, combining adjacent pairs (the leaves),
        // then doubles each iteration. 
        // i is the index that thread tid is responsible for writing on this
        // iteration. At a given level, the elements that get written are
        // spaced 2*stride apart, and we want to write into the right child
        // of each pair (the rightmost leaf of each subtree so far), so the
        // indices are 2*stride-1, 4*stride-1, 6*stride-1, etc. Thread tid 
        // handles the (tid+1)-th of those, giving i = (tid + 1) * (stride * 2) - 1
        uint i = (tid + 1) * (stride << 1) - 1;
        // take the value stride slots to the left (the left child's subtree
        // sum) and add it into s[i] (which currently holds the right child's
        // subtree su). After this, s[i] holds the sum of the combined, twice
        // as wide subtree, while s[i-stride] is left untouched.
        if (i < ELEMENTS_PER_GROUP) s[i] += s[i - stride];
        // wait for all threads to be done, since level k+1 reads from indices
        // that level k just wrote
        GroupMemoryBarrierWithGroupSync(); 
    }

    // Thread 0 records the group total, then zeroes the last slot so the
    // down-sweep produces an exclusive (not inclusive) prefix sum.
    if (tid == 0)
    {
        groupSums[groupID.x] = s[ELEMENTS_PER_GROUP - 1];
        s[ELEMENTS_PER_GROUP - 1] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    
    // Down-sweep phase.
    // Walk the same reduction tree back down, and at each node perform a "butterfly":
    //   left = s[i - stride] (save left child)
    //   s[i - stride] = s[i] (left child becomes parent's current value)
    //   s[i] = s[i] + left (right child becomes parent + left child's old value)
    // If we seed the root with 0 and run this, each leaf k ends up holding the sum
    // of all leaves with index less than k - the exclusive prefix sum.
    // We start from the root, descending to the leaves. At the last entry, the root,
    // we have the newly written 0, and the rest of s still holds whatever the up-sweep
    // left there: subtree sums at internal-node positions and original/partial values 
    // at leaf positions.
    // At each tree node we hold a running "prefix-sum-from-the-left" value in the
    // right-child slot, which the up-sweep wrote ti. The butterfly pushes that
    // value down into the two children: the left child receives the parent's value
    // verbatim (because the left child's exclusive prefix is whatever the parent's
    // exclusive prefix was), and the right child receives the parent's value plus the
    // left subtree's total (because by the time we reach the right child, we've 
    // passed everything in the left subtree). The left subtree's total is exactly what
    // the up-sweep saved at the left.child slot.
    for (uint stride = ELEMENTS_PER_GROUP >> 1; stride >= 1; stride >>= 1)
    {
        // stride is again the distance between the two children of a butterfly, 
        // and again it's the level indicator — but reversed compared to the up-sweep.
        // For example, with ELEMENTS_PER_GROUP=512, we start at stride 256, which
        // means the root level, i.e. one butterfly involving s[255] and s[511],
        // and halve down through 128, 64, etc. The active thread pattern is reversed
        // here as opposed to the up-sweep: only thread 0 is active first iteration, 
        // because only with tid=0 is i<512 passing.
        uint i = (tid + 1) * (stride << 1) - 1;
        if (i < ELEMENTS_PER_GROUP)
        {
            // s[i - stride] is the slot the up-sweep wrote to one level deeper on 
            // the left subtree, so it currently holds the total sum of the left subtree 
            // (the sum of all original inputs in the left half of this butterfly's range).
            uint left = s[i - stride]; 
            // The left child receives the parent's current value. s[i] at this moment 
            // holds the parent's exclusive prefix — the sum of all original inputs that 
            //lie strictly to the left of the parent's range. The left child's range starts 
            // at exactly the same place as the parent's range, so the left child's 
            // exclusive prefix is identical to the parent's. Pure copy.
            s[i - stride] = s[i];
            // The right child receives the parent's value plus the left subtree's total. 
            //s[i] already holds the parent's value (we haven't touched it yet this 
            // iteration — we only read it). Adding left, which is the sum of everything in 
            // the left subtree, gives the exclusive prefix at the right child's position: 
            // everything before the parent, plus everything in the left subtree, equals 
            // everything before the right child.
            s[i] += left;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Write the intra-group exclusive prefix sums.
    // Global offsets (from pass 2) are added in pass 3.
    cellPrefixSum[baseIdx + 2 * tid] = s[2 * tid];
    cellPrefixSum[baseIdx + 2 * tid + 1] = s[2 * tid + 1];
}

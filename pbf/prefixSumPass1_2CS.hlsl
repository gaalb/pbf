// Parallel exclusive prefix sum (Blelloch scan) — local scan kernel.
// Used for both pass 1 (N groups) and pass 2 (M groups) of the 5-pass design,
// with different descriptor bindings each time:
//
//   Pass 1: u0=cellCount, u1=cellPrefixSum, u2=groupSums        (N groups)
//   Pass 2: u0=groupSums, u1=groupPrefixSum, u2=superGroupSums  (M groups)
//
// Each group of THREAD_GROUP_SIZE (256) threads processes ELEMENTS_PER_GROUP (512)
// consecutive entries: each thread is responsible for two adjacent elements.
// The local exclusive prefix sum is written to u1[]. Cross-group offsets are NOT
// added here; passes 4 and 5 propagate them via prefixSumPass4CS.
// The per-group total is stored in u2[groupID.x] for the next level's scan.
//
// Dispatch: N = numCells / ELEMENTS_PER_GROUP groups (pass 1),
//           M = N / ELEMENTS_PER_GROUP groups (pass 2).

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

// Each particle computes its grid cell, atomically claims a slot within that
// cell (via InterlockedAdd on cellCount, which was cleared after the counting
// pass), and records its destination index in the permutation buffer perm[].
// A separate gatherCS pass reads perm[] and scatters each field to the sorted
// position. This keeps sortCS field-agnostic: adding or removing a particle
// field requires no changes here.
//
// In: cellCount (as atomic counter), cellPrefixSum, predictedPosition
// Out: perm (old index -> new sorted index), cellCount (side-effect of InterlockedAdd)

#define SortRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 7), UAV(u7, numDescriptors = 2), UAV(u16, numDescriptors = 1))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"
#include "GridUtils.hlsli" // posToCell(), cellIndex()

// Only predictedPosition (u2) is read from the particle fields table; the
// rest of u0..u6 are bound but unused by this shader.
RWStructuredBuffer<float3> predictedPosition : register(u2);

// Grid buffers
RWStructuredBuffer<uint> cellCount : register(u7);
RWStructuredBuffer<uint> cellPrefixSum : register(u8);

// Permutation output: perm[i] = sorted destination index for particle i
RWStructuredBuffer<uint> perm : register(u16);

[RootSignature(SortRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    int3 cell = posToCell(predictedPosition[i]); // 3D cell grid position of the float position
    uint ci = cellIndex(cell); // 1D index of the cell this particle belongs to

    // this shader uses cellCount as a running index for a given cell, incremented atomically!
    // slot is the index that this particle is going to take in the cell
    uint slot;
    InterlockedAdd(cellCount[ci], 1, slot); // our slot = index before the increment (so first is 0)

    // cellPrefixSum[ci] is basically the offset of the cell we're in, to which we add
    // the current running index (slot) to find where this particle's data shall get inserted
    perm[i] = cellPrefixSum[ci] + slot;
}

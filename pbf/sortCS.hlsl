// Each particle computes its grid cell, atomically claims a slot within that
// cell (via InterlockedAdd on cellCount, which was cleared after the counting
// pass), and records its destination index in the permutation buffer perm[].
// A separate gatherCS pass reads perm[] and scatters each field to the sorted
// position. This keeps sortCS field-agnostic: adding or removing a particle
// field requires no changes here.
//
// In: cellCount (as atomic counter), cellPrefixSum, predictedPosition
// Out: perm (old index -> new sorted index), cellCount (side-effect of InterlockedAdd)

#define SortRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 4))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"
#include "GridUtils.hlsli" // posToCell(), cellIndex()

RWStructuredBuffer<float3> predictedPosition : register(u0);
RWStructuredBuffer<uint>   cellCount         : register(u1);
RWStructuredBuffer<uint>   cellPrefixSum     : register(u2);
// Permutation output: perm[i] = sorted destination index for particle i
RWStructuredBuffer<uint>   perm              : register(u3);

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

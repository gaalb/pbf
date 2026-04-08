// Computes cellPrefixSum[i] = sum of cellCount[0..i-1] for all active cells.
// This gives the starting offset of each cell's particles in the sorted buffer:
// cell 0's particles go to sorted indices [0, cellCount[0]),
// cell 1's go to [cellCount[0], cellCount[0]+cellCount[1]), etc.
//
// Runs every frame as part of the spatial reorder pass. A simple serial loop
// (single thread) is used. For 32,768 cells (32^3) this takes microseconds.
//
// In: cellCount
// Out: cellPrefixSum


#define PrefixSumRootSig "CBV(b0), DescriptorTable(UAV(u7, numDescriptors = 2))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"
#include "GridUtils.hlsli" // gridDim()

RWStructuredBuffer<uint> cellCount : register(u7);
RWStructuredBuffer<uint> cellPrefixSum : register(u8);

[RootSignature(PrefixSumRootSig)]
[numthreads(1, 1, 1)]
void main()
{
    int dim = gridDim();
    uint totalCells = dim * dim * dim;

    // prefix sum: for each cell i, store the running total so far (which is
    // the sum of cellCount[0] through cellCount[i-1]), then add cellCount[i] to the running
    // total. After this loop, cellPrefixSum[ci] holds the starting index in the sorted buffer
    // for cell ci's particles. For example, if cellCount = [3, 0, 5, 2, ...], then
    // cellPrefixSum = [0, 3, 3, 8, ...].
    uint runningTotal = 0;
    for (uint i = 0; i < totalCells; i++)
    {
        cellPrefixSum[i] = runningTotal;
        runningTotal += cellCount[i];
    }
}

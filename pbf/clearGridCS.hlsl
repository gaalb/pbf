// Zeros the cellCount array.
// Dispatched over cells (not particles): one thread per cell.
//
// In: -
// Out: cellCount

#define ClearGridRootSig "CBV(b0), DescriptorTable(UAV(u7, numDescriptors = 2))"

#include "ComputeCb.hlsli"

#include "GridUtils.hlsli" // gridDim()

RWStructuredBuffer<uint> cellCount : register(u7);

[RootSignature(ClearGridRootSig)]
[numthreads(256, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;

    int dim = gridDim(); // how many cells there are along each axis
    uint totalCells = dim * dim * dim; // grid is cubic

    if (i >= totalCells)
        return; // discard threads that don't belong to cells

    cellCount[i] = 0;
}

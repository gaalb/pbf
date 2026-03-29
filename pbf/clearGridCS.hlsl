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

    int dim = gridDim();
    uint totalCells = dim * dim * dim;

    if (i >= totalCells)
        return;

    cellCount[i] = 0;
}

// Zeros the cellCount array.
// Dispatched over cells (not particles): one thread per cell.
//
// In: cellCount
// Out: cellCount
// a note on having cellCount as an input: even though the shader only writes to it, cellCount
// has to be flushed before we zero it out, since other shaders may still be using it, due to the
// fact that it has two uses (in one pass we use it to calculate how many particles are in a cell,
// then we zero it and use it as a counter to keep track of the next free slot in the cell)
// TODO: check if we can achieve an fps improvement by separating these two uses of cellCount,
// and having a new buffer to use as a "next free cell" buffer for the sort shader
// see the related note in ComputeShader::barrier_then_dispatch
//

#define ClearGridRootSig "CBV(b0), DescriptorTable(UAV(u7, numDescriptors = 2))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"
#include "GridUtils.hlsli"

RWStructuredBuffer<uint> cellCount : register(u7);

[RootSignature(ClearGridRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;

    uint totalCells = GRID_DIM * GRID_DIM * GRID_DIM;

    if (i >= totalCells)
        return; // discard threads that don't belong to cells

    cellCount[i] = 0;
}

// First of two grid-build passes. Each particle computes which cell it belongs to
// (from its predictedPosition) and atomically increments that cell's count.
//
// In: predictedPosition, cellCount
// Out: cellCount

#define CountGridRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 7)), DescriptorTable(UAV(u7, numDescriptors = 2))"

#include "ComputeCb.hlsli"

#include "GridUtils.hlsli" // posToCell(), cellIndex()

RWStructuredBuffer<float3> predictedPosition : register(u2);
RWStructuredBuffer<uint> cellCount : register(u7);

[RootSignature(CountGridRootSig)]
[numthreads(256, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    int3 cell = posToCell(predictedPosition[i]);
    uint ci = cellIndex(cell);

    InterlockedAdd(cellCount[ci], 1);
}

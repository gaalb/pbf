// Exclusive prefix sum of cellCount:
//
// Computes cellPrefixSum[i] = sum of cellCount[0..i-1] for all active cells.
// This gives the starting offset of each cell's particles in the sorted buffer:
// cell 0's particles go to sorted indices [0, cellCount[0]),
// cell 1's go to [cellCount[0], cellCount[0]+cellCount[1]), etc.
//
// Only runs every N frames as part of the spatial reorder pass, so a simple
// serial loop (single thread) is used. For ~5,700 cells at typical settings
// this takes microseconds.
//
// Root signature:
//   CBV(b0)                                    -- ComputeCb (for boxMin, boxMax, h -> gridDims)
//   DescriptorTable(UAV(u0, numDescriptors=5)) -- u1: cellCount (read), u4: cellPrefixSum (write)

#define PrefixSumRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 4))"

cbuffer ComputeCb : register(b0)
{
    float dt;
    uint numParticles;
    float h;
    float rho0;
    float3 boxMin;
    float epsilon;
    float3 boxMax;
    float viscosity;
    float sCorrK;
    float sCorrDeltaQ;
    float sCorrN;
    float vorticityEpsilon;
    float3 externalForce;
};

#include "GridUtils.hlsli" // gridDims()

RWStructuredBuffer<uint> cellCount : register(u1);
RWStructuredBuffer<uint> cellPrefixSum : register(u3);

[RootSignature(PrefixSumRootSig)]
[numthreads(1, 1, 1)]
void main()
{
    int3 dims = gridDims();
    uint numCells = dims.x * dims.y * dims.z;

    // prefix sum: for each cell i, store the running total so far (which is
    // the sum of cellCount[0] through cellCount[i-1]), then add cellCount[i] to the running
    // total. After this loop, cellPrefixSum[ci] holds the starting index in the sorted buffer
    // for cell ci's particles. For example, if cellCount = [3, 0, 5, 2, ...], then
    // cellPrefixSum = [0, 3, 3, 8, ...].
    uint runningTotal = 0;
    for (uint i = 0; i < numCells; i++)
    {
        cellPrefixSum[i] = runningTotal;
        runningTotal += cellCount[i];
    }
}

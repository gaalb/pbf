// Each particle computes its grid cell, atomically claims a slot within that
// cell (via InterlockedAdd on cellCount, which was cleared after the counting
// pass), and copies its field data directly to the sorted position in the
// sorted buffers. After this pass, the sorted buffers contain all particles
// arranged so that particles in the same grid cell are contiguous in memory.
// A CopyBufferRegion per field then copies the sorted data back to the main
// particle field buffers.
//
// TODO: switch to an index-based sort: compute a permutation buffer
// (old index -> new index) in this pass, then apply it with a separate gather
// shader. This avoids scattering every field here and makes adding/removing
// fields trivial.
//
// In: cellCount?, cellPrefixSum, position, velocity, predictedPosition, lambda, density, omega, scratch
// Out: sortedPosition, sortedVelocity, sortedPredictedPosition, sortedLambda, sortedDensity, sortedOmega, sortedScratch

#define SortRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 7)), DescriptorTable(UAV(u7, numDescriptors = 2)), DescriptorTable(UAV(u9, numDescriptors = 7))"

#include "ComputeCb.hlsli"

#include "GridUtils.hlsli" // posToCell(), cellIndex()

// Particle field buffers (read)
RWStructuredBuffer<float3> position : register(u0);
RWStructuredBuffer<float3> velocity : register(u1);
RWStructuredBuffer<float3> predictedPosition : register(u2);
RWStructuredBuffer<float> lambda : register(u3);
RWStructuredBuffer<float> density : register(u4);
RWStructuredBuffer<float3> omega : register(u5);
RWStructuredBuffer<float3> scratch : register(u6);

// Grid buffers
RWStructuredBuffer<uint> cellCount : register(u7);
RWStructuredBuffer<uint> cellPrefixSum : register(u8);

// Sorted particle field buffers (write)
RWStructuredBuffer<float3> sortedPosition : register(u9);
RWStructuredBuffer<float3> sortedVelocity : register(u10);
RWStructuredBuffer<float3> sortedPredictedPosition : register(u11);
RWStructuredBuffer<float> sortedLambda : register(u12);
RWStructuredBuffer<float> sortedDensity : register(u13);
RWStructuredBuffer<float3> sortedOmega : register(u14);
RWStructuredBuffer<float3> sortedScratch : register(u15);

[RootSignature(SortRootSig)]
[numthreads(256, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    int3 cell = posToCell(predictedPosition[i]); // 3D cell grid position of the float position
    uint ci = cellIndex(cell); // 1D index of the cell this particle belongs to

    // this shader uses cellCount as a running index for a given cell
    // slot is the index that this particle is going to take in the cell
    uint slot; 
    InterlockedAdd(cellCount[ci], 1, slot); // our slot = index before the increment (so first is 0)

    // cellPrefixSum[ci] is basically the offset of the cell we're in, to which we add
    // the current running index (slot) to find where this particle's data shall get inserted
    uint dest = cellPrefixSum[ci] + slot;
    sortedPosition[dest] = position[i];
    sortedVelocity[dest] = velocity[i];
    sortedPredictedPosition[dest] = predictedPosition[i];
    sortedLambda[dest] = lambda[i];
    sortedDensity[dest] = density[i];
    sortedOmega[dest] = omega[i];
    sortedScratch[dest] = scratch[i];
}

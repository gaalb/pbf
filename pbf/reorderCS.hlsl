// Spatial reorder pass:
//
// Scatters particles into a sorted buffer so that particles in the same grid
// cell are contiguous in memory. This restores the spatial cache coherence
// that degrades as particles move during simulation.
//
// Dispatched over (cell, slot) pairs rather than over particles. Each thread:
//   1. Computes its cell index and slot from the global thread ID.
//   2. If the slot is occupied (slot < cellCount[cell]), reads the particle
//      index from cellParticles and copies the particle to the sorted buffer.
//
// This approach uses the grid built earlier in the frame (cellCount +
// cellParticles) directly, avoiding a mismatch between grid-build-time
// positions and current positions. No atomics needed — each (cell, slot)
// pair maps to a unique sorted position deterministically.
//
// After this pass, CopyBufferRegion copies sortedParticles back to particles.
//
// Root signature:
//   CBV(b0)                                    -- ComputeCb
//   DescriptorTable(UAV(u0, numDescriptors=5)) -- u0: particles (read), u1: cellCount (read),
//                                                 u2: cellParticles (read), u3: sortedParticles (write),
//                                                 u4: cellPrefixSum (read)

#define ReorderRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 5))"

#include "Particle.hlsli" // Particle struct

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
    uint maxPerCell;
};

#include "GridUtils.hlsli" // gridDims()

RWStructuredBuffer<Particle> particles : register(u0);
RWStructuredBuffer<uint> cellCount : register(u1);
RWStructuredBuffer<uint> cellParticles : register(u2);
RWStructuredBuffer<Particle> sortedParticles : register(u3);
RWStructuredBuffer<uint> cellPrefixSum : register(u4);

[RootSignature(ReorderRootSig)]
[numthreads(256, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint threadIdx = dispatchID.x;

    // Map flat thread index to (cell, slot) pair
    uint ci = threadIdx / maxPerCell;
    uint slot = threadIdx % maxPerCell;

    // Skip threads beyond the current grid
    int3 dims = gridDims();
    uint numCells = dims.x * dims.y * dims.z;
    if (ci >= numCells)
        return;

    // Skip empty slots in this cell
    if (slot >= cellCount[ci])
        return;

    // Read the particle index from the grid and copy the particle to sorted order
    uint particleIndex = cellParticles[ci * maxPerCell + slot];
    sortedParticles[cellPrefixSum[ci] + slot] = particles[particleIndex];
}

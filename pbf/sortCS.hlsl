// Direct sort pass:
//
// Each particle computes its grid cell, atomically claims a slot within that
// cell (via InterlockedAdd on cellCount, which was cleared after the counting
// pass), and copies its own data directly to the sorted position in
// sortedParticles. This merges what used to be a separate grid-insert step
// and a scatter-reorder step into a single dispatch, eliminating the
// cellParticles buffer entirely.
//
// After this pass, sortedParticles contains all particles arranged so that
// particles in the same grid cell are contiguous in memory. A CopyBufferRegion
// then copies sortedParticles back to particles.
//
// Root signature:
//   CBV(b0)                                    -- ComputeCb
//   DescriptorTable(UAV(u0, numDescriptors=4)) -- u0: particles (read), u1: cellCount (write),
//                                                 u2: sortedParticles (write), u3: cellPrefixSum (read)

#define SortRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 4))"

#include "Particle.hlsli"

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
    uint fountainEnabled;
};

#include "GridUtils.hlsli" // posToCell(), cellIndex()

RWStructuredBuffer<Particle> particles : register(u0);
RWStructuredBuffer<uint> cellCount : register(u1);
RWStructuredBuffer<Particle> sortedParticles : register(u2);
RWStructuredBuffer<uint> cellPrefixSum : register(u3);

[RootSignature(SortRootSig)]
[numthreads(256, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    int3 cell = posToCell(particles[i].predictedPosition);
    uint ci = cellIndex(cell);

    uint slot;
    InterlockedAdd(cellCount[ci], 1, slot);

    sortedParticles[cellPrefixSum[ci] + slot] = particles[i];
}

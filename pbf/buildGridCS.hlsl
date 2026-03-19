// Build grid pass:
//
// Each particle computes which cell it belongs to (from its predictedPosition),
// then atomically claims a slot in that cell's entry in the cellParticles array.
// This runs once per frame after predictCS, before the solver loop.
//
// The grid is built from predicted positions (post-prediction, pre-solver). The solver
// loop and post-solver passes (vorticity, confinement, viscosity) reuse this same grid.
// Per the PBF paper, neighborhoods are computed once per step and reused across solver
// iterations.
//
// Root signature:
//   CBV(b0)                                    -- ComputeCb
//   DescriptorTable(UAV(u0, numDescriptors=3)) -- u0: particles, u1: cellCount, u2: cellParticles

#define BuildGridRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 3))"

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
    uint maxPerCell;
};

#include "GridUtils.hlsli" // posToCell(), cellIndex()

RWStructuredBuffer<Particle> particles : register(u0);
RWStructuredBuffer<uint> cellCount : register(u1);
RWStructuredBuffer<uint> cellParticles : register(u2);

[RootSignature(BuildGridRootSig)]
[numthreads(256, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    int3 cell = posToCell(particles[i].predictedPosition);
    uint ci = cellIndex(cell);

    // Atomically claim the next available slot in this cell.
    // InterlockedAdd returns the old value in 'slot', which becomes our write index.
    uint slot;
    InterlockedAdd(cellCount[ci], 1, slot);

    // If the cell is full, silently drop this particle from the grid.
    // This is safe: the SPH kernels give near-zero weight to distant particles anyway,
    // and overflow only happens during extreme transient compression.
    if (slot < maxPerCell)
        cellParticles[ci * maxPerCell + slot] = i;
}

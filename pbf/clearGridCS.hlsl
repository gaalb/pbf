// Clear grid pass:
//
// Zeros the cellCount array before buildGridCS repopulates it.
// Dispatched over cells (not particles): one thread per cell in the current grid.
// Cells beyond the current grid dimensions (which depend on the current h) are skipped.
//
// Root signature:
//   CBV(b0)                                    -- ComputeCb
//   DescriptorTable(UAV(u0, numDescriptors=5)) -- u0: particles (unused), u1: cellCount, u2-u4: unused

#define ClearGridRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 5))"

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

RWStructuredBuffer<uint> cellCount : register(u1);

[RootSignature(ClearGridRootSig)]
[numthreads(256, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;

    // The current grid may be smaller than the worst-case allocation.
    // Only clear cells that belong to the current grid.
    int3 dims = gridDims();
    uint numCells = dims.x * dims.y * dims.z;

    if (i >= numCells)
        return;

    cellCount[i] = 0;
}

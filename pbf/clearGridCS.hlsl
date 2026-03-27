// Clear grid pass:
//
// Zeros the cellCount array before countGridCS repopulates it.
// Dispatched over cells (not particles): one thread per cell.
//
// Root signature:
//   CBV(b0)                        -- ComputeCb
//   DescriptorTable(UAV(u0..u6))   -- particle field buffers (unused here)
//   DescriptorTable(UAV(u7..u8))   -- grid buffers: u7 = cellCount
//   DescriptorTable(UAV(u9..u15))  -- sorted particle field buffers (unused here)

#define ClearGridRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 7)), DescriptorTable(UAV(u7, numDescriptors = 2)), DescriptorTable(UAV(u9, numDescriptors = 7))"

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

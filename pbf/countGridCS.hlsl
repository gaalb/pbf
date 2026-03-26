// Count grid pass:
//
// First of two grid-build passes. Each particle computes which cell it belongs to
// (from its predictedPosition) and atomically increments that cell's count.
//
// The count is used by prefixSumCS to compute scatter offsets, after which
// cellCount is cleared and reused as a per-cell atomic slot counter by sortCS.
//
// Root signature:
//   CBV(b0)                        -- ComputeCb
//   DescriptorTable(UAV(u0..u6))   -- particle field buffers: u2 = predictedPosition (read)
//   DescriptorTable(UAV(u7..u8))   -- grid buffers: u7 = cellCount (write)
//   DescriptorTable(UAV(u9..u15))  -- sorted particle field buffers (unused here)

#define CountGridRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 7)), DescriptorTable(UAV(u7, numDescriptors = 2)), DescriptorTable(UAV(u9, numDescriptors = 7))"

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

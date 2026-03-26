// Count grid pass:
//
// First of two grid-build passes. Each particle computes which cell it belongs to
// (from its predictedPosition) and atomically increments that cell's count.
//
// The count is used by prefixSumCS to compute scatter offsets, after which
// cellCount is cleared and reused as a per-cell atomic slot counter by sortCS.
//
// Root signature:
//   CBV(b0)                                    -- ComputeCb
//   DescriptorTable(UAV(u0, numDescriptors=4)) -- u0: particles (read), u1: cellCount (write), u2-u3: unused

#define CountGridRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 4))"

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

[RootSignature(CountGridRootSig)]
[numthreads(256, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    int3 cell = posToCell(particles[i].predictedPosition);
    uint ci = cellIndex(cell);

    InterlockedAdd(cellCount[ci], 1);
}

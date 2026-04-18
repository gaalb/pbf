// setLodMaxCS: fills every particle's LOD countdown with maxLOD so all particles
// run the full solverIterations each frame.
//
// In: -
// Out: lod

#define SetLodMaxRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 1))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"

RWStructuredBuffer<uint> lod : register(u0);

[RootSignature(SetLodMaxRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles) return;
    lod[i] = maxLOD;
}

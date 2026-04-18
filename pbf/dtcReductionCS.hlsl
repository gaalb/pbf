// Calculates the min and max camera distance (DTC) across all particles
//
// In: predictedPosition, lodReduction
// Out: lodReduction

#define DtcReductionRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 7), UAV(u7, numDescriptors = 1))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"

RWStructuredBuffer<float3> predictedPosition : register(u2);
// uint buffer, even though it's really storing float values, because
// InterlockedMin/Max only works on uints
// the globallycoherent qualifier ensures that all threads must make
// sure to read the device-level L2 memory for this resource, rather than
// the potentially stale L1 group level cacjes
// technically this isn't necessary for correctness here, since
// InterlockedXXX already does this out of the box - but I like to remind myself :)
globallycoherent RWStructuredBuffer<uint> lodReduction : register(u7);

[RootSignature(DtcReductionRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles) return;

    float dtc = length(predictedPosition[i] - cameraPos);
    // dtc is float, but the buffer is uint to enable InterlockedMin/Max,
    // so cast into uint for Interlocked operations
    uint dtcBits = asuint(dtc); 
    InterlockedMin(lodReduction[0], dtcBits);
    InterlockedMax(lodReduction[1], dtcBits);
}

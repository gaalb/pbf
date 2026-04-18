// Sets the level of detail (lod) for a particle, based on its
// distance to the camera (dtc) and the configured dtcMin/dtcMax range. 
//
// In: predictedPosition, lodReduction
// Out: lod

#define LodRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 3))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"

RWStructuredBuffer<float3> predictedPosition : register(u0);
RWStructuredBuffer<uint>   lod               : register(u1);
RWStructuredBuffer<uint>   lodReduction      : register(u2);

[RootSignature(LodRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles) return;

    float dtcMin = asfloat(lodReduction[0]); // stored as uint bits, but really a float value
    float dtcMax = asfloat(lodReduction[1]); // stored as uint bits, but really a float value
    float dtc = length(predictedPosition[i] - cameraPos);
    // interpolation factor in [0, 1] range, with a guard against the dtcMin == dtcMax edge case (which would cause divide-by-zero)
    float t = (dtcMax > dtcMin) ? saturate((dtc - dtcMin) / (dtcMax - dtcMin)) : 0.0; 
    // lerp to get a value between maxLOD and minLOD, then round to nearest integer and cast to uint
    uint lodVal = (uint) round(lerp((float) maxLOD, (float) minLOD, t)); 
    lod[i] = lodVal;
}

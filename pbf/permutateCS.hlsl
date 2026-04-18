// Reads the permutation table computed by sortCS (perm[i] = sorted destination of
// particle i) and scatters all particle field data to the corresponding sorted
// positions in the sortedFields buffers.
//
// Adding or removing a particle field only requires changing this shader;
// sortCS is field-agnostic and remains untouched.
//
// In: perm, position, velocity, predictedPosition, lambda, density, omega, scratch
// Out: sortedPosition, sortedVelocity, sortedPredictedPosition,
//      sortedLambda, sortedDensity, sortedOmega, sortedScratch

#define GatherRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 15))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"

// Particle field buffers (read)
RWStructuredBuffer<float3> position          : register(u0);
RWStructuredBuffer<float3> velocity          : register(u1);
RWStructuredBuffer<float3> predictedPosition : register(u2);
RWStructuredBuffer<float>  lambda            : register(u3);
RWStructuredBuffer<float>  density           : register(u4);
RWStructuredBuffer<float3> omega             : register(u5);
RWStructuredBuffer<float3> scratch           : register(u6);

// Sorted particle field buffers (write)
RWStructuredBuffer<float3> sortedPosition          : register(u7);
RWStructuredBuffer<float3> sortedVelocity          : register(u8);
RWStructuredBuffer<float3> sortedPredictedPosition : register(u9);
RWStructuredBuffer<float>  sortedLambda            : register(u10);
RWStructuredBuffer<float>  sortedDensity           : register(u11);
RWStructuredBuffer<float3> sortedOmega             : register(u12);
RWStructuredBuffer<float3> sortedScratch           : register(u13);

// Permutation table (read): perm[i] is the sorted destination for particle i
RWStructuredBuffer<uint> perm : register(u14);

[RootSignature(GatherRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint i = dispatchID.x;
    if (i >= numParticles)
        return;

    uint dest = perm[i];
    sortedPosition[dest]          = position[i];
    sortedVelocity[dest]          = velocity[i];
    sortedPredictedPosition[dest] = predictedPosition[i];
    sortedLambda[dest]            = lambda[i];
    sortedDensity[dest]           = density[i];
    sortedOmega[dest]             = omega[i];
    sortedScratch[dest]           = scratch[i];
}

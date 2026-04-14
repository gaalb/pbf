// Clear the DTVS reduction buffer to initial values (0.0f bits) 
//
// In: -
// Out: lodReduction

#define ClearDtvsReductionRootSig \
    "CBV(b0), " \
    "DescriptorTable(UAV(u0, numDescriptors = 1))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"

RWStructuredBuffer<uint> lodReduction : register(u0);

[RootSignature(ClearDtvsReductionRootSig)]
[numthreads(1, 1, 1)]
void main()
{
    lodReduction[0] = 0u; // 0.0f bits: initial max-DTVS accumulator (grows via InterlockedMax)
}

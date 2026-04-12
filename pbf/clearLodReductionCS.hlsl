#define ClearLodReductionRootSig \
    "CBV(b0), " \
    "DescriptorTable(UAV(u0, numDescriptors = 1))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"

RWStructuredBuffer<uint> lodReduction : register(u0);

[RootSignature(ClearLodReductionRootSig)]
[numthreads(1, 1, 1)]
void main()
{
    lodReduction[0] = 0x7F7FFFFFu; // FLT_MAX bits (min DTC accumulator)
    lodReduction[1] = 0u;          // 0.0f bits    (max DTC accumulator)
}

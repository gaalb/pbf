// splatDensityVolumeResolveCS.hlsl
// Per-voxel resolve pass of the splat density pipeline.
//
// Reads the raw float bits accumulated by splatDensityVolumeCS into splatAccum (uint),
// reinterprets them as float via asfloat, writes to densityVolume, then clears splatAccum
// to 0 (= 0.0f in IEEE 754) for the next frame. Piggybacks the clear onto the
// already-required per-voxel iteration, avoiding a separate clear dispatch.
//
// Dispatched immediately after splatDensityVolumeCS, separated by a UAV barrier on
// splatAccum (issued automatically by dispatch_then_barrier since splatAccumTexture
// is listed in splatDensityShader's outputs).
//
// In:  splatAccumTexture (u1) — RWTexture3D<uint>, UAV state; raw float bits of density sums
// Out: densityVolume     (u0) — RWTexture3D<float>, UAV state; float density for liquidPS
//      splatAccumTexture (u1) — cleared to 0 (= 0.0f) for the next frame

#define SplatResolveRootSig \
    "CBV(b0), " \
    "DescriptorTable(UAV(u0, numDescriptors = 1)), " \
    "DescriptorTable(UAV(u1, numDescriptors = 1))"

#include "SharedConfig.hlsli"

RWTexture3D<float> densityVolume : register(u0);
RWTexture3D<uint>  splatAccum    : register(u1);

[RootSignature(SplatResolveRootSig)]
[numthreads(8, 8, 4)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint3 voxel  = id;
    uint  volDim = (uint)VOL_DIM;
    if (voxel.x >= volDim || voxel.y >= volDim || voxel.z >= volDim)
        return;

    densityVolume[voxel] = asfloat(splatAccum[voxel]); // reinterpret raw bits as float: no arithmetic, no quantization
    splatAccum[voxel] = 0u; // 0u = 0.0f in IEEE 754: clears accumulator for the next frame's splat pass
}

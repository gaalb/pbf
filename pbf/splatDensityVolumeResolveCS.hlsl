// splatDensityVolumeResolveCS.hlsl
// Per-voxel resolve pass of the splat density pipeline.
//
// Reads the fixed-point uint accumulator filled by splatDensityVolumeCS, converts it
// to a float density value in densityVolume, then writes 0 back to splatAccum so the
// accumulator is clean for the next frame's splat pass. This piggybacks the clear onto
// the already-required per-voxel iteration, avoiding a separate clear dispatch.
//
// Dispatched immediately after splatDensityVolumeCS, separated by a UAV barrier on
// splatAccum (issued automatically by dispatch_then_barrier since splatAccumTexture
// is listed in splatDensityShader's outputs).
//
// In:  splatAccumTexture (u1) — RWTexture3D<uint>, UAV state; fixed-point density sums
// Out: densityVolume     (u0) — RWTexture3D<float>, UAV state; float density for liquidPS
//      splatAccumTexture (u1) — cleared to 0 for the next frame

#define SplatResolveRootSig \
    "CBV(b0), " \
    "DescriptorTable(UAV(u0, numDescriptors = 1)), " \
    "DescriptorTable(UAV(u1, numDescriptors = 1))"

#include "SharedConfig.hlsli"
#include "splatDensityVolumeCommon.hlsli"

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

    uint accum = splatAccum[voxel];
    densityVolume[voxel] = float(accum) / SPLAT_SCALE_F;
    splatAccum[voxel] = 0u; // clear for the next frame's splat pass
}

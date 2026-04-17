// splatDensityVolumeCS.hlsl
// Per-particle pass of the splat density pipeline.
//
// Algorithm:
//   For each particle, compute the bounding box of voxels whose centers lie within
//   the SPH kernel radius H, then evaluate Poly6 for each candidate voxel and
//   atomically accumulate the contribution (as a float stored in raw bits) into splatAccum
//   via a compare-and-swap loop. The resolve pass (splatDensityVolumeResolveCS) reinterprets
//   the accumulated bits as float and clears the accumulator for the next frame.
//
// This is the inverse of the per-voxel approach (densityVolumeCS): instead of each
// voxel querying all nearby particles, each particle writes to all nearby voxels.
// For typical PBF particle counts the number of particle threads is far smaller than
// VOL_DIM^3 voxel threads, making this approach more efficient.
//
// Dispatched from the GRAPHICS queue (DIRECT command list) in DrawLiquidSurface().
// In:  posSnapshot[readIdx]  (t0) — StructuredBuffer<float3>, NON_PIXEL|PIXEL SRV state
// Out: splatAccumTexture      (u0) — RWTexture3D<uint>, UAV state; stores raw float bits

#define SplatDensityVolumeRootSig \
    "CBV(b0), " \
    "DescriptorTable(SRV(t0, numDescriptors = 1)), " \
    "DescriptorTable(UAV(u0, numDescriptors = 1))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"
#include "SphKernels.hlsli"
#include "GridUtils.hlsli"

StructuredBuffer<float3> position   : register(t0);
RWTexture3D<uint>        splatAccum : register(u0); // raw float bits; asfloat(splatAccum[v]) gives density

// Atomically adds 'value' to splatAccum[coord] using a compare-and-swap loop.
// splatAccum stores raw IEEE 754 float bits as uint. asfloat/asuint are pure bit
// reinterpretations (no arithmetic conversion), so the float addition is exact.
// The loop retries if another thread modified the slot between our read and swap.
// Poly6 contributions are always positive, so asfloat is always >= 0: no sign-bit issues.
void FloatInterlockedAdd(uint3 coord, float value)
{
    uint assumed, old;
    old = splatAccum[coord];
    [loop] do {
        assumed = old;
        InterlockedCompareExchange(splatAccum[coord], assumed, asuint(asfloat(assumed) + value), old);
    } while (old != assumed);
}

[RootSignature(SplatDensityVolumeRootSig)]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint pIdx = id.x;
    if (pIdx >= numParticles)
        return;

    float3 pos = position[pIdx];

    // pos-GRID_MIN is the float world coordinate position of this particle, 
    // dividing by VOXEL_SIZE gets us that in voxel steps. Voxel v has its center
    // at world position GRID_MIN + (v + 0.5) * voxelSize.
    float3 voxelF = (pos - GRID_MIN) / VOXEL_SIZE;

    // Conservative bounding box in integer voxel indices.
    // voxelF - H_IN_VOXEL is the continuous voxel coordinate of the point that
    // is exactly one SPH radius away in the negative direction along each axis,
    // the floor of that gives its integer voxel coordinate. We clamp it to 0 to
    // avoid negative indices. Same idea in the positive direction for vMax.
    int3 vMin = max(int3(0, 0, 0), int3(floor(voxelF - H_IN_VOXEL)));
    int3 vMax = min(int3(VOL_DIM - 1, VOL_DIM - 1, VOL_DIM - 1), int3(floor(voxelF  + H_IN_VOXEL)));

    for (int vz = vMin.z; vz <= vMax.z; vz++)
    for (int vy = vMin.y; vy <= vMax.y; vy++)
    for (int vx = vMin.x; vx <= vMax.x; vx++)
    {
        float3 voxelCenter = GRID_MIN + (float3(vx, vy, vz) + 0.5f) * VOXEL_SIZE;
        float3 r  = voxelCenter - pos;
        float  r2 = dot(r, r);

        float w = Poly6(r, r2); // 0.0 if r2 > H*H
        if (w <= 0.0f)
            continue;

        FloatInterlockedAdd(uint3(vx, vy, vz), w);
    }
}

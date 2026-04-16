// densityVolumeCS.hlsl
// Fills a Texture3D<float4> volume with float4(density, gradX, gradY, gradZ) at each voxel.
// The volume covers the fixed simulation grid (GRID_MIN..GRID_MAX) at resolution VOL_DIM^3,
// giving a voxel spacing of exactly H/2 per axis.
//
// For each voxel, this shader:
//   1. Maps the voxel's 3D index to a world-space position
//   2. Looks up the 27 neighboring grid cells (same spatial grid as lambdaCS / deltaCS)
//   3. Accumulates density (Poly6 kernel) and density gradient (Poly6Grad) from all particles
//   4. Writes float4(rho, gradX, gradY, gradZ) to the UAV
//
// The result is read by liquidPS.hlsl for ray-marched liquid surface rendering.
// Dispatched from the GRAPHICS queue (on commandList) in DrawLiquidSurface(), reading from
// the previous frame's position and grid snapshots. The density volume is filled and consumed
// within the same graphics frame (fill → SRV transition → liquidPS draw), so no double-buffering
// of the volume is required.
//
// In:  position snapshot (t0), cellCount snapshot (t1), cellPrefixSum snapshot (t2)
// Out: densityVolume (u16)

#define DensityVolumeRootSig \
    "CBV(b0), " \
    "DescriptorTable(SRV(t0, numDescriptors = 1)), " \
    "DescriptorTable(SRV(t1, numDescriptors = 2)), " \
    "DescriptorTable(UAV(u16, numDescriptors = 1))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"
#include "SphKernels.hlsli"
#include "GridUtils.hlsli"

StructuredBuffer<float3>  position        : register(t0);
StructuredBuffer<uint>    cellCount       : register(t1);
StructuredBuffer<uint>    cellPrefixSum   : register(t2);
RWTexture3D<float4>       densityVolume   : register(u16);

[RootSignature(DensityVolumeRootSig)]
[numthreads(8, 8, 4)]  // 8*8*4 = 256 threads per group, same total as THREAD_GROUP_SIZE
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    // dispatchID maps directly to voxel (X, Y, Z) coordinates.
    // Dispatched as (ceil(VOL_DIM/8), ceil(VOL_DIM/8), ceil(VOL_DIM/4)) groups,
    // so the guard below catches any excess threads on non-multiple VOL_DIM values.
    // This 3D layout avoids the 65535-per-axis group limit that a flat 1D dispatch
    // would hit when VOL_DIM = GRID_DIM*4 = 256 (256^3/256 = 65536 > 65535).
    uint3 voxel = dispatchID;
    uint volDim = (uint)VOL_DIM;
    if (voxel.x >= volDim || voxel.y >= volDim || voxel.z >= volDim)
        return;

    // Map voxel center to world-space position within [GRID_MIN, GRID_MAX]
    float3 uvw = (float3(voxel) + 0.5) / float(volDim);
    float3 worldPos = lerp(GRID_MIN, GRID_MAX, uvw);

    // Accumulate SPH density and gradient by iterating over neighboring grid cells
    float  rho     = 0.0;
    float3 gradRho = float3(0.0, 0.0, 0.0);

    NeighborCells nCells = NeighborCellIndices(worldPos);
    for (uint c = 0; c < nCells.count; c++)
    {
        uint ci     = nCells.indices[c];
        uint count  = cellCount[ci];
        uint offset = cellPrefixSum[ci];

        for (uint s = 0; s < count; s++)
        {
            uint  j = offset + s;
            float3 r  = worldPos - position[j]; // vector from particle j to voxel center
            float  r2 = dot(r, r);

            rho     += Poly6(r, r2);
            gradRho += Poly6Grad(r, r2);
        }
    }

    densityVolume[voxel] = float4(rho, gradRho);
}

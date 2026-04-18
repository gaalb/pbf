// Cell-centric variant of deltaCS.
//
// For each particle i, computes:
//   delta_p_i = (1/rho0) * sum_{j != i}( (lambda_i + lambda_j + s_corr) * grad_W_spiky(r_ij) )
// and writes:
//   scratch[i] = predictedPosition[i] + delta_p_i
//
// Unlike the particle-centric deltaCS, this shader dispatches one thread GROUP per
// spatial grid cell (SV_GroupID.x == cell index). All CC_THREADS threads in a group
// cooperate to load every neighbor position AND lambda into groupshared memory (GSM)
// before computing, eliminating most scattered VRAM reads in the inner neighbor loop.
//
// Phase 1 — cooperative GSM load:
//   All threads stride through the sorted buffer covering all 27 neighbor cells,
//   packing predictedPosition (xyz) and lambda (w) into float4 slots.
//   Consecutive threads access consecutive addresses -> coalesced reads.
//   One GroupMemoryBarrierWithGroupSync follows.
//
// Phase 2 — compute:
//   Threads with localIdx >= cellCount[myCellIdx] exit. The rest compute delta_p
//   reading pj from gs_posLambda[gsIdx].xyz and lambdaJ from gs_posLambda[gsIdx].w.
//
// Dispatch: (numCells, 1, 1)  — one group per cell, empty cells return immediately.
// GSM:      gs_posLambda[CC_GSM_SIZE] float4  = 1500 * 16 = 24 KB < 32 KB minimum.
//
// In: predictedPosition, lambda, cellCount, cellPrefixSum
// Out: scratch (new predicted position, Jacobi-style; copied to predictedPosition by positionFromScratchCS)

#define DeltaRootSig "CBV(b0), DescriptorTable(UAV(u0, numDescriptors = 6))"

#include "SharedConfig.hlsli"
#include "ComputeCb.hlsli"
#include "SphKernels.hlsli"
#include "GridUtils.hlsli"

// Cell-centric dispatch parameters (CC_deltaCS, CC_lambdaCS, ...)
#define CC_THREADS  96  // threads per group; must be a multiple of 32 (warp size)
#define CC_GSM_SIZE 1500  // groupshared float4 slots: xyz = predictedPosition, w = lambda
                          // 1500 * 16 bytes = 24 KB < 32 KB minimum GSM guarantee
                          // headroom: ~34 particles/cell * 27 cells = ~918 at rest density

// Like NeighborCellIndices but takes 3D cell coordinates directly,
// skipping the position-to-cell quantization step.
// Used by cell-centric shaders where the cell index comes from SV_GroupID.
NeighborCells NeighborCellIndicesByCell(int3 myCell)
{
    NeighborCells result;
    result.count = 0;
    for (int dz = -1; dz <= 1; dz++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++)
            {
                int3 nc = myCell + int3(dx, dy, dz);
                if (nc.x < 0 || nc.x >= GRID_DIM ||
                    nc.y < 0 || nc.y >= GRID_DIM ||
                    nc.z < 0 || nc.z >= GRID_DIM)
                    continue;
                result.indices[result.count++] = cellIndex(nc);
            }
    return result;
}

#ifdef USE_MORTON_CODES
// Extracts every 3rd bit starting at bit 0 (inverse of expandBits).
// Used to decode a Morton code back to a single axis coordinate.
uint compactBits(uint x)
{
    x &= 0x09249249;
    x = (x | (x >> 2)) & 0x030C30C3;
    x = (x | (x >> 4)) & 0x0300F00F;
    x = (x | (x >> 8)) & 0x030000FF;
    x = (x | (x >> 16)) & 0x000003FF;
    return x;
}

// Decodes a Morton code back to 3D cell coordinates (inverse of cellIndex).
int3 cellToCoord(uint morton)
{
    return int3(compactBits(morton), compactBits(morton >> 1), compactBits(morton >> 2));
}
#else
// Decodes a flat row-major index back to 3D cell coordinates (inverse of cellIndex).
int3 cellToCoord(uint idx)
{
    return int3(int(idx) % GRID_DIM, (int(idx) / GRID_DIM) % GRID_DIM, int(idx) / (GRID_DIM * GRID_DIM));
}

#endif

RWStructuredBuffer<float3> predictedPosition : register(u0);
RWStructuredBuffer<float>  lambda            : register(u1);
RWStructuredBuffer<float3> scratch           : register(u2);
RWStructuredBuffer<uint>   cellCount         : register(u3);
RWStructuredBuffer<uint>   cellPrefixSum     : register(u4);
RWStructuredBuffer<uint>   lod               : register(u5);

groupshared float4 gs_posLambda[CC_GSM_SIZE]; // xyz = predictedPosition, w = lambda

[RootSignature(DeltaRootSig)]
[numthreads(CC_THREADS, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint localIdx : SV_GroupIndex)
{
    uint myCellIdx   = groupID.x;
    uint myCellCount = cellCount[myCellIdx];

    // All threads in the group share the same myCellIdx and therefore the same myCellCount.
    // If the cell is empty, all threads return together before the barrier — valid in HLSL
    // because no thread crosses the barrier (no divergence hazard).
    if (myCellCount == 0)
        return;

    int3          myCell3D = cellToCoord(myCellIdx);
    NeighborCells nCells   = NeighborCellIndicesByCell(myCell3D);

    // ---- Phase 1: cooperative GSM load ------------------------------------------
    // All CC_THREADS threads iterate the 27 neighbor cells in the same order.
    // gsOffset advances identically for every thread (same cellCount reads -> same values).
    // The inner loop strides by CC_THREADS: thread 0 loads s=0, thread 1 loads s=1, etc.
    // -> consecutive threads access consecutive predictedPosition addresses -> coalesced.
    uint gsOffset = 0;
    for (uint c = 0; c < nCells.count; c++)
    {
        uint ci    = nCells.indices[c];
        uint count = cellCount[ci];
        uint base  = cellPrefixSum[ci];
        for (uint s = localIdx; s < count; s += CC_THREADS)
        {
            uint gsIdx = gsOffset + s;
            if (gsIdx < CC_GSM_SIZE)    // guard: skip overflow on extreme transient density spikes
                gs_posLambda[gsIdx] = float4(predictedPosition[base + s], lambda[base + s]);
        }
        gsOffset += count;
    }
    GroupMemoryBarrierWithGroupSync();

    // ---- Phase 2: compute delta_p ------------------------------------------------
    // Threads beyond the cell's own particle count have finished loading but have no
    // particle to process.
    if (localIdx >= myCellCount)
        return;

    uint   i       = cellPrefixSum[myCellIdx] + localIdx;
    if (lod[i] == 0)
        return;
    float3 pi      = predictedPosition[i];
    float  lambdaI = lambda[i];

    float  poly6AtDeltaQ = Poly6(float3(SCORR_DELTA_Q, 0, 0), SCORR_DELTA_Q * SCORR_DELTA_Q);
    float3 deltaP        = float3(0, 0, 0);

    // Iterate the same 27-cell list in the same order as the load phase so that
    // gsOff mirrors the gsOffset computed there and gs_positions[gsOff + s] is
    // the position of the s-th particle in cell ci.
    uint gsOff = 0;
    for (uint c = 0; c < nCells.count; c++)
    {
        uint ci    = nCells.indices[c];
        uint count = cellCount[ci];
        uint base  = cellPrefixSum[ci];

        for (uint s = 0; s < count; s++)
        {
            uint j = base + s;
            if (j == i)
                continue;

            // Read from GSM if within budget, fall back to VRAM for overflow particles.
            uint   gsIdx   = gsOff + s;
            bool   inGSM   = gsIdx < CC_GSM_SIZE;
            float3 pj      = inGSM ? gs_posLambda[gsIdx].xyz : predictedPosition[j];
            float  lambdaJ = inGSM ? gs_posLambda[gsIdx].w   : lambda[j];
            float3 r       = pi - pj;
            float  r2      = dot(r, r);

            // Overlapping particles: normal gradient is zero, nudge apart instead.
            if (r2 < EPSILON * EPSILON)
            {
                deltaP += overlapJitter(i, j) * (H * 0.001);
                continue;
            }

            // Artificial pressure (Eq. 13): purely repulsive term to suppress tensile instability.
            float wRatio = Poly6(r, r2) / poly6AtDeltaQ;
            float sCorr  = -sCorrK * pow(wRatio, SCORR_N);

            // Position correction (Eq. 12 + 13).
            deltaP += (lambdaI + lambdaJ + sCorr) * SpikyGrad(r, r2);
        }
        gsOff += count;
    }
    deltaP /= RHO0;

    scratch[i] = pi + deltaP;
}

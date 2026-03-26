// Spatial grid helper functions for neighbor lookups.
//
// The simulation box is subdivided into cells of side length h (the SPH kernel radius).
// Each particle belongs to exactly one cell based on its position. To find all potential
// neighbors of a particle, we check its own cell and the 26 surrounding cells (3x3x3 block).
// Since cell size = h = the kernel support radius, this guarantees that every particle
// within distance h is found in one of those 27 cells.
//
// The grid is centered on the simulation box so that both the boxMin and boxMax
// boundaries are inset equally into their respective cells. Without centering,
// the grid origin was boxMin, placing boxMin exactly on a cell edge: adjacent-cell
// particles were at distance >= h (kernel contribution zero). Meanwhile boxMax
// fell mid-cell, giving its neighbors full kernel contributions. This asymmetry
// caused visibly worse boiling at the boxMin corner (-x, -z). Centering the grid
// ensures both boundaries have the same inset, making the neighbor contribution
// symmetric at all walls and corners.
//
// Must be included after the cbuffer declaration, as these functions reference
// boxMin, boxMax, and h from ComputeCb.

#ifndef GRID_UTILS_HLSLI
#define GRID_UTILS_HLSLI

// Computes the grid dimensions (number of cells along each axis) from the current
// box size and cell size h. This is done per-thread rather than passed in the CB
// because it's derivable from values already there (boxMin, boxMax, h).
int3 gridDims()
{
    return int3(ceil((boxMax - boxMin) / h));
}

// Grid origin: the grid is centered on the simulation box. dims * h >= boxExtent,
// so the excess (dims * h - boxExtent) is split equally on both sides.
float3 gridOrigin()
{
    float3 extent = boxMax - boxMin;
    float3 gridExtent = float3(gridDims()) * h;
    return boxMin - (gridExtent - extent) * 0.5;
}

// Maps a world-space position to its 3D grid cell coordinates.
// Clamped to [0, gridDims-1] so that particles exactly on the boundary
// don't index out of bounds.
int3 posToCell(float3 pos)
{
    int3 dims = gridDims();
    return clamp(int3((pos - gridOrigin()) / h), int3(0, 0, 0), dims - int3(1, 1, 1));
}

// Converts 3D cell coordinates to a flat 1D index.
// Row-major layout: X changes fastest, then Y, then Z.
uint cellIndex(int3 cell)
{
    int3 dims = gridDims();
    return (uint)cell.x + (uint)cell.y * dims.x + (uint)cell.z * dims.x * dims.y;
}

#endif // GRID_UTILS_HLSLI

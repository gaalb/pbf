// Uncomment to use Morton code (Z-order curve) cell indexing instead of row-major.
#define USE_MORTON_CODES

// Spatial grid helper functions for neighbor lookups.
//
// The simulation box is subdivided into cells of side length h (the SPH kernel radius).
// Each particle belongs to exactly one cell based on its position. To find all potential
// neighbors of a particle, we check its own cell and the 26 surrounding cells (3x3x3 block).
// Since cell size = h = the kernel support radius, this guarantees that every particle
// within distance h is found in one of those 27 cells.
//
// The grid dimensions are a power of two along each axis (equal on all three axes),
// and the box is sized to exactly match: boxExtent = gridDim * h. This means boxMin
// is exactly the grid origin — no centering offset is needed, and both boundaries
// are symmetric by construction.
//
// Two cell indexing schemes are available, selected by the USE_MORTON_CODES preprocessor
// define. Both expose the same interface: gridDim(), posToCell(), cellIndex().
//
// Row-major (default):
//   cellIndex = x + y * dim + z * dim * dim
//   Simple and predictable, but spatially distant cells can be adjacent in memory.
//
// Morton codes (Z-order curve, #define USE_MORTON_CODES before including):
//   cellIndex = bit-interleave(x, y, z)
//   Preserves 3D spatial locality better than row-major: cells close in 3D tend to
//   have close indices, so after the counting sort, particles in nearby cells end up
//   near each other in the buffer, improving GPU cache hit rates during the neighbor
//   loops in lambdaCS, deltaCS, vorticityCS, confinementCS, viscosityCS.
//   Requires power-of-two grid dimensions for a dense index space (no wasted codes).
//
// Must be included after the cbuffer declaration, as these functions reference
// boxMin, boxMax, and h from ComputeCb.

#ifndef GRID_UTILS_HLSLI
#define GRID_UTILS_HLSLI

// ─── Common functions (shared by both indexing schemes) ───────────────────────

// Computes the grid dimension (number of cells along each axis) from the
// box size and cell size h. Since the box is constructed as gridDim * h on
// each axis, this always yields a clean integer (the same power of two).
int gridDim()
{
    return int((boxMax.x - boxMin.x) / h + 0.5);
}

// Maps a world-space position to its 3D grid cell coordinates.
// The grid origin is boxMin (no centering offset needed since box = gridDim * h).
// Clamped to [0, gridDim-1] so particles exactly on the boundary don't index out of bounds.
int3 posToCell(float3 pos)
{
    int dim = gridDim();
    return clamp(int3((pos - boxMin) / h), int3(0, 0, 0), int3(dim - 1, dim - 1, dim - 1));
}

// ─── Cell indexing ────────────────────────────────────────────────────────────

#ifdef USE_MORTON_CODES

// Spreads the 10 low bits of x into every-third-bit positions.
// E.g. input bits  ...9876543210
//      output bits ...9..8..7..6..5..4..3..2..1..0
// This is the standard "magic number" bit-interleave used in Morton code construction.
// Each step doubles the spacing between bits by shifting and masking with a pattern
// that isolates the bits that need to move.
uint expandBits(uint x)
{
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x <<  8)) & 0x0300F00F;
    x = (x | (x <<  4)) & 0x030C30C3;
    x = (x | (x <<  2)) & 0x09249249;
    return x;
}

// Converts 3D cell coordinates to a Morton code (Z-order curve index).
// Interleaves the bits of x, y, z: the resulting uint has bits
// ...z9 y9 x9 z8 y8 x8 ... z1 y1 x1 z0 y0 x0
// This preserves 3D spatial locality better than row-major, so cells close
// in 3D tend to have close indices and their particles end up nearby in the
// sorted buffer.
uint cellIndex(int3 cell)
{
    return expandBits((uint)cell.x)
         | (expandBits((uint)cell.y) << 1)
         | (expandBits((uint)cell.z) << 2);
}

#else // Row-major indexing (default)

// Converts 3D cell coordinates to a flat 1D index.
// Row-major layout: X changes fastest, then Y, then Z.
uint cellIndex(int3 cell)
{
    int dim = gridDim();
    return (uint)cell.x + (uint)cell.y * dim + (uint)cell.z * dim * dim;
}

#endif // USE_MORTON_CODES

#endif // GRID_UTILS_HLSLI

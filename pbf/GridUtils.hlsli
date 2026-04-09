#include "SharedConfig.hlsli"

// Uncomment to use Morton code (Z-order curve) cell indexing instead of row-major.
#define USE_MORTON_CODES

// Spatial grid helper functions for neighbor lookups.
//
// The simulation box is subdivided into cells of side length H/CELL_PER_H.
// Each particle belongs to exactly one cell based on its position. To find all potential
// neighbors of a particle, we check its own cell and the surrounding cells in a
// (2*CELL_PER_H+1)^3 block (±CELL_PER_H in each axis). Since a particle
// up to H away is at most CELL_PER_H cells away per axis, this guarantees every
// particle within the kernel support distance H is found in one of those checked cells.
//
// The grid dimensions are a power of two along each axis (equal on all three axes),
// and the box is sized to exactly match: boxExtent = GRID_DIM * (H/CELL_PER_H).
// This means boxMin is exactly the grid origin — no centering offset is needed, and both
// boundaries are symmetric by construction.
// With CELL_PER_H=1 (default), this reduces to the original cell size = H.
//
// Two cell indexing schemes are available, selected by the USE_MORTON_CODES preprocessor
// define. Both expose the same interface: posToCell(), cellIndex().
//
// Row-major (default):
//   cellIndex = x + y * GRID_DIM + z * GRID_DIM * GRID_DIM
//
// Morton codes (Z-order curve, #define USE_MORTON_CODES before including):
//   cellIndex = bit-interleave(x, y, z)

#ifndef GRID_UTILS_HLSLI
#define GRID_UTILS_HLSLI

// Compile-time grid origin and extent (world space), derived from GRID_DIM, H, CELL_PER_H.
// These replace the former gridMin/gridMax fields that were in the constant buffer.
static const float3 GRID_MIN = float3(-BOX_HALF_EXTENT, -BOX_HALF_EXTENT, -BOX_HALF_EXTENT);
static const float3 GRID_MAX = float3( BOX_HALF_EXTENT,  BOX_HALF_EXTENT,  BOX_HALF_EXTENT);

// Maps a world-space position to its 3D grid cell coordinates.
// Dividing (pos - GRID_MIN) by H gives position in H-units; multiplying by CELL_PER_H
// converts to cell-units where each cell is H/CELL_PER_H wide.
// Clamped to [0, GRID_DIM-1] so particles exactly on the boundary don't index out of bounds.
int3 posToCell(float3 pos)
{
    return clamp(int3((pos - GRID_MIN) / H * float(CELL_PER_H)), int3(0, 0, 0), int3(GRID_DIM - 1, GRID_DIM - 1, GRID_DIM - 1));
}

#ifdef USE_MORTON_CODES

// Morton Codes preserve 3D spatial locality better than row-major: cells close in 3D tend to
// have close indices, so after the counting sort, particles in nearby cells end up
// near each other in the buffer, improving GPU cache hit rates during the neighbor
// loops in lambdaCS, deltaCS, vorticityCS, confinementCS, viscosityCS.
// In order for this to produce a dense index space - meaning every integer from 0
// to numCells-1 maps to exactly one cell with no gaps - we need the number of
// reachable Morton Codes to equal the number of cells. A Morton code for a 3D cell
// (x, y, z) is formed by bit-interleaving the three coordinates: z1 y1 x1 z0 y0 x0.
// If each coordinate has a range that's a power of two, say, 32=2^5, each coordinate
// fits exactly 5 bits without waste. This means that interleaving three 5-bit values
// gives a 15-bit result in range 0..2^15-1, so each code maps to a valid cell in the
// index range 0..32767. If gridDim isn't a power of two, let's say it's 30, then
// each dimension still requires 5 bits but only uses 0..29, leaving 30 and 31 unused.
// This means that when you interleave, combinations that include 30 or 31 for any of
// the coordiantes don't map to any actual cells. The code still spans to 0..32767 but
// it will be interspersed with invalid codes. This is an issue because cellCount and
// cellPrefixSum are indexed by Morton codes directly, so they must be allocated large
// enough to cover the full code range. With power of two dimensions, that allocation size
// equals gridDim^3 exactly without waste, but with a non power of two dimension, we'd
// need to over-allocate to the next power of two, cubed, and only use gridDim^3 of them.
// In addition, every cell lookup would need bounds checks to not look in the "holes",
// becuase the holes containing invalid values aren't at the end of the index range, but
// interspersed in the middle.



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
    return (uint)cell.x + (uint)cell.y * GRID_DIM + (uint)cell.z * GRID_DIM * GRID_DIM;
}

#endif // USE_MORTON_CODES

// Maximum number of cells to check for neighbors: (2*CELL_PER_H+1)^3.
// Each cell is H/CELL_PER_H wide, so a particle up to H away is at most
// CELL_PER_H cells away per axis, requiring a (2i+1)^3 search volume.
#define NEIGHBOR_CELL_COUNT \
    ((2*CELL_PER_H+1)*(2*CELL_PER_H+1)*(2*CELL_PER_H+1))

struct NeighborCells {
    uint indices[NEIGHBOR_CELL_COUNT];
    uint count;
};

// Given a world-space position, returns the flat cell indices of all valid
// (in-bounds) neighboring cells.
// TODO: when GRID_DIM is large enough, drop the cells on the very corners
NeighborCells NeighborCellIndices(float3 pos)
{
    NeighborCells result;
    result.count = 0;
    int3 myCell = posToCell(pos);
    for (int dz = -CELL_PER_H; dz <= CELL_PER_H; dz++)
    for (int dy = -CELL_PER_H; dy <= CELL_PER_H; dy++)
    for (int dx = -CELL_PER_H; dx <= CELL_PER_H; dx++)
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

#endif // GRID_UTILS_HLSLI

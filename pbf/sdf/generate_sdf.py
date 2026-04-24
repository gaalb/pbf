"""
Offline SDF (Signed Distance Field) generator for PBF solid obstacles.

Loads a mesh, samples a regular 3D grid of signed distances in the mesh's
own coordinate space, and saves a compact binary .sdf file for the PBF runtime.

Setup (run once, inside this folder's venv):
    pip install .

Usage:
    python generate_sdf.py <mesh_path> <output.sdf> [--res N] [--padding F]

    mesh_path   - any format trimesh can load: OBJ, FBX, GLB, STL, PLY, ...
    output.sdf  - binary output file consumed by SolidObstacle::Load()
    --res N     - voxel grid resolution (default 64, meaning 64^3 voxels)
    --padding F - fractional padding around the mesh AABB (default 0.1 = 10%)

Output binary format (.sdf):
    int32   nx, ny, nz          grid dimensions (all equal to --res)
    float32 sdfMin[3]           object-space AABB min (x, y, z)
    float32 sdfMax[3]           object-space AABB max (x, y, z)
    float32 data[nz*ny*nx * 4]  interleaved float4 per voxel: (d, gx, gy, gz)
                                z-major layout (x varies fastest, matches D3D12 Texture3D)
                                d  : signed distance (negative = inside, positive = outside)
                                gx,gy,gz : outward unit gradient in object space

Notes:
  - Uses trimesh's BVH-accelerated closest-point query for unsigned distances,
    then ray-casting for sign determination. Works best on watertight meshes.
    Non-watertight meshes may produce sign errors near open boundaries.
  - The SDF is in the mesh's original coordinate space (same units as the OBJ).
    Place the SolidObstacle in world space by calling SetTransform() at runtime.
"""

import argparse
import struct
import numpy as np
import trimesh
import trimesh.proximity
from tqdm import tqdm

_CHUNK = 1 << 16   # 65 536 query points per batch — keeps peak temp alloc ≈ 200 MB


def generate_sdf(mesh_path: str, output_path: str, resolution: int, padding_factor: float) -> None:
    print(f"Loading mesh: {mesh_path}")
    scene_or_mesh = trimesh.load(mesh_path, force='mesh')

    # Some formats (GLB, DAE) produce a Scene with multiple sub-meshes; merge them.
    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = list(scene_or_mesh.geometry.values())
        if not meshes:
            raise ValueError("Scene contains no geometry")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene_or_mesh

    print(f"  Vertices : {len(mesh.vertices):,}")
    print(f"  Faces    : {len(mesh.faces):,}")
    print(f"  Watertight: {mesh.is_watertight}")

    bounds_min = mesh.bounds[0].astype(np.float64)
    bounds_max = mesh.bounds[1].astype(np.float64)
    extent = bounds_max - bounds_min
    print(f"  Bounds   : {bounds_min}  to  {bounds_max}")
    print(f"  Extent   : {extent}")

    # Padded SDF bounding region (stored in the file for runtime UVW mapping)
    padding = np.maximum(extent * padding_factor, 1e-4)
    sdf_min = (bounds_min - padding).astype(np.float32)
    sdf_max = (bounds_max + padding).astype(np.float32)
    print(f"  SDF region after padding: {sdf_min}  to  {sdf_max}")

    N = resolution
    # Sample at texel center positions, not at the sdfMin/sdfMax endpoints.
    # A Texture3D with N texels along an axis places texel k's center at UV
    # (k + 0.5) / N.  The shader maps position P to UV = (P - sdfMin) / (sdfMax - sdfMin),
    # so texel k's center corresponds to position sdfMin + (k + 0.5)/N * (sdfMax - sdfMin).
    # Using np.linspace(sdfMin, sdfMax, N) would place data at k/(N-1) fractions instead,
    # creating a half-texel offset that shifts sampled distances by ~0.5 voxels — enough
    # to cause visible collision penetration at typical grid resolutions.
    half_voxel = 0.5 * (sdf_max - sdf_min) / N
    xs = np.linspace(float(sdf_min[0] + half_voxel[0]), float(sdf_max[0] - half_voxel[0]), N)
    ys = np.linspace(float(sdf_min[1] + half_voxel[1]), float(sdf_max[1] - half_voxel[1]), N)
    zs = np.linspace(float(sdf_min[2] + half_voxel[2]), float(sdf_max[2] - half_voxel[2]), N)

    # Build query points in D3D12 Texture3D layout: z outermost, x innermost.
    # After meshgrid with indexing='ij', arrays have shape (nz, ny, nx) so that
    # C-order ravel gives z-major order (x varies fastest) — exactly what D3D12 expects.
    ZZ, YY, XX = np.meshgrid(zs, ys, xs, indexing='ij')        # each shape (nz, ny, nx)
    query_points = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)  # (N^3, 3)

    total_pts = len(query_points)
    print(f"Computing SDF for {N}^3 = {N**3:,} query points...")

    closest_pts = np.empty((total_pts, 3), dtype=np.float64)
    distances   = np.empty(total_pts,      dtype=np.float64)
    inside      = np.empty(total_pts,      dtype=bool)

    with tqdm(total=total_pts, desc="  Computing", unit="vox", unit_scale=True) as pbar:
        for lo in range(0, total_pts, _CHUNK):
            hi    = min(lo + _CHUNK, total_pts)
            chunk = query_points[lo:hi]
            cp, d, _ = trimesh.proximity.closest_point(mesh, chunk)
            closest_pts[lo:hi] = cp
            distances[lo:hi]   = d
            inside[lo:hi]      = mesh.contains(chunk)
            pbar.update(hi - lo)

    sdf_values = np.where(inside, -distances, distances).astype(np.float32)

    n_inside = int(inside.sum())
    print(f"  Distance range : [{sdf_values.min():.4f}, {sdf_values.max():.4f}]")
    print(f"  Inside voxels  : {n_inside:,} / {N**3:,}  ({100.0 * n_inside / N**3:.1f}%)")

    # Geometric gradient: vector from closest surface point to query voxel, normalised.
    # For exterior voxels this already points outward; for interior voxels it points
    # inward (into the solid), so we flip with the sign of the SDF.
    direction = query_points - closest_pts                          # shape (N³, 3)
    norm      = np.linalg.norm(direction, axis=-1, keepdims=True)  # shape (N³, 1)
    sign      = np.where(inside, -1.0, 1.0)[:, np.newaxis]        # +1 outside, -1 inside
    gradient  = (sign * direction / np.maximum(norm, 1e-8)).astype(np.float32)  # shape (N³, 3)

    # Pack into float4 per voxel: (d, gx, gy, gz), z-major (x varies fastest)
    data = np.stack([sdf_values, gradient[:, 0], gradient[:, 1], gradient[:, 2]], axis=1)
    # data shape: (N³, 4) — C-order write gives the correct z-major interleaved layout

    with open(output_path, 'wb') as f:
        f.write(struct.pack('<iii', N, N, N))
        f.write(struct.pack('<fff', float(sdf_min[0]), float(sdf_min[1]), float(sdf_min[2])))
        f.write(struct.pack('<fff', float(sdf_max[0]), float(sdf_max[1]), float(sdf_max[2])))
        data.tofile(f)

    header_bytes = 3 * 4 + 3 * 4 + 3 * 4          # 36 bytes
    data_bytes   = N * N * N * 4 * 4               # 4 floats per voxel
    total_kb     = (header_bytes + data_bytes) / 1024
    print(f"Written {total_kb:.1f} KB  →  {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a binary SDF file from a mesh for PBF solid-obstacle collision.")
    parser.add_argument("mesh_path",   help="Input mesh file (OBJ, FBX, GLB, STL, ...)")
    parser.add_argument("output_path", help="Output .sdf file")
    parser.add_argument("--res",     type=int,   default=64,  metavar="N",
                        help="Grid resolution (default: 64 → 64^3 voxels)")
    parser.add_argument("--padding", type=float, default=0.1, metavar="F",
                        help="Fractional AABB padding (default: 0.1 = 10%%)")
    args = parser.parse_args()

    generate_sdf(args.mesh_path, args.output_path, args.res, args.padding)


if __name__ == "__main__":
    main()
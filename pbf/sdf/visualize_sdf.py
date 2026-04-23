"""
Visualize an existing .sdf file as a 3D zero-isosurface.

Shows the d=0 level set extracted from the signed distance field via marching
cubes. Use this to sanity-check two things:
  1. Does the shape match the original mesh?
  2. Is the interior sealed? (holes will appear as gaps or missing patches)

Usage:
    python visualize_sdf.py <input.sdf>

Requires:
    pip install scikit-image matplotlib
"""

import argparse
import struct
import sys

import numpy as np


def _load(path: str):
    with open(path, 'rb') as f:
        nx, ny, nz = struct.unpack('<iii', f.read(12))
        sdf_min = np.array(struct.unpack('<fff', f.read(12)), dtype=np.float32)
        sdf_max = np.array(struct.unpack('<fff', f.read(12)), dtype=np.float32)
        raw = np.frombuffer(f.read(), dtype=np.float32)

    expected = nz * ny * nx * 4
    if len(raw) != expected:
        raise ValueError(f"Expected {expected} floats, got {len(raw)} — file may be corrupt.")

    # Layout: z-major interleaved float4 (d, gx, gy, gz) per voxel.
    # Reshape to (nz, ny, nx, 4) then take channel 0 (signed distance).
    distances = np.ascontiguousarray(raw.reshape(nz, ny, nx, 4)[:, :, :, 0])
    return distances, sdf_min, sdf_max


def visualize(sdf_path: str) -> None:
    try:
        from skimage.measure import marching_cubes
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("Requires scikit-image and matplotlib:")
        print("  pip install scikit-image matplotlib")
        sys.exit(1)

    print(f"Loading: {sdf_path}")
    distances, sdf_min, sdf_max = _load(sdf_path)
    nz, ny, nx = distances.shape
    print(f"  Grid  : {nx} x {ny} x {nz}")
    print(f"  Bounds: {sdf_min}  to  {sdf_max}")
    print(f"  Range : [{distances.min():.4f}, {distances.max():.4f}]")

    dx = (float(sdf_max[0]) - float(sdf_min[0])) / nx
    dy = (float(sdf_max[1]) - float(sdf_min[1])) / ny
    dz = (float(sdf_max[2]) - float(sdf_min[2])) / nz

    print("Extracting isosurface...")
    try:
        verts, faces, _, _ = marching_cubes(
            distances, level=0.0, spacing=(dz, dy, dx), allow_degenerate=False)
    except (ValueError, TypeError):
        try:
            verts, faces, _, _ = marching_cubes(distances, level=0.0, spacing=(dz, dy, dx))
        except ValueError:
            print("No zero-crossing found — interior may be empty or sign determination failed.")
            sys.exit(1)

    # distances axes: (z, y, x) → marching_cubes verts are (z-offset, y-offset, x-offset).
    world = np.column_stack([
        float(sdf_min[0]) + verts[:, 2],
        float(sdf_min[1]) + verts[:, 1],
        float(sdf_min[2]) + verts[:, 0],
    ])
    print(f"  {len(verts):,} vertices, {len(faces):,} faces")

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')
    poly = Poly3DCollection(world[faces], alpha=0.85, linewidth=0)
    poly.set_facecolor('steelblue')
    ax.add_collection3d(poly)

    ax.set_xlim(float(sdf_min[0]), float(sdf_max[0]))
    ax.set_ylim(float(sdf_min[1]), float(sdf_max[1]))
    ax.set_zlim(float(sdf_min[2]), float(sdf_max[2]))
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f'SDF zero-isosurface: {sdf_path}')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a .sdf file as a 3D isosurface.")
    parser.add_argument("sdf_path", help="Input .sdf file")
    args = parser.parse_args()
    visualize(args.sdf_path)


if __name__ == "__main__":
    main()

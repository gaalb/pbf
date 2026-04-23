"""
Offline SDF (Signed Distance Field) generator for PBF solid obstacles.

Loads a mesh, repairs topology, samples a regular 3D grid of signed distances
in the mesh's own coordinate space, and saves a compact binary .sdf file for
the PBF runtime.

Setup (run once, inside this folder's venv):
    pip install trimesh scipy scikit-image matplotlib tqdm

Usage:
    python generate_sdf.py <mesh_path> <output.sdf> [--res N] [--padding F] [--skin F] [--no-preview]

    mesh_path    - any format trimesh can load: OBJ, FBX, GLB, STL, PLY, ...
    output.sdf   - binary output file consumed by SolidObstacle::Load()
    --res N      - voxel grid resolution (default 128, meaning 128^3 voxels)
    --padding F  - fractional padding around the mesh AABB (default 0.1 = 10%)
    --skin F     - dilation in mesh units (default 0, see Notes below)
    --no-preview - skip the 3D isosurface preview after generation

Output binary format (.sdf):
    int32   nx, ny, nz          grid dimensions (all equal to --res)
    float32 sdfMin[3]           object-space AABB min (x, y, z)
    float32 sdfMax[3]           object-space AABB max (x, y, z)
    float32 data[nz*ny*nx * 4]  interleaved float4 per voxel: (d, gx, gy, gz)
                                z-major layout (x varies fastest, matches D3D12 Texture3D)
                                d  : signed distance (negative = inside, positive = outside)
                                gx,gy,gz : outward unit gradient in object space

Notes:
  - Unsigned distances are computed via BVH-accelerated closest-point queries,
    processed in batches to keep memory usage bounded.

  - Sign (inside/outside) is determined by a 6-connectivity flood-fill from all
    bounding-box faces rather than ray-casting. This seals mesh gaps up to ~1 voxel
    wide and is robust against non-watertight topology.

  - --skin F (dilation / caulking):
    At a sharp concave edge — e.g. the inner corner where the bottom of a slide
    meets its side wall — the SDF gradient is discontinuous. The field correctly
    reports the distance to the nearest surface point, but because two faces meet
    at a point, it only enforces one face's normal at a time. A fast-moving
    particle can slip through such an edge between timesteps.

    --skin subtracts F from every voxel's signed distance, shifting the d=0
    surface outward by F units uniformly. At a concave corner this fills the
    sharp notch with a rounded fillet of radius F (like running a bead of caulk
    into the corner), giving the collision response a smooth gradient to work
    against rather than a knife-edge discontinuity.

    The cost is that every interior cavity shrinks by F units on all sides.
    A good starting value is 0.5–1.0 × voxel size (= extent / res). Use
    --no-preview to regenerate quickly while tuning, then drop --no-preview
    to inspect the result: the isosurface will visibly pull away from sharp
    inner edges by roughly F units.

  - The SDF is in the mesh's original coordinate space (same units as the OBJ).
    Place the SolidObstacle in world space by calling SetTransform() at runtime.
"""

import argparse
import struct

import numpy as np
import trimesh
import trimesh.proximity
import trimesh.repair

try:
    from tqdm import tqdm as _tqdm
    def _progress(it, **kw):
        return _tqdm(it, **kw)
except ImportError:
    def _progress(it, **kw):
        return it

# Number of query points processed per closest-point batch.
# Keeps the BVH candidate expansion array well under 1 GB even for dense meshes.
_BATCH = 50_000


def _load_and_repair(mesh_path: str) -> trimesh.Trimesh:
    raw = trimesh.load(mesh_path, force='mesh', process=False)
    if isinstance(raw, trimesh.Scene):
        parts = list(raw.geometry.values())
        if not parts:
            raise ValueError("Scene contains no geometry")
        mesh = trimesh.util.concatenate(parts)
    else:
        mesh = raw

    # OBJ (and similar formats) splits one geometric vertex into multiple records
    # when adjacent faces reference it with different UV / normal indices.
    # Merging back to unique positions fixes the phantom "open edges" that cause
    # is_watertight to return False even on visually closed meshes.
    mesh.merge_vertices()
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fill_holes(mesh)
    return mesh


def _unsigned_distances(mesh: trimesh.Trimesh, pts: np.ndarray):
    """Return (distances float32 (N,), closest_pts float32 (N,3)) via batched BVH queries."""
    n = len(pts)
    dists   = np.empty(n,       dtype=np.float32)
    closest = np.empty((n, 3),  dtype=np.float32)

    batches = list(range(0, n, _BATCH))
    for start in _progress(batches, desc="  distances", unit="batch"):
        end = min(start + _BATCH, n)
        cp, d, _ = trimesh.proximity.closest_point(mesh, pts[start:end])
        dists[start:end]   = d
        closest[start:end] = cp

    return dists, closest


def _flood_fill_outside(unsigned_grid: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    6-connectivity flood-fill from all six bounding-box faces.
    Returns boolean array of shape (nz, ny, nx): True = outside (positive SDF).

    A voxel is passable only if its unsigned distance exceeds half a voxel,
    which blocks any mesh gap narrower than one voxel from leaking the fill.
    """
    from scipy import ndimage

    walkable = unsigned_grid > (voxel_size * 0.5)

    struct6 = ndimage.generate_binary_structure(3, 1)   # 6-connectivity
    labeled, _ = ndimage.label(walkable, structure=struct6)

    # Any connected component that touches a face of the bounding box is exterior.
    outside_labels = set()
    outside_labels.update(np.unique(labeled[0,  :,  :]))
    outside_labels.update(np.unique(labeled[-1, :,  :]))
    outside_labels.update(np.unique(labeled[:,  0,  :]))
    outside_labels.update(np.unique(labeled[:, -1,  :]))
    outside_labels.update(np.unique(labeled[:, :,   0]))
    outside_labels.update(np.unique(labeled[:, :,  -1]))
    outside_labels.discard(0)   # 0 = non-walkable (surface or interior)

    return np.isin(labeled, list(outside_labels))


def _visualize(title: str, distances: np.ndarray,
               sdf_min: np.ndarray, sdf_max: np.ndarray) -> None:
    try:
        from skimage.measure import marching_cubes
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("  Preview requires scikit-image and matplotlib — skipping.")
        print("  Install with: pip install scikit-image matplotlib")
        return

    nz, ny, nx = distances.shape
    dx = (float(sdf_max[0]) - float(sdf_min[0])) / nx
    dy = (float(sdf_max[1]) - float(sdf_min[1])) / ny
    dz = (float(sdf_max[2]) - float(sdf_min[2])) / nz

    try:
        verts, faces, _, _ = marching_cubes(
            distances, level=0.0, spacing=(dz, dy, dx), allow_degenerate=False)
    except (ValueError, TypeError):
        try:
            verts, faces, _, _ = marching_cubes(distances, level=0.0, spacing=(dz, dy, dx))
        except ValueError:
            print("  Marching cubes found no zero-crossing — interior may be empty.")
            return

    # distances has axes (z, y, x), so marching_cubes verts are in (z, y, x) offsets.
    world = np.column_stack([
        float(sdf_min[0]) + verts[:, 2],
        float(sdf_min[1]) + verts[:, 1],
        float(sdf_min[2]) + verts[:, 0],
    ])
    print(f"  Isosurface: {len(verts):,} verts, {len(faces):,} faces")

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')
    poly = Poly3DCollection(world[faces], alpha=0.85, linewidth=0)
    poly.set_facecolor('steelblue')
    ax.add_collection3d(poly)

    ax.set_xlim(float(sdf_min[0]), float(sdf_max[0]))
    ax.set_ylim(float(sdf_min[1]), float(sdf_max[1]))
    ax.set_zlim(float(sdf_min[2]), float(sdf_max[2]))
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def generate_sdf(mesh_path: str, output_path: str, resolution: int,
                 padding_factor: float, skin: float, preview: bool) -> None:
    print(f"Loading mesh: {mesh_path}")
    mesh = _load_and_repair(mesh_path)

    print(f"  Vertices  : {len(mesh.vertices):,}")
    print(f"  Faces     : {len(mesh.faces):,}")
    print(f"  Watertight: {mesh.is_watertight}")

    bounds_min = mesh.bounds[0].astype(np.float64)
    bounds_max = mesh.bounds[1].astype(np.float64)
    extent     = bounds_max - bounds_min
    print(f"  Bounds    : {bounds_min}  to  {bounds_max}")
    print(f"  Extent    : {extent}")

    padding = np.maximum(extent * padding_factor, 1e-4)
    sdf_min = (bounds_min - padding).astype(np.float32)
    sdf_max = (bounds_max + padding).astype(np.float32)
    print(f"  SDF region: {sdf_min}  to  {sdf_max}")

    N = resolution
    half_voxel = 0.5 * (sdf_max - sdf_min) / N
    xs = np.linspace(float(sdf_min[0] + half_voxel[0]), float(sdf_max[0] - half_voxel[0]), N)
    ys = np.linspace(float(sdf_min[1] + half_voxel[1]), float(sdf_max[1] - half_voxel[1]), N)
    zs = np.linspace(float(sdf_min[2] + half_voxel[2]), float(sdf_max[2] - half_voxel[2]), N)

    # Build query points in z-major (x varies fastest) order to match D3D12 Texture3D layout.
    ZZ, YY, XX = np.meshgrid(zs, ys, xs, indexing='ij')
    pts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)

    print(f"Computing SDF for {N}^3 = {N**3:,} query points  (batch {_BATCH:,})...")
    dists, closest = _unsigned_distances(mesh, pts)

    print("  Determining inside/outside via flood-fill...")
    voxel_size = float(np.max((sdf_max - sdf_min) / N))
    outside = _flood_fill_outside(dists.reshape(N, N, N), voxel_size)
    inside  = ~outside.ravel()

    sdf_vals = np.where(inside, -dists, dists).astype(np.float32)

    if skin != 0.0:
        # Dilation: shift every distance value down by `skin` units, moving the
        # zero-crossing outward. At concave inner edges this fills in the corner
        # like caulk, giving particles a safe margin before reaching the edge.
        sdf_vals -= np.float32(skin)
        print(f"  Skin/dilation  : {skin:.4f} units applied")

    n_inside = int((sdf_vals < 0).sum())
    print(f"  Distance range : [{sdf_vals.min():.4f}, {sdf_vals.max():.4f}]")
    print(f"  Inside voxels  : {n_inside:,} / {N**3:,}  ({100.0 * n_inside / N**3:.1f}%)")

    # Gradient: unit vector pointing outward from the nearest surface point.
    direction = pts - closest
    norm      = np.linalg.norm(direction, axis=-1, keepdims=True)
    sign      = np.where(inside, -1.0, 1.0)[:, np.newaxis]
    gradient  = (sign * direction / np.maximum(norm, 1e-8)).astype(np.float32)

    data = np.stack([sdf_vals, gradient[:, 0], gradient[:, 1], gradient[:, 2]], axis=1)

    with open(output_path, 'wb') as f:
        f.write(struct.pack('<iii', N, N, N))
        f.write(struct.pack('<fff', float(sdf_min[0]), float(sdf_min[1]), float(sdf_min[2])))
        f.write(struct.pack('<fff', float(sdf_max[0]), float(sdf_max[1]), float(sdf_max[2])))
        data.tofile(f)

    total_kb = (36 + N**3 * 16) / 1024
    print(f"Written {total_kb:.1f} KB  →  {output_path}")

    if preview:
        print("Rendering preview...")
        _visualize(output_path, sdf_vals.reshape(N, N, N), sdf_min, sdf_max)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a binary SDF file from a mesh for PBF solid-obstacle collision.")
    parser.add_argument("mesh_path",    help="Input mesh file (OBJ, FBX, GLB, STL, ...)")
    parser.add_argument("output_path",  help="Output .sdf file")
    parser.add_argument("--res",        type=int,   default=128, metavar="N",
                        help="Grid resolution (default: 128 → 128^3 voxels)")
    parser.add_argument("--padding",    type=float, default=0.1, metavar="F",
                        help="Fractional AABB padding (default: 0.1 = 10%%)")
    parser.add_argument("--skin",       type=float, default=0.0, metavar="F",
                        help="Dilation in mesh units (default: 0). Subtracts F from every "
                             "SDF value, expanding the solid outward and caulking concave "
                             "inner edges. Start around 0.5× voxel size.")
    parser.add_argument("--no-preview", action="store_true",
                        help="Skip the 3D isosurface preview after generation")
    args = parser.parse_args()

    generate_sdf(args.mesh_path, args.output_path, args.res, args.padding,
                 args.skin, preview=not args.no_preview)


if __name__ == "__main__":
    main()

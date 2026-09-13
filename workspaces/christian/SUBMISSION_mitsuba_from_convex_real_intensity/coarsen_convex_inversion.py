"""
Coarsen a raw convex-inversion (simulated intensity) reconstruction STL to
~1000 vertices, no other remeshing -- same recipe used for asteroids 1-3
this session (gpytoolbox.remesh_botsch isotropic remeshing, edge-length
factor tuned per-mesh to land near the target vertex count), which
consistently outperformed finer/full-resolution meshes as an initial guess
on the official challenge metric.

Usage: python coarsen_convex_inversion.py <model_num> [<model_num> ...]

Reads asteroid{model_num}_simulated_intensity.stl from CONVEX_INVERSION_DIR
(env var, defaults to the location used in this project).
"""

import os
import sys
from pathlib import Path

import numpy as np
import trimesh
from gpytoolbox import remesh_botsch

SOURCE_DIR = Path(
    os.environ.get("CONVEX_INVERSION_DIR", "/zhome/95/1/155570/HAC-DTU/HAC-collab/reconstructions/convex/intensity")
)
OUTPUT_DIR = Path(__file__).parent / "coarse_initial_guess"
TARGET_VERTEX_COUNT = 1000
TARGET_TOLERANCE = 150  # accept 850-1150 vertices


def coarsen(model_num: int) -> None:
    src_path = SOURCE_DIR / f"asteroid{model_num}_simulated_intensity.stl"
    mesh = trimesh.load_mesh(src_path, force="mesh")
    v, f = mesh.vertices.copy(), mesh.faces.copy()
    mean_edge = mesh.edges_unique_length.mean()

    factor = 1.0
    best = None
    for _ in range(25):
        nv, nf = remesh_botsch(v, f, i=5, h=mean_edge * factor, project=True)
        candidate = trimesh.Trimesh(vertices=nv, faces=nf, process=False)
        n_verts = len(candidate.vertices)
        if best is None or abs(n_verts - TARGET_VERTEX_COUNT) < abs(best[0] - TARGET_VERTEX_COUNT):
            best = (n_verts, candidate, factor)
        if abs(n_verts - TARGET_VERTEX_COUNT) <= TARGET_TOLERANCE:
            break
        # too many vertices -> coarsen more (bigger edge length); too few -> finer
        factor *= 1.15 if n_verts > TARGET_VERTEX_COUNT else 0.9

    n_verts, candidate, used_factor = best
    watertight = candidate.is_watertight
    winding_consistent = candidate.is_winding_consistent
    vol_ratio = candidate.volume / candidate.convex_hull.volume if candidate.is_watertight else float("nan")

    out_path = OUTPUT_DIR / f"asteroid{model_num}_simulated_intensity_coarse1000.stl"
    candidate.export(out_path)
    print(
        f"asteroid{model_num}: {n_verts}v, {len(candidate.faces)}f, factor={used_factor:.3f}, "
        f"watertight={watertight}, winding_consistent={winding_consistent}, "
        f"vol/hull={vol_ratio:.4f} -> {out_path}"
    )
    if not (watertight and winding_consistent):
        print(f"  WARNING: asteroid{model_num} coarse mesh is not clean (watertight/winding)")


if __name__ == "__main__":
    model_nums = [int(a) for a in sys.argv[1:]]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for n in model_nums:
        coarsen(n)

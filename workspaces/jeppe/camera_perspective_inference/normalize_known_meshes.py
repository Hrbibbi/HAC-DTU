#!/usr/bin/env python3
"""Normalize the three public asteroid meshes for camera experiments.

The normalized meshes are written to
``WORKSPACE_DIR / "camera_perspective_inference" / "meshes"`` as
``model_1.stl`` through ``model_3.stl``.
"""

from pathlib import Path

import trimesh

from hac.constants import MODEL_BASE_RADII, WORKSPACE_DIR
from hac.data import get_model_stl_path
from hac.mesh import normalize_mesh_to_challenge_cylinder


KNOWN_MODEL_NUMBERS = (1, 2, 3)
OUTPUT_DIR = WORKSPACE_DIR / "camera_perspective_inference" / "meshes"


def normalize_known_mesh(model_num: int, output_dir: Path = OUTPUT_DIR) -> Path:
    """Normalize one public model and save it as a binary STL file."""
    source_path = get_model_stl_path(model_num)
    output_path = output_dir / f"model_{model_num}.stl"

    mesh = trimesh.load(source_path, force="mesh")
    normalized_mesh = normalize_mesh_to_challenge_cylinder(
        mesh,
        target_R=MODEL_BASE_RADII[model_num],
        verbose=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_mesh.export(output_path, file_type="stl")
    return output_path


def main() -> None:
    for model_num in KNOWN_MODEL_NUMBERS:
        output_path = normalize_known_mesh(model_num)
        print(f"Saved normalized model {model_num}: {output_path}")


if __name__ == "__main__":
    main()

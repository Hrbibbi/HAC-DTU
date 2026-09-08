from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import drjit as dr
import trimesh

from hac.config import config
from hac.mesh import convert_stl_to_obj
from hac.mitsuba.patch import mi


@dataclass(frozen=True, slots=True)
class MitsubaMesh:
    obj_path: Path

    @classmethod
    def from_stl_path(cls, stl_path: Path) -> MitsubaMesh:
        if config.store_temp_obj_files_in_tempdir:
            temp_dir = Path(tempfile.gettempdir())
            obj_path = temp_dir / stl_path.with_suffix(".obj").name
        else:
            obj_path = stl_path.with_suffix(".obj")

        convert_stl_to_obj(stl_path, obj_path)

        return cls(obj_path=obj_path)

    def to_filename(self) -> str:
        return str(self.obj_path)

    def to_trimesh(self) -> trimesh.Trimesh:
        mesh = trimesh.load_mesh(self.obj_path, file_type="obj")
        return mesh


def normalize_positions_to_challenge_cylinder(flat_positions, target_radius: float):
    """Differentiably normalize positions into the known challenge cylinder."""
    vertex_count = dr.width(flat_positions) // 3
    vertex_indices = dr.arange(mi.UInt, vertex_count)
    vertices = mi.Point3f(
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices),
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices + 1),
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices + 2),
    )

    x_center = 0.5 * (dr.min(vertices.x) + dr.max(vertices.x))
    y_center = 0.5 * (dr.min(vertices.y) + dr.max(vertices.y))
    centered_x = vertices.x - x_center
    centered_y = vertices.y - y_center
    radial_max = dr.max(dr.sqrt(dr.square(centered_x) + dr.square(centered_y)))

    z_min = dr.min(vertices.z)
    z_max = dr.max(vertices.z)
    z_center = 0.5 * (z_min + z_max)

    normalized_vertices = mi.Point3f(
        centered_x * (target_radius / radial_max),
        centered_y * (target_radius / radial_max),
        (vertices.z - z_center) * (2.0 / (z_max - z_min)),
    )
    return dr.ravel(normalized_vertices)

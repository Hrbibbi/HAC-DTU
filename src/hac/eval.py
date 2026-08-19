import numpy as np
import trimesh


def calculate_voxel_measure(
    true_mesh: trimesh.Trimesh, pred_mesh: trimesh.Trimesh, pitch: float = 0.02
) -> float:
    """
    Fast voxel-based similarity measure.

    Parameters
    ----------
    true_mesh : trimesh.Trimesh
        Ground truth mesh.

    pred_mesh : trimesh.Trimesh
        Predicted mesh.

    pitch : float
        Voxel size. Smaller values give higher accuracy but require
        more memory and computation.

    Returns
    -------
    float
        Similarity score in [0, 1].
    """
    # Voxelize both meshes
    vox_true = true_mesh.voxelized(pitch)
    vox_pred = pred_mesh.voxelized(pitch)

    # Get occupied voxel indices
    true_voxels = set(map(tuple, vox_true.sparse_indices))
    pred_voxels = set(map(tuple, vox_pred.sparse_indices))

    if len(true_voxels) == 0 and len(pred_voxels) == 0:
        return 0.0

    # Symmetric difference
    mismatch_count = len(true_voxels.symmetric_difference(pred_voxels))

    # Denominator from your original formula
    total_volume_sum = len(true_voxels) + len(pred_voxels)

    score = 1.0 - mismatch_count / total_volume_sum

    return np.clip(score, 0.0, 1.0)


def calculate_voxel_measure_solid(
    true_mesh: trimesh.Trimesh, pred_mesh: trimesh.Trimesh, pitch: float = 0.02
) -> float:
    """
    Solid-volume analogue of `calculate_voxel_measure`: same symmetric-
    difference / (|A|+|B|) Dice-style formula, but on
    `.voxelized(pitch).fill()` (interior-filled) voxel sets instead of the
    hollow surface shell.
    """
    true_voxels = set(map(tuple, true_mesh.voxelized(pitch).fill().sparse_indices))
    pred_voxels = set(map(tuple, pred_mesh.voxelized(pitch).fill().sparse_indices))
    if len(true_voxels) == 0 and len(pred_voxels) == 0:
        return 0.0
    mismatch_count = len(true_voxels.symmetric_difference(pred_voxels))
    total = len(true_voxels) + len(pred_voxels)
    return float(np.clip(1.0 - mismatch_count / total, 0.0, 1.0))

import time
from pathlib import Path

import numpy as np
import pyvista as pv
from skimage.filters import threshold_otsu
from tqdm.auto import tqdm

from hac.constants import *
from hac.data import get_model_stl_path
from hac.geometry import *


def simulate_rotation_frames(
    mesh,
    cam_idx,
    out_path=None,
    rot_angles=np.arange(360),
    show_edges=False,
    cam_dist=30,
    focal_point=(0, 0, 0),
    image_size=(512, 512),
    light_dist=1000,
    verbose=True,
):
    """
    Simulate grayscale rendered frames of a rotating object.

    If out_path is None, frames are stored in memory and returned as a NumPy array.
    If out_path is provided, frames are written to a .npy memmap file and also
    returned as a NumPy array.

    Returns
    -------
    np.ndarray
        Array with shape (n_frames, height, width) and dtype uint8.
    """
    rot_angles = np.asarray(rot_angles, dtype=float)
    focal_point = np.asarray(focal_point, dtype=float)

    width, height = image_size
    frames_shape = (len(rot_angles), height, width)

    old_multi_samples = pv.global_theme.multi_samples
    pv.global_theme.multi_samples = 0

    if out_path is None:
        frames = np.empty(frames_shape, dtype=np.uint8)
    else:
        out_path = Path(out_path)
        frames = np.lib.format.open_memmap(
            out_path,
            mode="w+",
            dtype=np.uint8,
            shape=frames_shape,
        )

    plotter = pv.Plotter(
        lighting="none",
        off_screen=True,
        window_size=image_size,
    )
    plotter.set_background("black")

    plotter.add_mesh(
        mesh,
        color="white",
        show_edges=show_edges,
        smooth_shading=False,
        ambient=0.0,
        diffuse=1.0,
        specular=0.0,
        pbr=True,
    )

    light = pv.Light(
        light_type="scene light",
        positional=False,
        intensity=1.0,
    )
    plotter.add_light(light)

    start = time.perf_counter()

    try:
        for i, rot_angle in enumerate(tqdm(rot_angles, desc=f"Rendering {cam_idx}")):
            direction = direction_from_angles(
                XY_ANGLES[cam_idx] + rot_angle,
                ELEV_ANGLES[cam_idx],
            )
            light_direction = direction_from_angles(rot_angle, 0)

            camera_position = focal_point + direction * cam_dist
            light_position = focal_point + light_direction * light_dist

            plotter.camera_position = [
                camera_position,
                focal_point,
                (0, 0, 1),
            ]

            # Orthographic projection (commented out as it gave worse results):
            # plotter.camera.parallel_projection = True
            # plotter.camera.parallel_scale = 6

            light.position = light_position
            light.focal_point = focal_point

            plotter.render()

            rgb = plotter.screenshot(
                return_img=True,
                transparent_background=False,
            )

            if rgb.shape[-1] != 3:
                raise ValueError(
                    f"Expected RGB image with 3 channels, got shape {rgb.shape}"
                )

            frames[i] = rgb[..., :3].mean(axis=-1).astype(np.uint8)

    finally:
        plotter.close()
        pv.global_theme.multi_samples = old_multi_samples

    if out_path is not None:
        frames.flush()

    elapsed = time.perf_counter() - start
    fps = len(rot_angles) / elapsed

    if verbose:
        if out_path is None:
            print("Generated frames in memory")
        else:
            print(f"Saved frames: {out_path}")

        print(f"Shape: {frames.shape}, dtype: {frames.dtype}")
        print(f"Elapsed: {elapsed:.2f} s")
        print(f"Speed: {fps:.2f} frames/s")

    return np.asarray(frames)

def generate_intensity_curve(frames: np.ndarray):
    thresholded_frames = np.where(
        frames > INTENSITY_CURVE_THRESHOLD,
        frames,
        0,
    )
    lightcurve = np.sum(thresholded_frames, axis=(1, 2)).astype(float)
    mean_intensity = lightcurve.mean()
    lightcurve = lightcurve / mean_intensity
    return lightcurve

def generate_binary_curve(frames: np.ndarray):
    binary_threshold = threshold_otsu(frames[0])
    thresholded_frames = frames > binary_threshold
    lightcurve = np.sum(thresholded_frames, axis=(1, 2)).astype(float)
    mean_intensity = lightcurve.mean()
    lightcurve = lightcurve / mean_intensity
    return lightcurve

def generate_lightcurves_from_mesh(
    mesh,
    cam_idx: str,
    out_dir: str | Path | None = None,
    rot_angles=np.arange(360),
    image_size=(1024, 1024),
    verbose=True,
) -> dict[str, np.ndarray]:
    """
    Generate HAC-style lightcurves from an already prepared mesh.

    The caller is responsible for ensuring that the mesh is in the desired
    coordinate system and scale. In particular, external meshes such as
    DAMIT models should be normalized before calling this function.

    Parameters
    ----------
    mesh
        PyVista mesh or trimesh.Trimesh.

    cam_idx
        Challenge observer/camera position.

    out_dir
        Optional directory in which the generated lightcurves are saved.

        Files are written as::

            cam_<cam_idx>_intensity.npy
            cam_<cam_idx>_binary.npy

    rot_angles
        Rotation angles used during simulation.

    image_size
        Rendering resolution.

    verbose
        Print rendering and output information.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing ``intensity`` and ``binary`` lightcurves.
    """
    frames = simulate_rotation_frames(
        mesh=mesh,
        cam_idx=cam_idx,
        rot_angles=rot_angles,
        show_edges=False,
        cam_dist=30,
        focal_point=(0, 0, 0),
        image_size=image_size,
        light_dist=1000,
        verbose=verbose,
    )

    intensity_curve = (
        generate_intensity_curve(
            frames
        )
    )

    binary_curve = (
        generate_binary_curve(
            frames
        )
    )

    if out_dir is not None:
        out_dir = Path(out_dir)

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        intensity_path = (
            out_dir
            / f"cam_{cam_idx}_intensity.npy"
        )

        binary_path = (
            out_dir
            / f"cam_{cam_idx}_binary.npy"
        )

        np.save(
            intensity_path,
            intensity_curve,
        )

        np.save(
            binary_path,
            binary_curve,
        )

        if verbose:
            print(
                f"Saved intensity curve: "
                f"{intensity_path}"
            )

            print(
                f"Saved binary curve:    "
                f"{binary_path}"
            )

    return {
        "intensity": intensity_curve,
        "binary": binary_curve,
    }


def generate_challenge_lightcurves(
    model_num: int,
    cam_idx: str,
    sim_root: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """
    Generate lightcurves for one original Helsinki Asteroid Challenge model.

    This is a convenience wrapper around ``generate_lightcurves_from_mesh``.
    It resolves and loads the challenge STL before performing the common
    simulation.

    Parameters
    ----------
    model_num
        Challenge model number.

    cam_idx
        Challenge observer/camera position.

    sim_root
        Optional root directory for generated challenge-model lightcurves.

        If provided, output is written under::

            sim_root / model_<model_num>/

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing ``intensity`` and ``binary`` lightcurves.
    """
    stl_path = get_model_stl_path(
        model_num
    )

    mesh = pv.read(
        stl_path
    )

    if sim_root is None:
        out_dir = None
    else:
        out_dir = (
            Path(sim_root)
            / f"model_{model_num:02d}"
        )

    return generate_lightcurves_from_mesh(
        mesh=mesh,
        cam_idx=cam_idx,
        out_dir=out_dir,
    )

if __name__ == "__main__":
    generate_challenge_lightcurves(
        model_num=2,
        cam_idx="45_top",
        sim_root="workspaces/david/out/sim_lc"
    )
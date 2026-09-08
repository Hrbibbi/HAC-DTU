"""Optimize an HAC asteroid shape with Mitsuba Large Steps.

Only supports the `direct` integrator (orthographic assumption).
"""

import argparse
import copy
import datetime
import json
import math
import os
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import trimesh

DataSource = Literal["real", "simulated"]
CurveType = Literal["intensity", "binary"]

INTEGRATOR: dict[str, Any] = {"type": "direct"}
PROJECTION: Literal["orthographic"] = "orthographic"


@dataclass(frozen=True, slots=True)
class RunConfiguration:
    model_num: int
    data_source: DataSource
    curve_type: CurveType
    bsdf_json: Path | None
    bsdf_name: str | None
    initial_mesh: Path | None
    output_dir: Path | None
    variant: str
    cuda_device: str
    phases: int | None
    width: int
    height: int
    spp: int
    spp_grad: int
    iterations: int
    learning_rate: float
    large_steps_lambda: float
    seed: int
    validation_interval: int
    validation_spp: int
    early_stop_patience: int
    initial_subdivisions: int
    initial_face_count: int
    checkpoint_interval: int
    reduction_temperature: float
    evaluate_reference: bool
    render_movies: bool
    movie_observer: str
    movie_phases: int
    movie_width: int
    movie_height: int
    movie_spp: int
    movie_fps: int

    @property
    def data_sources(self) -> tuple[DataSource, ...]:
        return (self.data_source,)


@dataclass(frozen=True, slots=True)
class RunPaths:
    output_dir: Path
    resolved_config: Path
    initial_mesh: Path
    best_mesh: Path
    last_mesh: Path
    normalized_reference_mesh: Path
    checkpoints: Path
    history: Path
    curves: Path
    lightcurve_plot: Path
    loss_plot: Path
    initial_views: Path
    best_views: Path
    reference_views: Path
    summary: Path
    initial_movie: Path
    best_movie: Path
    reference_movie: Path
    comparison_movie: Path

    @classmethod
    def from_output_dir(cls, output_dir: Path) -> "RunPaths":
        return cls(
            output_dir=output_dir,
            resolved_config=output_dir / "resolved_config.json",
            initial_mesh=output_dir / "initial_mesh.obj",
            best_mesh=output_dir / "best_mesh.obj",
            last_mesh=output_dir / "last_mesh.obj",
            normalized_reference_mesh=output_dir / "normalized_reference_mesh.obj",
            checkpoints=output_dir / "checkpoints",
            history=output_dir / "history.jsonl",
            curves=output_dir / "curves.npz",
            lightcurve_plot=output_dir / "lightcurve_fit.png",
            loss_plot=output_dir / "optimization_loss.png",
            initial_views=output_dir / "initial_views.png",
            best_views=output_dir / "best_views.png",
            reference_views=output_dir / "reference_views.png",
            summary=output_dir / "summary.json",
            initial_movie=output_dir / "initial_shape.mp4",
            best_movie=output_dir / "best_shape.mp4",
            reference_movie=output_dir / "reference_shape.mp4",
            comparison_movie=output_dir / "comparison.mp4",
        )


def _positive(parser: argparse.ArgumentParser, name: str, value: float) -> None:
    if value <= 0:
        parser.error(f"{name} must be positive")


def parse_args(argv: Sequence[str] | None = None) -> RunConfiguration:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_num", type=int, choices=range(1, 11))
    parser.add_argument(
        "--data-source",
        choices=("real", "simulated"),
        default="simulated",
    )
    parser.add_argument(
        "--curve-type",
        choices=("intensity", "binary"),
        default="intensity",
    )
    parser.add_argument(
        "--bsdf-json",
        type=Path,
        help="fixed Mitsuba BSDF dictionary or BSDF trial-list JSON",
    )
    parser.add_argument(
        "--bsdf-name",
        help="entry name when --bsdf-json contains multiple trial configurations",
    )
    parser.add_argument(
        "--initial-mesh",
        type=Path,
        help="optional warm-start mesh; otherwise construct an ellipsoid",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--variant",
        default=os.environ.get("MITSUBA_VARIANT", "cuda_ad_mono"),
    )
    parser.add_argument(
        "--cuda-device",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    )

    parser.add_argument(
        "--phases",
        type=int,
        default=None,
        help=(
            "subsample the light curves to this many frames by index (uneven, "
            "no averaging); omit to use every native frame (~841 real, 360 blender)"
        ),
    )
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--spp", type=int, default=8)
    parser.add_argument("--spp-grad", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=2e-2)
    parser.add_argument("--large-steps-lambda", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--validation-spp", type=int, default=16)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="validation checks without improvement; 0 disables early stopping",
    )
    parser.add_argument("--initial-subdivisions", type=int, default=5)
    parser.add_argument("--initial-face-count", type=int, default=2_500)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=25,
        help="iterations between mesh checkpoints; 0 disables them",
    )
    parser.add_argument("--reduction-temperature", type=float, default=0.01)
    parser.add_argument("--evaluate-reference", action="store_true")
    parser.add_argument("--render-movies", action="store_true")
    parser.add_argument("--movie-observer", default="0_hor1")
    parser.add_argument("--movie-phases", type=int, default=360)
    parser.add_argument("--movie-width", type=int, default=512)
    parser.add_argument("--movie-height", type=int, default=512)
    parser.add_argument("--movie-spp", type=int, default=16)
    parser.add_argument("--movie-fps", type=int, default=30)
    args = parser.parse_args(argv)

    for name in (
        "width",
        "height",
        "spp",
        "iterations",
        "validation_interval",
        "validation_spp",
        "initial_face_count",
        "movie_phases",
        "movie_width",
        "movie_height",
        "movie_spp",
        "movie_fps",
    ):
        _positive(parser, f"--{name.replace('_', '-')}", getattr(args, name))
    for name in ("learning_rate", "large_steps_lambda", "reduction_temperature"):
        _positive(parser, f"--{name.replace('_', '-')}", getattr(args, name))
    if args.spp_grad < 0:
        parser.error("--spp-grad must be non-negative")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.early_stop_patience < 0:
        parser.error("--early-stop-patience must be non-negative")
    if args.initial_subdivisions < 0:
        parser.error("--initial-subdivisions must be non-negative")
    if args.checkpoint_interval < 0:
        parser.error("--checkpoint-interval must be non-negative")
    if args.phases is not None and args.phases <= 0:
        parser.error("--phases must be positive")
    if args.bsdf_name is not None and args.bsdf_json is None:
        parser.error("--bsdf-name requires --bsdf-json")
    if args.evaluate_reference and args.model_num > 3:
        parser.error("--evaluate-reference is only available for models 1-3")

    return RunConfiguration(
        model_num=args.model_num,
        data_source=args.data_source,
        curve_type=args.curve_type,
        bsdf_json=args.bsdf_json,
        bsdf_name=args.bsdf_name,
        initial_mesh=args.initial_mesh,
        output_dir=args.output_dir,
        variant=args.variant,
        cuda_device=args.cuda_device,
        phases=args.phases,
        width=args.width,
        height=args.height,
        spp=args.spp,
        spp_grad=args.spp_grad,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        large_steps_lambda=args.large_steps_lambda,
        seed=args.seed,
        validation_interval=args.validation_interval,
        validation_spp=args.validation_spp,
        early_stop_patience=args.early_stop_patience,
        initial_subdivisions=args.initial_subdivisions,
        initial_face_count=args.initial_face_count,
        checkpoint_interval=args.checkpoint_interval,
        reduction_temperature=args.reduction_temperature,
        evaluate_reference=args.evaluate_reference,
        render_movies=args.render_movies,
        movie_observer=args.movie_observer,
        movie_phases=args.movie_phases,
        movie_width=args.movie_width,
        movie_height=args.movie_height,
        movie_spp=args.movie_spp,
        movie_fps=args.movie_fps,
    )


def initialize_mitsuba(configuration: RunConfiguration):
    """Select the runtime before importing anything from ``hac.mitsuba``."""
    os.environ["CUDA_VISIBLE_DEVICES"] = configuration.cuda_device
    import mitsuba as mi

    mi.set_variant(configuration.variant)
    return mi


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(value), indent=2) + "\n")


def _resolve_output_dir(configuration: RunConfiguration) -> Path:
    if configuration.output_dir is not None:
        return configuration.output_dir.resolve()
    from hac.constants import WORKSPACE_DIR

    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    return (
        WORKSPACE_DIR
        / "optimize_shape_v5"
        / (
            f"model{configuration.model_num}_{configuration.data_source}_"
            f"{configuration.curve_type}_{timestamp}"
        )
    )


def _load_bsdf(configuration: RunConfiguration) -> dict[str, Any] | None:
    if configuration.bsdf_json is None:
        return None
    path = configuration.bsdf_json
    if not path.is_file():
        raise FileNotFoundError(f"BSDF JSON not found: {path}")
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "type" in payload:
        return copy.deepcopy(payload)
    if isinstance(payload, dict) and isinstance(payload.get("bsdf"), dict):
        if configuration.bsdf_name not in (None, payload.get("name")):
            raise ValueError(
                f"BSDF entry {configuration.bsdf_name!r} is not present in {path}."
            )
        return copy.deepcopy(payload["bsdf"])
    if isinstance(payload, list):
        entries = [entry for entry in payload if isinstance(entry, dict)]
        if configuration.bsdf_name is None:
            if len(entries) != 1:
                raise ValueError(
                    f"{path} contains {len(entries)} entries; select one with --bsdf-name."
                )
            entry = entries[0]
        else:
            matches = [
                entry
                for entry in entries
                if entry.get("name") == configuration.bsdf_name
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one BSDF named {configuration.bsdf_name!r} in {path}; "
                    f"found {len(matches)}."
                )
            entry = matches[0]
        if not isinstance(entry.get("bsdf"), dict):
            raise ValueError(f"Selected entry in {path} has no BSDF dictionary.")
        return copy.deepcopy(entry["bsdf"])
    raise ValueError(f"{path} must contain a Mitsuba BSDF dictionary or trial list.")


def _prepare_initial_mesh(
    configuration: RunConfiguration,
    cylinder_radius: float,
) -> trimesh.Trimesh:
    from hac.mesh import (
        construct_initial_ellipsoid_mesh,
        normalize_mesh_to_challenge_cylinder,
    )

    if configuration.initial_mesh is not None:
        if not configuration.initial_mesh.is_file():
            raise FileNotFoundError(
                f"Initial mesh not found: {configuration.initial_mesh}"
            )
        mesh = trimesh.load_mesh(configuration.initial_mesh, force="mesh")
    else:
        mesh = construct_initial_ellipsoid_mesh(
            radius=cylinder_radius,
            subdivisions=configuration.initial_subdivisions,
        )
        if len(mesh.faces) > configuration.initial_face_count:
            mesh = mesh.simplify_quadric_decimation(
                face_count=configuration.initial_face_count
            )
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise ValueError("Initial mesh must be a non-empty triangle mesh.")
    return normalize_mesh_to_challenge_cylinder(mesh, target_R=cylinder_radius)


def _build_forward(
    mesh_path: Path,
    configuration: RunConfiguration,
    bsdf: Mapping[str, Any] | None,
    *,
    n_phases: int,
    width: int | None = None,
    height: int | None = None,
    spp: int | None = None,
):
    from hac.constants import OBSERVERS
    from hac.mitsuba.forward import ForwardModel
    from hac.mitsuba.mesh import MitsubaMesh
    from hac.mitsuba.scene import RenderingSettings

    return ForwardModel(
        MitsubaMesh(mesh_path),
        n_phases=n_phases,
        observers=dict(OBSERVERS),
        rendering_settings=RenderingSettings(
            width=configuration.width if width is None else width,
            height=configuration.height if height is None else height,
            spp=configuration.spp if spp is None else spp,
        ),
        normalize_mesh=False,
        bsdf=bsdf,
        integrator=INTEGRATOR,
        projection=PROJECTION,
    )


def _configure_reduction(
    forward,
    configuration: RunConfiguration,
    *,
    binary_thresholds: np.ndarray | None = None,
) -> np.ndarray | None:
    import drjit as dr
    import mitsuba as mi

    from hac.mitsuba.forward import (
        calibrate_binary_thresholds,
        reduction_binary_smooth,
        reduction_intensity_smooth,
    )

    if configuration.curve_type == "intensity":
        forward.reduction_function = lambda frames: reduction_intensity_smooth(
            frames,
            temperature=configuration.reduction_temperature,
        )
        return None

    if binary_thresholds is None:
        with dr.suspend_grad():
            initial_frames = forward.render(
                seed=configuration.seed,
                spp=configuration.validation_spp,
            )
        binary_thresholds = calibrate_binary_thresholds(initial_frames)
    # One calibrated threshold per observer, broadcast over (phase, y, x, channel).
    # Built once here so it stays a constant inside the frozen training step.
    threshold_tensor = mi.TensorXf(binary_thresholds.reshape(-1, 1, 1, 1, 1))
    forward.reduction_function = lambda frames: reduction_binary_smooth(
        frames,
        threshold=threshold_tensor,
        temperature=configuration.reduction_temperature,
    )
    return binary_thresholds


def _positions_numpy(forward) -> np.ndarray:
    positions = np.asarray(forward.params[forward.position_key], dtype=np.float32)
    return positions.reshape(-1, 3).copy()


def _faces_numpy(forward) -> np.ndarray:
    faces = np.asarray(forward.params[forward.face_key], dtype=np.int64)
    return faces.reshape(-1, 3).copy()


def _export_mesh(path: Path, positions: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=positions, faces=faces, process=False).export(path)


def _apply_optimizer_positions(
    forward,
    optimizer,
    large_steps,
    cylinder_radius: float,
):
    from hac.mitsuba.mesh import normalize_positions_to_challenge_cylinder

    positions = normalize_positions_to_challenge_cylinder(
        large_steps.from_differential(optimizer["u"]),
        cylinder_radius,
    )
    forward.params[forward.position_key] = positions
    forward.params.update()
    return positions


def _source_balanced_loss(curves, target, group_slices: Mapping[str, slice]):
    import drjit as dr

    group_losses = [
        dr.mean(dr.square(curves[group_slice] - target[group_slice]))
        for group_slice in group_slices.values()
    ]
    return sum(group_losses) / len(group_losses)


def _reduce_lightcurves_and_backward(
    frames,
    reduction_function,
    target,
    group_slices: Mapping[str, slice],
):
    """Reduce rendered frames and backpropagate within a frozen AD scope.

    ``frames`` retains the external render graph, so backward propagation from
    the frozen reduction still reaches the current Large Steps variable.
    """
    import drjit as dr

    curves = reduction_function(frames)
    loss = _source_balanced_loss(curves, target, group_slices)

    # Frozen functions have an isolated gradient scope, so backward must be
    # part of the recording rather than applied to the returned loss later.
    dr.backward(loss)
    return loss


def _numpy_source_losses(
    curves: np.ndarray,
    target: np.ndarray,
    group_slices: Mapping[str, slice],
) -> dict[str, float]:
    return {
        source: float(np.mean(np.square(curves[group_slice] - target[group_slice])))
        for source, group_slice in group_slices.items()
    }


def _evaluate(
    forward,
    target: np.ndarray,
    group_slices: Mapping[str, slice],
    *,
    seed: int,
    spp: int,
) -> tuple[float, dict[str, float], np.ndarray]:
    import drjit as dr

    with dr.suspend_grad():
        curves = np.asarray(forward.render_lightcurves(seed=seed, spp=spp)).copy()
    losses = _numpy_source_losses(curves, target, group_slices)
    return float(np.mean(list(losses.values()))), losses, curves


def _check_finite(name: str, values: Any) -> None:
    array = np.asarray(values)
    if not np.all(np.isfinite(array)):
        raise RuntimeError(f"{name} contains non-finite values.")


def _compare_reloaded_checkpoint_loss(
    reloaded_loss: float,
    in_memory_loss: float,
    *,
    relative_tolerance: float = 1e-3,
    absolute_tolerance: float = 1e-5,
) -> dict[str, float | bool]:
    """Compare checkpoint renders without invalidating a completed run."""
    absolute_difference = abs(reloaded_loss - in_memory_loss)
    relative_difference = absolute_difference / max(abs(in_memory_loss), 1e-15)
    matches = math.isclose(
        reloaded_loss,
        in_memory_loss,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    )
    if not matches:
        warnings.warn(
            "Fresh ForwardModel loss differs from the in-memory best checkpoint: "
            f"{reloaded_loss} versus {in_memory_loss} "
            f"(absolute difference {absolute_difference:.6g}, "
            f"relative difference {relative_difference:.3%}). Continuing with "
            "the reloaded checkpoint; mesh serialization, visibility changes, and "
            "Monte Carlo rendering can cause small differences.",
            RuntimeWarning,
            stacklevel=2,
        )
    return {
        "matches_tolerance": matches,
        "absolute_loss_difference": absolute_difference,
        "relative_loss_difference": relative_difference,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
    }


def _append_history(file, entry: Mapping[str, Any]) -> None:
    file.write(json.dumps(_json_value(entry)) + "\n")
    file.flush()


def _save_diagnostic_views(
    mesh_path: Path,
    output_path: Path,
    configuration: RunConfiguration,
    bsdf: Mapping[str, Any] | None,
    *,
    title: str,
) -> None:
    from hac.constants import OBSERVERS
    from hac.mitsuba.forward import ForwardModel
    from hac.mitsuba.mesh import MitsubaMesh
    from hac.mitsuba.scene import RenderingSettings
    from hac.mitsuba.visualize import render_forward_views

    names = ("0_top", "0_bottom", "0_hor1")
    observers = {name: OBSERVERS[name] for name in names}
    diagnostic = ForwardModel(
        MitsubaMesh(mesh_path),
        n_phases=1,
        observers=observers,
        rendering_settings=RenderingSettings(
            width=max(128, configuration.width),
            height=max(128, configuration.height),
            spp=configuration.validation_spp,
        ),
        normalize_mesh=False,
        bsdf=bsdf,
        projection=PROJECTION,
        integrator=INTEGRATOR,
    )
    figure, _ = render_forward_views(
        diagnostic,
        title=title,
        output_path=output_path,
        seed=configuration.seed,
        spp=configuration.validation_spp,
    )
    import matplotlib.pyplot as plt

    plt.close(figure)


def _movie_frames(
    mesh_path: Path,
    configuration: RunConfiguration,
    bsdf: Mapping[str, Any] | None,
):
    from hac.constants import OBSERVERS
    from hac.mitsuba.forward import ForwardModel
    from hac.mitsuba.mesh import MitsubaMesh
    from hac.mitsuba.scene import RenderingSettings
    from hac.mitsuba.visualize import bitmap_frames

    if "/" in configuration.movie_observer:
        source, observer_name = configuration.movie_observer.split("/", 1)
        if source not in configuration.data_sources:
            raise ValueError(f"Movie source {source!r} is not active in this run.")
    else:
        observer_name = configuration.movie_observer
    if observer_name not in OBSERVERS:
        raise ValueError(f"Unknown movie observer {observer_name!r}.")
    forward = ForwardModel(
        MitsubaMesh(mesh_path),
        n_phases=configuration.movie_phases,
        observers={observer_name: OBSERVERS[observer_name]},
        rendering_settings=RenderingSettings(
            width=configuration.movie_width,
            height=configuration.movie_height,
            spp=configuration.movie_spp,
        ),
        normalize_mesh=False,
        bsdf=bsdf,
        projection=PROJECTION,
        integrator=INTEGRATOR,
    )
    return bitmap_frames(
        forward.render(seed=configuration.seed, spp=configuration.movie_spp)[0]
    )


def _render_movies(
    paths: RunPaths,
    configuration: RunConfiguration,
    bsdf: Mapping[str, Any] | None,
    reference_available: bool,
) -> dict[str, str]:
    from hac.mitsuba.visualize import write_movie

    initial = _movie_frames(paths.initial_mesh, configuration, bsdf)
    best = _movie_frames(paths.best_mesh, configuration, bsdf)
    panels = [initial, best]
    write_movie(paths.initial_movie, initial, configuration.movie_fps)
    write_movie(paths.best_movie, best, configuration.movie_fps)
    artifacts = {
        "initial_movie": str(paths.initial_movie),
        "best_movie": str(paths.best_movie),
    }
    if reference_available:
        reference = _movie_frames(
            paths.normalized_reference_mesh,
            configuration,
            bsdf,
        )
        panels.insert(0, reference)
        write_movie(paths.reference_movie, reference, configuration.movie_fps)
        artifacts["reference_movie"] = str(paths.reference_movie)
    comparison = np.concatenate(panels, axis=2)
    write_movie(paths.comparison_movie, comparison, configuration.movie_fps)
    artifacts["comparison_movie"] = str(paths.comparison_movie)
    return artifacts


def run(configuration: RunConfiguration) -> dict[str, Any]:
    mi = initialize_mitsuba(configuration)
    import drjit as dr

    from hac.constants import MODEL_BASE_RADII, OBSERVERS
    from hac.data import get_model_stl_path, prepare_lightcurve_targets
    from hac.mesh import normalize_mesh_to_challenge_cylinder
    from hac.visualize import plot_lightcurve_comparison, plot_optimization_loss

    output_dir = _resolve_output_dir(configuration)
    paths = RunPaths.from_output_dir(output_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    cylinder_radius = MODEL_BASE_RADII[configuration.model_num]
    bsdf = _load_bsdf(configuration)
    targets = prepare_lightcurve_targets(
        model_num=configuration.model_num,
        data_source=configuration.data_source,
        curve_type=configuration.curve_type,
        n_phases=configuration.phases,
        observer_names=tuple(OBSERVERS),
    )
    # One render phase per kept light curve frame (index-subsampled, not resampled).
    n_phases = targets.n_phases
    # See the training loop: only the intensity reduction is dr.freeze-safe.
    freeze_training = configuration.curve_type == "intensity"
    _write_json(
        paths.resolved_config,
        {
            **asdict(configuration),
            "resolved_output_dir": output_dir,
            "data_sources": configuration.data_sources,
            "n_phases": n_phases,
            "projection": PROJECTION,
            "integrator": INTEGRATOR,
            "frozen_training": freeze_training,
        },
    )
    initial_mesh = _prepare_initial_mesh(configuration, cylinder_radius)
    initial_mesh.export(paths.initial_mesh)

    forward = _build_forward(
        paths.initial_mesh, configuration, bsdf, n_phases=n_phases
    )
    if tuple(forward.observer_names) != targets.observer_names:
        raise RuntimeError(
            "Target/render observer order differs: "
            f"{targets.observer_names} versus {forward.observer_names}."
        )
    position_keys = [
        key for key in forward.params.keys() if key.endswith("vertex_positions")
    ]
    if position_keys != [forward.position_key]:
        raise RuntimeError(
            f"Expected one shared optimizable mesh; found {position_keys}."
        )
    binary_thresholds = _configure_reduction(forward, configuration)
    faces = _faces_numpy(forward)
    target_tensor = mi.TensorXf(targets.values.copy())

    initial_loss, initial_source_losses, initial_curves = _evaluate(
        forward,
        targets.values,
        targets.group_slices,
        seed=configuration.seed,
        spp=configuration.validation_spp,
    )
    _save_diagnostic_views(
        paths.initial_mesh,
        paths.initial_views,
        configuration,
        bsdf,
        title=f"Model {configuration.model_num}: initial shape",
    )

    large_steps = mi.ad.LargeSteps(
        forward.params[forward.position_key],
        forward.params[forward.face_key],
        configuration.large_steps_lambda,
    )

    # Faces and the other scene parameters are constant during optimization.
    # Excluding them here keeps mi.render() from traversing the batch scene's
    # thousands of static sensor and instance parameters on every iteration.
    # Flush access-tracking candidates first so keep() cannot leave a removed
    # parameter pending for the next update().
    forward.params.update()
    forward.params.keep([forward.position_key])

    optimizer = mi.ad.Adam(lr=configuration.learning_rate, uniform=True)
    optimizer["u"] = large_steps.to_differential(forward.params[forward.position_key])

    # dr.freeze cannot traverse drjit state captured in reduction_function's
    # closure. The binary reduction closes over a per-observer threshold tensor,
    # so only the intensity path (closure holds plain Python scalars) is frozen.
    training_step = (
        dr.freeze(_reduce_lightcurves_and_backward)
        if freeze_training
        else _reduce_lightcurves_and_backward
    )

    best_loss = initial_loss
    best_iteration = 0
    best_positions = _positions_numpy(forward)
    _export_mesh(paths.best_mesh, best_positions, faces)
    training_losses: list[float] = []
    validation_history: list[dict[str, Any]] = [
        {
            "iteration": 0,
            "loss": initial_loss,
            "source_losses": initial_source_losses,
        }
    ]
    validation_checks_without_improvement = 0
    iterations_completed = 0

    with paths.history.open("w") as history_file:
        _append_history(
            history_file,
            {
                "kind": "validation",
                "iteration": 0,
                "loss": initial_loss,
                "source_losses": initial_source_losses,
                "best": True,
            },
        )
        for iteration in range(1, configuration.iterations + 1):
            positions = _apply_optimizer_positions(
                forward,
                optimizer,
                large_steps,
                cylinder_radius,
            )
            _check_finite("vertex positions", positions)
            frames = forward.render(
                seed=configuration.seed + iteration,
                spp=configuration.spp,
                spp_grad=configuration.spp_grad,
            )
            loss = training_step(
                frames,
                forward.reduction_function,
                target_tensor,
                targets.group_slices,
            )
            dr.eval(loss)
            loss_value = float(np.asarray(loss).reshape(-1)[0])
            if not math.isfinite(loss_value):
                raise RuntimeError(
                    f"Non-finite training loss at iteration {iteration}."
                )
            gradient = dr.grad(optimizer["u"])
            _check_finite("Large Steps gradient", gradient)
            optimizer.step()

            training_losses.append(loss_value)
            iterations_completed = iteration
            entry: dict[str, Any] = {
                "kind": "training",
                "iteration": iteration,
                "loss": loss_value,
            }
            _append_history(history_file, entry)
            print(
                f"Iteration {iteration:04d}/{configuration.iterations}: "
                f"training_loss={loss_value:.8f}"
            )

            should_validate = (
                iteration % configuration.validation_interval == 0
                or iteration == configuration.iterations
            )
            should_checkpoint = (
                configuration.checkpoint_interval > 0
                and iteration % configuration.checkpoint_interval == 0
            )
            if should_validate or should_checkpoint:
                _apply_optimizer_positions(
                    forward,
                    optimizer,
                    large_steps,
                    cylinder_radius,
                )
            if should_checkpoint:
                checkpoint_path = paths.checkpoints / f"iteration_{iteration:04d}.obj"
                _export_mesh(checkpoint_path, _positions_numpy(forward), faces)

            if should_validate:
                validation_loss, source_losses, _ = _evaluate(
                    forward,
                    targets.values,
                    targets.group_slices,
                    seed=configuration.seed,
                    spp=configuration.validation_spp,
                )
                improved = validation_loss < best_loss
                if improved:
                    best_loss = validation_loss
                    best_iteration = iteration
                    best_positions = _positions_numpy(forward)
                    _export_mesh(paths.best_mesh, best_positions, faces)
                    validation_checks_without_improvement = 0
                else:
                    validation_checks_without_improvement += 1
                validation_entry = {
                    "kind": "validation",
                    "iteration": iteration,
                    "loss": validation_loss,
                    "source_losses": source_losses,
                    "best": improved,
                }
                validation_history.append(
                    {
                        "iteration": iteration,
                        "loss": validation_loss,
                        "source_losses": source_losses,
                    }
                )
                _append_history(history_file, validation_entry)
                print(
                    f"  validation_loss={validation_loss:.8f}, "
                    f"source_losses={source_losses}, best={improved}"
                )
                if (
                    configuration.early_stop_patience > 0
                    and validation_checks_without_improvement
                    >= configuration.early_stop_patience
                ):
                    print(
                        "Stopping early after "
                        f"{validation_checks_without_improvement} validation checks "
                        "without improvement."
                    )
                    break

    _apply_optimizer_positions(forward, optimizer, large_steps, cylinder_radius)
    last_positions = _positions_numpy(forward)
    _export_mesh(paths.last_mesh, last_positions, faces)
    _export_mesh(paths.best_mesh, best_positions, faces)

    frozen_training_recordings = (
        training_step.n_recordings if freeze_training else 0
    )
    del training_step
    del forward
    dr.flush_malloc_cache()
    final_forward = _build_forward(
        paths.best_mesh, configuration, bsdf, n_phases=n_phases
    )
    _configure_reduction(
        final_forward,
        configuration,
        binary_thresholds=binary_thresholds,
    )
    final_loss, final_source_losses, final_curves = _evaluate(
        final_forward,
        targets.values,
        targets.group_slices,
        seed=configuration.seed,
        spp=configuration.validation_spp,
    )
    checkpoint_reconstruction = _compare_reloaded_checkpoint_loss(
        final_loss,
        best_loss,
    )
    _save_diagnostic_views(
        paths.best_mesh,
        paths.best_views,
        configuration,
        bsdf,
        title=f"Model {configuration.model_num}: optimized shape",
    )

    reference_loss = None
    reference_source_losses = None
    reference_available = False
    if configuration.evaluate_reference:
        reference_path = get_model_stl_path(configuration.model_num)
        reference_mesh = normalize_mesh_to_challenge_cylinder(
            trimesh.load_mesh(reference_path, force="mesh"),
            target_R=cylinder_radius,
        )
        reference_mesh.export(paths.normalized_reference_mesh)
        reference_forward = _build_forward(
            paths.normalized_reference_mesh,
            configuration,
            bsdf,
            n_phases=n_phases,
        )
        _configure_reduction(
            reference_forward,
            configuration,
            binary_thresholds=binary_thresholds,
        )
        reference_loss, reference_source_losses, _ = _evaluate(
            reference_forward,
            targets.values,
            targets.group_slices,
            seed=configuration.seed,
            spp=configuration.validation_spp,
        )
        reference_available = True
        _save_diagnostic_views(
            paths.normalized_reference_mesh,
            paths.reference_views,
            configuration,
            bsdf,
            title=f"Model {configuration.model_num}: reference shape",
        )
        del reference_forward
        dr.flush_malloc_cache()

    plot_lightcurve_comparison(
        targets.values,
        initial_curves,
        final_curves,
        targets.observer_names,
        ("HAC target", "Initial shape", "Optimized shape"),
        paths.lightcurve_plot,
        title=(
            f"Model {configuration.model_num}: {configuration.data_source} "
            f"{configuration.curve_type}, validation MSE={final_loss:.6g}"
        ),
    )
    plot_optimization_loss(
        training_losses,
        paths.loss_plot,
        reference_loss=reference_loss,
        reference_label=(
            "Reference-shape forward MSE" if reference_loss is not None else None
        ),
    )
    np.savez_compressed(
        paths.curves,
        target=targets.values,
        initial=initial_curves,
        best=final_curves,
        observer_names=np.asarray(targets.observer_names),
    )

    movie_artifacts: dict[str, str] = {}
    if configuration.render_movies:
        movie_artifacts = _render_movies(
            paths,
            configuration,
            bsdf,
            reference_available,
        )

    radial = np.linalg.norm(best_positions[:, :2], axis=1)
    summary = {
        "configuration": asdict(configuration),
        "model_num": configuration.model_num,
        "observer_names": targets.observer_names,
        "observer_group_slices": {
            source: [group_slice.start, group_slice.stop]
            for source, group_slice in targets.group_slices.items()
        },
        "n_phases": n_phases,
        "projection": PROJECTION,
        "integrator": INTEGRATOR,
        "frozen_training": freeze_training,
        "frozen_training_recordings": frozen_training_recordings,
        "shared_mesh_parameter": final_forward.position_key,
        "shared_bsdf": bsdf,
        "vertex_count": len(best_positions),
        "face_count": len(faces),
        "iterations_completed": iterations_completed,
        "initial_validation_loss": initial_loss,
        "initial_source_losses": initial_source_losses,
        "best_iteration": best_iteration,
        "best_validation_loss": best_loss,
        "final_validation_loss": final_loss,
        "final_source_losses": final_source_losses,
        "checkpoint_reconstruction": checkpoint_reconstruction,
        "reference_loss": reference_loss,
        "reference_source_losses": reference_source_losses,
        "training_loss_history": training_losses,
        "validation_history": validation_history,
        "binary_thresholds": (
            None
            if binary_thresholds is None
            else {
                name: float(binary_thresholds[index])
                for index, name in enumerate(targets.observer_names)
            }
        ),
        "constraint": {
            "target_radius": cylinder_radius,
            "radial_max": float(np.max(radial)),
            "z_min": float(np.min(best_positions[:, 2])),
            "z_max": float(np.max(best_positions[:, 2])),
        },
        "artifacts": {
            "resolved_config": str(paths.resolved_config),
            "initial_mesh": str(paths.initial_mesh),
            "best_mesh": str(paths.best_mesh),
            "last_mesh": str(paths.last_mesh),
            "history": str(paths.history),
            "curves": str(paths.curves),
            "lightcurve_plot": str(paths.lightcurve_plot),
            "loss_plot": str(paths.loss_plot),
            "initial_views": str(paths.initial_views),
            "best_views": str(paths.best_views),
            "reference_mesh": (
                str(paths.normalized_reference_mesh) if reference_available else None
            ),
            "reference_views": (
                str(paths.reference_views) if reference_available else None
            ),
            **movie_artifacts,
        },
        "elapsed_time_seconds": time.perf_counter() - start_time,
    }
    _write_json(paths.summary, summary)
    print(json.dumps(_json_value(summary), indent=2))
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    configuration = parse_args(argv)
    run(configuration)


if __name__ == "__main__":
    main()

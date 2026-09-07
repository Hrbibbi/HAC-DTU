"""
Jointly fit shared canonical BSDF parameters across several known models.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import argparse
import copy
import datetime
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

import drjit as dr
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import trimesh
from skimage.filters import threshold_otsu

mi.set_variant(os.environ.get("MITSUBA_VARIANT", "cuda_ad_mono"))

from hac.constants import MODEL_BASE_RADII, OBSERVERS, WORKSPACE_DIR
from hac.data import get_model_stl_path, load_data
from hac.mesh import normalize_mesh_to_challenge_cylinder
from hac.mitsuba.forward import (
    ForwardModel,
    reduction_binary_smooth,
    reduction_intensity_smooth,
)
from hac.mitsuba.mesh import MitsubaMesh
from hac.mitsuba.scene import Projection, RenderingSettings
from hac.mitsuba.visualize import render_forward_views
from hac.types import ObserverGeometry


EXPERIMENTS = tuple(OBSERVERS)

OPTIMIZATION_PHASES = 30
OPTIMIZATION_WIDTH = 128
OPTIMIZATION_HEIGHT = 128
OPTIMIZATION_SPP = 16
OPTIMIZATION_SPP_GRAD = 16
DEFAULT_ITERATIONS = 100
DEFAULT_INFO_INTERVAL = 10
LEARNING_RATE = 8e-3
EARLY_STOP_PATIENCE = 15

FINAL_SPP = 64
FINAL_SEED = 20260819

BINARY_SOFT_THRESHOLD_TEMPERATURE = 0.02
BOUND_SATURATION_FRACTION = 1e-3
MAX_KNOWN_MODEL_FACE_COUNT = 100_000

DEFAULT_MODEL_NUMS = (1, 2, 3)
DEFAULT_VALIDATION_SPP = 32
VALIDATION_SEED = 20260821

DataSource = Literal["real", "simulated", "both"]
CurveType = Literal["intensity", "binary"]


@dataclass(frozen=True, slots=True)
class OptimizedParameter:
    path: str
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class BsdfConfiguration:
    name: str
    bsdf: dict[str, Any]
    optimize: tuple[OptimizedParameter, ...]


@dataclass(frozen=True, slots=True)
class RunConfiguration:
    model_nums: tuple[int, ...]
    data_source: DataSource
    curve_type: CurveType
    camera_distance: float | None
    simulated_distance: float | None
    bsdf_configurations: tuple[BsdfConfiguration, ...]
    output_dir: Path
    iterations: int
    info_interval: int
    validation_spp: int

    @property
    def observer_distances(self) -> dict[str, float | None]:
        distances: dict[str, float | None] = {}
        if self.data_source in ("real", "both"):
            distances["real"] = self.camera_distance
        if self.data_source in ("simulated", "both"):
            distances["simulated"] = self.simulated_distance
        return distances

    @property
    def projection(self) -> Projection:
        perspective_flags = {
            distance is not None for distance in self.observer_distances.values()
        }
        if len(perspective_flags) != 1:
            raise ValueError(
                "All active observer groups must use the same projection; provide "
                "both distances for perspective or neither for orthographic."
            )
        return "perspective" if perspective_flags.pop() else "orthographic"


@dataclass(frozen=True, slots=True)
class PreparedModel:
    model_num: int
    known_model_stl: Path
    mesh_path: Path
    render_face_count: int
    target_array: np.ndarray
    target_tensor: Any


@dataclass(slots=True)
class ModelState:
    prepared: PreparedModel
    forward: ForwardModel
    binary_thresholds: np.ndarray | None
    parameter_keys: dict[str, str]


def _data_sources(data_source: DataSource) -> tuple[Literal["real", "simulated"], ...]:
    return ("real", "simulated") if data_source == "both" else (data_source,)


def _experiment_names(data_source: DataSource) -> tuple[str, ...]:
    sources = _data_sources(data_source)
    if len(sources) == 1:
        return EXPERIMENTS
    return tuple(
        f"{source}/{experiment}" for source in sources for experiment in EXPERIMENTS
    )


def _scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _load_bsdf_configurations(path: Path) -> tuple[BsdfConfiguration, ...]:
    entries = json.loads(path.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} must contain a non-empty JSON list.")

    configurations = []
    for entry in entries:
        name = entry["name"]
        bsdf = copy.deepcopy(entry["bsdf"])
        optimize = tuple(
            OptimizedParameter(
                path=parameter_path,
                lower=float(lower),
                upper=float(upper),
            )
            for parameter_path, (lower, upper) in entry["optimize"].items()
        )
        if not optimize:
            raise ValueError(f"BSDF configuration {name!r} optimizes no parameters.")
        if any(parameter.lower >= parameter.upper for parameter in optimize):
            raise ValueError(f"BSDF configuration {name!r} has invalid bounds.")
        if bsdf["type"] == "diffuse" and any(
            parameter.path in {"reflectance", "reflectance.value"}
            for parameter in optimize
        ):
            raise ValueError(
                f"BSDF configuration {name!r} cannot optimize diffuse reflectance: "
                "per-observer normalization removes that global scale."
            )
        configurations.append(BsdfConfiguration(name, bsdf, optimize))
    return tuple(configurations)


def _observers(camera_distance: float | None) -> dict[str, ObserverGeometry]:
    if camera_distance is None:
        return dict(OBSERVERS)
    return {
        name: ObserverGeometry(
            theta=observer.theta,
            alpha=observer.alpha,
            r_obs=camera_distance,
        )
        for name, observer in OBSERVERS.items()
    }


def _forward_observers(
    configuration: RunConfiguration,
) -> tuple[
    dict[str, ObserverGeometry] | None,
    dict[str, dict[str, ObserverGeometry]] | None,
]:
    groups = {
        source: _observers(configuration.observer_distances[source])
        for source in _data_sources(configuration.data_source)
    }
    if configuration.data_source == "both":
        return None, groups
    return next(iter(groups.values())), None


def _resolve_parameter_key(params, path: str) -> str:
    suffix = path if path.startswith(".") else f".{path}"
    matches = [key for key in params.keys() if key == path or key.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one scene parameter matching {path!r}; got {matches}."
        )
    return matches[0]


def _resolve_bounds(
    forward: ForwardModel,
    configuration: BsdfConfiguration,
) -> dict[str, tuple[float, float]]:
    return {
        _resolve_parameter_key(forward.params, parameter.path): (
            parameter.lower,
            parameter.upper,
        )
        for parameter in configuration.optimize
    }


def _calibrate_binary_thresholds(
    forward: ForwardModel,
    *,
    seed: int,
    spp: int,
) -> np.ndarray:
    with dr.suspend_grad():
        frames = np.asarray(forward.render(seed=seed, spp=spp))
    thresholds = np.empty(forward.observer_count, dtype=np.float32)
    for observer_index in range(forward.observer_count):
        phase_zero = frames[observer_index, 0, :, :, 0]
        thresholds[observer_index] = (
            float(phase_zero.max())
            if phase_zero.max() <= phase_zero.min()
            else float(threshold_otsu(phase_zero))
        )
    return thresholds


def _set_reduction(
    forward: ForwardModel,
    curve_type: CurveType,
    binary_thresholds: np.ndarray | None,
) -> None:
    if curve_type == "intensity":
        forward.reduction_function = lambda x: reduction_intensity_smooth(
            x, temperature=0.01
        )
        return

    if binary_thresholds is None:
        raise ValueError("Binary reduction requires calibrated thresholds.")
    expected_shape = (forward.observer_count,)
    if binary_thresholds.shape != expected_shape:
        raise ValueError(
            f"Expected binary thresholds with shape {expected_shape}; "
            f"got {binary_thresholds.shape}."
        )
    threshold_tensor = mi.TensorXf(
        binary_thresholds.reshape(forward.observer_count, 1, 1, 1, 1)
    )
    forward.reduction_function = partial(
        reduction_binary_smooth,
        threshold=threshold_tensor,
        temperature=BINARY_SOFT_THRESHOLD_TEMPERATURE,
    )


def _build_forward(
    mesh_path: Path,
    bsdf_configuration: BsdfConfiguration,
    configuration: RunConfiguration,
    settings: RenderingSettings,
    *,
    binary_thresholds: np.ndarray | None = None,
) -> tuple[
    ForwardModel,
    np.ndarray | None,
    dict[str, tuple[float, float]],
]:
    observers, observer_groups = _forward_observers(configuration)
    forward = ForwardModel(
        MitsubaMesh(mesh_path),
        n_phases=OPTIMIZATION_PHASES,
        observers=observers,
        observer_groups=observer_groups,
        rendering_settings=settings,
        normalize_mesh=False,
        bsdf=bsdf_configuration.bsdf,
        projection=configuration.projection,
    )
    bounds = _resolve_bounds(forward, bsdf_configuration)
    forward.params.keep(list(bounds))
    for key in bounds:
        if np.asarray(forward.params[key]).size != 1:
            raise ValueError(f"Optimized parameter {key} is not scalar.")

    if configuration.curve_type == "binary" and binary_thresholds is None:
        binary_thresholds = _calibrate_binary_thresholds(
            forward,
            seed=0,
            spp=FINAL_SPP,
        )
    _set_reduction(forward, configuration.curve_type, binary_thresholds)
    return forward, binary_thresholds, bounds


def _fitted_bsdf(
    configuration: BsdfConfiguration,
    parameter_keys: Mapping[str, str],
    parameter_values: Mapping[str, float],
) -> dict[str, Any]:
    fitted = copy.deepcopy(configuration.bsdf)
    for parameter in configuration.optimize:
        cursor = fitted
        parts = parameter.path.split(".")
        value = parameter_values[parameter_keys[parameter.path]]
        assigned = False
        for index, part in enumerate(parts[:-1]):
            child = cursor.get(part)
            if not isinstance(child, dict):
                # Mitsuba traverses a scalar plugin property such as roughness
                # as roughness.value even when its input dictionary uses
                # roughness: 0.5. Materialize that fitted scalar directly.
                if parts[index + 1 :] == ["value"] and part in cursor:
                    cursor[part] = value
                    assigned = True
                    break
                raise ValueError(
                    f"Cannot assign BSDF parameter path {parameter.path!r}: "
                    f"{part!r} is not a dictionary."
                )
            cursor = child
        if assigned:
            continue

        leaf = parts[-1]
        if leaf not in cursor:
            raise ValueError(
                f"Cannot assign BSDF parameter path {parameter.path!r}: "
                f"missing leaf {leaf!r}."
            )
        cursor[leaf] = value
    return fitted


def _save_render_angles(
    forward: ForwardModel,
    path: Path,
    *,
    title: str,
    seed: int = FINAL_SEED,
    spp: int = OPTIMIZATION_SPP,
) -> None:
    with dr.suspend_grad():
        figure, _ = render_forward_views(
            forward,
            seed=seed,
            spp=spp,
            title=title,
            output_path=path,
        )
    plt.close(figure)


def _evaluate(
    forward: ForwardModel,
    target,
    *,
    seed: int,
    spp: int | None = None,
    spp_grad: int = 0,
    backpropagate: bool = False,
) -> tuple[float, np.ndarray]:
    frames = forward.render(seed=seed, spp=spp, spp_grad=spp_grad)
    curves = forward.reduction_function(frames)
    loss = dr.mean(dr.square(curves - target))
    if backpropagate:
        dr.backward(loss)

    return _scalar(loss), np.asarray(curves).copy()


def _plot_lightcurves(
    path: Path,
    target: np.ndarray,
    initial: np.ndarray,
    final: np.ndarray,
    *,
    title: str,
    target_label: str,
    curve_type: CurveType,
    observer_names: Sequence[str] = EXPERIMENTS,
) -> None:
    observer_names = tuple(observer_names)
    if target.shape[0] != len(observer_names):
        raise ValueError(
            f"Expected {len(observer_names)} target curves for the observer labels; "
            f"got {target.shape[0]}."
        )
    row_count = math.ceil(len(observer_names) / 4)
    figure, axes = plt.subplots(
        row_count,
        4,
        figsize=(14, 2.6 * row_count),
        sharex=True,
        squeeze=False,
    )
    mark_every = max(1, target.shape[1] // 12)
    for index, experiment in enumerate(observer_names):
        axis = axes.flat[index]
        axis.plot(
            target[index],
            color="black",
            linewidth=1.8,
            alpha=0.7,
            label=target_label,
            zorder=2,
        )
        axis.plot(
            initial[index],
            color="tab:orange",
            linestyle="--",
            marker="o",
            markerfacecolor="none",
            markersize=3,
            markevery=mark_every,
            label="Before fitting",
            zorder=4,
        )
        axis.plot(
            final[index],
            color="tab:blue",
            linewidth=1.3,
            label="Fitted canonical model",
            zorder=3,
        )
        axis.set_title(experiment)
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(observer_names) :]:
        axis.set_visible(False)

    figure.supxlabel("Optimization phase sample")
    figure.supylabel(
        "Normalized intensity"
        if curve_type == "intensity"
        else "Normalized pixel count"
    )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.97),
    )
    figure.suptitle(title, y=0.997)
    figure.tight_layout(rect=(0.02, 0.03, 1.0, 0.93))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _resample_curve(values: np.ndarray, n_phases: int) -> np.ndarray:
    if len(values) < n_phases:
        raise ValueError(
            f"Cannot resample {len(values)} native frames onto {n_phases} phases."
        )
    bins = np.round(np.arange(len(values)) * n_phases / len(values)).astype(int)
    bins %= n_phases
    counts = np.bincount(bins, minlength=n_phases)
    sums = np.bincount(bins, weights=values, minlength=n_phases)
    return sums / counts


def _load_targets(
    model_num: int,
    data_source: DataSource,
    curve_type: CurveType,
) -> tuple[np.ndarray, Any]:
    targets = []
    for source in _data_sources(data_source):
        data = load_data(
            model_num=model_num,
            origin=source,
            curve_type=curve_type,
        ).data
        print(
            f"Loaded {source} {curve_type} target lightcurves: "
            f"{len(data)} native frames, {len(data.columns)} observers."
        )
        source_target = np.stack(
            [
                _resample_curve(
                    np.asarray(data[experiment], dtype=float),
                    OPTIMIZATION_PHASES,
                )
                for experiment in EXPERIMENTS
            ]
        ).astype(np.float32)
        source_target /= np.mean(source_target, axis=1, keepdims=True)
        targets.append(source_target)

    target = np.concatenate(targets, axis=0)
    return target, mi.TensorXf(target.copy())


def _parse_args(argv: Sequence[str] | None = None) -> RunConfiguration:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--models",
        type=int,
        nargs="+",
        choices=DEFAULT_MODEL_NUMS,
        default=DEFAULT_MODEL_NUMS,
        metavar="MODEL",
        help="known models to fit jointly (default: 1 2 3)",
    )
    parser.add_argument(
        "--data-source",
        choices=("real", "simulated", "both"),
        required=True,
    )
    parser.add_argument(
        "--curve-type",
        choices=("intensity", "binary"),
        required=True,
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        metavar="DISTANCE",
        help=(
            "use canonical perspective projection at this shared distance; "
            "omit for canonical orthographic projection"
        ),
    )
    parser.add_argument(
        "--simulated-distance",
        type=float,
        metavar="DISTANCE",
        help=(
            "perspective distance for simulated observers; defaults to "
            "--camera-distance so both groups can share one distance"
        ),
    )
    parser.add_argument("--bsdf-configs", type=Path, required=True)
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"maximum optimizer iterations (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--info-interval",
        type=int,
        default=DEFAULT_INFO_INTERVAL,
        help=(
            "run deterministic validation and save renders every N iterations "
            f"(default: {DEFAULT_INFO_INTERVAL})"
        ),
    )
    parser.add_argument(
        "--validation-spp",
        type=int,
        default=DEFAULT_VALIDATION_SPP,
        help=(
            "samples per pixel for fixed-seed checkpoint selection "
            f"(default: {DEFAULT_VALIDATION_SPP})"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    model_nums = tuple(args.models)
    if len(set(model_nums)) != len(model_nums):
        parser.error("--models cannot contain duplicates")
    if args.camera_distance is not None and args.camera_distance <= 0.0:
        parser.error("--camera-distance must be positive")
    if args.simulated_distance is not None and args.simulated_distance <= 0.0:
        parser.error("--simulated-distance must be positive")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.info_interval <= 0:
        parser.error("--info-interval must be positive")
    if args.validation_spp <= 0:
        parser.error("--validation-spp must be positive")

    camera_distance = (
        args.camera_distance if args.data_source in ("real", "both") else None
    )
    simulated_distance = (
        (
            args.simulated_distance
            if args.simulated_distance is not None
            else args.camera_distance
        )
        if args.data_source in ("simulated", "both")
        else None
    )
    active_distances = tuple(
        distance
        for source, distance in (
            ("real", camera_distance),
            ("simulated", simulated_distance),
        )
        if args.data_source in (source, "both")
    )
    if len({distance is None for distance in active_distances}) != 1:
        parser.error(
            "real and simulated observers must both be orthographic or both be perspective"
        )
    projection = "orthographic" if active_distances[0] is None else "perspective"
    distance_parts = []
    if camera_distance is not None:
        distance_parts.append(f"real{camera_distance:g}".replace(".", "p"))
    if simulated_distance is not None:
        distance_parts.append(f"simulated{simulated_distance:g}".replace(".", "p"))
    distance_label = "" if not distance_parts else "_" + "_".join(distance_parts)
    model_label = "-".join(str(model_num) for model_num in model_nums)
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    default_output = (
        WORKSPACE_DIR
        / "mitsuba_forward_optimization_v4"
        / (
            f"models{model_label}_{args.data_source}_{args.curve_type}_"
            f"{projection}{distance_label}_{timestamp}"
        )
    )
    return RunConfiguration(
        model_nums=model_nums,
        data_source=args.data_source,
        curve_type=args.curve_type,
        camera_distance=camera_distance,
        simulated_distance=simulated_distance,
        bsdf_configurations=_load_bsdf_configurations(args.bsdf_configs),
        output_dir=args.output_dir or default_output,
        iterations=args.iterations,
        info_interval=args.info_interval,
        validation_spp=args.validation_spp,
    )


def _prepare_model(
    model_num: int,
    configuration: RunConfiguration,
) -> PreparedModel:
    model_directory = configuration.output_dir / f"model{model_num}"
    model_directory.mkdir(parents=True, exist_ok=True)

    known_model_stl = get_model_stl_path(model_num=model_num)
    known_mesh = trimesh.load_mesh(known_model_stl)
    original_face_count = len(known_mesh.faces)
    if original_face_count > MAX_KNOWN_MODEL_FACE_COUNT:
        print(
            f"Model {model_num}: decimating known model from "
            f"{original_face_count:,} to {MAX_KNOWN_MODEL_FACE_COUNT:,} faces."
        )
        known_mesh = known_mesh.simplify_quadric_decimation(
            face_count=MAX_KNOWN_MODEL_FACE_COUNT
        )
    known_mesh = normalize_mesh_to_challenge_cylinder(
        known_mesh,
        target_R=MODEL_BASE_RADII[model_num],
    )
    mesh_path = model_directory / "fixed_known_model.obj"
    known_mesh.export(mesh_path)

    target_array, target_tensor = _load_targets(
        model_num,
        configuration.data_source,
        configuration.curve_type,
    )
    return PreparedModel(
        model_num=model_num,
        known_model_stl=known_model_stl,
        mesh_path=mesh_path,
        render_face_count=len(known_mesh.faces),
        target_array=target_array,
        target_tensor=target_tensor,
    )


def _logical_bounds(
    bsdf_configuration: BsdfConfiguration,
) -> dict[str, tuple[float, float]]:
    return {
        parameter.path: (parameter.lower, parameter.upper)
        for parameter in bsdf_configuration.optimize
    }


def _build_model_states(
    prepared_models: Sequence[PreparedModel],
    bsdf_configuration: BsdfConfiguration,
    configuration: RunConfiguration,
    settings: RenderingSettings,
) -> list[ModelState]:
    states = []
    expected_logical_bounds = _logical_bounds(bsdf_configuration)
    for prepared in prepared_models:
        forward, binary_thresholds, scene_bounds = _build_forward(
            prepared.mesh_path,
            bsdf_configuration,
            configuration,
            settings,
        )
        parameter_keys = {
            parameter.path: _resolve_parameter_key(
                forward.params,
                parameter.path,
            )
            for parameter in bsdf_configuration.optimize
        }
        resolved_bounds = {
            path: scene_bounds[scene_key] for path, scene_key in parameter_keys.items()
        }
        if resolved_bounds != expected_logical_bounds:
            raise RuntimeError(
                f"Model {prepared.model_num} exposed inconsistent bounds: "
                f"{resolved_bounds} versus {expected_logical_bounds}."
            )
        states.append(
            ModelState(
                prepared=prepared,
                forward=forward,
                binary_thresholds=binary_thresholds,
                parameter_keys=parameter_keys,
            )
        )
    return states


def _state_parameter_values(state: ModelState) -> dict[str, float]:
    return {
        path: _scalar(state.forward.params[scene_key])
        for path, scene_key in state.parameter_keys.items()
    }


def _initialize_shared_optimizer(
    states: Sequence[ModelState],
) -> tuple[Any, dict[str, float]]:
    if not states:
        raise ValueError("Joint optimization requires at least one model.")

    initial_parameters = _state_parameter_values(states[0])
    for state in states[1:]:
        values = _state_parameter_values(state)
        for path, expected in initial_parameters.items():
            if not math.isclose(values[path], expected, rel_tol=1e-7, abs_tol=1e-8):
                raise RuntimeError(
                    f"Model {state.prepared.model_num} initialized shared parameter "
                    f"{path!r} to {values[path]}, expected {expected}."
                )

    optimizer = mi.ad.Adam(lr=LEARNING_RATE)
    first_state = states[0]
    for path, scene_key in first_state.parameter_keys.items():
        optimizer[path] = first_state.forward.params[scene_key]
    return optimizer, initial_parameters


def _optimizer_values(optimizer, parameter_paths: Sequence[str]) -> dict[str, float]:
    return {path: _scalar(optimizer[path]) for path in parameter_paths}


def _apply_shared_parameters(state: ModelState, optimizer) -> None:
    state.forward.params.update(
        {scene_key: optimizer[path] for path, scene_key in state.parameter_keys.items()}
    )


def _loss_tensor(state: ModelState, *, seed: int, spp_grad: int):
    frames = state.forward.render(seed=seed, spp_grad=spp_grad)
    curves = state.forward.reduction_function(frames)
    loss = dr.mean(dr.square(curves - state.prepared.target_tensor))
    return loss, curves


def _evaluate_states(
    states: Sequence[ModelState],
    optimizer,
    *,
    seed: int,
    spp: int,
) -> tuple[float, dict[int, float], dict[int, np.ndarray]]:
    losses: dict[int, float] = {}
    curves: dict[int, np.ndarray] = {}
    with dr.suspend_grad():
        for model_index, state in enumerate(states):
            _apply_shared_parameters(state, optimizer)
            loss, model_curves = _evaluate(
                state.forward,
                state.prepared.target_tensor,
                seed=seed + model_index,
                spp=spp,
            )
            model_num = state.prepared.model_num
            losses[model_num] = loss
            curves[model_num] = model_curves
    return float(np.mean(list(losses.values()))), losses, curves


def _save_render_history(
    states: Sequence[ModelState],
    optimizer,
    configuration: RunConfiguration,
    trial_name: str,
    iteration: int,
    iteration_width: int,
) -> dict[int, str]:
    paths = {}
    for state in states:
        _apply_shared_parameters(state, optimizer)
        model_num = state.prepared.model_num
        path = (
            configuration.output_dir
            / f"trial_{trial_name}_render_angles_history"
            / f"model{model_num}"
            / f"iteration_{iteration:0{iteration_width}d}.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        _save_render_angles(
            state.forward,
            path,
            title=(
                f"Joint fit model {model_num}: {trial_name}, "
                f"{configuration.projection}, iteration {iteration}"
            ),
        )
        paths[model_num] = str(path)
    return paths


def _saturated_shared_parameters(
    values: Mapping[str, float],
    bounds: Mapping[str, tuple[float, float]],
) -> list[str]:
    saturated = []
    for path, (lower, upper) in bounds.items():
        margin = BOUND_SATURATION_FRACTION * (upper - lower)
        if values[path] <= lower + margin or values[path] >= upper - margin:
            saturated.append(path)
    return saturated


def _threshold_mapping(
    thresholds: np.ndarray | None,
    data_source: DataSource,
) -> dict[str, float] | None:
    if thresholds is None:
        return None
    return {
        experiment: float(thresholds[index])
        for index, experiment in enumerate(_experiment_names(data_source))
    }


def _plot_joint_loss(
    path: Path,
    trial_results: Sequence[Mapping[str, Any]],
) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    positive_losses = []
    for result in trial_results:
        training = np.asarray(result["loss_history"], dtype=float)
        positive_losses.extend(training.tolist())
        (line,) = axis.plot(
            np.arange(1, len(training) + 1),
            training,
            linewidth=1.2,
            alpha=0.65,
            label=f"{result['name']} training",
        )
        validation = result["validation_history"]
        validation_iterations = [entry["iteration"] for entry in validation]
        validation_losses = [entry["aggregate_loss"] for entry in validation]
        positive_losses.extend(validation_losses)
        axis.plot(
            validation_iterations,
            validation_losses,
            marker="o",
            linestyle="--",
            color=line.get_color(),
            label=f"{result['name']} fixed-seed validation",
        )
    if positive_losses and all(loss > 0.0 for loss in positive_losses):
        axis.set_yscale("log")
    axis.set(
        xlabel="Iteration",
        ylabel="Equal-weight model-average MSE",
        title="Joint forward-parameter optimization loss",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _optimize_trial(
    prepared_models: Sequence[PreparedModel],
    bsdf_configuration: BsdfConfiguration,
    configuration: RunConfiguration,
    settings: RenderingSettings,
) -> dict[str, Any]:
    name = bsdf_configuration.name
    print(
        f"\nStarting joint canonical BSDF trial {name!r} for models "
        f"{configuration.model_nums}"
    )
    states = _build_model_states(
        prepared_models,
        bsdf_configuration,
        configuration,
        settings,
    )
    optimizer, initial_parameters = _initialize_shared_optimizer(states)
    parameter_paths = tuple(initial_parameters)
    bounds = _logical_bounds(bsdf_configuration)
    model_count = len(states)

    initial_loss, initial_model_losses, initial_curves = _evaluate_states(
        states,
        optimizer,
        seed=VALIDATION_SEED,
        spp=configuration.validation_spp,
    )
    best_loss = initial_loss
    best_iteration = 0
    best_parameters = initial_parameters.copy()

    loss_history: list[float] = []
    model_loss_history = {state.prepared.model_num: [] for state in states}
    parameter_history: list[dict[str, float]] = []
    validation_history: list[dict[str, Any]] = [
        {
            "iteration": 0,
            "aggregate_loss": initial_loss,
            "model_losses": initial_model_losses,
        }
    ]
    render_history = {state.prepared.model_num: [] for state in states}
    iteration_width = max(3, len(str(configuration.iterations)))
    initial_render_paths = _save_render_history(
        states,
        optimizer,
        configuration,
        name,
        0,
        iteration_width,
    )
    for model_num, path in initial_render_paths.items():
        render_history[model_num].append(path)

    for iteration in range(1, configuration.iterations + 1):
        current_parameters = _optimizer_values(optimizer, parameter_paths)
        iteration_model_losses = {}

        # Backward passes are deliberately sequential. They accumulate into the
        # same optimizer leaves while allowing each model's render graph to be
        # released before the next geometry is rendered.
        for model_index, state in enumerate(states):
            _apply_shared_parameters(state, optimizer)
            loss, _ = _loss_tensor(
                state,
                seed=iteration * model_count + model_index,
                spp_grad=OPTIMIZATION_SPP_GRAD,
            )
            dr.eval(loss)
            model_loss = _scalar(loss)
            if not math.isfinite(model_loss):
                raise RuntimeError(
                    f"Non-finite training loss in trial {name!r}, "
                    f"model {state.prepared.model_num}, iteration {iteration}."
                )
            iteration_model_losses[state.prepared.model_num] = model_loss
            dr.backward(loss / model_count)

        aggregate_loss = float(np.mean(list(iteration_model_losses.values())))
        loss_history.append(aggregate_loss)
        parameter_history.append(current_parameters)
        for model_num, model_loss in iteration_model_losses.items():
            model_loss_history[model_num].append(model_loss)

        non_finite_gradient_paths = [
            path
            for path in parameter_paths
            if not math.isfinite(_scalar(dr.grad(optimizer[path])))
        ]
        if non_finite_gradient_paths:
            raise RuntimeError(
                f"Non-finite accumulated gradients in trial {name!r} for "
                f"parameters {non_finite_gradient_paths}."
            )

        optimizer.step()
        for path, (lower, upper) in bounds.items():
            optimizer[path] = dr.clip(optimizer[path], lower, upper)

        should_validate = (
            iteration % configuration.info_interval == 0
            or iteration == configuration.iterations
        )
        if should_validate:
            validation_loss, validation_model_losses, _ = _evaluate_states(
                states,
                optimizer,
                seed=VALIDATION_SEED,
                spp=configuration.validation_spp,
            )
            validation_history.append(
                {
                    "iteration": iteration,
                    "aggregate_loss": validation_loss,
                    "model_losses": validation_model_losses,
                }
            )
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_iteration = iteration
                best_parameters = _optimizer_values(optimizer, parameter_paths)

            paths = _save_render_history(
                states,
                optimizer,
                configuration,
                name,
                iteration,
                iteration_width,
            )
            for model_num, path in paths.items():
                render_history[model_num].append(path)

            print(
                f"{name}: validation iteration {iteration:03d}, "
                f"aggregate={validation_loss:.8f}, "
                f"models={validation_model_losses}"
            )

        print(
            f"{name}: {iteration:03d}/{configuration.iterations}, "
            f"training aggregate={aggregate_loss:.8f}, "
            f"models={iteration_model_losses}, "
            f"parameters={current_parameters} -> "
            f"{_optimizer_values(optimizer, parameter_paths)}"
        )
        if (
            should_validate
            and iteration - best_iteration >= EARLY_STOP_PATIENCE
            and iteration < configuration.iterations
        ):
            print(
                f"{name}: stopping early after {iteration} iterations "
                f"({EARLY_STOP_PATIENCE} without improving fixed-seed "
                "validation loss)."
            )
            break

    iterations_completed = len(loss_history)
    fitted_bsdf = _fitted_bsdf(
        bsdf_configuration,
        {path: path for path in parameter_paths},
        best_parameters,
    )
    fitted_configuration = BsdfConfiguration(
        name=name,
        bsdf=fitted_bsdf,
        # Equal constant textures may be deduplicated by Mitsuba and therefore
        # need not retain distinct traversal keys. Final verification only needs
        # to rebuild and render the materialized, portable BSDF.
        optimize=(),
    )

    final_model_results = {}
    final_losses = {}
    for state in states:
        model_num = state.prepared.model_num
        del state.forward
        dr.flush_malloc_cache()
        final_forward, final_thresholds, _ = _build_forward(
            state.prepared.mesh_path,
            fitted_configuration,
            configuration,
            settings,
            binary_thresholds=state.binary_thresholds,
        )
        with dr.suspend_grad():
            final_loss, final_curve = _evaluate(
                final_forward,
                state.prepared.target_tensor,
                seed=FINAL_SEED + model_num,
                spp=FINAL_SPP,
            )
        final_losses[model_num] = final_loss
        final_render_path = (
            configuration.output_dir
            / f"model{model_num}"
            / f"trial_{name}_render_angles.png"
        )
        _save_render_angles(
            final_forward,
            final_render_path,
            title=f"Joint fit model {model_num}: {name}, best shared fit",
            seed=FINAL_SEED + model_num,
            spp=FINAL_SPP,
        )
        lightcurve_path = (
            configuration.output_dir
            / f"model{model_num}"
            / f"trial_{name}_lightcurve_fit.png"
        )
        _plot_lightcurves(
            lightcurve_path,
            state.prepared.target_array,
            initial_curves[model_num],
            final_curve,
            title=(
                f"Joint canonical fit, model {model_num} "
                f"({configuration.data_source}/{configuration.curve_type}, "
                f"{configuration.projection}): {name}, MSE={final_loss:.6g}"
            ),
            target_label=(
                f"HAC {configuration.data_source} {configuration.curve_type} target"
            ),
            curve_type=configuration.curve_type,
            observer_names=_experiment_names(configuration.data_source),
        )
        final_model_results[model_num] = {
            "initial_validation_loss": initial_model_losses[model_num],
            "final_loss": final_loss,
            "binary_otsu_thresholds": _threshold_mapping(
                final_thresholds,
                configuration.data_source,
            ),
            "render_grid": str(final_render_path),
            "render_angle_history": render_history[model_num],
            "lightcurve_plot": str(lightcurve_path),
            "validated_with_fresh_canonical_forward_model": True,
        }
        del final_forward
        dr.flush_malloc_cache()

    final_aggregate_loss = float(np.mean(list(final_losses.values())))
    saturated = _saturated_shared_parameters(best_parameters, bounds)
    if saturated:
        print(
            f"WARNING: {name}: shared parameters {saturated} sit on configured "
            "bounds. The fit may be determined by the bounds rather than data."
        )

    return {
        "name": name,
        "model_nums": list(configuration.model_nums),
        "model_weights": {
            str(model_num): 1.0 / len(configuration.model_nums)
            for model_num in configuration.model_nums
        },
        "initial_bsdf": bsdf_configuration.bsdf,
        "fitted_bsdf": fitted_bsdf,
        "bounds": {path: list(value) for path, value in bounds.items()},
        "initial_parameters": initial_parameters,
        "best_parameters": best_parameters,
        "saturated_parameters": saturated,
        "projection": configuration.projection,
        "observer_distances": configuration.observer_distances,
        "curve_type": configuration.curve_type,
        "binary_soft_threshold_temperature": (
            BINARY_SOFT_THRESHOLD_TEMPERATURE
            if configuration.curve_type == "binary"
            else None
        ),
        "initial_validation_loss": initial_loss,
        "best_iteration": best_iteration,
        "best_validation_loss": best_loss,
        "final_aggregate_loss": final_aggregate_loss,
        "final_worst_model_loss": max(final_losses.values()),
        "max_iterations": configuration.iterations,
        "iterations_completed": iterations_completed,
        "stopped_early": iterations_completed < configuration.iterations,
        "info_interval": configuration.info_interval,
        "validation_seed": VALIDATION_SEED,
        "validation_spp": configuration.validation_spp,
        "loss_history": loss_history,
        "model_loss_history": model_loss_history,
        "parameter_history": parameter_history,
        "validation_history": validation_history,
        "models": final_model_results,
    }


def main(configuration: RunConfiguration) -> None:
    configuration.output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    prepared_models = [
        _prepare_model(model_num, configuration)
        for model_num in configuration.model_nums
    ]
    settings = RenderingSettings(
        width=OPTIMIZATION_WIDTH,
        height=OPTIMIZATION_HEIGHT,
        spp=OPTIMIZATION_SPP,
    )

    trial_results = []
    for bsdf_configuration in configuration.bsdf_configurations:
        result = _optimize_trial(
            prepared_models,
            bsdf_configuration,
            configuration,
            settings,
        )
        result_path = configuration.output_dir / f"trial_{result['name']}.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        trial_results.append(result)
        print(f"Saved joint trial to {result_path}")
        dr.flush_malloc_cache()

    loss_plot = configuration.output_dir / "optimization_loss.png"
    _plot_joint_loss(loss_plot, trial_results)
    best = min(trial_results, key=lambda result: result["final_aggregate_loss"])
    summary = {
        "model_nums": list(configuration.model_nums),
        "target_data_source": configuration.data_source,
        "target_curve_type": configuration.curve_type,
        "projection": configuration.projection,
        "observer_distances": configuration.observer_distances,
        "shared_parameters": "BSDF parameters listed in each trial configuration",
        "model_specific_quantities": [
            "known mesh",
            "target lightcurves",
            "binary Otsu thresholds",
        ],
        "models": {
            str(prepared.model_num): {
                "known_model_stl": str(prepared.known_model_stl),
                "fixed_render_mesh": str(prepared.mesh_path),
                "render_face_count": prepared.render_face_count,
            }
            for prepared in prepared_models
        },
        "forward_model": "hac.mitsuba.forward.ForwardModel",
        "integrator": "direct_projective",
        "optimization_phases": OPTIMIZATION_PHASES,
        "max_iterations": configuration.iterations,
        "info_interval": configuration.info_interval,
        "learning_rate": LEARNING_RATE,
        "optimization_spp": OPTIMIZATION_SPP,
        "gradient_spp": OPTIMIZATION_SPP_GRAD,
        "validation_seed": VALIDATION_SEED,
        "validation_spp": configuration.validation_spp,
        "final_spp": FINAL_SPP,
        "best_trial_name": best["name"],
        "best_trial_final_aggregate_loss": best["final_aggregate_loss"],
        "best_trial_final_worst_model_loss": best["final_worst_model_loss"],
        "loss_plot": str(loss_plot),
        "trials": trial_results,
        "elapsed_time_seconds": time.perf_counter() - start_time,
    }
    summary_path = configuration.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"\nBest joint trial: {best['name']} "
        f"(aggregate MSE={best['final_aggregate_loss']:.6g}, "
        f"worst model MSE={best['final_worst_model_loss']:.6g})"
    )
    if best["saturated_parameters"]:
        print(
            "WARNING: the best trial saturated "
            f"{best['saturated_parameters']}; widen bounds or treat those "
            "parameters as weakly identified."
        )
    print(f"Wrote results to {configuration.output_dir}")


if __name__ == "__main__":
    run_configuration = _parse_args()
    projection_description = (
        "canonical orthographic"
        if run_configuration.projection == "orthographic"
        else f"canonical perspective at distances {run_configuration.observer_distances}"
    )
    print(
        f"Jointly optimizing canonical ForwardModel BSDF parameters for models "
        f"{run_configuration.model_nums} with {projection_description} using "
        f"{run_configuration.data_source} {run_configuration.curve_type} targets"
    )
    main(run_configuration)

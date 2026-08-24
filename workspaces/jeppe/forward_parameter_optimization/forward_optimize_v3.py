"""Fit canonical Mitsuba ForwardModel BSDF parameters with a fixed known shape.

The optimizer constructs hac.mitsuba.forward.ForwardModel directly. It supports
the canonical orthographic projection and the canonical perspective projection
at a caller-supplied camera distance. No script-local renderer is used.
"""

# Runtime setup must precede imports from hac.mitsuba.
# ruff: noqa: E402

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

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
    model_num: int
    data_source: DataSource
    curve_type: CurveType
    camera_distance: float | None
    simulated_distance: float | None
    bsdf_configurations: tuple[BsdfConfiguration, ...]
    output_dir: Path
    iterations: int
    info_interval: int

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


def _parse_args(argv: Sequence[str] | None = None) -> RunConfiguration:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("model_num", type=int, choices=(1, 2, 3))
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
            "use canonical perspective projection at this distance; "
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
        help=f"save render-angle history every N iterations (default: {DEFAULT_INFO_INTERVAL})",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    if args.camera_distance is not None and args.camera_distance <= 0.0:
        parser.error("--camera-distance must be positive")
    if args.simulated_distance is not None and args.simulated_distance <= 0.0:
        parser.error("--simulated-distance must be positive")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.info_interval <= 0:
        parser.error("--info-interval must be positive")

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
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    default_output = (
        WORKSPACE_DIR
        / "mitsuba_forward_optimization_v3"
        / (
            f"model{args.model_num}_{args.data_source}_{args.curve_type}_"
            f"{projection}{distance_label}_{timestamp}"
        )
    )
    return RunConfiguration(
        model_num=args.model_num,
        data_source=args.data_source,
        curve_type=args.curve_type,
        camera_distance=camera_distance,
        simulated_distance=simulated_distance,
        bsdf_configurations=_load_bsdf_configurations(args.bsdf_configs),
        output_dir=args.output_dir or default_output,
        iterations=args.iterations,
        info_interval=args.info_interval,
    )


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


def _parameter_values(
    forward: ForwardModel,
    bounds: Mapping[str, tuple[float, float]],
) -> dict[str, float]:
    return {key: _scalar(forward.params[key]) for key in bounds}


def _saturated_parameters(
    values: Mapping[str, float],
    bounds: Mapping[str, tuple[float, float]],
) -> list[str]:
    saturated = []
    for key, (lower, upper) in bounds.items():
        margin = BOUND_SATURATION_FRACTION * (upper - lower)
        if values[key] <= lower + margin or values[key] >= upper - margin:
            saturated.append(key)
    return saturated


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


def _optimize_trial(
    mesh_path: Path,
    bsdf_configuration: BsdfConfiguration,
    configuration: RunConfiguration,
    target,
    settings: RenderingSettings,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    name = bsdf_configuration.name
    print(f"\nStarting canonical BSDF trial {name!r}")
    forward, binary_thresholds, bounds = _build_forward(
        mesh_path,
        bsdf_configuration,
        configuration,
        settings,
    )

    parameter_keys = {
        parameter.path: _resolve_parameter_key(forward.params, parameter.path)
        for parameter in bsdf_configuration.optimize
    }
    initial_parameters = _parameter_values(forward, bounds)
    optimizer = mi.ad.Adam(lr=LEARNING_RATE)
    for key, value in forward.params.items():
        optimizer[key] = value

    def optimizer_values() -> dict[str, float]:
        return {key: _scalar(optimizer[key]) for key in bounds}

    loss_history: list[float] = []
    parameter_history: list[dict[str, float]] = []
    initial_curve = None
    best_loss = math.inf
    best_iteration = 0
    best_parameters = initial_parameters.copy()
    render_history_directory = (
        configuration.output_dir / f"trial_{name}_render_angles_history"
    )
    render_history_paths: list[str] = []
    iteration_width = max(3, len(str(configuration.iterations)))

    def save_render_history(forward_model: ForwardModel, iteration: int) -> None:
        path = (
            render_history_directory / f"iteration_{iteration:0{iteration_width}d}.png"
        )
        _save_render_angles(
            forward_model,
            path,
            title=(
                f"Canonical model {configuration.model_num}: {name}, "
                f"{configuration.projection}, iteration {iteration}"
            ),
        )
        render_history_paths.append(str(path))

    save_render_history(forward, 0)
    last_render_history_iteration = 0

    for iteration in range(1, configuration.iterations + 1):
        forward.params.update(optimizer)
        loss, curves = _evaluate(
            forward,
            target,
            seed=iteration,
            spp_grad=OPTIMIZATION_SPP_GRAD,
            backpropagate=True,
        )
        if initial_curve is None:
            initial_curve = curves.copy()

        current = optimizer_values()
        loss_history.append(loss)
        parameter_history.append(current)
        if loss < best_loss:
            best_loss = loss
            best_iteration = iteration
            best_parameters = current.copy()

        non_finite_gradient_keys = [
            key for key in bounds if not math.isfinite(_scalar(dr.grad(optimizer[key])))
        ]
        if non_finite_gradient_keys:
            raise RuntimeError(
                f"Non-finite gradients in trial {name} for parameters "
                f"{non_finite_gradient_keys}."
            )

        optimizer.step()
        for key, (lower, upper) in bounds.items():
            optimizer[key] = dr.clip(optimizer[key], lower, upper)

        if iteration % configuration.info_interval == 0:
            forward.params.update(optimizer)
            save_render_history(forward, iteration)
            last_render_history_iteration = iteration

        print(
            f"{name}: {iteration:03d}/{configuration.iterations}, "
            f"loss={loss:.8f}, "
            f"parameters={current} -> {optimizer_values()}"
        )
        if iteration - best_iteration >= EARLY_STOP_PATIENCE:
            print(
                f"{name}: stopping early after {iteration} iterations "
                f"({EARLY_STOP_PATIENCE} without improving best loss)."
            )
            break

    iterations_completed = len(loss_history)
    if last_render_history_iteration != iterations_completed:
        forward.params.update(optimizer)
        save_render_history(forward, iterations_completed)

    assert initial_curve is not None

    # The final score is deliberately produced by a newly constructed canonical
    # ForwardModel loaded from a materialized fitted BSDF dictionary. This verifies
    # that the saved BSDF itself is portable, not just the traversed parameter map.
    fitted_bsdf = _fitted_bsdf(
        bsdf_configuration,
        parameter_keys,
        best_parameters,
    )
    fitted_configuration = BsdfConfiguration(
        name=bsdf_configuration.name,
        bsdf=fitted_bsdf,
        optimize=bsdf_configuration.optimize,
    )
    del forward
    dr.flush_malloc_cache()
    final_forward, final_thresholds, final_bounds = _build_forward(
        mesh_path,
        fitted_configuration,
        configuration,
        settings,
        binary_thresholds=binary_thresholds,
    )
    if tuple(final_bounds) != tuple(bounds):
        raise RuntimeError(
            "Fresh canonical ForwardModel exposed different BSDF parameter keys: "
            f"{tuple(bounds)} versus {tuple(final_bounds)}."
        )
    loaded_parameters = _parameter_values(final_forward, final_bounds)
    if not all(
        math.isclose(
            loaded_parameters[key],
            best_parameters[key],
            rel_tol=1e-6,
            abs_tol=1e-7,
        )
        for key in final_bounds
    ):
        raise RuntimeError(
            "Fresh canonical ForwardModel did not load fitted BSDF values: "
            f"{loaded_parameters} versus {best_parameters}."
        )

    with dr.suspend_grad():
        final_loss, final_curve = _evaluate(
            final_forward,
            target,
            seed=FINAL_SEED,
            spp=FINAL_SPP,
        )
    final_render_path = configuration.output_dir / f"trial_{name}_render_angles.png"
    _save_render_angles(
        final_forward,
        final_render_path,
        title=f"Canonical model {configuration.model_num}: {name}, best fit",
        spp=FINAL_SPP,
    )
    saturated = _saturated_parameters(best_parameters, bounds)
    if saturated:
        print(
            f"WARNING: {name}: parameters {saturated} sit on configured bounds. "
            "The fit may be determined by the bounds rather than the data."
        )

    threshold_mapping = (
        None
        if final_thresholds is None
        else {
            experiment: float(final_thresholds[index])
            for index, experiment in enumerate(
                _experiment_names(configuration.data_source)
            )
        }
    )
    result = {
        "name": name,
        "initial_bsdf": bsdf_configuration.bsdf,
        "fitted_bsdf": fitted_bsdf,
        "bounds": {key: list(value) for key, value in bounds.items()},
        "initial_parameters": initial_parameters,
        "best_parameters": best_parameters,
        "saturated_parameters": saturated,
        "projection": configuration.projection,
        "observer_distances": configuration.observer_distances,
        "curve_type": configuration.curve_type,
        "binary_otsu_thresholds": threshold_mapping,
        "binary_soft_threshold_temperature": (
            BINARY_SOFT_THRESHOLD_TEMPERATURE
            if configuration.curve_type == "binary"
            else None
        ),
        "initial_loss": loss_history[0],
        "best_iteration": best_iteration,
        "best_loss": best_loss,
        "final_loss": final_loss,
        "max_iterations": configuration.iterations,
        "info_interval": configuration.info_interval,
        "render_grid": str(final_render_path),
        "render_angle_history_directory": str(render_history_directory),
        "render_angle_history": render_history_paths,
        "iterations_completed": iterations_completed,
        "stopped_early": iterations_completed < configuration.iterations,
        "validated_with_fresh_canonical_forward_model": True,
        "loss_history": loss_history,
        "parameter_history": parameter_history,
    }
    return result, initial_curve, final_curve


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


def _plot_loss(path: Path, trial_results: Sequence[Mapping[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    losses = []
    for result in trial_results:
        values = np.asarray(result["loss_history"], dtype=float)
        losses.extend(values.tolist())
        axis.plot(
            np.arange(1, len(values) + 1),
            values,
            marker="o",
            markersize=3,
            markevery=max(1, len(values) // 12),
            label=result["name"],
        )
    if all(loss > 0.0 for loss in losses):
        axis.set_yscale("log")
    axis.set(
        xlabel="Iteration",
        ylabel="Curve MSE",
        title="Canonical forward-parameter optimization loss",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
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


def main(configuration: RunConfiguration) -> None:
    model_num = configuration.model_num
    output_dir = configuration.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    model_stl_path = get_model_stl_path(model_num=model_num)
    known_mesh = trimesh.load_mesh(model_stl_path)
    original_face_count = len(known_mesh.faces)
    if original_face_count > MAX_KNOWN_MODEL_FACE_COUNT:
        print(
            f"Decimating known model from {original_face_count:,} to "
            f"{MAX_KNOWN_MODEL_FACE_COUNT:,} faces for canonical rendering."
        )
        known_mesh = known_mesh.simplify_quadric_decimation(
            face_count=MAX_KNOWN_MODEL_FACE_COUNT
        )
    known_mesh = normalize_mesh_to_challenge_cylinder(
        known_mesh,
        target_R=MODEL_BASE_RADII[model_num],
    )
    mesh_path = output_dir / "fixed_known_model.obj"
    known_mesh.export(mesh_path)

    target_array, target_tensor = _load_targets(
        model_num,
        configuration.data_source,
        configuration.curve_type,
    )
    settings = RenderingSettings(
        width=OPTIMIZATION_WIDTH,
        height=OPTIMIZATION_HEIGHT,
        spp=OPTIMIZATION_SPP,
    )

    trial_results = []
    for bsdf_configuration in configuration.bsdf_configurations:
        result, initial_curve, final_curve = _optimize_trial(
            mesh_path,
            bsdf_configuration,
            configuration,
            target_tensor,
            settings,
        )
        name = bsdf_configuration.name
        lightcurve_path = output_dir / f"trial_{name}_lightcurve_fit.png"
        _plot_lightcurves(
            lightcurve_path,
            target_array,
            initial_curve,
            final_curve,
            title=(
                f"Canonical model {model_num} "
                f"({configuration.data_source}/{configuration.curve_type}, "
                f"{configuration.projection}): {name}, "
                f"MSE={result['final_loss']:.6g}"
            ),
            target_label=(
                f"HAC {configuration.data_source} {configuration.curve_type} target"
            ),
            curve_type=configuration.curve_type,
            observer_names=_experiment_names(configuration.data_source),
        )
        result["lightcurve_plot"] = str(lightcurve_path)
        (output_dir / f"trial_{name}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        trial_results.append(result)
        print(f"Saved {name} plots to {lightcurve_path} and {result['render_grid']}")
        dr.flush_malloc_cache()

    loss_plot = output_dir / "optimization_loss.png"
    _plot_loss(loss_plot, trial_results)

    best = min(trial_results, key=lambda result: result["final_loss"])
    summary = {
        "model_num": model_num,
        "target_data_source": configuration.data_source,
        "target_curve_type": configuration.curve_type,
        "projection": configuration.projection,
        "observer_distances": configuration.observer_distances,
        "known_model_stl": str(model_stl_path),
        "fixed_render_mesh": str(mesh_path),
        "render_face_count": len(known_mesh.faces),
        "forward_model": "hac.mitsuba.forward.ForwardModel",
        "integrator": "direct_projective",
        "optimization_phases": OPTIMIZATION_PHASES,
        "max_iterations": configuration.iterations,
        "info_interval": configuration.info_interval,
        "learning_rate": LEARNING_RATE,
        "optimization_spp": OPTIMIZATION_SPP,
        "gradient_spp": OPTIMIZATION_SPP_GRAD,
        "final_spp": FINAL_SPP,
        "best_trial_name": best["name"],
        "loss_plot": str(loss_plot),
        "trials": trial_results,
        "elapsed_time_seconds": time.perf_counter() - start_time,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nBest trial: {best['name']} (MSE={best['final_loss']:.6g})")
    if best["saturated_parameters"]:
        print(
            "WARNING: the best trial saturated "
            f"{best['saturated_parameters']}; widen bounds or treat those "
            "parameters as not identified by the normalized lightcurves."
        )
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    run_configuration = _parse_args()
    projection_description = (
        "canonical orthographic"
        if run_configuration.projection == "orthographic"
        else f"canonical perspective at distances {run_configuration.observer_distances}"
    )
    print(
        f"Optimizing canonical ForwardModel BSDF parameters for model "
        f"{run_configuration.model_num} with {projection_description} using "
        f"{run_configuration.data_source} {run_configuration.curve_type} targets"
    )
    main(run_configuration)

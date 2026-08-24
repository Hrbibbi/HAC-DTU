import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import argparse
import datetime
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import drjit as dr
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import trimesh

mi.set_variant(os.environ.get("MITSUBA_VARIANT", "cuda_ad_mono"))

import forward_optimize_v3 as v3
from hac.constants import MODEL_BASE_RADII, WORKSPACE_DIR
from hac.data import get_model_stl_path
from hac.mesh import normalize_mesh_to_challenge_cylinder
from hac.mitsuba.forward import ForwardModel
from hac.mitsuba.scene import RenderingSettings


DEFAULT_MODEL_NUMS = (1, 2, 3)
DEFAULT_VALIDATION_SPP = 32
VALIDATION_SEED = 20260821


@dataclass(frozen=True, slots=True)
class RunConfiguration:
    model_nums: tuple[int, ...]
    data_source: v3.DataSource
    curve_type: v3.CurveType
    camera_distance: float | None
    simulated_distance: float | None
    bsdf_configurations: tuple[v3.BsdfConfiguration, ...]
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
    def projection(self) -> v3.Projection:
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
        default=v3.DEFAULT_ITERATIONS,
        help=f"maximum optimizer iterations (default: {v3.DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--info-interval",
        type=int,
        default=v3.DEFAULT_INFO_INTERVAL,
        help=(
            "run deterministic validation and save renders every N iterations "
            f"(default: {v3.DEFAULT_INFO_INTERVAL})"
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
        bsdf_configurations=v3._load_bsdf_configurations(args.bsdf_configs),
        output_dir=args.output_dir or default_output,
        iterations=args.iterations,
        info_interval=args.info_interval,
        validation_spp=args.validation_spp,
    )


def _single_model_configuration(
    configuration: RunConfiguration,
    model_num: int,
    bsdf_configuration: v3.BsdfConfiguration,
) -> v3.RunConfiguration:
    return v3.RunConfiguration(
        model_num=model_num,
        data_source=configuration.data_source,
        curve_type=configuration.curve_type,
        camera_distance=configuration.camera_distance,
        simulated_distance=configuration.simulated_distance,
        bsdf_configurations=(bsdf_configuration,),
        output_dir=configuration.output_dir,
        iterations=configuration.iterations,
        info_interval=configuration.info_interval,
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
    if original_face_count > v3.MAX_KNOWN_MODEL_FACE_COUNT:
        print(
            f"Model {model_num}: decimating known model from "
            f"{original_face_count:,} to {v3.MAX_KNOWN_MODEL_FACE_COUNT:,} faces."
        )
        known_mesh = known_mesh.simplify_quadric_decimation(
            face_count=v3.MAX_KNOWN_MODEL_FACE_COUNT
        )
    known_mesh = normalize_mesh_to_challenge_cylinder(
        known_mesh,
        target_R=MODEL_BASE_RADII[model_num],
    )
    mesh_path = model_directory / "fixed_known_model.obj"
    known_mesh.export(mesh_path)

    target_array, target_tensor = v3._load_targets(
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
    bsdf_configuration: v3.BsdfConfiguration,
) -> dict[str, tuple[float, float]]:
    return {
        parameter.path: (parameter.lower, parameter.upper)
        for parameter in bsdf_configuration.optimize
    }


def _build_model_states(
    prepared_models: Sequence[PreparedModel],
    bsdf_configuration: v3.BsdfConfiguration,
    configuration: RunConfiguration,
    settings: RenderingSettings,
) -> list[ModelState]:
    states = []
    expected_logical_bounds = _logical_bounds(bsdf_configuration)
    for prepared in prepared_models:
        model_configuration = _single_model_configuration(
            configuration,
            prepared.model_num,
            bsdf_configuration,
        )
        forward, binary_thresholds, scene_bounds = v3._build_forward(
            prepared.mesh_path,
            bsdf_configuration,
            model_configuration,
            settings,
        )
        parameter_keys = {
            parameter.path: v3._resolve_parameter_key(
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
        path: v3._scalar(state.forward.params[scene_key])
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

    optimizer = mi.ad.Adam(lr=v3.LEARNING_RATE)
    first_state = states[0]
    for path, scene_key in first_state.parameter_keys.items():
        optimizer[path] = first_state.forward.params[scene_key]
    return optimizer, initial_parameters


def _optimizer_values(optimizer, parameter_paths: Sequence[str]) -> dict[str, float]:
    return {path: v3._scalar(optimizer[path]) for path in parameter_paths}


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
            loss, model_curves = v3._evaluate(
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
        v3._save_render_angles(
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
        margin = v3.BOUND_SATURATION_FRACTION * (upper - lower)
        if values[path] <= lower + margin or values[path] >= upper - margin:
            saturated.append(path)
    return saturated


def _threshold_mapping(
    thresholds: np.ndarray | None,
    data_source: v3.DataSource,
) -> dict[str, float] | None:
    if thresholds is None:
        return None
    return {
        experiment: float(thresholds[index])
        for index, experiment in enumerate(v3._experiment_names(data_source))
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
    bsdf_configuration: v3.BsdfConfiguration,
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
                spp_grad=v3.OPTIMIZATION_SPP_GRAD,
            )
            dr.eval(loss)
            model_loss = v3._scalar(loss)
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
            if not math.isfinite(v3._scalar(dr.grad(optimizer[path])))
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
            and iteration - best_iteration >= v3.EARLY_STOP_PATIENCE
            and iteration < configuration.iterations
        ):
            print(
                f"{name}: stopping early after {iteration} iterations "
                f"({v3.EARLY_STOP_PATIENCE} without improving fixed-seed "
                "validation loss)."
            )
            break

    iterations_completed = len(loss_history)
    fitted_bsdf = v3._fitted_bsdf(
        bsdf_configuration,
        {path: path for path in parameter_paths},
        best_parameters,
    )
    fitted_configuration = v3.BsdfConfiguration(
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
        final_forward, final_thresholds, _ = v3._build_forward(
            state.prepared.mesh_path,
            fitted_configuration,
            _single_model_configuration(
                configuration,
                model_num,
                fitted_configuration,
            ),
            settings,
            binary_thresholds=state.binary_thresholds,
        )
        with dr.suspend_grad():
            final_loss, final_curve = v3._evaluate(
                final_forward,
                state.prepared.target_tensor,
                seed=v3.FINAL_SEED + model_num,
                spp=v3.FINAL_SPP,
            )
        final_losses[model_num] = final_loss
        final_render_path = (
            configuration.output_dir
            / f"model{model_num}"
            / f"trial_{name}_render_angles.png"
        )
        v3._save_render_angles(
            final_forward,
            final_render_path,
            title=f"Joint fit model {model_num}: {name}, best shared fit",
            seed=v3.FINAL_SEED + model_num,
            spp=v3.FINAL_SPP,
        )
        lightcurve_path = (
            configuration.output_dir
            / f"model{model_num}"
            / f"trial_{name}_lightcurve_fit.png"
        )
        v3._plot_lightcurves(
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
            observer_names=v3._experiment_names(configuration.data_source),
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
            v3.BINARY_SOFT_THRESHOLD_TEMPERATURE
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
        width=v3.OPTIMIZATION_WIDTH,
        height=v3.OPTIMIZATION_HEIGHT,
        spp=v3.OPTIMIZATION_SPP,
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
        "optimization_phases": v3.OPTIMIZATION_PHASES,
        "max_iterations": configuration.iterations,
        "info_interval": configuration.info_interval,
        "learning_rate": v3.LEARNING_RATE,
        "optimization_spp": v3.OPTIMIZATION_SPP,
        "gradient_spp": v3.OPTIMIZATION_SPP_GRAD,
        "validation_seed": VALIDATION_SEED,
        "validation_spp": configuration.validation_spp,
        "final_spp": v3.FINAL_SPP,
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

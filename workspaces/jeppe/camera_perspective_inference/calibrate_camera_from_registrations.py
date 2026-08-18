#!/usr/bin/env python3
"""Fit perspective cameras from ``registrations.json5``.

The JSON file next to this script groups manually identified registrations
by model and source frame. Each point contains ``image_coords`` in pixels and
the corresponding ``mesh_coords`` in world coordinates. For example:

    {
      "model_1_simulation": {
        "0center_frame0": [
          {"name": "tip", "image_coords": [320, 240],
           "mesh_coords": [1.0, 0.0, 0.0]}
        ]
      }
    }

Run the complete calibration with:

    uv run python workspaces/jeppe/camera_perspective_inference/calibrate_camera_from_registrations.py

Every model and frame in the file is calibrated. Each model is written to
``WORKSPACE_DIR/camera_perspective_inference/results/model_NUM.json5``. An
annotated ``model_NUM_FRAME_debug.png`` is written beside it for every frame.

The camera uses Mitsuba sensor-local coordinates, square pixels, and the image
centre as its fixed principal point. Distance and angles are reported relative
to the world origin. At least four non-collinear points are required; six or
more well-spread, non-coplanar points are strongly preferable.

Distance and field-of-view uncertainty estimates use a local linearisation of
the reprojection equations. By default, supplied image coordinates are assumed
to have a one-pixel standard deviation. The value can be changed with
``--pixel-sigma``. The calculation does not include systematic uncertainty in
the mesh coordinates, object centre, principal point, or camera model.
"""

import argparse
import math
import re
import ujson5 as json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from skimage import io

from hac.constants import WORKSPACE_DIR

SCRIPT_DIR = Path(__file__).resolve().parent
REGISTRATIONS_PATH = SCRIPT_DIR / "registrations.json5"
CAMERA_WORKSPACE_DIR = WORKSPACE_DIR / "camera_perspective_inference"
FIRST_FRAMES_DIR = CAMERA_WORKSPACE_DIR / "first_frames"
RESULTS_DIR = CAMERA_WORKSPACE_DIR / "results"
DEFAULT_IMAGE_POINT_STANDARD_DEVIATION_PIXELS = 1.0
MODEL_KEY_PATTERN = re.compile(r"^model_(\d+)_(.+)$")


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    image_width: int
    image_height: int
    registration_count: int
    focal_length_pixels: float
    horizontal_fov_degrees: float
    vertical_fov_degrees: float
    principal_point_pixels: list[float]
    camera_position_world: list[float]
    camera_distance_to_object_center: float
    camera_elevation_degrees: float
    camera_azimuth_degrees: float
    world_to_camera_rotation: list[list[float]]
    world_to_camera_translation: list[float]
    rms_reprojection_error_pixels: float
    max_reprojection_error_pixels: float
    assumed_image_point_standard_deviation_pixels: float
    residual_image_point_standard_deviation_pixels: float
    effective_image_point_standard_deviation_pixels: float
    camera_distance_standard_error: float
    camera_distance_95_percent_confidence_interval: list[float]
    horizontal_fov_standard_error_degrees: float
    horizontal_fov_95_percent_confidence_interval_degrees: list[float]
    vertical_fov_standard_error_degrees: float
    vertical_fov_95_percent_confidence_interval_degrees: list[float]
    registrations: list[dict[str, list[float] | float]]


def _numerical_jacobian(function, parameters: np.ndarray) -> np.ndarray:
    """Evaluate a central finite-difference Jacobian at ``parameters``."""
    parameters = np.asarray(parameters, dtype=np.float64)
    value = np.atleast_1d(np.asarray(function(parameters), dtype=np.float64))
    jacobian = np.empty((len(value), len(parameters)), dtype=np.float64)
    relative_step = np.cbrt(np.finfo(np.float64).eps)

    for index, parameter in enumerate(parameters):
        step = relative_step * max(1.0, abs(float(parameter)))
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += step
        minus[index] -= step
        jacobian[:, index] = (
            np.atleast_1d(function(plus)) - np.atleast_1d(function(minus))
        ) / (2.0 * step)

    return jacobian


def _confidence_interval(value: float, standard_error: float) -> list[float]:
    half_width = 1.96 * standard_error
    return [value - half_width, value + half_width]


def _project(
    world_points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    focal_length: float,
    principal_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points using Mitsuba left/up/forward sensor coordinates."""
    camera_points = world_points @ rotation.T + translation
    depth = camera_points[:, 2]
    pixels = np.empty((len(world_points), 2), dtype=np.float64)
    pixels[:, 0] = principal_point[0] - focal_length * camera_points[:, 0] / depth
    pixels[:, 1] = principal_point[1] - focal_length * camera_points[:, 1] / depth
    return pixels, depth


def _look_at_rotation(camera_position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a world-to-camera rotation in Mitsuba sensor coordinates."""
    forward = target - camera_position
    forward /= np.linalg.norm(forward)
    world_up = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, world_up))) > 0.95:
        world_up = np.asarray([0.0, 1.0, 0.0])
    left = np.cross(world_up, forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    up /= np.linalg.norm(up)
    return np.stack([left, up, forward])


def _initial_rotations(
    object_center: np.ndarray, radius: float
) -> tuple[np.ndarray, ...]:
    """Provide broad look-at starts; least squares refines all pose variables."""
    directions = []
    for elevation_degrees in (-45.0, -20.0, 0.0, 20.0, 45.0):
        elevation = math.radians(elevation_degrees)
        for azimuth_degrees in range(0, 360, 45):
            azimuth = math.radians(azimuth_degrees)
            directions.append(
                np.asarray(
                    [
                        math.cos(azimuth) * math.cos(elevation),
                        math.sin(azimuth) * math.cos(elevation),
                        math.sin(elevation),
                    ]
                )
            )
    return tuple(
        _look_at_rotation(object_center + 3.0 * radius * direction, object_center)
        for direction in directions
    )


def calibrate_camera(
    image_points: np.ndarray,
    world_points: np.ndarray,
    image_size: tuple[int, int],
    object_center: np.ndarray,
    principal_point: np.ndarray,
    fit_principal_point: bool,
    robust: bool,
    image_point_standard_deviation_pixels: float = (
        DEFAULT_IMAGE_POINT_STANDARD_DEVIATION_PIXELS
    ),
) -> CalibrationResult:
    """Fit the camera that minimises pixel reprojection error."""
    if (
        not math.isfinite(image_point_standard_deviation_pixels)
        or image_point_standard_deviation_pixels <= 0.0
    ):
        raise ValueError("Image-point standard deviation must be positive and finite")
    if len(image_points) != len(world_points):
        raise ValueError("Image and world point counts differ")
    minimum_points = 5 if fit_principal_point else 4
    if len(image_points) < minimum_points:
        raise ValueError(
            f"Need at least {minimum_points} registrations for this fit; got "
            f"{len(image_points)}"
        )
    if np.linalg.matrix_rank(world_points - world_points[0]) < 2:
        raise ValueError("The supplied world coordinates must not be collinear")

    width, height = image_size
    centred_world = world_points - object_center
    radius = float(np.max(np.linalg.norm(centred_world, axis=1)))
    if radius <= 0.0:
        raise ValueError("World points have zero spatial extent about object center")
    pixel_span = max(
        float(np.ptp(image_points[:, 0])), float(np.ptp(image_points[:, 1]))
    )
    world_span = float(np.max(np.ptp(world_points, axis=0)))
    if pixel_span <= 0.0 or world_span <= 0.0:
        raise ValueError("Registrations need non-zero image and world extent")

    parameter_count = 9 if fit_principal_point else 7

    def unpack(parameters: np.ndarray):
        rotation = Rotation.from_rotvec(parameters[:3]).as_matrix()
        translation = np.asarray(
            [
                parameters[3],
                parameters[4],
                radius + math.exp(parameters[5]),
            ]
        )
        focal_length = math.exp(parameters[6])
        fitted_principal_point = (
            parameters[7:9] if fit_principal_point else principal_point
        )
        return rotation, translation, focal_length, fitted_principal_point

    def residual(parameters: np.ndarray) -> np.ndarray:
        rotation, translation, focal_length, fitted_principal_point = unpack(parameters)
        reprojected, depth = _project(
            centred_world,
            rotation,
            translation,
            focal_length,
            fitted_principal_point,
        )
        residuals = (reprojected - image_points).ravel()
        # A finite penalty makes invalid trial poses unattractive while keeping
        # least_squares numerically well behaved.
        behind_camera = np.minimum(depth, 0.0)
        return np.concatenate([residuals, 10_000.0 * behind_camera])

    def reprojection_residual(parameters: np.ndarray) -> np.ndarray:
        rotation, translation, focal_length, fitted_principal_point = unpack(parameters)
        reprojected, _ = _project(
            centred_world,
            rotation,
            translation,
            focal_length,
            fitted_principal_point,
        )
        return (reprojected - image_points).ravel()

    maximum_dimension = max(width, height)
    focal_starts = maximum_dimension * np.asarray([0.5, 1.0, 2.0, 4.0])
    best = None
    for rotation0 in _initial_rotations(object_center, radius):
        rotvec0 = Rotation.from_matrix(rotation0).as_rotvec()
        for focal0 in focal_starts:
            distance0 = max(
                2.0 * radius,
                focal0 * world_span / pixel_span,
            )
            mean_offset = np.mean(image_points, axis=0) - principal_point
            parameters0 = [
                *rotvec0,
                -mean_offset[0] * distance0 / focal0,
                -mean_offset[1] * distance0 / focal0,
                math.log(distance0 - radius),
                math.log(focal0),
            ]
            if fit_principal_point:
                parameters0.extend(principal_point)
            candidate = least_squares(
                residual,
                parameters0,
                method="trf",
                loss="soft_l1" if robust else "linear",
                f_scale=2.0,
                max_nfev=5_000,
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
            )
            if best is None or np.sum(candidate.fun**2) < np.sum(best.fun**2):
                best = candidate

    if best is None:
        raise RuntimeError("Camera optimisation did not run")
    reprojection_jacobian = _numerical_jacobian(reprojection_residual, best.x)
    _, singular_values, right_singular_vectors = np.linalg.svd(
        reprojection_jacobian,
        full_matrices=False,
    )
    if len(singular_values) < parameter_count or singular_values[-1] <= (
        singular_values[0] * 1e-10
    ):
        raise ValueError(
            "This registration set does not separately identify all fitted "
            "camera parameters. Add non-coplanar or more oblique points, fix "
            "the principal point, or fix either FOV or distance."
        )

    rotation, translation, focal_length, fitted_principal_point = unpack(best.x)
    reprojected, depth = _project(
        centred_world,
        rotation,
        translation,
        focal_length,
        fitted_principal_point,
    )
    if np.any(depth <= 0.0):
        raise RuntimeError("Optimised camera placed one or more points behind it")

    camera_position = object_center - rotation.T @ translation
    offset = camera_position - object_center
    distance = float(np.linalg.norm(offset))
    elevation = math.degrees(math.atan2(offset[2], np.linalg.norm(offset[:2])))
    azimuth = math.degrees(math.atan2(offset[1], offset[0]))
    errors = np.linalg.norm(reprojected - image_points, axis=1)

    residual_degrees_of_freedom = 2 * len(image_points) - parameter_count
    residual_pixel_sigma = math.sqrt(
        float(np.sum((reprojected - image_points) ** 2)) / residual_degrees_of_freedom
    )
    effective_pixel_sigma = max(
        image_point_standard_deviation_pixels,
        residual_pixel_sigma,
    )
    inverse_squared_singular_values = np.diag(1.0 / singular_values**2)
    parameter_covariance = (
        effective_pixel_sigma**2
        * right_singular_vectors.T
        @ inverse_squared_singular_values
        @ right_singular_vectors
    )

    def derived_camera_quantities(parameters: np.ndarray) -> np.ndarray:
        fitted_rotation, fitted_translation, fitted_focal_length, _ = unpack(parameters)
        fitted_camera_position = object_center - fitted_rotation.T @ fitted_translation
        fitted_distance = np.linalg.norm(fitted_camera_position - object_center)
        horizontal_fov = math.degrees(
            2.0 * math.atan(width / (2.0 * fitted_focal_length))
        )
        vertical_fov = math.degrees(
            2.0 * math.atan(height / (2.0 * fitted_focal_length))
        )
        return np.asarray([fitted_distance, horizontal_fov, vertical_fov])

    derived_jacobian = _numerical_jacobian(derived_camera_quantities, best.x)
    derived_covariance = derived_jacobian @ parameter_covariance @ derived_jacobian.T
    derived_standard_errors = np.sqrt(np.maximum(np.diag(derived_covariance), 0.0))
    distance_standard_error = float(derived_standard_errors[0])
    horizontal_fov_standard_error = float(derived_standard_errors[1])
    vertical_fov_standard_error = float(derived_standard_errors[2])
    horizontal_fov = math.degrees(2.0 * math.atan(width / (2.0 * focal_length)))
    vertical_fov = math.degrees(2.0 * math.atan(height / (2.0 * focal_length)))

    return CalibrationResult(
        image_width=width,
        image_height=height,
        registration_count=len(image_points),
        focal_length_pixels=float(focal_length),
        horizontal_fov_degrees=horizontal_fov,
        vertical_fov_degrees=vertical_fov,
        principal_point_pixels=np.asarray(fitted_principal_point).tolist(),
        camera_position_world=camera_position.tolist(),
        camera_distance_to_object_center=distance,
        camera_elevation_degrees=elevation,
        camera_azimuth_degrees=azimuth,
        world_to_camera_rotation=rotation.tolist(),
        world_to_camera_translation=translation.tolist(),
        rms_reprojection_error_pixels=float(np.sqrt(np.mean(errors**2))),
        max_reprojection_error_pixels=float(np.max(errors)),
        assumed_image_point_standard_deviation_pixels=(
            image_point_standard_deviation_pixels
        ),
        residual_image_point_standard_deviation_pixels=residual_pixel_sigma,
        effective_image_point_standard_deviation_pixels=effective_pixel_sigma,
        camera_distance_standard_error=distance_standard_error,
        camera_distance_95_percent_confidence_interval=_confidence_interval(
            distance,
            distance_standard_error,
        ),
        horizontal_fov_standard_error_degrees=horizontal_fov_standard_error,
        horizontal_fov_95_percent_confidence_interval_degrees=_confidence_interval(
            horizontal_fov,
            horizontal_fov_standard_error,
        ),
        vertical_fov_standard_error_degrees=vertical_fov_standard_error,
        vertical_fov_95_percent_confidence_interval_degrees=_confidence_interval(
            vertical_fov,
            vertical_fov_standard_error,
        ),
        registrations=[
            {
                "pixel": pixel.tolist(),
                "world": world.tolist(),
                "reprojected_pixel": projected.tolist(),
                "reprojection_error_pixels": float(error),
            }
            for pixel, world, projected, error in zip(
                image_points, world_points, reprojected, errors
            )
        ],
    )


def _parse_registrations(
    frame_name: str,
    values: object,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not isinstance(values, list) or not values:
        raise ValueError(
            f"{frame_name!r} must contain a non-empty list of registrations"
        )

    image_points: list[np.ndarray] = []
    world_points: list[np.ndarray] = []
    point_names: list[str] = []
    for index, item in enumerate(values, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                f"{frame_name!r} registration {index} must be a JSON object"
            )
        point_name = str(item.get("name", index))
        try:
            image_point = np.asarray(item["image_coords"], dtype=np.float64)
            world_point = np.asarray(item["mesh_coords"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{frame_name!r} registration {point_name!r} must contain "
                "numeric image_coords and mesh_coords"
            ) from error
        if image_point.shape != (2,) or world_point.shape != (3,):
            raise ValueError(
                f"{frame_name!r} registration {point_name!r} needs two "
                "image coordinates and three mesh coordinates"
            )
        if not np.all(np.isfinite(image_point)) or not np.all(np.isfinite(world_point)):
            raise ValueError(
                f"{frame_name!r} registration {point_name!r} contains a "
                "non-finite coordinate"
            )
        image_points.append(image_point)
        world_points.append(world_point)
        point_names.append(point_name)

    return np.stack(image_points), np.stack(world_points), point_names


def _load_models() -> dict[str, tuple[int, str, dict[str, object]]]:
    try:
        data = json.loads(REGISTRATIONS_PATH.read_text())
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Registration file not found: {REGISTRATIONS_PATH}"
        ) from error

    if not isinstance(data, dict) or not data:
        raise ValueError(f"{REGISTRATIONS_PATH} must contain a non-empty JSON object")

    models: dict[str, tuple[int, str, dict[str, object]]] = {}
    for model_key, raw_frames in data.items():
        match = MODEL_KEY_PATTERN.match(model_key) if isinstance(model_key, str) else None
        if match is None:
            raise ValueError(
                f"Invalid model key {model_key!r}; expected model_NUM_ORIGIN"
            )
        if not isinstance(raw_frames, dict) or not raw_frames:
            raise ValueError(f"{model_key!r} must contain a non-empty object of frames")

        frames: dict[str, object] = {}
        for frame_name, values in raw_frames.items():
            if not isinstance(frame_name, str):
                raise ValueError(f"{model_key!r} contains a non-string frame name")
            frames[frame_name] = values
        models[model_key] = (
            int(match.group(1)),
            match.group(2),
            frames,
        )

    return models


def _frame_image_path(model_key: str, frame_name: str) -> Path:
    frame_path = Path(frame_name)
    if frame_path.name != frame_name or frame_path.suffix not in ("", ".png"):
        raise ValueError(
            f"Frame name {frame_name!r} must be a filename with an optional .png suffix"
        )

    image_name = frame_name if frame_path.suffix else f"{frame_name}.png"
    image_path = FIRST_FRAMES_DIR / model_key / image_name
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Source frame for {frame_name!r} not found: {image_path}"
        )
    return image_path


def _write_debug_image(
    path: Path,
    image: np.ndarray,
    image_points: np.ndarray,
    point_names: list[str],
    result: CalibrationResult,
) -> None:
    reprojected = np.asarray(
        [item["reprojected_pixel"] for item in result.registrations]
    )
    errors = np.asarray(
        [item["reprojection_error_pixels"] for item in result.registrations]
    )

    figure, axis = plt.subplots(figsize=(10, 8))
    axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
    axis.scatter(
        image_points[:, 0],
        image_points[:, 1],
        c="red",
        marker="x",
        s=70,
        label="Supplied pixel",
    )
    axis.scatter(
        reprojected[:, 0],
        reprojected[:, 1],
        edgecolors="cyan",
        facecolors="none",
        marker="o",
        s=70,
        label="Fitted projection",
    )
    for point_name, pixel, projected, error in zip(
        point_names, image_points, reprojected, errors, strict=True
    ):
        axis.plot(
            [pixel[0], projected[0]],
            [pixel[1], projected[1]],
            color="yellow",
            linewidth=0.8,
        )
        axis.annotate(
            f"{point_name} ({error:.2f}px)",
            pixel + np.asarray([5.0, -5.0]),
            color="yellow",
        )

    axis.set_xlim(-0.5, image.shape[1] - 0.5)
    axis.set_ylim(image.shape[0] - 0.5, -0.5)
    axis.set_title(
        "Camera calibration: "
        f"RMS {result.rms_reprojection_error_pixels:.2f}px, "
        f"max {result.max_reprojection_error_pixels:.2f}px"
    )
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pixel-sigma",
        type=float,
        default=DEFAULT_IMAGE_POINT_STANDARD_DEVIATION_PIXELS,
        help=(
            "assumed standard deviation of each manually marked image "
            "coordinate in pixels (default: %(default)s)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    models = _load_models()
    object_center = np.zeros(3, dtype=np.float64)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for model_key, (model_number, origin, frames) in sorted(
        models.items(),
        key=lambda item: (item[1][0], item[1][1]),
    ):
        calibrations: dict[str, object] = {}
        for frame_name, values in frames.items():
            image_path = _frame_image_path(model_key, frame_name)
            image = io.imread(image_path)
            height, width = image.shape[:2]
            image_points, world_points, point_names = _parse_registrations(
                frame_name,
                values,
            )
            principal_point = np.asarray(
                ((width - 1) / 2.0, (height - 1) / 2.0),
                dtype=np.float64,
            )
            try:
                result = calibrate_camera(
                    image_points=image_points,
                    world_points=world_points,
                    image_size=(width, height),
                    object_center=object_center,
                    principal_point=principal_point,
                    fit_principal_point=False,
                    robust=False,
                    image_point_standard_deviation_pixels=args.pixel_sigma,
                )
            except (RuntimeError, ValueError) as error:
                raise ValueError(
                    f"Could not calibrate {model_key}/{frame_name}: {error}"
                ) from error

            debug_path = RESULTS_DIR / f"{model_key}_{Path(frame_name).stem}_debug.png"
            _write_debug_image(
                debug_path,
                image,
                image_points,
                point_names,
                result,
            )
            calibrations[frame_name] = {
                "source_image": str(image_path),
                "debug_image": str(debug_path),
                **asdict(result),
            }

        payload = {
            "model_number": model_number,
            "origin": origin,
            "source_registrations": str(REGISTRATIONS_PATH),
            "settings": {
                "object_center": object_center.tolist(),
                "fit_principal_point": False,
                "robust": False,
                "assumed_image_point_standard_deviation_pixels": args.pixel_sigma,
                "uncertainty_method": (
                    "local linear covariance with the larger of the assumed "
                    "pixel sigma and residual pixel sigma"
                ),
            },
            "calibrations": calibrations,
        }
        result_path = RESULTS_DIR / f"{model_key}.json5"
        result_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {result_path}")
        for calibration in calibrations.values():
            print(f"Wrote {calibration['debug_image']}")


if __name__ == "__main__":
    main()

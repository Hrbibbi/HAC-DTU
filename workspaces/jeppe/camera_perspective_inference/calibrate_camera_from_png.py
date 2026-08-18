#!/usr/bin/env python3
"""Estimate a pinhole camera from four silhouette corners in a PNG.

The foreground is separated from the border colour and its four diagonal
extrema are returned in TL, TR, BR, BL order.  The caller supplies the 3D
point corresponding to each detected corner.  A perspective camera with
square pixels and a fixed principal point is then fitted by reprojection-error
minimisation.

Example for a cube whose detected corners correspond to the supplied vertices:

    uv run python workspaces/jeppe/calibrate_camera_from_png.py cube.png \
        --corner TL -1 -1  1 \
        --corner TR -1  1  1 \
        --corner BR -1  1 -1 \
        --corner BL -1 -1 -1 \
        --annotated cube_corners.png

Coordinates and the reported camera distance use the same units.  By default,
distance is measured to world-coordinate origin; use ``--object-center`` when
the object is centred elsewhere.
"""
import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from skimage import io
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects


CORNER_ORDER = ("TL", "TR", "BR", "BL")


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    image_width: int
    image_height: int
    focal_length_pixels: float
    horizontal_fov_degrees: float
    vertical_fov_degrees: float
    principal_point_pixels: list[float]
    camera_position_world: list[float]
    camera_distance_to_object_center: float
    camera_elevation_degrees: float
    world_to_camera_rotation: list[list[float]]
    world_to_camera_translation: list[float]
    detected_corners_pixels: dict[str, list[float]]
    reprojected_corners_pixels: dict[str, list[float]]
    rms_reprojection_error_pixels: float


def _normalise_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float64) / np.iinfo(image.dtype).max
    else:
        image = image.astype(np.float64)
    return np.clip(image, 0.0, 1.0)


def foreground_mask(
    image: np.ndarray,
    threshold: float | None,
    minimum_component_size: int,
) -> np.ndarray:
    """Separate the object from a nearly uniform image-border background."""
    image = _normalise_image(image)
    if image.ndim == 2:
        channels = image[..., None]
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        channels = image[..., :3]
    else:
        raise ValueError(f"Unsupported PNG array shape: {image.shape}")

    border = np.concatenate(
        [channels[0], channels[-1], channels[:, 0], channels[:, -1]], axis=0
    )
    background = np.median(border, axis=0)
    difference = np.linalg.norm(channels - background, axis=2)

    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[..., 3]
        border_alpha = np.median(
            np.concatenate([alpha[0], alpha[-1], alpha[:, 0], alpha[:, -1]])
        )
        difference = np.maximum(difference, np.abs(alpha - border_alpha))

    if threshold is None:
        nonzero = difference[difference > np.finfo(float).eps]
        if nonzero.size == 0:
            raise ValueError("The PNG is indistinguishable from its border colour")
        threshold = float(threshold_otsu(difference))
    if threshold < 0.0:
        raise ValueError("Foreground threshold must be non-negative")

    mask = difference > threshold
    if minimum_component_size > 1:
        mask = remove_small_objects(mask, max_size=minimum_component_size - 1)
    if not np.any(mask):
        raise ValueError(
            "Foreground detection produced an empty mask; try --threshold with "
            "a smaller value"
        )
    return mask


def diagonal_extrema(mask: np.ndarray) -> np.ndarray:
    """Return foreground extrema in TL, TR, BR, BL pixel order."""
    rows, columns = np.nonzero(mask)
    points = np.column_stack([columns, rows]).astype(np.float64)
    x = points[:, 0]
    y = points[:, 1]
    indices = (
        int(np.argmin(x + y)),
        int(np.argmax(x - y)),
        int(np.argmax(x + y)),
        int(np.argmin(x - y)),
    )
    corners = points[list(indices)]
    if len(np.unique(corners, axis=0)) != 4:
        raise ValueError(
            "The four diagonal foreground extrema are not distinct. The object "
            "may not have a quadrilateral silhouette."
        )
    return corners


def _initial_rotation(world_points: np.ndarray) -> np.ndarray:
    """Construct a weak-perspective rotation from ordered quadrilateral edges."""
    right = (world_points[1] - world_points[0]) + (
        world_points[2] - world_points[3]
    )
    down = (world_points[3] - world_points[0]) + (
        world_points[2] - world_points[1]
    )
    right /= np.linalg.norm(right)
    image_up = -down + np.dot(down, right) * right
    image_up /= np.linalg.norm(image_up)
    forward = np.cross(right, image_up)
    forward /= np.linalg.norm(forward)
    rotation = np.stack([right, image_up, forward])
    if np.linalg.det(rotation) < 0.0:
        rotation[2] *= -1.0
    return rotation


def _project(
    centred_world_points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    focal_length: float,
    principal_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    camera_points = centred_world_points @ rotation.T + translation
    depth = camera_points[:, 2]
    pixels = np.empty((len(camera_points), 2), dtype=np.float64)
    pixels[:, 0] = principal_point[0] + focal_length * camera_points[:, 0] / depth
    pixels[:, 1] = principal_point[1] - focal_length * camera_points[:, 1] / depth
    return pixels, depth


def calibrate_camera(
    image_points: np.ndarray,
    world_points: np.ndarray,
    image_size: tuple[int, int],
    principal_point: np.ndarray,
    object_center: np.ndarray,
) -> CalibrationResult:
    """Fit focal length and a six-degree-of-freedom camera pose."""
    if image_points.shape != (4, 2) or world_points.shape != (4, 3):
        raise ValueError("Exactly four 2D corners and four 3D points are required")
    if np.linalg.matrix_rank(world_points - world_points[0]) < 2:
        raise ValueError("The supplied 3D corners must not be collinear")

    width, height = image_size
    centred = world_points - object_center
    object_radius = float(np.max(np.linalg.norm(centred, axis=1)))
    if object_radius <= 0.0:
        raise ValueError("The supplied 3D points have zero spatial extent")

    rotation0 = _initial_rotation(world_points)
    rotvec0 = Rotation.from_matrix(rotation0).as_rotvec()
    pixel_span = max(float(np.ptp(image_points[:, 0])), float(np.ptp(image_points[:, 1])))
    world_span = max(
        np.linalg.norm(world_points[1] - world_points[0]),
        np.linalg.norm(world_points[3] - world_points[0]),
    )

    def unpack(parameters: np.ndarray):
        rotation = Rotation.from_rotvec(parameters[:3]).as_matrix()
        focal_length = math.exp(parameters[6])
        # Keeping translation-z beyond a bounding sphere guarantees that every
        # calibration point remains in front of the pinhole camera.
        translation = np.asarray(
            [
                parameters[3],
                parameters[4],
                object_radius + math.exp(parameters[5]),
            ]
        )
        return rotation, translation, focal_length

    def residual(parameters: np.ndarray) -> np.ndarray:
        rotation, translation, focal_length = unpack(parameters)
        pixels, _ = _project(
            centred, rotation, translation, focal_length, principal_point
        )
        return (pixels - image_points).ravel()

    best = None
    maximum_dimension = max(width, height)
    for focal0 in maximum_dimension * np.asarray([0.5, 1.0, 2.0, 4.0, 8.0]):
        distance0 = max(
            2.0 * object_radius,
            focal0 * world_span / max(pixel_span, 1.0),
        )
        mean_offset = np.mean(image_points, axis=0) - principal_point
        tx0 = mean_offset[0] * distance0 / focal0
        ty0 = -mean_offset[1] * distance0 / focal0
        parameters0 = np.concatenate(
            [
                rotvec0,
                [tx0, ty0, math.log(distance0 - object_radius), math.log(focal0)],
            ]
        )
        candidate = least_squares(
            residual,
            parameters0,
            method="trf",
            max_nfev=20_000,
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
        )
        if best is None or np.sum(candidate.fun**2) < np.sum(best.fun**2):
            best = candidate

    if best is None:
        raise RuntimeError("Camera optimisation did not run")

    singular_values = np.linalg.svd(best.jac, compute_uv=False)
    if singular_values[-1] <= singular_values[0] * 1e-10:
        raise ValueError(
            "FOV and distance are not separately identifiable from these "
            "corners. Use a tilted view, non-coplanar corners, or fix one of "
            "the two quantities."
        )

    rotation, translation, focal_length = unpack(best.x)
    reprojected, depth = _project(
        centred, rotation, translation, focal_length, principal_point
    )
    if np.any(depth <= 0.0):
        raise RuntimeError("Optimised camera placed calibration points behind it")

    camera_position = object_center - rotation.T @ translation
    camera_offset = camera_position - object_center
    distance = float(np.linalg.norm(camera_offset))
    elevation = math.degrees(
        math.atan2(camera_offset[2], np.linalg.norm(camera_offset[:2]))
    )
    rms = float(np.sqrt(np.mean(np.sum((reprojected - image_points) ** 2, axis=1))))

    return CalibrationResult(
        image_width=width,
        image_height=height,
        focal_length_pixels=float(focal_length),
        horizontal_fov_degrees=math.degrees(
            2.0 * math.atan(width / (2.0 * focal_length))
        ),
        vertical_fov_degrees=math.degrees(
            2.0 * math.atan(height / (2.0 * focal_length))
        ),
        principal_point_pixels=principal_point.tolist(),
        camera_position_world=camera_position.tolist(),
        camera_distance_to_object_center=distance,
        camera_elevation_degrees=elevation,
        world_to_camera_rotation=rotation.tolist(),
        world_to_camera_translation=translation.tolist(),
        detected_corners_pixels={
            name: point.tolist() for name, point in zip(CORNER_ORDER, image_points)
        },
        reprojected_corners_pixels={
            name: point.tolist() for name, point in zip(CORNER_ORDER, reprojected)
        },
        rms_reprojection_error_pixels=rms,
    )


def _parse_world_corners(values: list[list[str]]) -> np.ndarray:
    supplied: dict[str, np.ndarray] = {}
    for label, x, y, z in values:
        label = label.upper()
        if label not in CORNER_ORDER:
            raise ValueError(f"Corner label must be one of {CORNER_ORDER}; got {label!r}")
        if label in supplied:
            raise ValueError(f"Corner {label} was supplied more than once")
        supplied[label] = np.asarray([float(x), float(y), float(z)])
    missing = set(CORNER_ORDER) - supplied.keys()
    if missing:
        raise ValueError(f"Missing 3D coordinates for corners: {sorted(missing)}")
    return np.stack([supplied[name] for name in CORNER_ORDER])


def _write_annotation(
    path: Path,
    image: np.ndarray,
    detected: np.ndarray,
    reprojected: np.ndarray,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 8))
    axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
    axis.scatter(detected[:, 0], detected[:, 1], c="red", marker="x", s=90)
    axis.scatter(
        reprojected[:, 0],
        reprojected[:, 1],
        facecolors="none",
        edgecolors="cyan",
        marker="o",
        s=90,
    )
    for name, point in zip(CORNER_ORDER, detected):
        axis.annotate(name, point + np.asarray([5.0, -5.0]), color="yellow")
    axis.set_xlim(0, image.shape[1])
    axis.set_ylim(image.shape[0], 0)
    axis.set_title("Detected (red x) and reprojected (cyan circle) corners")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("png", type=Path, help="Input PNG with a uniform background")
    parser.add_argument(
        "--corner",
        nargs=4,
        action="append",
        required=True,
        metavar=("LABEL", "X", "Y", "Z"),
        help="3D coordinate for detected corner LABEL in TL, TR, BR, BL",
    )
    parser.add_argument(
        "--object-center",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Point to which camera distance is reported (default: origin)",
    )
    parser.add_argument(
        "--principal-point",
        nargs=2,
        type=float,
        metavar=("CX", "CY"),
        help="Principal point in pixels (default: image centre)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Foreground/background colour-distance threshold in [0, 1]",
    )
    parser.add_argument(
        "--minimum-component-size",
        type=int,
        default=64,
        help="Remove foreground components smaller than this many pixels",
    )
    parser.add_argument("--annotated", type=Path, help="Write an annotated PNG")
    parser.add_argument("--json", type=Path, help="Also write results as JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = io.imread(args.png)
    height, width = image.shape[:2]
    mask = foreground_mask(image, args.threshold, args.minimum_component_size)
    image_points = diagonal_extrema(mask)
    world_points = _parse_world_corners(args.corner)
    principal_point = np.asarray(
        args.principal_point
        if args.principal_point is not None
        else ((width - 1) / 2.0, (height - 1) / 2.0),
        dtype=np.float64,
    )
    result = calibrate_camera(
        image_points=image_points,
        world_points=world_points,
        image_size=(width, height),
        principal_point=principal_point,
        object_center=np.asarray(args.object_center, dtype=np.float64),
    )
    result_dict = asdict(result)
    output = json.dumps(result_dict, indent=2)
    print(output)

    if args.json is not None:
        args.json.write_text(output + "\n")
    if args.annotated is not None:
        reprojected = np.stack(
            [result.reprojected_corners_pixels[name] for name in CORNER_ORDER]
        )
        _write_annotation(args.annotated, image, image_points, reprojected)


if __name__ == "__main__":
    main()

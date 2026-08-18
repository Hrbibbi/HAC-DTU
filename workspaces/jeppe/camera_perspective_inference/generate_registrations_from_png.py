#!/usr/bin/env python3
"""Generate registrations.json entries from a square-ish model-2 PNG.

The foreground detector and TL, TR, BR, BL diagonal extrema come from
calibrate_camera_from_png.py. By default, the extrema are paired with the
normalized model-2 mesh vertices visible in 0center_frame0.png and emitted
as a complete JSON fragment under model_2.

The JSON fragment is printed to stdout. An annotated image is always written
beside the input as ``INPUT_STEM_registrations.png``.

Example, optionally redirecting the JSON into a file:

    uv run python \
        workspaces/jeppe/camera_perspective_inference/\
generate_registrations_from_png.py \
        /work3/projects/HAC-DTU/workspaces/jeppe/\
camera_perspective_inference/first_frames/model_2/0center_frame0.png \
        > /tmp/model_2.json

Use four --corner LABEL X Y Z options to override the model-2 defaults for
another view. The labels must be TL, TR, BR, and BL.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from calibrate_camera_from_png import (
    CORNER_ORDER,
    diagonal_extrema,
    foreground_mask,
)
from skimage import io

MODEL2_0CENTER_WORLD_CORNERS = np.asarray(
    [
        [-0.9967411756515503, 0.9967411756515503, 1.0],
        [-0.9967411756515503, -0.9967411756515503, 1.0],
        [-0.9968111515045166, -0.9968111515045166, -1.0],
        [-0.9968111515045166, 0.9968111515045166, -1.0],
    ],
    dtype=np.float64,
)


def _parse_world_corners(values: list[list[str]] | None) -> np.ndarray:
    if values is None:
        return MODEL2_0CENTER_WORLD_CORNERS.copy()

    supplied: dict[str, np.ndarray] = {}
    for label, x, y, z in values:
        label = label.upper()
        if label not in CORNER_ORDER:
            raise ValueError(
                f"Corner label must be one of {CORNER_ORDER}; got {label!r}"
            )
        if label in supplied:
            raise ValueError(f"Corner {label} was supplied more than once")
        supplied[label] = np.asarray(
            [float(x), float(y), float(z)],
            dtype=np.float64,
        )

    missing = set(CORNER_ORDER) - supplied.keys()
    if missing:
        raise ValueError(f"Missing 3D coordinates for corners: {sorted(missing)}")
    return np.stack([supplied[name] for name in CORNER_ORDER])


def _entries(
    image_points: np.ndarray,
    world_points: np.ndarray,
) -> list[dict[str, str | list[int] | list[float]]]:
    return [
        {
            "name": name,
            "mesh_coords": world_point.tolist(),
            "image_coords": [int(value) for value in image_point],
        }
        for name, image_point, world_point in zip(
            CORNER_ORDER,
            image_points,
            world_points,
            strict=True,
        )
    ]


def _write_annotation(
    path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    corners: np.ndarray,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 8))
    axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
    axis.contour(mask, levels=[0.5], colors=["lime"], linewidths=0.6)
    axis.scatter(
        corners[:, 0],
        corners[:, 1],
        c="red",
        marker="x",
        s=90,
    )
    for name, point in zip(CORNER_ORDER, corners, strict=True):
        axis.annotate(
            name,
            point + np.asarray([5.0, -5.0]),
            color="yellow",
        )
    axis.set_xlim(-0.5, image.shape[1] - 0.5)
    axis.set_ylim(image.shape[0] - 0.5, -0.5)
    axis.set_title("Detected model-2 registration corners")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "png",
        type=Path,
        help="Input PNG with a uniform background",
    )
    parser.add_argument(
        "--corner",
        nargs=4,
        action="append",
        metavar=("LABEL", "X", "Y", "Z"),
        help=(
            "Override a default model-2 corner coordinate; supply TL, TR, BR, and BL"
        ),
    )
    parser.add_argument(
        "--model-key",
        default="model_2",
        help="Top-level JSON model key (default: model_2)",
    )
    parser.add_argument(
        "--frame-name",
        help="JSON frame key (default: PNG filename without its suffix)",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = io.imread(args.png)
    mask = foreground_mask(
        image,
        args.threshold,
        args.minimum_component_size,
    )
    image_points = diagonal_extrema(mask)
    world_points = _parse_world_corners(args.corner)
    frame_name = args.frame_name or args.png.stem

    payload = {
        args.model_key: {
            frame_name: _entries(image_points, world_points),
        }
    }
    annotation_path = args.png.with_name(f"{args.png.stem}_registrations.png")
    _write_annotation(
        annotation_path,
        image,
        mask,
        image_points,
    )
    print(json.dumps(payload, indent=2))
    print(f"Debug image: {annotation_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

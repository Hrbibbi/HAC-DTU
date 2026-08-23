"""Visualization helpers for Mitsuba renders."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

import drjit as dr
import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from hac.mitsuba.patch import mi

if TYPE_CHECKING:
    from hac.mitsuba.forward import ForwardModel


RenderSelection = tuple[str, int]

DEFAULT_RENDER_SELECTIONS: tuple[RenderSelection, ...] = (
    ("0_top", 0),
    ("0_bottom", 0),
    ("0_hor1", 0),
)

_OBSERVER_ALIASES = {
    "0center": "0hor1",
}


def bitmap_frames(rendered_frames: Any) -> NDArray[np.uint8]:
    """Convert ``(phase, height, width, 1)`` renders to uint8 RGB frames."""
    shape = tuple(rendered_frames.shape)
    if len(shape) != 4 or shape[-1] != 1:
        raise ValueError(
            f"rendered_frames must have shape (phase, height, width, 1); got {shape}."
        )

    frame_count, height, width, _ = shape
    if frame_count < 1 or height < 1 or width < 1:
        raise ValueError(f"rendered_frames dimensions must be positive; got {shape}.")

    strip = dr.transpose(rendered_frames, axes=(1, 0, 2, 3))
    strip = dr.reshape(strip, (height, frame_count * width, 1))
    bitmap = np.asarray(mi.util.convert_to_bitmap(strip))
    return np.transpose(
        bitmap.reshape(height, frame_count, width, 3),
        axes=(1, 0, 2, 3),
    )


def render_forward_views(
    forward_model: ForwardModel,
    selections: Sequence[RenderSelection] | None = None,
    *,
    seed: int = 0,
    spp: int | None = None,
    spp_grad: int = 0,
    title: str | None = None,
    output_path: str | PathLike[str] | None = None,
    dpi: int = 160,
) -> tuple[Figure, tuple[Axes, ...]]:
    """Render selected observer/phase views and display them side by side.

    Each selection is an ``(observer_name, phase_index)`` pair. Observer names
    are matched exactly first and then without underscores, hyphens, or case;
    ``"0center"`` is also accepted as an alias for ``"0_hor1"``. The default
    selections are ``0_top``, ``0_bottom``, and ``0_hor1`` at phase index 0.

    The function renders the full batch once, optionally saves the figure, and
    returns the figure and axes so callers can further customize or display it.
    """
    selected = tuple(DEFAULT_RENDER_SELECTIONS if selections is None else selections)
    if not selected:
        raise ValueError("selections must contain at least one observer/phase pair.")
    if dpi <= 0:
        raise ValueError(f"dpi must be positive; got {dpi}.")

    observer_names = tuple(forward_model.observer_names)
    resolved: list[tuple[str, int, int]] = []
    for selection in selected:
        if not isinstance(selection, tuple) or len(selection) != 2:
            raise ValueError(
                "Each selection must be an (observer_name, phase_index) tuple; "
                f"got {selection!r}."
            )
        observer_name, phase_index = selection
        if not isinstance(observer_name, str):
            raise TypeError(
                f"Selection observer names must be strings; got {observer_name!r}."
            )
        if not isinstance(phase_index, int) or isinstance(phase_index, bool):
            raise TypeError(
                "Selection phase indices must be integers; "
                f"got {phase_index!r} for {observer_name!r}."
            )
        if not 0 <= phase_index < forward_model.n_phases:
            raise IndexError(
                f"Phase index {phase_index} for {observer_name!r} is outside "
                f"[0, {forward_model.n_phases})."
            )

        resolved_name = _resolve_observer_name(observer_name, observer_names)
        resolved.append(
            (resolved_name, observer_names.index(resolved_name), phase_index)
        )

    rendered = forward_model.render(seed=seed, spp=spp, spp_grad=spp_grad)
    frame_cache: dict[int, NDArray[np.uint8]] = {}
    figure, axes_array = plt.subplots(
        1,
        len(resolved),
        figsize=(3.2 * len(resolved), 3.2),
        squeeze=False,
    )
    axes = tuple(axes_array[0])
    for axis, (observer_name, observer_index, phase_index) in zip(
        axes, resolved, strict=True
    ):
        if observer_index not in frame_cache:
            frame_cache[observer_index] = bitmap_frames(rendered[observer_index])
        axis.imshow(frame_cache[observer_index][phase_index])
        phase_degrees = 360.0 * phase_index / forward_model.n_phases
        axis.set_title(
            f"{observer_name}\nphase {phase_index} ({phase_degrees:g}\N{DEGREE SIGN})"
        )
        axis.axis("off")

    if title is not None:
        figure.suptitle(title)
    figure.tight_layout()
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=dpi)
    return figure, axes


def _normalize_observer_name(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "")


def _resolve_observer_name(requested: str, available: tuple[str, ...]) -> str:
    if requested in available:
        return requested

    normalized = _normalize_observer_name(requested)
    normalized = _OBSERVER_ALIASES.get(normalized, normalized)
    matches = [
        name for name in available if _normalize_observer_name(name) == normalized
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Observer name {requested!r} is ambiguous; it matches {matches}."
        )
    raise ValueError(
        f"Observer {requested!r} is not present in this ForwardModel. "
        f"Available observers: {available}."
    )


def write_movie(
    path: str | PathLike[str],
    frames: ArrayLike,
    fps: int,
) -> None:
    """Write uint8 RGB ``(frame, height, width, 3)`` data as an MP4 movie."""
    output_path = Path(path)
    frame_array = np.asarray(frames)
    if frame_array.ndim != 4 or frame_array.shape[-1] != 3:
        raise ValueError(
            "frames must have shape (frame, height, width, 3); "
            f"got {frame_array.shape}."
        )
    if 0 in frame_array.shape[:3]:
        raise ValueError(
            f"frames dimensions must be positive; got {frame_array.shape}."
        )
    if frame_array.dtype != np.uint8:
        raise ValueError(f"frames must have dtype uint8; got {frame_array.dtype}.")
    if fps <= 0:
        raise ValueError(f"fps must be positive; got {fps}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame_array.shape[1:3]
    writer = imageio_ffmpeg.write_frames(
        str(output_path),
        size=(width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "18"],
    )
    writer.send(None)
    try:
        for frame in frame_array:
            writer.send(np.ascontiguousarray(frame))
    finally:
        writer.close()

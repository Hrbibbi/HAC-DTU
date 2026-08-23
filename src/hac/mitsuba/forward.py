"""Differentiable forward rendering of asteroid light curves."""

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

import drjit as dr

from hac.mitsuba.mesh import MitsubaMesh
from hac.mitsuba.patch import mi
from hac.mitsuba.scene import (
    MIN_PROJECTED_PHASE_CENTER_SEPARATION,
    BatchSceneLayout,
    Projection,
    RenderingSettings,
    make_batch_scene,
)
from hac.constants import INTENSITY_CURVE_THRESHOLD
from hac.types import ObserverGeometry

# Lazy type alias to indicate that object is an array of shape:
# (observer, phase, y, x, channel)
# Note we only do luminance, so channel is always 1
RenderedFramesT = Any

# Lazy type alias to indicate that type is lightcurves with shape:
# (observer, phase)
LightcurvesT = Any

T = TypeVar("T")

_INTENSITY_THRESHOLD_UNIT_INTERVAL_FIPS = (
    INTENSITY_CURVE_THRESHOLD / 255.0
)  # https://fips.fi/data-challenges/helsinki-asteroid-challenge-2026/


def normalize_lightcurves(lightcurves: LightcurvesT) -> LightcurvesT:
    observer_count = lightcurves.shape[0]
    lightcurves_mean_observer = dr.reshape(
        dr.mean(lightcurves, axis=1),
        (observer_count, 1),
    )
    normalized_lightcurves = lightcurves / lightcurves_mean_observer
    return normalized_lightcurves


def threshold_sigmoid_gate(
    x: T,
    threshold: float = _INTENSITY_THRESHOLD_UNIT_INTERVAL_FIPS,
    temperature: float = 0.01,
) -> T:
    """
    Smoothed thresholding function.
    Smaller temperatures give sharper transitions about the threshold.
    """
    return 1.0 / (1.0 + dr.exp(-(x - threshold) / temperature))


def threshold_hard_gate(
    x: T,
    threshold: float = _INTENSITY_THRESHOLD_UNIT_INTERVAL_FIPS,
) -> T:
    """
    Hard thresholding function.
    """
    return dr.select(x > threshold, x, 0.0)


def threshold_otsu_smooth(
    image: T,
    nbins: int = 256,
    *,
    histogram_sigma: float | None = None,
    temperature: float = 1e-3,
) -> T:
    """Return a differentiable approximation of Otsu's threshold.

    Pixels are softly assigned to Gaussian histogram bins on [0, 1], and the
    threshold is a softmax-weighted average of the between-class variance
    instead of a discrete argmax.

    Args:
        image: Non-empty floating-point DrJit array with values in [0, 1].
        nbins: Number of histogram bins.
        histogram_sigma: Width of the Gaussian histogram kernel. By default,
            this is the distance between adjacent bin centers.
        temperature: Softmax temperature used to select the threshold. Lower
            values more closely approximate the discrete Otsu argmax.
    """
    if nbins < 2:
        raise ValueError(f"nbins must be at least 2; got {nbins}.")
    if histogram_sigma is None:
        histogram_sigma = 1.0 / (nbins - 1)
    elif histogram_sigma <= 0:
        raise ValueError(f"histogram_sigma must be positive; got {histogram_sigma}.")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive; got {temperature}.")

    pixels = dr.ravel(image)
    pixel_count = dr.width(pixels)
    if pixel_count == 0:
        raise ValueError("threshold_otsu requires a non-empty image.")

    float_type = type(pixels)
    tensor_type = dr.tensor_t(float_type)
    pixels_tensor = dr.reshape(tensor_type(pixels), (pixel_count, 1))
    bin_centers = dr.reshape(
        dr.linspace(tensor_type, 0.0, 1.0, nbins),
        (1, nbins),
    )

    histogram_logits = -0.5 * dr.square((pixels_tensor - bin_centers) / histogram_sigma)
    histogram_logits -= dr.max(histogram_logits, axis=1, keepdims=True)
    memberships = dr.exp(histogram_logits)
    memberships /= dr.sum(memberships, axis=1, keepdims=True)

    probabilities = dr.sum(memberships, axis=0)
    probabilities /= dr.sum(probabilities)
    bin_centers = dr.reshape(bin_centers, (nbins,))

    cumulative_probability = dr.cumsum(probabilities)
    cumulative_moment = dr.cumsum(probabilities * bin_centers)
    probability1 = cumulative_probability[:-1]
    probability2 = dr.maximum(1.0 - probability1, 0.0)
    moment1 = cumulative_moment[:-1]
    total_moment = dr.sum(probabilities * bin_centers)

    epsilon = 1e-12
    mean1 = moment1 / dr.maximum(probability1, epsilon)
    mean2 = (total_moment - moment1) / dr.maximum(probability2, epsilon)
    between_class_variance = probability1 * probability2 * dr.square(mean1 - mean2)

    threshold_logits = between_class_variance / temperature
    threshold_logits -= dr.max(threshold_logits)
    threshold_weights = dr.exp(threshold_logits)
    soft_threshold = dr.sum(threshold_weights * bin_centers[:-1]) / dr.sum(
        threshold_weights
    )

    # Otsu's threshold is ill-defined for nearly constant images. Smoothly
    # approach the image mean as its variance falls below a few bin widths.
    image_mean = dr.mean(pixels)
    image_variance = dr.mean(dr.square(pixels - image_mean))
    variance_floor = (4.0 * histogram_sigma) ** 2
    contrast_weight = image_variance / (image_variance + variance_floor)
    threshold = contrast_weight * soft_threshold + (1.0 - contrast_weight) * image_mean
    return dr.ravel(threshold)


def reduction_intensity_sum(
    frames: RenderedFramesT,
) -> LightcurvesT:
    """
    Simple lightcurve reduction that just sums intensities and normalizes according
    to the challenge description (division by curve mean)
    """
    lightcurves = dr.sum(frames, axis=(2, 3, 4))
    return normalize_lightcurves(lightcurves)


def reduction_intensity_hard(
    frames: RenderedFramesT,
) -> LightcurvesT:
    """
    Matches exactly 'intensity' reduction as given by organisers,
    including the hard, non-differentiable thresholding step.
    """
    thresholded_frames = threshold_hard_gate(
        frames, threshold=_INTENSITY_THRESHOLD_UNIT_INTERVAL_FIPS
    )
    lightcurves = dr.sum(thresholded_frames, axis=(2, 3, 4))
    return normalize_lightcurves(lightcurves)


def reduction_intensity_smooth(
    frames: RenderedFramesT,
    temperature: float = 0.01,
) -> LightcurvesT:
    """
    Matches 'intensity' reduction as given by organisers, but smoothens transition.

    Temperature controls the sharpness of the transition.
    Lower temperatures give sharper transitions.

    See also the `reduction_comparison.py` notebook.
    """
    threshold = (
        INTENSITY_CURVE_THRESHOLD / 255.0
    )  # https://fips.fi/data-challenges/helsinki-asteroid-challenge-2026/
    thresholded_frames = (
        threshold_sigmoid_gate(frames, threshold=threshold, temperature=temperature)
        * frames
    )
    lightcurves = dr.sum(thresholded_frames, axis=(2, 3, 4))
    return normalize_lightcurves(lightcurves)


def reduction_binary_hard(frames: RenderedFramesT, threshold: float) -> LightcurvesT:
    """
    Matches exactly the 'binary' reduction as given by the organisers,
    though requires the threshold value to be passed in.
    The threshold value should be determined using Otsu's method,
    but as this is cumbersome to make differentiable, we approximate this
    with a fixed threshold that can be calculated initially or
    at discrete points in the optimisation.

    This includes the hard, non-differentiable thresholding step.
    """
    thresholded_frames = dr.select(frames > threshold, 1.0, 0.0)
    lightcurves = dr.sum(thresholded_frames, axis=(2, 3, 4))
    return normalize_lightcurves(lightcurves)


def reduction_binary_smooth(
    frames: RenderedFramesT,
    threshold: float,
    temperature: float = 0.01,
) -> LightcurvesT:
    """
    Matches exactly the 'binary' reduction as given by the organisers,
    though requires the threshold value to be passed in.
    The threshold value should be determined using Otsu's method,
    but as this is cumbersome to make differentiable, we approximate this
    with a fixed threshold that can be calculated initially or
    at discrete points in the optimisation.

    Temperature controls the sharpness of the transition.
    Lower temperatures give sharper transitions.

    See also the `reduction_comparison.py` notebook.
    """
    thresholded_frames = threshold_sigmoid_gate(
        frames, threshold=threshold, temperature=temperature
    )
    lightcurves = dr.sum(thresholded_frames, axis=(2, 3, 4))
    return normalize_lightcurves(lightcurves)


def reduction_binary_otsu_smooth(
    frames: RenderedFramesT,
    nbins: int = 256,
    *,
    histogram_sigma: float | None = None,
    temperature: float = 1e-3,
) -> LightcurvesT:
    """
    Matches exactly the 'binary' reduction as given by the organisers,
    though requires the threshold value to be passed in.
    The threshold value should be determined using Otsu's method,
    but as this is cumbersome to make differentiable, we approximate this
    with a fixed threshold that can be calculated initially or
    at discrete points in the optimisation.

    Temperature controls the sharpness of the transition.
    Lower temperatures give sharper transitions.

    See also the `reduction_comparison.py` notebook.
    """
    threshold = threshold_otsu_smooth(
        frames, nbins=nbins, histogram_sigma=histogram_sigma, temperature=temperature
    )
    thresholded_frames = threshold_sigmoid_gate(
        frames, threshold=threshold, temperature=temperature
    )
    lightcurves = dr.sum(thresholded_frames, axis=(2, 3, 4))
    return normalize_lightcurves(lightcurves)


class ForwardModel:
    """A batch scene together with the mutable mesh parameters it renders."""

    def __init__(
        self,
        mesh: MitsubaMesh,
        *,
        n_phases: int,
        observers: Mapping[str, ObserverGeometry],
        rendering_settings: RenderingSettings,
        normalize_mesh: bool = False,
        bounding_cylinder_radius: float | None = None,
        minimum_projected_phase_center_separation: float = (
            MIN_PROJECTED_PHASE_CENTER_SEPARATION
        ),
        reduction_function: Callable[
            [RenderedFramesT], LightcurvesT
        ] = reduction_intensity_sum,
        bsdf: Mapping[str, Any] | None = None,
        integrator: Mapping[str, Any] | None = None,
        projection: Projection = "orthographic",
        orthographic_view_scale_override: float | None = None,
    ) -> None:
        # Materialize the mapping so its insertion order remains the sample order
        # throughout scene construction, rendering, and light-curve comparison.
        ordered_observers = dict(observers)
        result = make_batch_scene(
            mesh=mesh,
            n_phases=n_phases,
            observers=ordered_observers,
            rendering_settings=rendering_settings,
            normalize_mesh=normalize_mesh,
            bounding_cylinder_radius=bounding_cylinder_radius,
            minimum_projected_phase_center_separation=(
                minimum_projected_phase_center_separation
            ),
            bsdf=bsdf,
            integrator=integrator,
            projection=projection,
            orthographic_view_scale_override=orthographic_view_scale_override,
        )

        self.scene = result.scene
        self.layout: BatchSceneLayout = result.layout
        self.params: Any = mi.traverse(self.scene)
        self.n_phases = n_phases
        self.observer_names = tuple(ordered_observers)
        self.observer_count = len(ordered_observers)
        self.rendering_settings = rendering_settings
        self.reduction_function = reduction_function
        self.projection = projection

        position_keys = [
            key for key in self.params.keys() if key.endswith("vertex_positions")
        ]
        if len(position_keys) != 1:
            raise RuntimeError(
                "Expected exactly one optimizable mesh in the batch scene, "
                f"found {len(position_keys)}."
            )

        self.position_key = position_keys[0]
        self.face_key = f"{self.position_key.removesuffix('vertex_positions')}faces"
        if self.face_key not in self.params:
            raise RuntimeError(
                f"Could not find faces corresponding to {self.position_key!r}."
            )

    def render(
        self,
        *,
        seed: int = 0,
        spp: int | None = None,
        spp_grad: int = 0,
    ):
        """Render frames as ``(observer, phase, y, x, channel)``."""
        if seed < 0:
            raise ValueError(f"seed must be non-negative; got {seed}.")
        if spp is not None and spp <= 0:
            raise ValueError(f"spp must be positive when provided; got {spp}.")
        if spp_grad < 0:
            raise ValueError(f"spp_grad must be non-negative; got {spp_grad}.")

        settings = self.rendering_settings
        sample_count = self.observer_count * self.n_phases
        strip = mi.render(
            self.scene,
            params=self.params,
            spp=spp or settings.spp,
            spp_grad=spp_grad,
            seed=seed,
            seed_grad=seed + 1,
        )

        # Mitsuba's batch sensor lays samples out horizontally in one long film.
        # Split that strip and restore the observer-major ordering used to build
        # the scene: all phases for observer 0, then all phases for observer 1.
        frames = dr.transpose(
            dr.reshape(
                strip,
                (settings.height, sample_count, settings.width, 1),
            ),
            axes=(1, 0, 2, 3),
        )
        return dr.reshape(
            frames,
            (
                self.observer_count,
                self.n_phases,
                settings.height,
                settings.width,
                1,
            ),
        )

    def render_lightcurves(
        self,
        *,
        seed: int = 0,
        spp: int | None = None,
        spp_grad: int = 0,
    ):
        """Render and reduce mean-normalized curves for every observer."""
        return self.reduction_function(
            self.render(seed=seed, spp=spp, spp_grad=spp_grad)
        )

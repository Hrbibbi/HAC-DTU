"""Mitsuba differentiable inverse-rendering shape optimization, seeded from
a convex-inversion reconstruction instead of an ellipsoid.

Also includes remeshing and concavity masking experiments, plus a binary-lightcurve
"""

# Runtime setup must precede imports from hac.mitsuba.
# ruff: noqa: E402

# %%
import os

# Updated challenge lightcurve data (see module docstring). Must be set
# before any hac import -- hac.constants resolves CHALLENGE_DATA_DIR from
# DSDIR at import time.
os.environ["DSDIR"] = "/work3/projects/HAC-DTU"

import datetime
import json
import math
import time
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np
import trimesh
from skimage.filters import threshold_otsu

mi.set_variant("cuda_ad_mono")

from hac.constants import CHALLENGE_DATA_DIR, MODEL_BASE_RADII, OBSERVERS
from hac.data import get_model_stl_path, load_data
from hac.mesh import normalize_mesh_to_challenge_cylinder
from hac.mitsuba.forward import ForwardModel
from hac.mitsuba.mesh import MitsubaMesh
from hac.mitsuba.scene import RenderingSettings
from hac.mitsuba.visualize import bitmap_frames, write_movie
from hac.visualize import plot_lightcurve_comparison, plot_optimization_loss

print(f"CHALLENGE_DATA_DIR resolved to: {CHALLENGE_DATA_DIR}")
assert str(CHALLENGE_DATA_DIR) == "/work3/projects/HAC-DTU/challenge_data", (
    f"DSDIR did not take effect -- reading from {CHALLENGE_DATA_DIR} instead "
    "of the expected updated HAC-collab data."
)

# %%
IS_NOTEBOOK = True
try:
    get_ipython()
except NameError:
    IS_NOTEBOOK = False

MODEL_NUM = 3
TARGET_ORIGIN = "simulated"
TARGET_CURVE_TYPE = "intensity"

# Which convex-inversion reconstruction to seed from -- independent of
# TARGET_ORIGIN above
INITIAL_GUESS_ORIGIN = "simulated"
INITIAL_GUESS_STL_PATH = Path(
    f"/work3/projects/HAC-DTU/reconstructions/convex/joint/"
    f"asteroid{MODEL_NUM}_{INITIAL_GUESS_ORIGIN}_joint.stl"
)

INITIAL_GUESS_SUBDIVISION_PASSES = 1

EXPERIMENTS = tuple(OBSERVERS)
MOVIE_EXPERIMENT = "0_hor1"

OPTIMIZATION_PHASES = 180 #180
OPTIMIZATION_WIDTH = 128 # 64
OPTIMIZATION_HEIGHT = 128 # 64
OPTIMIZATION_SPP = 8 # 8
OPTIMIZATION_ITERATIONS = 300 #300
LEARNING_RATE = 1e-2 #2e-2
LARGE_STEPS_LAMBDA = 25.0  # lambda=5.0 test gave a noisier path to a similar final loss, reverted
LOSS_TYPE = "mse"  # "l1" or "mse"

# Used when TARGET_CURVE_TYPE == "binary"
BINARY_SOFT_THRESHOLD_TEMPERATURE = 0.02

# Botsch-Kobbelt isotropic remeshing (gpytoolbox.remesh_botsch), applied
# mid-optimization to add resolution so the optimizer has local vertices to
# carve out concavities with -- mirrors Mitsuba's own shape-optimization
# tutorial (mitsuba.readthedocs.io/en/stable/src/inverse_rendering/
# shape_optimization.html)
REMESH_AT_ITERATIONS = (OPTIMIZATION_ITERATIONS // 2,) if OPTIMIZATION_ITERATIONS >= 20 else ()
REMESH_BOTSCH_KOBBELT_ITERATIONS = 5
REMESH_TARGET_EDGE_LENGTH_FACTOR = 0.5

# Discourage the optimized shape from bulging outward past the initial
# convex-inversion hull --
# - "hard_clamp": positions are clamped to max_radii every iteration before
#   rendering. 
# - "soft_penalty": a smooth quadratic-overshoot term (see
#   convex_hull_penalty) is added to the loss instead of clamping, keeping
#   the whole pipeline differentiable end-to-end. CONVEX_HULL_PENALTY_WEIGHT
#   trades off strictness vs. smoothness, the same way LARGE_STEPS_LAMBDA
#   trades off detail vs. smoothness.
ENFORCE_CONVEX_HULL_CONSTRAINT = True
CONVEX_HULL_CONSTRAINT_MODE = "hard_clamp"  # "soft_penalty" or "hard_clamp"
CONVEX_HULL_PENALTY_WEIGHT = 10.0

# Oracle experiment: only let vertices move where we KNOW (from the true
# shape, ray-cast against it) there's a genuine concavity -- everything
# else stays pinned exactly at its initial-guess position. This is a
# cheating/upper-bound experiment
CONCAVITY_MASK_EXPERIMENT = False
# Minimum relative radial deficit (true-shape radius vs. hull radius, same
# direction from the hull centroid) to classify a vertex as "inside a real
# concavity" (free to move) rather than "just the usual gap between a flat
# hull facet and the mildly curved true surface"
CONCAVITY_RELATIVE_THRESHOLD = 0.05
# Seed from the TRUE reference shape's OWN convex hull instead of the
# convex-inversion reconstruction
INITIAL_GUESS_SOURCE_TRUE_CONVEX_HULL = False


MOVIE_PHASES = 360
MOVIE_WIDTH = 512
MOVIE_HEIGHT = 512
MOVIE_SPP = 16
MOVIE_FPS = 30

MIN_PROJECTED_PHASE_CENTER_SEPARATION = 5.0

OUTPUT_DIR = Path(
    f"mitsuba_opt_from_convex_model{MODEL_NUM}_{TARGET_ORIGIN}_{TARGET_CURVE_TYPE}_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}"
)
TRUE_MOVIE_PATH = OUTPUT_DIR / "true_object.mp4"
INITIAL_GUESS_MOVIE_PATH = OUTPUT_DIR / "initial_guess.mp4"
OPTIMIZED_MOVIE_PATH = OUTPUT_DIR / "optimized_object.mp4"
COMPARISON_MOVIE_PATH = OUTPUT_DIR / "true_vs_optimized.mp4"
OPTIMIZED_MESH_PATH = OUTPUT_DIR / "optimized_asteroid.obj"
INITIAL_GUESS_MESH_PATH = OUTPUT_DIR / "initial_guess.obj"
NORMALIZED_REFERENCE_MESH_PATH = OUTPUT_DIR / "normalized_reference.obj"
LIGHTCURVE_PLOT_PATH = OUTPUT_DIR / "lightcurve_fit.png"
LOSS_PLOT_PATH = OUTPUT_DIR / "optimization_loss.png"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"


def dr_loss(curve, target):
    diff = curve - target
    return dr.mean(dr.abs(diff)) if LOSS_TYPE == "l1" else dr.mean(dr.square(diff))


def np_loss(curve, target):
    diff = curve - target
    return float(np.mean(np.abs(diff)) if LOSS_TYPE == "l1" else np.mean(np.square(diff)))


def compute_otsu_threshold_per_observer(frames_tensor, observer_count):
    """Per-observer Otsu threshold from each observer's own phase-0 frame --
    mirrors hac.simulate.generate_binary_curve's threshold_otsu(frames[0]),
    computed once from a reference render and reused as a fixed constant.
    Returns an (observer_count, 1, 1, 1, 1) mi.TensorXf, ready to broadcast
    against ForwardModel.render()'s (observer, phase, y, x, channel)
    output. See BINARY_SOFT_THRESHOLD_TEMPERATURE."""
    frames_np = np.asarray(frames_tensor)
    thresholds = np.empty(observer_count, dtype=np.float32)
    for i in range(observer_count):
        frame0 = frames_np[i, 0, :, :, 0]
        if frame0.max() <= frame0.min():
            # Degenerate (e.g. fully unlit) frame -- Otsu is undefined on a
            # constant image. Fall back to that constant value so nothing
            # ever counts as "lit" for this observer.
            thresholds[i] = frame0.max()
        else:
            thresholds[i] = threshold_otsu(frame0)
    return mi.TensorXf(thresholds.reshape(observer_count, 1, 1, 1, 1))


def soft_binary_curve_from_frames(frames_tensor, threshold, temperature, observer_count):
    """Mean-normalized soft-binary curve from an ALREADY-rendered frames
    tensor -- same math as ForwardModel.render_binary_lightcurves, but
    reused here to avoid a redundant re-render when the frames were already
    produced (e.g. to compute the Otsu threshold itself)."""
    soft_lit = 1.0 / (1.0 + dr.exp(-(frames_tensor - threshold) / temperature))
    lightcurves = dr.sum(soft_lit, axis=(2, 3, 4))
    observer_means = dr.reshape(dr.mean(lightcurves, axis=1), (observer_count, 1))
    return lightcurves / observer_means


def normalize_positions_to_challenge_cylinder(flat_positions, target_radius):
    """Differentiably normalize positions into the known challenge cylinder."""
    vertex_count = dr.width(flat_positions) // 3
    vertex_indices = dr.arange(mi.UInt, vertex_count)
    vertices = mi.Point3f(
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices),
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices + 1),
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices + 2),
    )

    x_center = 0.5 * (dr.min(vertices.x) + dr.max(vertices.x))
    y_center = 0.5 * (dr.min(vertices.y) + dr.max(vertices.y))
    centered_x = vertices.x - x_center
    centered_y = vertices.y - y_center
    radial_max = dr.max(dr.sqrt(dr.square(centered_x) + dr.square(centered_y)))

    z_min = dr.min(vertices.z)
    z_max = dr.max(vertices.z)
    z_center = 0.5 * (z_min + z_max)

    normalized_vertices = mi.Point3f(
        centered_x * (target_radius / radial_max),
        centered_y * (target_radius / radial_max),
        (vertices.z - z_center) * (2.0 / (z_max - z_min)),
    )
    return dr.ravel(normalized_vertices)


def compute_convex_hull_face_data(mesh: trimesh.Trimesh):
    """Outward face normals and plane offsets (dot(v, n_i) = offset_i for
    any vertex v on face i) for a convex mesh -- fixed for the whole
    optimization run, computed once from the pristine initial guess (before
    any subdivision/remeshing changes its vertex/face resolution; those
    don't change the underlying convex shape)."""
    face_normals = mesh.face_normals
    face_offsets = np.einsum("ij,ij->i", mesh.vertices[mesh.faces[:, 0]], face_normals)
    return face_normals, face_offsets


def compute_radial_bounds(vertices_np, centroid_np, hull_face_normals, hull_face_offsets):
    """For each vertex, the distance from centroid_np to the convex hull's
    boundary along that vertex's OWN current direction -- the minimum
    positive distance to any hull face plane along the ray, which is exact
    for a convex polytope containing the centroid. Recomputed whenever the
    vertex set changes (initially, and after each remesh)."""
    directions = vertices_np - centroid_np
    radii = np.linalg.norm(directions, axis=1)
    unit_directions = directions / np.maximum(radii, 1e-8)[:, None]

    denom = unit_directions @ hull_face_normals.T  # (V, F)
    numer = hull_face_offsets[None, :] - (centroid_np @ hull_face_normals.T)  # (F,) -> (1, F)
    with np.errstate(divide="ignore", invalid="ignore"):
        candidate = np.where(denom > 1e-9, numer / denom, np.inf)
    return candidate.min(axis=1)  # (V,)


def convex_hull_penalty(flat_positions, centroid, max_radii):
    """Mean squared radial overshoot past max_radii, along each vertex's own
    current direction from centroid -- a smooth, fully differentiable
    stand-in for a hard clamp. Zero for any vertex still inside the hull,
    so it exerts no gradient pressure until a vertex actually violates the
    constraint. See ENFORCE_CONVEX_HULL_CONSTRAINT."""
    vertex_count = dr.width(flat_positions) // 3
    vertex_indices = dr.arange(mi.UInt, vertex_count)
    vertices = mi.Point3f(
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices),
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices + 1),
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices + 2),
    )

    radius = dr.norm(vertices - centroid)
    overshoot = dr.maximum(radius - max_radii, 0.0)
    return dr.mean(dr.square(overshoot))


def clamp_positions_to_convex_hull(flat_positions, centroid, max_radii):
    """Pull any vertex back along its own current direction from centroid
    if it has moved past max_radii -- see ENFORCE_CONVEX_HULL_CONSTRAINT."""
    vertex_count = dr.width(flat_positions) // 3
    vertex_indices = dr.arange(mi.UInt, vertex_count)
    vertices = mi.Point3f(
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices),
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices + 1),
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices + 2),
    )

    centered = vertices - centroid
    radius = dr.norm(centered)
    unit_direction = centered / dr.maximum(radius, 1e-8)
    clamped_radius = dr.minimum(radius, max_radii)
    clamped_vertices = centroid + unit_direction * clamped_radius
    return dr.ravel(clamped_vertices)


def compute_true_shape_radial_distances(unit_directions_np, centroid_np, true_mesh):
    """For each unit direction from centroid_np, the distance to the TRUE
    reference mesh's surface along that ray -- inf for any ray that misses
    (shouldn't happen for a watertight, roughly star-shaped-from-centroid
    mesh, but handled defensively). See CONCAVITY_MASK_EXPERIMENT."""
    ray_origins = np.tile(centroid_np, (len(unit_directions_np), 1))
    locations, index_ray, _ = true_mesh.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=unit_directions_np, multiple_hits=False,
    )
    true_radius = np.full(len(unit_directions_np), np.inf)
    if len(index_ray) > 0:
        dists = np.linalg.norm(locations - centroid_np, axis=1)
        for ray_i, dist in zip(index_ray, dists):
            true_radius[ray_i] = min(true_radius[ray_i], dist)
    return true_radius


def compute_concavity_mask(vertices_np, centroid_np, max_radii_np, true_mesh):
    """Classify each vertex as "free to move" (True, inside a genuine
    concavity of the true shape) or "frozen" (False, the hull is already
    correct there) -- see CONCAVITY_MASK_EXPERIMENT and
    CONCAVITY_RELATIVE_THRESHOLD."""
    directions = vertices_np - centroid_np
    radii = np.linalg.norm(directions, axis=1)
    unit_directions = directions / np.maximum(radii, 1e-8)[:, None]
    true_radius = compute_true_shape_radial_distances(unit_directions, centroid_np, true_mesh)
    relative_deficit = (max_radii_np - true_radius) / np.maximum(max_radii_np, 1e-8)
    concave = relative_deficit > CONCAVITY_RELATIVE_THRESHOLD
    return np.where(np.isfinite(true_radius), concave, False)


def apply_concavity_freeze(flat_positions, frozen_flat_positions, concave_mask):
    """Overwrite any vertex NOT classified as concave back to its frozen
    (true-convex-hull) position -- see CONCAVITY_MASK_EXPERIMENT."""
    vertex_count = dr.width(flat_positions) // 3
    vertex_indices = dr.arange(mi.UInt, vertex_count)
    positions = mi.Point3f(
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices),
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices + 1),
        dr.gather(mi.Float, flat_positions, 3 * vertex_indices + 2),
    )
    frozen = mi.Point3f(
        dr.gather(mi.Float, frozen_flat_positions, 3 * vertex_indices),
        dr.gather(mi.Float, frozen_flat_positions, 3 * vertex_indices + 1),
        dr.gather(mi.Float, frozen_flat_positions, 3 * vertex_indices + 2),
    )
    blended = mi.Point3f(
        dr.select(concave_mask, positions.x, frozen.x),
        dr.select(concave_mask, positions.y, frozen.y),
        dr.select(concave_mask, positions.z, frozen.z),
    )
    return dr.ravel(blended)


def movie_frames(mesh_path: Path):
    forward = ForwardModel(
        MitsubaMesh(obj_path=mesh_path),
        n_phases=MOVIE_PHASES,
        observers={MOVIE_EXPERIMENT: OBSERVERS[MOVIE_EXPERIMENT]},
        rendering_settings=RenderingSettings(
            width=MOVIE_WIDTH,
            height=MOVIE_HEIGHT,
            spp=MOVIE_SPP,
        ),
        normalize_mesh=False,
        minimum_projected_phase_center_separation=(
            MIN_PROJECTED_PHASE_CENTER_SEPARATION
        ),
    )
    return bitmap_frames(forward.render(seed=1)[0])


def combine_movie_frames(known_model_frames, initial_guess_frames, optimized_frames):
    """Return the known-model panel and the three-panel comparison frames."""
    if known_model_frames is None:
        known_model_frames = np.zeros_like(initial_guess_frames)

    comparison_frames = np.concatenate(
        [known_model_frames, initial_guess_frames, optimized_frames],
        axis=2,
    )
    return known_model_frames, comparison_frames


def remesh_botsch_kobbelt(vertices_np, faces_np):
    """Isotropic Botsch-Kobbelt remeshing via gpytoolbox, same call as
    Mitsuba's shape-optimization tutorial: target edge length is half the
    mesh's current mean edge length."""
    from gpytoolbox import remesh_botsch

    v0, v1, v2 = (vertices_np[faces_np[:, i]] for i in range(3))
    mean_edge_length = np.mean(
        np.concatenate(
            [
                np.linalg.norm(v1 - v0, axis=1),
                np.linalg.norm(v2 - v1, axis=1),
                np.linalg.norm(v0 - v2, axis=1),
            ]
        )
    )
    target_edge_length = float(mean_edge_length) * REMESH_TARGET_EDGE_LENGTH_FACTOR

    v_new, f_new = remesh_botsch(
        vertices_np.astype(np.float64),
        faces_np.astype(np.int32),
        i=REMESH_BOTSCH_KOBBELT_ITERATIONS,
        h=target_edge_length,
        project=True,
    )
    return v_new.astype(np.float32), f_new.astype(np.int64)


def build_forward(mesh_path, optimization_observers, optimization_rendering_settings):
    forward = ForwardModel(
        MitsubaMesh(obj_path=mesh_path),
        n_phases=OPTIMIZATION_PHASES,
        observers=optimization_observers,
        rendering_settings=optimization_rendering_settings,
        normalize_mesh=False,
        minimum_projected_phase_center_separation=MIN_PROJECTED_PHASE_CENTER_SEPARATION,
    )
    faces_np = (
        forward.params[forward.face_key].numpy().reshape(-1, 3).astype(np.int64).copy()
    )
    return forward, faces_np


def build_optimization_state(mesh_path, optimization_observers, optimization_rendering_settings):
    """(Re)build the forward model, LargeSteps parametrization, and Adam
    optimizer from a mesh file. Needed both for the initial mesh and after
    each remesh: LargeSteps/Adam are tied to a fixed vertex/face count, so a
    topology change requires rebuilding both from scratch."""
    forward, faces_np = build_forward(
        mesh_path, optimization_observers, optimization_rendering_settings
    )
    large_steps = mi.ad.LargeSteps(
        forward.params[forward.position_key],
        forward.params[forward.face_key],
        LARGE_STEPS_LAMBDA,
    )
    optimizer = mi.ad.Adam(lr=LEARNING_RATE, uniform=True)
    optimizer["u"] = large_steps.to_differential(forward.params[forward.position_key])
    return forward, faces_np, large_steps, optimizer


def main():
    if CONCAVITY_MASK_EXPERIMENT and not ENFORCE_CONVEX_HULL_CONSTRAINT:
        raise ValueError(
            "CONCAVITY_MASK_EXPERIMENT reuses the hull constraint's max_radii -- "
            "requires ENFORCE_CONVEX_HULL_CONSTRAINT=True."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    cylinder_radius = MODEL_BASE_RADII[MODEL_NUM]
    model_stl_path = get_model_stl_path(model_num=MODEL_NUM) if MODEL_NUM <= 3 else None
    if model_stl_path is not None:
        reference_mesh = normalize_mesh_to_challenge_cylinder(
            trimesh.load_mesh(model_stl_path),
            target_R=cylinder_radius,
        )
        reference_mesh.export(NORMALIZED_REFERENCE_MESH_PATH)
    else:
        print(
            f"Model {MODEL_NUM} has no known reference shape; "
            "skipping reference mesh and reference loss."
        )

    if INITIAL_GUESS_SOURCE_TRUE_CONVEX_HULL:
        if model_stl_path is None:
            raise ValueError(
                "INITIAL_GUESS_SOURCE_TRUE_CONVEX_HULL needs a known true reference "
                f"shape, but model {MODEL_NUM} has none (MODEL_NUM must be <= 3)."
            )
        initial_guess = reference_mesh.convex_hull
        print(
            "Initial guess (TRUE convex hull of the reference shape, pre-subdivision, "
            f"bypassing convex inversion) -- {len(initial_guess.vertices)} vertices, "
            f"{len(initial_guess.faces)} faces"
        )
    else:
        if not INITIAL_GUESS_STL_PATH.exists():
            raise FileNotFoundError(
                f"Convex-inversion initial guess not found at {INITIAL_GUESS_STL_PATH} -- "
                "run convex_inversion_multi_asteroid3.py (or the matching asteroid/data-kind "
                "script) first to produce it."
            )
        initial_guess = normalize_mesh_to_challenge_cylinder(
            trimesh.load_mesh(INITIAL_GUESS_STL_PATH, force="mesh"),
            target_R=cylinder_radius,
        )
        print(
            f"Initial guess (convex inversion, pre-subdivision): {INITIAL_GUESS_STL_PATH} -- "
            f"{len(initial_guess.vertices)} vertices, {len(initial_guess.faces)} faces"
        )
    for _ in range(INITIAL_GUESS_SUBDIVISION_PASSES):
        initial_guess = initial_guess.subdivide()
    if INITIAL_GUESS_SUBDIVISION_PASSES > 0:
        print(
            f"Subdivided {INITIAL_GUESS_SUBDIVISION_PASSES} pass(es) -- "
            f"{len(initial_guess.vertices)} vertices, {len(initial_guess.faces)} faces"
        )
    initial_guess.export(INITIAL_GUESS_MESH_PATH)

    optimization_observers = {
        experiment: OBSERVERS[experiment] for experiment in EXPERIMENTS
    }
    optimization_rendering_settings = RenderingSettings(
        width=OPTIMIZATION_WIDTH,
        height=OPTIMIZATION_HEIGHT,
        spp=OPTIMIZATION_SPP,
    )
    forward, faces_np, large_steps, optimizer = build_optimization_state(
        INITIAL_GUESS_MESH_PATH, optimization_observers, optimization_rendering_settings
    )

    if ENFORCE_CONVEX_HULL_CONSTRAINT:
        # Fixed for the whole run: subdivision/remeshing change vertex/face
        # resolution but not the underlying convex shape, so the hull face
        # planes computed here stay valid throughout. Computed from
        # Mitsuba's OWN loaded vertex/face order (forward.params), NOT the
        # separately trimesh-loaded `initial_guess` object -- Mitsuba's OBJ
        # loader does not preserve trimesh's OBJ-export vertex order (same
        # point set, different permutation -- confirmed directly via
        # workspaces/christian/test_vertex_order.py), so per-vertex data
        # (max_radii) computed from one ordering and applied against the
        # other silently clamps/penalizes each vertex using a DIFFERENT
        # vertex's bound. That mismatch was the actual root cause of the
        # "irregular shape" symptom and the inflated/climbing loss seen in
        # both the hard-clamp and soft-penalty variants -- not a genuine
        # flaw in either constraint approach.
        mitsuba_vertices_np = (
            forward.params[forward.position_key].numpy().reshape(-1, 3).astype(np.float64)
        )
        hull_mesh = trimesh.Trimesh(
            vertices=mitsuba_vertices_np, faces=faces_np, process=False
        )
        hull_face_normals, hull_face_offsets = compute_convex_hull_face_data(hull_mesh)
        hull_centroid_np = np.asarray(hull_mesh.centroid, dtype=np.float64)
        hull_centroid = mi.Point3f(*hull_centroid_np.tolist())
        max_radii = mi.Float(
            compute_radial_bounds(
                mitsuba_vertices_np, hull_centroid_np,
                hull_face_normals, hull_face_offsets,
            ).astype(np.float32)
        )
        print(
            f"Convex hull constraint enabled: {len(hull_face_offsets)} hull faces, "
            f"centroid={hull_centroid_np.round(4).tolist()}"
        )

        if CONCAVITY_MASK_EXPERIMENT:
            concave_mask_np = compute_concavity_mask(
                mitsuba_vertices_np, hull_centroid_np, np.array(max_radii), reference_mesh
            )
            concave_mask = mi.Bool(concave_mask_np.tolist())
            frozen_positions = mi.Float(
                mitsuba_vertices_np.astype(np.float32).flatten().tolist()
            )
            print(
                f"Concavity mask: {int(concave_mask_np.sum())}/{len(concave_mask_np)} "
                f"vertices free to move ({100 * concave_mask_np.mean():.1f}%), "
                f"threshold={CONCAVITY_RELATIVE_THRESHOLD}"
            )

    target_data = load_data(
        model_num=MODEL_NUM,
        origin=TARGET_ORIGIN,
        curve_type=TARGET_CURVE_TYPE,
    ).data
    target_phase_indices = (
        np.arange(OPTIMIZATION_PHASES) * len(target_data) // OPTIMIZATION_PHASES
    )
    target = mi.TensorXf(
        np.stack(
            [
                np.asarray(target_data[experiment], dtype=np.float32)[
                    target_phase_indices
                ]
                for experiment in EXPERIMENTS
            ]
        )
    )
    target = target / dr.reshape(dr.mean(target, axis=1), (forward.observer_count, 1))

    if TARGET_CURVE_TYPE == "binary" and model_stl_path is None:
        raise ValueError(
            "TARGET_CURVE_TYPE='binary' needs a known reference shape (MODEL_NUM <= 3) "
            "to calibrate the per-observer Otsu threshold against."
        )

    binary_threshold = None
    true_shape_forward_curve_mse = None
    if model_stl_path is not None:
        reference_forward = ForwardModel(
            MitsubaMesh(obj_path=NORMALIZED_REFERENCE_MESH_PATH),
            n_phases=OPTIMIZATION_PHASES,
            observers=optimization_observers,
            rendering_settings=optimization_rendering_settings,
            normalize_mesh=False,
            minimum_projected_phase_center_separation=(
                MIN_PROJECTED_PHASE_CENTER_SEPARATION
            ),
        )
        if TARGET_CURVE_TYPE == "binary":
            reference_frames = reference_forward.render(seed=0)
            binary_threshold = compute_otsu_threshold_per_observer(
                reference_frames, reference_forward.observer_count
            )
            reference_curve = np.asarray(
                soft_binary_curve_from_frames(
                    reference_frames, binary_threshold,
                    BINARY_SOFT_THRESHOLD_TEMPERATURE, reference_forward.observer_count,
                )
            ).copy()
            print(
                "Binary soft-threshold (per observer, from true-shape render): "
                f"{np.asarray(binary_threshold).flatten().round(4).tolist()}"
            )
        else:
            reference_curve = np.asarray(reference_forward.render_lightcurves(seed=0)).copy()
        true_shape_forward_curve_mse = np_loss(reference_curve, np.asarray(target))
        del reference_forward
        print(f"True-shape forward curve {LOSS_TYPE.upper()}: {true_shape_forward_curve_mse:.8f}")

    loss_history = []
    remesh_events = []
    initial_guess_curve = None
    best_loss = math.inf
    best_iteration = None
    best_positions_np = None
    best_faces_np = None

    for iteration in range(OPTIMIZATION_ITERATIONS):
        positions = normalize_positions_to_challenge_cylinder(
            large_steps.from_differential(optimizer["u"]),
            cylinder_radius,
        )
        if ENFORCE_CONVEX_HULL_CONSTRAINT and CONVEX_HULL_CONSTRAINT_MODE == "hard_clamp":
            positions = clamp_positions_to_convex_hull(positions, hull_centroid, max_radii)
        if CONCAVITY_MASK_EXPERIMENT:
            positions = apply_concavity_freeze(positions, frozen_positions, concave_mask)
        forward.params[forward.position_key] = positions
        forward.params.update()

        if TARGET_CURVE_TYPE == "binary":
            curve = forward.render_binary_lightcurves(
                threshold=binary_threshold,
                temperature=BINARY_SOFT_THRESHOLD_TEMPERATURE,
                seed=0,
            )
        else:
            curve = forward.render_lightcurves(seed=0)
        if initial_guess_curve is None:
            initial_guess_curve = np.asarray(curve).copy()

        curve_loss = dr_loss(curve, target)
        loss = curve_loss
        hull_penalty_value = 0.0
        if ENFORCE_CONVEX_HULL_CONSTRAINT and CONVEX_HULL_CONSTRAINT_MODE == "soft_penalty":
            hull_penalty = convex_hull_penalty(positions, hull_centroid, max_radii)
            hull_penalty_value = float(hull_penalty.array[0])
            loss = loss + CONVEX_HULL_PENALTY_WEIGHT * hull_penalty
        loss_value = float(curve_loss.array[0])
        if loss_value < best_loss:
            best_loss = loss_value
            best_iteration = iteration + 1
            best_positions_np = (
                forward.params[forward.position_key]
                .numpy()
                .reshape(-1, 3)
                .astype(np.float32)
                .copy()
            )
            best_faces_np = faces_np.copy()

        dr.backward(loss)
        optimizer.step()

        loss_history.append(loss_value)
        hull_penalty_str = (
            f" hull_penalty={hull_penalty_value:.8f}"
            if ENFORCE_CONVEX_HULL_CONSTRAINT and CONVEX_HULL_CONSTRAINT_MODE == "soft_penalty"
            else ""
        )
        print(
            f"Iteration {iteration + 1:03d}/{OPTIMIZATION_ITERATIONS}: "
            f"loss={loss_value:.8f}{hull_penalty_str}"
        )

        if (iteration + 1) in REMESH_AT_ITERATIONS:
            current_positions_np = (
                forward.params[forward.position_key]
                .numpy()
                .reshape(-1, 3)
                .astype(np.float32)
                .copy()
            )
            remeshed_vertices, remeshed_faces = remesh_botsch_kobbelt(
                current_positions_np, faces_np
            )
            remeshed_mesh = trimesh.Trimesh(
                vertices=remeshed_vertices, faces=remeshed_faces, process=False
            )
            remeshed_mesh_path = OUTPUT_DIR / f"remeshed_after_iter{iteration + 1:03d}.obj"
            remeshed_mesh.export(remeshed_mesh_path)
            print(
                f"Remeshed after iteration {iteration + 1}: "
                f"{current_positions_np.shape[0]} -> {remeshed_vertices.shape[0]} vertices, "
                f"{faces_np.shape[0]} -> {remeshed_faces.shape[0]} faces"
            )
            forward, faces_np, large_steps, optimizer = build_optimization_state(
                remeshed_mesh_path, optimization_observers, optimization_rendering_settings
            )
            if ENFORCE_CONVEX_HULL_CONSTRAINT:
                # Topology changed -- max_radii is per-vertex, must be
                # rebuilt for the new vertex set. The hull itself (face
                # normals/offsets/centroid) is unchanged. Uses Mitsuba's OWN
                # post-reload vertex order (forward.params), NOT
                # remeshed_vertices (gpytoolbox/trimesh-export order) --
                # same vertex-order mismatch as the initial setup, see the
                # comment there.
                mitsuba_vertices_np = (
                    forward.params[forward.position_key]
                    .numpy()
                    .reshape(-1, 3)
                    .astype(np.float64)
                )
                max_radii = mi.Float(
                    compute_radial_bounds(
                        mitsuba_vertices_np, hull_centroid_np,
                        hull_face_normals, hull_face_offsets,
                    ).astype(np.float32)
                )
                if CONCAVITY_MASK_EXPERIMENT:
                    concave_mask_np = compute_concavity_mask(
                        mitsuba_vertices_np, hull_centroid_np,
                        np.array(max_radii), reference_mesh,
                    )
                    concave_mask = mi.Bool(concave_mask_np.tolist())
                    frozen_positions = mi.Float(
                        mitsuba_vertices_np.astype(np.float32).flatten().tolist()
                    )
                    print(
                        f"Concavity mask rebuilt after remesh: "
                        f"{int(concave_mask_np.sum())}/{len(concave_mask_np)} "
                        f"vertices free to move ({100 * concave_mask_np.mean():.1f}%)"
                    )
            remesh_events.append(
                {
                    "after_iteration": iteration + 1,
                    "vertex_count": int(remeshed_vertices.shape[0]),
                    "face_count": int(remeshed_faces.shape[0]),
                    "mesh_path": str(remeshed_mesh_path),
                }
            )

    if best_positions_np is None or best_faces_np is None:
        raise RuntimeError("Shape optimization did not produce a checkpoint")
    if initial_guess_curve is None:
        raise RuntimeError("Shape optimization did not render an initial curve")

    # best_positions_np/best_faces_np may predate a later remesh (different
    # topology than the live `forward`), so rebuild a forward from the saved
    # checkpoint mesh rather than assuming it still matches `forward`.
    optimized_mesh = trimesh.Trimesh(
        vertices=best_positions_np, faces=best_faces_np, process=False
    )
    optimized_mesh.export(OPTIMIZED_MESH_PATH)

    final_forward, _ = build_forward(
        OPTIMIZED_MESH_PATH, optimization_observers, optimization_rendering_settings
    )
    if TARGET_CURVE_TYPE == "binary":
        final_curve = np.asarray(
            final_forward.render_binary_lightcurves(
                threshold=binary_threshold,
                temperature=BINARY_SOFT_THRESHOLD_TEMPERATURE,
                seed=0,
            )
        ).copy()
    else:
        final_curve = np.asarray(final_forward.render_lightcurves(seed=0)).copy()
    final_loss = np_loss(final_curve, np.asarray(target))

    plot_lightcurve_comparison(
        np.asarray(target),
        initial_guess_curve,
        final_curve,
        EXPERIMENTS,
        ("HAC target", "Initial guess (convex inversion)", "Optimized shape"),
        LIGHTCURVE_PLOT_PATH,
        title=(
            f"Model {MODEL_NUM} ({TARGET_ORIGIN}/{TARGET_CURVE_TYPE}): all "
            f"{forward.observer_count} observers, final {LOSS_TYPE.upper()}={final_loss:.6g}"
        ),
    )
    plot_optimization_loss(
        loss_history,
        LOSS_PLOT_PATH,
        reference_loss=true_shape_forward_curve_mse,
        reference_label=(
            f"True-shape forward {LOSS_TYPE.upper()}" if true_shape_forward_curve_mse is not None else None
        ),
    )

    true_movie_frames = (
        movie_frames(NORMALIZED_REFERENCE_MESH_PATH) if model_stl_path is not None else None
    )
    initial_guess_movie_frames = movie_frames(INITIAL_GUESS_MESH_PATH)
    optimized_movie_frames = movie_frames(OPTIMIZED_MESH_PATH)
    true_movie_frames, comparison_movie_frames = combine_movie_frames(
        true_movie_frames, initial_guess_movie_frames, optimized_movie_frames
    )
    write_movie(TRUE_MOVIE_PATH, true_movie_frames, MOVIE_FPS)
    write_movie(INITIAL_GUESS_MOVIE_PATH, initial_guess_movie_frames, MOVIE_FPS)
    write_movie(OPTIMIZED_MOVIE_PATH, optimized_movie_frames, MOVIE_FPS)
    write_movie(COMPARISON_MOVIE_PATH, comparison_movie_frames, MOVIE_FPS)

    elapsed_time = time.perf_counter() - start_time
    summary = {
        "model_num": MODEL_NUM,
        "target_origin": TARGET_ORIGIN,
        "target_curve_type": TARGET_CURVE_TYPE,
        "initial_guess_source": (
            "true_convex_hull" if INITIAL_GUESS_SOURCE_TRUE_CONVEX_HULL else "convex_inversion"
        ),
        "initial_guess_stl": (
            None if INITIAL_GUESS_SOURCE_TRUE_CONVEX_HULL else str(INITIAL_GUESS_STL_PATH)
        ),
        "experiments": list(EXPERIMENTS),
        "observer_count": forward.observer_count,
        "optimization_phases_per_observer": OPTIMIZATION_PHASES,
        "optimization_target_count": forward.observer_count * OPTIMIZATION_PHASES,
        "optimization_iterations": OPTIMIZATION_ITERATIONS,
        "learning_rate": LEARNING_RATE,
        "large_steps_lambda": LARGE_STEPS_LAMBDA,
        "loss_type": LOSS_TYPE,
        "binary_soft_threshold_temperature": (
            BINARY_SOFT_THRESHOLD_TEMPERATURE if TARGET_CURVE_TYPE == "binary" else None
        ),
        "binary_threshold_per_observer": (
            np.asarray(binary_threshold).flatten().tolist()
            if binary_threshold is not None else None
        ),
        "cylinder_radius": cylinder_radius,
        "shape_scale_constraint": "differentiable_challenge_cylinder_normalization",
        "convex_hull_constraint_enabled": ENFORCE_CONVEX_HULL_CONSTRAINT,
        "convex_hull_constraint_type": CONVEX_HULL_CONSTRAINT_MODE,
        "convex_hull_penalty_weight": CONVEX_HULL_PENALTY_WEIGHT,
        "concavity_mask_experiment_enabled": CONCAVITY_MASK_EXPERIMENT,
        "concavity_relative_threshold": (
            CONCAVITY_RELATIVE_THRESHOLD if CONCAVITY_MASK_EXPERIMENT else None
        ),
        "concavity_free_vertex_fraction": (
            float(concave_mask_np.mean()) if CONCAVITY_MASK_EXPERIMENT else None
        ),
        "final_z_min": float(np.min(best_positions_np[:, 2])),
        "final_z_max": float(np.max(best_positions_np[:, 2])),
        "final_radial_max": float(np.max(np.linalg.norm(best_positions_np[:, :2], axis=1))),
        "initial_guess_vertex_count": len(initial_guess.vertices),
        "initial_guess_face_count": len(initial_guess.faces),
        "initial_guess_mesh": str(INITIAL_GUESS_MESH_PATH),
        "remesh_at_iterations": list(REMESH_AT_ITERATIONS),
        "remesh_botsch_kobbelt_iterations": REMESH_BOTSCH_KOBBELT_ITERATIONS,
        "remesh_target_edge_length_factor": REMESH_TARGET_EDGE_LENGTH_FACTOR,
        "remesh_events": remesh_events,
        "best_checkpoint_vertex_count": len(best_positions_np),
        "best_checkpoint_face_count": len(best_faces_np),
        "reference_shape_available": model_stl_path is not None,
        "normalized_reference_mesh": (
            str(NORMALIZED_REFERENCE_MESH_PATH) if model_stl_path is not None else None
        ),
        "movie_experiment": MOVIE_EXPERIMENT,
        "true_shape_forward_curve_mse": true_shape_forward_curve_mse,
        "initial_guess_total_loss": loss_history[0],
        "best_iteration": best_iteration,
        "final_loss": final_loss,
        "true_movie": str(TRUE_MOVIE_PATH),
        "true_movie_is_placeholder": model_stl_path is None,
        "initial_guess_movie": str(INITIAL_GUESS_MOVIE_PATH),
        "optimized_movie": str(OPTIMIZED_MOVIE_PATH),
        "comparison_movie": str(COMPARISON_MOVIE_PATH),
        "optimized_mesh": str(OPTIMIZED_MESH_PATH),
        "minimum_projected_phase_center_separation": (
            forward.layout.minimum_projected_separation
        ),
        "orthographic_view_scale": forward.layout.orthographic_view_scale,
        "elapsed_time_seconds": elapsed_time,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


# %%
if __name__ == "__main__":
    print(f"Running mitsuba shape optimization seeded from convex inversion on model {MODEL_NUM}")
    main()

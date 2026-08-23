# %%
# %load_ext autoreload
# %autoreload 2

import os

os.environ["DSDIR"] = "/work3/projects/HAC-DTU"

import matplotlib.pyplot as plt
import numpy as np

from hac.constants import MODEL_BASE_RADII, OBSERVERS
from hac.data import get_model_stl_path, load_data
from hac.mitsuba import MitsubaMesh, RenderingSettings
from hac.mitsuba.forward import (
    ForwardModel,
    reduction_intensity_hard,
    reduction_intensity_smooth,
    reduction_intensity_sum,
    threshold_sigmoid_gate,
)
from hac.mitsuba.patch import mi

mi.set_variant("cuda_ad_mono")

# %%
# Configuration
MODEL_NUM = 3
N_PHASES = 180

mesh_stl_path = get_model_stl_path(MODEL_NUM)
mesh = MitsubaMesh.from_stl_path(mesh_stl_path)

rendering_settings = RenderingSettings(
    width=64,
    height=64,
    spp=32,
)

bounding_cylinder_radius = MODEL_BASE_RADII[MODEL_NUM]

model = ForwardModel(
    mesh,
    n_phases=N_PHASES,
    observers=OBSERVERS,
    rendering_settings=rendering_settings,
    normalize_mesh=True,
    bounding_cylinder_radius=bounding_cylinder_radius,
    reduction_function=reduction_intensity_sum,
)

# %%
# Render once; every reduction below reuses these frames so we're comparing
# apples to apples instead of re-rendering (and re-sampling noise) each time.
frames = model.render(seed=0)
print(f"frames shape: (observer, phase, y, x, channel) = {tuple(frames.shape)}")

# %%
# --- Image comparison: raw vs. hard threshold (FIPS) vs. sigmoid-gated ---
observer_index = 0
phase_index = 0
observer_name = model.layout.observer_names[observer_index]

threshold = 25.5 / 255.0
temperatures = [0.001, 0.01, 0.05, 0.2]

frame = frames[observer_index, phase_index, :, :, 0]
frame_np = np.array(frame)

hard_thresholded = np.where(frame_np > threshold, frame_np, 0.0)

sigmoid_gated = {}
sigmoid_gate_masks = {}
for temperature in temperatures:
    gate = threshold_sigmoid_gate(frame, threshold=threshold, temperature=temperature)
    sigmoid_gate_masks[temperature] = np.array(gate)
    sigmoid_gated[temperature] = np.array(gate * frame)

n_cols = 2 + len(temperatures)
fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

vmax = frame_np.max()

axes[0].imshow(frame_np, cmap="gray", vmin=0, vmax=vmax)
axes[0].set_title("raw (no threshold)")

axes[1].imshow(hard_thresholded, cmap="gray", vmin=0, vmax=vmax)
axes[1].set_title("hard threshold (fips)")

for ax, temperature in zip(axes[2:], temperatures):
    ax.imshow(sigmoid_gated[temperature], cmap="gray", vmin=0, vmax=vmax)
    ax.set_title(f"sigmoid gate (T={temperature})")

for ax in axes:
    ax.axis("off")

fig.suptitle(f"observer={observer_name!r}, phase={phase_index}")
fig.tight_layout()
plt.show()

# %%
# --- Sigmoid gate mask itself (the weight applied to each pixel, in [0, 1]) ---
fig, axes = plt.subplots(1, len(temperatures), figsize=(4 * len(temperatures), 4))

for ax, temperature in zip(axes, temperatures):
    im = ax.imshow(sigmoid_gate_masks[temperature], cmap="viridis", vmin=0, vmax=1)
    ax.set_title(f"gate mask (T={temperature})")
    ax.axis("off")

fig.colorbar(im, ax=axes, shrink=0.7, label="gate weight")
fig.suptitle(f"observer={observer_name!r}, phase={phase_index}")
plt.show()

# %%
# --- Real and simulated (Blender) reference lightcurves ---
# These have their own frame counts (unrelated to N_PHASES), so curves below
# are compared by phase *fraction* (0-1) rather than by phase index.
real_lightcurves = load_data(MODEL_NUM, origin="real", curve_type="intensity")
simulated_lightcurves = load_data(MODEL_NUM, origin="simulated", curve_type="intensity")

# %%
# --- Lightcurve comparison across reductions ---
lightcurves_sum = reduction_intensity_sum(frames)
lightcurves_fips = reduction_intensity_hard(frames)
lightcurves_fips_smooth = {
    temperature: reduction_intensity_smooth(frames, temperature=temperature)
    for temperature in temperatures
}

lightcurves_sum_np = np.array(lightcurves_sum)
lightcurves_fips_np = np.array(lightcurves_fips)
lightcurves_fips_smooth_np = {
    temperature: np.array(lc) for temperature, lc in lightcurves_fips_smooth.items()
}

phase_fraction = np.arange(N_PHASES) / N_PHASES

fig, axes = plt.subplots(
    model.observer_count, 1, figsize=(8, 3 * model.observer_count), sharex=True
)
if model.observer_count == 1:
    axes = [axes]

for obs_idx, ax in enumerate(axes):
    observer_name = model.layout.observer_names[obs_idx]

    ax.plot(
        phase_fraction, lightcurves_sum_np[obs_idx], "o-", label="sum (no threshold)"
    )
    ax.plot(
        phase_fraction,
        lightcurves_fips_np[obs_idx],
        "s-",
        label="hard threshold (fips)",
    )
    for temperature in temperatures:
        ax.plot(
            phase_fraction,
            lightcurves_fips_smooth_np[temperature][obs_idx],
            "--",
            label=f"sigmoid gate (T={temperature})",
        )

    if observer_name in real_lightcurves.data.columns:
        real_index = real_lightcurves.data.index.to_numpy()
        real_fraction = (real_index - real_index.min()) / len(real_index)
        ax.plot(
            real_fraction,
            real_lightcurves.data[observer_name].to_numpy(),
            ".",
            color="black",
            markersize=2,
            alpha=0.5,
            label="real",
        )

    if observer_name in simulated_lightcurves.data.columns:
        simulated_index = simulated_lightcurves.data.index.to_numpy()
        simulated_fraction = (
            simulated_index - simulated_index.min()
        ) / len(simulated_index)
        ax.plot(
            simulated_fraction,
            simulated_lightcurves.data[observer_name].to_numpy(),
            ".",
            color="gray",
            markersize=2,
            alpha=0.5,
            label="simulated (blender)",
        )

    ax.set_ylabel("normalized intensity")
    ax.set_title(observer_name)
    ax.legend(fontsize="small", loc="upper right")

axes[-1].set_xlabel("phase fraction")
fig.tight_layout()
plt.show()

# %%

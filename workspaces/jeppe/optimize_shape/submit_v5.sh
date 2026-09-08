#!/bin/bash
#BSUB -q c47511
#BSUB -J dtu-hac-optimize-shape-v5[1-10]
#BSUB -n 12
#BSUB -W 08:00
#BSUB -R 'span[hosts=1]'
#BSUB -gpu 'num=1:mode=exclusive_process'
#BSUB -R 'rusage[mem=2GB]'
#BSUB -e logs/dtu-hac-optimize-shape-v5.%J.%I.err
#BSUB -o logs/dtu-hac-optimize-shape-v5.%J.%I.out

# optimize_shape_v5: orthographic + `direct` integrator only (see the module
# docstring for why `direct` is free relative to `direct_projective` here).
#
# Submit from this directory so the log dir exists:
#     cd workspaces/jeppe/optimize_shape && mkdir -p logs && bsub < submit_v5.sh

set -euo pipefail

cd "${LS_SUBCWD:-$(dirname "$0")}"

nvidia-smi

# Match the CUDA toolkit to cupy-cuda13x / mitsuba in pyproject.toml.
module load cuda/13.3.0

# Array job: one asteroid model per index (see #BSUB -J ...[1-10]).
MODEL="${LSB_JOBINDEX:-1}"
[ "$MODEL" -ge 1 ] || MODEL=1  # LSF sets LSB_JOBINDEX=0 for non-array jobs

echo "Running job ${LSB_JOBID:-?}[${LSB_JOBINDEX:-0}] -> model $MODEL on $(hostname)"

# --phases subsamples the light curves by index (no averaging); drop it to keep
# every native frame (~841 real intensity, 360 blender) - 841 phases means a
# ~23.5k-sensor batch scene, far heavier per iteration.
#
# The surface material is fixed to hac.mitsuba.scene.DEFAULT_BSDF (roughplastic,
# mirrors bsdf.json); there is no longer a --bsdf-json option.
#
# Strategy: start deliberately coarse (--initial-face-count 250) and let
# --remesh-level add resolution (Botsch-Kobbelt, ~4x faces per event) only when
# the optimizer plateaus. Level 4 -> ~64k faces worst case.
#
# Every value below is the script default made explicit, except:
#   MODEL (positional)          from $LSB_JOBINDEX
#   --data-source real          (default: simulated)
#   --phases 360                 (default: every native frame)
#   --initial-face-count 250     (default: 2500)
#   --remesh-level 4             (default: fixed topology)
#   --evaluate-reference         added only for models 1-3 (the script rejects
#                                 it for models 4+); gives the reference-shape
#                                 forward MSE floor
args=(
    "$MODEL"
    --data-source real
    --curve-type intensity
    --variant cuda_ad_mono
    --cuda-device 0
    --phases 360
    --width 64
    --height 64
    --spp 8
    --spp-grad 8
    --iterations 300
    --learning-rate 2e-2
    --large-steps-lambda 25.0
    --seed 0
    --validation-interval 10
    --validation-spp 16
    --early-stop-patience 0
    --initial-subdivisions 5
    --initial-face-count 250
    --checkpoint-interval 25
    --reduction-temperature 0.01
    --remesh-level 4
)
if [ "$MODEL" -le 3 ]; then
    args+=(--evaluate-reference)
fi

uv run python optimize_shape_v5.py "${args[@]}"

# Remeshing alternatives: --remesh-interval N for a fixed every-N-iterations
# schedule instead of --remesh-level; omit both for fixed-topology optimization.

# Optional: append a 360-frame turntable of initial / best / reference shapes.
#   --render-movies --movie-observer 0_hor1 --movie-phases 360 \
#   --movie-width 512 --movie-height 512 --movie-spp 16 --movie-fps 30

# --output-dir is left unset on purpose: results land in
#   $WORKSPACE_DIR/optimize_shape_v5/model<N>_real_intensity_<UTC timestamp>/

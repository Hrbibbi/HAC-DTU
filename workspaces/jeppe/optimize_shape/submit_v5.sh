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

echo "Running job ${LSB_JOBID:-?}[${LSB_JOBINDEX:-0}] on $(hostname)"

# --phases subsamples the light curves by index (no averaging); drop it to keep
# every native frame (~841 real intensity, 360 blender) - 841 phases means a
# ~23.5k-sensor batch scene, far heavier per iteration.
#
# The surface material is fixed to hac.mitsuba.scene.DEFAULT_BSDF (roughplastic,
# mirrors bsdf.json); there is no longer a --bsdf-json option.
#
# Every value below is the script default made explicit, except:
#   --data-source real          (default: simulated)
#   --phases 360                 (default: every native frame)
#   --initial-face-count 250     (default: 2500)
#   --evaluate-reference         (default: off; models 1-3 only, gives the
#                                 reference-shape forward MSE floor)
uv run python optimize_shape_v5.py \
    1 \
    --data-source real \
    --curve-type intensity \
    --variant cuda_ad_mono \
    --cuda-device 0 \
    --phases 360 \
    --width 64 \
    --height 64 \
    --spp 8 \
    --spp-grad 8 \
    --iterations 300 \
    --learning-rate 2e-2 \
    --large-steps-lambda 25.0 \
    --seed 0 \
    --validation-interval 10 \
    --validation-spp 16 \
    --early-stop-patience 0 \
    --initial-subdivisions 5 \
    --initial-face-count 250 \
    --checkpoint-interval 25 \
    --reduction-temperature 0.01 \
    --evaluate-reference

# Optional: append a 360-frame turntable of initial / best / reference shapes.
#   --render-movies \
#   --movie-observer 0_hor1 \
#   --movie-phases 360 \
#   --movie-width 512 \
#   --movie-height 512 \
#   --movie-spp 16 \
#   --movie-fps 30

# --output-dir is left unset on purpose: results land in
#   $WORKSPACE_DIR/optimize_shape_v5/model1_real_intensity_<UTC timestamp>/

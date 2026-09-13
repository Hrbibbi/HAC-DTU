#!/bin/sh
### Mitsuba shape optimization from a coarse (~1000-vertex) convex-inversion
### initial guess, real intensity data, asteroids 1-3. Coarsens the raw
### convex-inversion reconstructions (CONVEX_INVERSION_DIR, expected to
### contain asteroidN_simulated_intensity.stl) before optimizing.
### Config: LARGE_STEPS_LAMBDA=12.5, 90 phases, 32x32, 600 iterations, no
### remeshing, convex-hull and cylinder constraints on.
### See mitsuba_from_convex.py, coarsen_convex_inversion.py.
### Submit with: bsub < workspaces/christian/SUBMISSION_mitsuba_from_convex_real_intensity/submit_mitsuba_from_convex_coarse_asteroids123.sh

#BSUB -q c47511
#BSUB -J mitsuba_from_convex_coarse_ast123
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 8:00
#BSUB -R "rusage[mem=15GB]"
#BSUB -o /zhome/95/1/155570/HAC-DTU/workspaces/christian/mitsuba_from_convex/logs/mitsuba_from_convex_coarse_ast123_%J.out
#BSUB -e /zhome/95/1/155570/HAC-DTU/workspaces/christian/mitsuba_from_convex/logs/mitsuba_from_convex_coarse_ast123_%J.err

set -eu
REPO_ROOT="/zhome/95/1/155570/HAC-DTU"
cd "$REPO_ROOT/workspaces/christian/SUBMISSION_mitsuba_from_convex_real_intensity"

module load cuda/13.3.0
source "$HOME/work3/miniforge3/bin/activate" hac

export LARGE_STEPS_LAMBDA=12.5
export OPTIMIZATION_PHASES=90
export OPTIMIZATION_WIDTH=32
export OPTIMIZATION_HEIGHT=32
export OPTIMIZATION_ITERATIONS=600
export INITIAL_GUESS_REMESH_AT_START=0
export INITIAL_GUESS_SUBDIVISION_PASSES=0
export ENABLE_MID_RUN_REMESH=0

# Set to wherever asteroidN_simulated_intensity.stl (raw convex-inversion
# reconstructions) live if not using this project's default location.
export CONVEX_INVERSION_DIR="${CONVEX_INVERSION_DIR:-/zhome/95/1/155570/HAC-DTU/HAC-collab/reconstructions/convex/intensity}"
python coarsen_convex_inversion.py 1 2 3

for MODEL_NUM in 1 2 3; do
    echo "=================== MODEL_NUM=${MODEL_NUM} ==================="
    export MODEL_NUM
    export INITIAL_GUESS_STL_PATH="$REPO_ROOT/workspaces/christian/SUBMISSION_mitsuba_from_convex_real_intensity/coarse_initial_guess/asteroid${MODEL_NUM}_simulated_intensity_coarse1000.stl"
    python -u mitsuba_from_convex.py
done

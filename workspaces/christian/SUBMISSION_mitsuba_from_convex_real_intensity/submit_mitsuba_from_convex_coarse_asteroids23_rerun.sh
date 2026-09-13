#!/bin/sh
### Rerun of submit_mitsuba_from_convex_coarse_asteroids123.sh for
### MODEL_NUM=2,3 only -- asteroid 1 already completed successfully in
### that job and does not need to be rerun. Coarsens the raw convex-
### inversion reconstructions (CONVEX_INVERSION_DIR, expected to contain
### asteroidN_simulated_intensity.stl) before optimizing.
### See mitsuba_from_convex.py, coarsen_convex_inversion.py.
### Submit with: bsub < workspaces/christian/SUBMISSION_mitsuba_from_convex_real_intensity/submit_mitsuba_from_convex_coarse_asteroids23_rerun.sh

#BSUB -q c47511
#BSUB -J mitsuba_from_convex_coarse_ast23_rerun
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 8:00
#BSUB -R "rusage[mem=15GB]"
#BSUB -o /zhome/95/1/155570/HAC-DTU/workspaces/christian/mitsuba_from_convex/logs/mitsuba_from_convex_coarse_ast23_rerun_%J.out
#BSUB -e /zhome/95/1/155570/HAC-DTU/workspaces/christian/mitsuba_from_convex/logs/mitsuba_from_convex_coarse_ast23_rerun_%J.err

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
python coarsen_convex_inversion.py 2 3

for MODEL_NUM in 2 3; do
    echo "=================== MODEL_NUM=${MODEL_NUM} ==================="
    export MODEL_NUM
    export INITIAL_GUESS_STL_PATH="$REPO_ROOT/workspaces/christian/SUBMISSION_mitsuba_from_convex_real_intensity/coarse_initial_guess/asteroid${MODEL_NUM}_simulated_intensity_coarse1000.stl"
    python -u mitsuba_from_convex.py
done

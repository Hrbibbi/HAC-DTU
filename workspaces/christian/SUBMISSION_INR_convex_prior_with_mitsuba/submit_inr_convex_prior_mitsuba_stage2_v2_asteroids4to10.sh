#!/bin/sh
### INR with convex prior, Mitsuba-rendered in both stages, real
### intensity data, held-out asteroids 4-10 (no ground truth).
### Config: STAGE1_STEPS=80, STAGE2_STEPS=150.
### See inr_convex_prior_with_mitsuba.py.
### Submit with: bsub < workspaces/christian/SUBMISSION_INR_convex_prior_with_mitsuba/submit_inr_convex_prior_mitsuba_stage2_v2_asteroids4to10.sh

#BSUB -q gpua100
#BSUB -J inr_stage2_v2_ast4to10
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 8:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o /zhome/95/1/155570/HAC-DTU/workspaces/christian/bjobs/inr_mitsuba_optimization/logs/inr_stage2_v2_ast4to10_%J.out
#BSUB -e /zhome/95/1/155570/HAC-DTU/workspaces/christian/bjobs/inr_mitsuba_optimization/logs/inr_stage2_v2_ast4to10_%J.err

set -eu
REPO_ROOT="/zhome/95/1/155570/HAC-DTU"
cd "$REPO_ROOT/workspaces/christian/SUBMISSION_INR_convex_prior_with_mitsuba"

module load cuda/13.3.0
source "$HOME/work3/miniforge3/bin/activate" hac

export OPTIX_CACHE_MAXSIZE=0
export STAGE1_STEPS=80
export STAGE2_STEPS=150
export EVAL_EVERY=10

for ASTEROID_ID in 4 5 6 7 8 9 10; do
    echo "=================== ASTEROID_ID=${ASTEROID_ID} ==================="
    export ASTEROID_ID
    python -u inr_convex_prior_with_mitsuba.py
done

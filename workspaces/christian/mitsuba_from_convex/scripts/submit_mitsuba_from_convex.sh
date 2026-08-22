#!/bin/sh
### Mitsuba differentiable inverse-rendering shape optimization, seeded from
### a convex-inversion reconstruction

#BSUB -q c47511
#BSUB -J mitsuba_from_convex
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -R "rusage[mem=15GB]"
#BSUB -o logs/mitsuba_from_convex_%J.out
#BSUB -e logs/mitsuba_from_convex_%J.err

REPO_ROOT="/zhome/95/1/155570/HAC-DTU"
cd "$REPO_ROOT/workspaces/christian/mitsuba_from_convex"

module load cuda/13.3.0
source "$HOME/work3/miniforge3/bin/activate" hac

export LARGE_STEPS_LAMBDA=12
export OPTIMIZATION_PHASES=360
python -u mitsuba_from_convex.py

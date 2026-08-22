#!/bin/sh
### Variant of mitsuba_from_convex.py: starts from a small sphere (not
### fitted to the cylinder) instead of the convex-inversion mesh, using the
### convex-inversion reconstruction as a soft Chamfer-distance prior

#BSUB -q c47511
#BSUB -J mitsuba_from_convex_small_sphere_chamfer_prior
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -R "rusage[mem=15GB]"
#BSUB -o /zhome/95/1/155570/HAC-DTU/workspaces/christian/mitsuba_from_convex/logs/mitsuba_from_convex_small_sphere_chamfer_prior_%J.out
#BSUB -e /zhome/95/1/155570/HAC-DTU/workspaces/christian/mitsuba_from_convex/logs/mitsuba_from_convex_small_sphere_chamfer_prior_%J.err

REPO_ROOT="/zhome/95/1/155570/HAC-DTU"
cd "$REPO_ROOT/workspaces/christian/mitsuba_from_convex"

module load cuda/13.3.0
source "$HOME/work3/miniforge3/bin/activate" hac

export LARGE_STEPS_LAMBDA=12
export ENFORCE_CYLINDER_CONSTRAINT=0
export LAMBDA_CHAMFER=0.15
export OPTIMIZATION_PHASES=360
python -u mitsuba_from_convex_small_sphere_chamfer_prior.py

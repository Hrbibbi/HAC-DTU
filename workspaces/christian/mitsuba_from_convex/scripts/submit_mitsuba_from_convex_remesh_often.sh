#!/bin/sh
### Variant of mitsuba_from_convex.py: remeshes (Botsch-Kobbelt isotropic
### remeshing) at multiple points during optimization instead of once at the
### midpoint

#BSUB -q c47511
#BSUB -J mitsuba_from_convex_remesh_often
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -R "rusage[mem=15GB]"
#BSUB -o logs/mitsuba_from_convex_remesh_often_%J.out
#BSUB -e logs/mitsuba_from_convex_remesh_often_%J.err

set -eu

module load cuda/13.3.0
source "$HOME/work3/miniforge3/bin/activate" hac

python -u mitsuba_from_convex_remesh_often.py

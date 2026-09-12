#!/bin/sh

#BSUB -q gpul40s
#BSUB -J multiconvex
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -B
#BSUB -N
#BSUB -o multiconvex_%J.out
#BSUB -e multiconvex_%J.err

# Load CUDA
module load cuda/13.1.0

# Go to project directory
cd ~/HelsinkiAsteroidChallenge/HAC-DTU || exit 1

# Activate virtual environment
source .venv/bin/activate

#
cd workspaces/magnus || exit 1

# Run your script
python multiconvex.py

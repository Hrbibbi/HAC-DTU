#!/bin/bash
#BSUB -q gpua10
#BSUB -J dtu-hac-optimization[5-10]
#BSUB -n 4
#BSUB -W 01:00
#BSUB -R 'span[hosts=1]'
#BSUB -gpu 'num=1:mode=exclusive_process'
#BSUB -R 'rusage[mem=16GB]'
#BSUB -e logs/dtu-hac-optimization.%J.%I.err
#BSUB -o logs/dtu-hac-optimization.%J.%I.out

nvidia-smi

# Load the cuda module
module load cuda/13.3.0

echo "Running job $LSB_JOBINDEX on $(hostname)"
uv run python -u mitsuba_shape_optimization.py $LSB_JOBINDEX

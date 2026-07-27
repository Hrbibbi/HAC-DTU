#!/bin/bash
#BSUB -q gpua10
#BSUB -J dtu-hac-optimization[1-10]
#BSUB -n 8
#BSUB -W 04:00
#BSUB -R 'span[hosts=1]'
#BSUB -gpu 'num=1:mode=exclusive_process'
#BSUB -R 'rusage[mem=4GB]'
#BSUB -e logs/dtu-hac-optimization.%J.%I.err
#BSUB -o logs/dtu-hac-optimization.%J.%I.out

nvidia-smi

# Load the cuda module
module load cuda/13.2

uv run python mitsuba_shape_optimization.py $LSB_JOBINDEX

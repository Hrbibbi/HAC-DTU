#!/bin/sh

#BSUB -q c47511
#BSUB -J "convexinitial[4-10]"
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 5:00
#BSUB -R 'span[hosts=1]'
#BSUB -R "rusage[mem=5GB]"
#BSUB -B
#BSUB -N
#BSUB -o convexinitial_%I_%J.out
#BSUB -e convexinitial_%I_%J.err

# Load CUDA
module load cuda/13.1.0

# Go to project directory
cd ~/HelsinkiAsteroidChallenge/HAC-DTU || exit 1

# Activate virtual environment
source .venv/bin/activate

# Go to workspace
cd convexinitial || exit 1

# Run asteroid number given by the job-array index
python convexinitial.py $LSB_JOBINDEX simulated
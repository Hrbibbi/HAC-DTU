#!/bin/bash
#BSUB -q gpua10
#BSUB -J dtu-hac-optimization-forward
#BSUB -n 12
#BSUB -W 08:00
#BSUB -R 'span[hosts=1]'
#BSUB -gpu 'num=1:mode=exclusive_process'
#BSUB -R 'rusage[mem=2GB]'
#BSUB -e logs/dtu-hac-optimization-forward.%J.%I.err
#BSUB -o logs/dtu-hac-optimization-forward.%J.%I.out

nvidia-smi

# Load the cuda module
module load cuda/13.3.0

echo "Running job $LSB_JOBINDEX on $(hostname)"
uv run python forward_optimize.py --data-source real --curve-type intensity --bsdf-configs bsdfs_roughplastic_extended.json
uv run python forward_optimize.py --data-source real --curve-type intensity --bsdf-configs bsdfs_roughplastic.json
uv run python forward_optimize.py --data-source simulated --curve-type intensity --bsdf-configs bsdfs_roughplastic_extended.json
uv run python forward_optimize.py --data-source simulated --curve-type intensity --bsdf-configs bsdfs_roughplastic.json
uv run python forward_optimize.py --data-source real --curve-type intensity --bsdf-configs bsdfs_blender_default.json
uv run python forward_optimize.py --data-source real --curve-type intensity --bsdf-configs bsdfs_principled_extended.json
uv run python forward_optimize.py --data-source simulated --curve-type intensity --bsdf-configs bsdfs_blender_default.json
uv run python forward_optimize.py --data-source simulated --curve-type intensity --bsdf-configs bsdfs_principled_extended.json

#!/bin/sh
#BSUB -J sim_damit[1-82]
#BSUB -q hpc
#BSUB -n 4
#BSUB -W 01:00
#BSUB -R "rusage[mem=4GB]"
# Make sure the logs folder exists!
#BSUB -o workspaces/david/bjobs/logs/sim_damit_%J_%I.out
#BSUB -e workspaces/david/bjobs/logs/sim_damit_%J_%I.err

uv run python -u workspaces/david/bjobs/sim_damit.py $LSB_JOBINDEX
#!/bin/bash

#SBATCH --job-name=finetune_chgnet
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=6G
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --error=job.err
#SBATCH --output=job.out

# For Grendel:
# check nodes available: gnodes
# check running jobs: mj
# submit job: sbatch <filename> (example: sbatch submit_job.sh)
# cancel running job: scancel <job ID>
# access node: ssh <node name>

sleep 1d

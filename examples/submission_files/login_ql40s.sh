#!/bin/bash

#SBATCH --job-name=login
#SBATCH --partition=ql40s
#SBATCH --nodes=1
#SBATCH --mem=754G
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=2
#SBATCH --time=08:00:00
#SBATCH --error=job.err
#SBATCH --output=job.out

# For Grendel:
# check nodes available: gnodes
# check running jobs: mj
# submit job: sbatch <filename> (example: sbatch submit_job.sh)
# cancel running job: scancel <job ID>
# access node: ssh <node name>

sleep 1d

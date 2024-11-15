#!/bin/bash

#SBATCH --job-name=finetune_chgnet
#SBATCH --partition=ql40s
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=6G
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --error=job.err
#SBATCH --output=job.out

source /comm/swstack/bin/modules.sh --force
source ~/.bashrc
conda activate chgnet

python finetune_chgnet.py

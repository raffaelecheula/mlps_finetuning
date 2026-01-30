#!/bin/bash

#SBATCH --job-name=chgnet
#SBATCH --partition=ql40s
#SBATCH --nodes=1
#SBATCH --mem=377G
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --error=job.err
#SBATCH --output=job.out

source /comm/swstack/bin/modules.sh --force
source ~/.bashrc
conda activate chgnet

python run_cross_validation.py

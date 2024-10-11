#!/bin/bash
#SBATCH -J long
#SBATCH -N 1
#SBATCH -p IAI_SLURM_HGX
#SBATCH --gres=gpu:4
#SBATCH --qos=16gpu-hgx
#SBATCH --time=72:00:00
#SBATCH -o logs/process-QuALITY-%j-out.log
#SBATCH -e logs/process-QuALITY-%j-err.log
#SBATCH -c 1

python scripts/datasets/process_quality.py

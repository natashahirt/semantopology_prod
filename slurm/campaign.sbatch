#!/bin/bash
# One campaign row per array task. CPU only. Skip rows that already have DONE.
#
#   python slurm/make_manifest.py
#   # set --array to 0-(N-1) from the printed row count
#   sbatch slurm/campaign.sbatch
#
# Activate the env inside this script. Slurm does not load .bashrc.

#SBATCH --job-name=semantopology
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-108%8
#SBATCH --output=logs/%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs results

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Pre-downloaded CLIP weights. Compute nodes may have no internet.
# export CLIP_CACHE="${SLURM_SUBMIT_DIR}/.cache/clip"
# export TORCH_HOME="${SLURM_SUBMIT_DIR}/.cache/torch"

# module load miniforge
# source activate semantopology

python slurm/run_row.py

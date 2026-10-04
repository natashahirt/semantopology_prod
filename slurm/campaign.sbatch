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
# mit_normal caps a base account at 96 cores per user (256 for Standard):
# 24 concurrent tasks x 4 CPUs = 96. Raise the % only after checking the cap.
#SBATCH --array=0-117%24
#SBATCH --output=logs/%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs results

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Pre-downloaded CLIP weights. Compute nodes may have no internet.
export CLIP_CACHE="${HOME}/.cache/clip"
export TORCH_HOME="${HOME}/.cache/torch"

module load miniforge/23.11.0-0
source activate semantopology
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

python slurm/run_row.py

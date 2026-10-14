#!/bin/bash
# One finer tee hybrid. The manifest is a single row, so the array is one task.
# The physics grid ends at 384×320; the first stage is the 192×160 tee.
#
#SBATCH --job-name=tee-fine
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0
#SBATCH --output=logs/%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs results

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export CAMPAIGN_MANIFEST=slurm/tee_fine.tsv

export CLIP_CACHE="${HOME}/.cache/clip"
export TORCH_HOME="${HOME}/.cache/torch"

module load miniforge/23.11.0-0
source activate semantopology
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

python slurm/run_row.py

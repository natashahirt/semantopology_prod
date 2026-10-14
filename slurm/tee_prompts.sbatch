#!/bin/bash
# Two full-grid tee hybrids: "spiral galaxies" and "creeping ivy".
# Same 192×160 recipe as the pinned tee_full solves, seed 101.
#
#SBATCH --job-name=tee-prompt
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs results

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export CAMPAIGN_MANIFEST=slurm/tee_prompts.tsv

export CLIP_CACHE="${HOME}/.cache/clip"
export TORCH_HOME="${HOME}/.cache/torch"

module load miniforge/23.11.0-0
source activate semantopology
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

python slurm/run_row.py

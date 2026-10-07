#!/bin/bash
# Phase 3 after the campaign arrays leave the queue:
#
#   sbatch --dependency=afterany:JOB slurm/diversity.sbatch
#
# CLIP scoring uses the CPUs this job requested. The diversity re-solves
# use CHOLMOD, so OpenMP is pinned to 1 before those processes start.

#SBATCH --job-name=diversity
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs analysis/out

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export CLIP_CACHE="${HOME}/.cache/clip"
export TORCH_HOME="${HOME}/.cache/torch"
export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_OFFLINE=1

module load miniforge/23.11.0-0
source activate semantopology
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
python -m analysis.evaluate --device cpu
python -m analysis.figures

# CHOLMOD must stay single-threaded. Each re-solve is a fresh interpreter.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -m analysis.diversity solve
python -m analysis.diversity report
python -m analysis.specificity report

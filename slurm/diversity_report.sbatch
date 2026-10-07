#!/bin/bash
# Retry of the diversity report after the evaluator pass in job 24954686.
# Scores and figures are already on disk. This re-solves only cache misses
# (CHOLMOD, one thread) and then writes the diversity and specificity tables.

#SBATCH --job-name=diversity-report
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs analysis/out

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

module load miniforge/23.11.0-0
source activate semantopology
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

python -m analysis.diversity solve
python -m analysis.diversity report
python -m analysis.specificity report

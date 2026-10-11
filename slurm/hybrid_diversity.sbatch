#!/bin/bash
# Diversity analysis for every tall hybrid. Re-solves are CHOLMOD, so
# OpenMP stays at 1. The semantic diversity tables are left in place;
# this writes analysis/out/diversity_hybrid/.

#SBATCH --job-name=hybrid-diversity
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
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

python -m analysis.diversity solve --hybrids
python -m analysis.diversity report --hybrids

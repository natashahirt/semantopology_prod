#!/bin/bash
# Semantic-over-hybrid sheet for the 24 M prompts. Submit after the Mh array:
#
#   sbatch --dependency=afterany:JOB slurm/m_hybrid_gallery.sbatch

#SBATCH --job-name=m-hybrid-gallery
#SBATCH --partition=mit_normal
#SBATCH --account=mit_general
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs analysis/out

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

module load miniforge/23.11.0-0
source activate semantopology
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

python -m analysis.figures --gallery-only

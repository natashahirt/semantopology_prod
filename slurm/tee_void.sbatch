#!/bin/bash
# Re-solve the three full-grid spiral-galaxy tees with void logits pinned.
# run_row.py skips a run id once any attempt has DONE, so this calls run.py
# directly and the next attempt lands beside attempt_1.
#
#SBATCH --job-name=tee-void
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs results

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

export CLIP_CACHE="${HOME}/.cache/clip"
export TORCH_HOME="${HOME}/.cache/torch"

module load miniforge/23.11.0-0
source activate semantopology
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

SEEDS=(101 202 303)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

python run.py \
  --run-id "tee/spiral_galaxy/seed-${SEED}/hybrid" \
  --experiment tee \
  --group hybrid \
  --mode hybrid \
  --structure tee_full \
  --clip "spiral galaxy" \
  --seed "${SEED}" \
  --physics-beta-max 8

#!/bin/bash
# One dream-layout job per line of slurm/manifest.txt.
#
# mit_normal_gpu jobs last at most 6 hours and the base GPU limit is 2, so
# two array tasks can run at once. mit_preemptable allows a longer limit and
# more GPUs, but a preempted task starts over: this solver does not resume.
#
# Replication does not need this script. The same command is:
#   python run.py --problem multistory_building --clip "unfurling fern fronds"
#
# Before submitting, set --array to 0 through (number of manifest lines - 1)
# and activate the environment inside this script. Slurm does not load .bashrc.

#SBATCH --job-name=semantopology
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs results

export OMP_NUM_THREADS=1

# module load miniforge
# source activate neural_structural_opt

mapfile -t JOBS < <(grep -vE '^[[:space:]]*(#|$)' slurm/manifest.txt)
LINE="${JOBS[$SLURM_ARRAY_TASK_ID]}"
PROBLEM="${LINE%%$'\t'*}"
CLIP="${LINE#*$'\t'}"
export CLIP_TEXT="${CLIP}"
SLUG="$(python -c 'import os; from recipe.preset import prompt_slug; print(prompt_slug(os.environ["CLIP_TEXT"]))')"

python run.py \
  --problem "${PROBLEM}" \
  --clip "${CLIP}" \
  --device cuda \
  --out "results/${SLUG}"

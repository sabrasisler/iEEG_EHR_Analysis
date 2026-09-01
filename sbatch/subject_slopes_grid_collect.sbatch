#!/bin/bash
#SBATCH --job-name=mm_slopes_collect
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=24GB
#SBATCH --output=logs/mm_slopes_collect_%j.out
#
# Concatenate the per-region unpooled slopes and reduce them to one
# sign-consistency number per cell.
#
#   sbatch --dependency=afterok:<array_jobid> --export=ALL,RUN_DIR="$RUN_DIR" \
#          sbatch/subject_slopes_grid_collect.sbatch

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

: "${RUN_DIR:?set RUN_DIR to the grid run directory}"

echo "=== subject slopes collect | $(date) ==="

python -m ieeg_ehr.analysis.compute_subject_slopes_grid \
    --stage collect \
    --run-dir "${RUN_DIR}"

echo "=== done ($(date)) ==="

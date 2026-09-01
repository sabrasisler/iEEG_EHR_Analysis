#!/bin/bash
#SBATCH --job-name=mm_slopes
#SBATCH --partition=normal
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --output=logs/mm_slopes_%A_%a.out
#SBATCH --array=0-20
#
# Unpooled per-subject pain slopes for every cell, one array task per region.
# Feeds the sign-consistency map.
#
#   sbatch --export=ALL,RUN_DIR="$RUN_DIR" sbatch/subject_slopes_grid_array.sbatch
#   sbatch --dependency=afterok:<jobid> --export=ALL,RUN_DIR="$RUN_DIR" \
#          sbatch/subject_slopes_grid_collect.sbatch
#
# This is NOT a refit. Each subject gets an ordinary least-squares line through
# their own epochs, which costs microseconds; the entire runtime is the Lustre
# read of the per-channel view, which is exactly why it is split by region -- a
# region's bins share one read.
#
# The BLUPs would have been free and are the wrong answer: partial pooling drags
# subjects toward the group, so BLUP-based sign consistency sits at 0.8-1.0
# everywhere and separates nothing.
#
# RESOURCES ARE DELIBERATELY SMALL. A first version asked 48 GB / 2 CPUs / 2 h and
# sat unscheduled behind an 80k-core backlog -- on `normal` 48 GB alone charges six
# cores, so the request was competing for a slot it never needed. A region's 38
# bins are ~1.4M rows of a few columns at the very worst (Lateral Temporal, 801
# contacts), the arithmetic is one polyfit per subject, and there is no mixed model
# anywhere in this job. 16 GB on one core backfills almost immediately.

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

: "${RUN_DIR:?set RUN_DIR to a grid run directory containing grid_cells.parquet}"

echo "=== subject slopes | region ${SLURM_ARRAY_TASK_ID} | $(date) ==="

python -m ieeg_ehr.analysis.compute_subject_slopes_grid \
    --stage fit \
    --run-dir "${RUN_DIR}" \
    --region-index "${SLURM_ARRAY_TASK_ID}"

echo "=== done ($(date)) ==="

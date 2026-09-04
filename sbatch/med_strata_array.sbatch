#!/bin/bash
#SBATCH --job-name=mm_medstrata
#SBATCH --partition=normal
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24GB
#SBATCH --output=logs/mm_medstrata_%A_%a.out
#SBATCH --array=0-20
#
# Medication-stratified grid, stage `fit`: ONE ARRAY TASK PER REGION.
#
#   RUN_DIR=$(python -m ieeg_ehr.analysis.run_mixed_model_med_strata \
#                 --stage prepare --drug-set analgesics)
#   sbatch --export=ALL,RUN_DIR="$RUN_DIR" sbatch/med_strata_array.sbatch
#   sbatch --dependency=afterok:<jobid> --export=ALL,RUN_DIR="$RUN_DIR" \
#          sbatch/med_strata_collect.sbatch
#
# THREE FITS PER CELL, not one: medicated, unmedicated, and the interaction that
# is the only thing licensing a comparison between the two maps.
#
# RESOURCES ARE MEASURED, not guessed. Insula -- 502 contacts, the second-largest
# region -- ran all 38 bins x 3 fits in ~80 seconds inside 16 GB. 24 GB leaves
# room for Lateral Temporal at 801 contacts and nothing more; an earlier job in
# this project sat unschedulable behind an 80k-core backlog purely because it
# asked for 48 GB it never used.
#
# Per region, not per cell: a region's bins share one Lustre read of 51 subject
# tables, which is the expensive part. Same reasoning as the unstratified grid.
#
# The per-subject unpooled slopes for the sign-consistency panel are computed
# HERE, while the frames are already in memory. Doing it afterwards with
# compute_subject_slopes_grid would mean six more full reads of the view (three
# strata x two drug sets) for numbers that are free at this point.

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

: "${RUN_DIR:?set RUN_DIR to the directory printed by --stage prepare}"
EXTRA=${EXTRA:-}

echo "=== med strata | region ${SLURM_ARRAY_TASK_ID} | $(date) ==="
echo "    run: ${RUN_DIR}"

python -m ieeg_ehr.analysis.run_mixed_model_med_strata \
    --stage fit \
    --run-dir "${RUN_DIR}" \
    --region-index "${SLURM_ARRAY_TASK_ID}" \
    ${EXTRA}

echo "=== done ($(date)) ==="

#!/bin/bash
#SBATCH --job-name=mm_changescore
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --output=logs/mm_changescore_%A_%a.out
#SBATCH --array=0-20
#
# Paired change-score grid, stage `fit`: ONE ARRAY TASK PER REGION.
#
#   RUN_DIR=$(python -m ieeg_ehr.analysis.run_change_score_grid \
#                 --stage prepare --drug-set analgesics \
#                 --min-gap-min 30 --max-gap-min 240)
#   sbatch --export=ALL,RUN_DIR="$RUN_DIR" sbatch/change_score_array.sbatch
#   sbatch --dependency=afterok:<jobid> --export=ALL,RUN_DIR="$RUN_DIR" \
#          sbatch/change_score_collect.sbatch
#
# Rows are (consecutive assessment pair x channel) and the outcome is the CHANGE
# in log10 power, so this is the only design in the project where a slow
# multiplicative drift on the signal -- impedance, reference, amplitude --
# cancels rather than being absorbed into the medication main effect.
#
# TWO FITS PER CELL: the full model and the VC_CHANGE_REDUCED variant, whose LRT
# is the mixture-corrected test on the random d_pain slope.
#
# RESOURCES ARE MEASURED HERE, not inherited. This file first carried the
# stratified grid's 24 GB / 90 min; the run of 2026-09-10 then measured the fit
# stage at MaxRSS 0.41-1.09 GB across 10 completed region tasks (median 0.91 GB),
# so 24 GB was ~22x oversized. That padding is not free -- it is what got a
# diagnostic rejected outright by QOSMaxMemoryPerUser on `dev` that same day, and
# an earlier job in this project sat unschedulable behind an 80k-core backlog
# purely because it asked for 48 GB it never used. 4 GB is ~4x the observed peak.
#
# 1 CPU, not 2: OMP/OPENBLAS/MKL are all pinned to one thread below, so the
# second core was never used, and 1 core packs 21 array tasks onto the 24-core
# ckeller1 node. Walltime 1 h is generous -- those same tasks took 35 min WITH
# the channel variance component, which was ~13 s/cell; without it a cell fits in
# ~0.55 s, so the grid is minutes now (see mixed_model.VC_CHANGE).
#
# Per region, not per cell: a region's bins share one Lustre read of the subject
# view tables, which is the expensive part. Same reasoning as the other grids.

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

# `fit` is the group mixed model per cell; `subjects` is the per-SUBJECT
# med_between coefficient the cluster permutation test needs (798 cells are not
# 798 independent tests -- see docs/cluster_permutation.md). Same region
# parallelism and the same Lustre read either way, so one file serves both.
STAGE=${STAGE:-fit}

echo "=== change score | stage ${STAGE} | region ${SLURM_ARRAY_TASK_ID} | $(date) ==="
echo "    run: ${RUN_DIR}"

python -m ieeg_ehr.analysis.run_change_score_grid \
    --stage "${STAGE}" \
    --run-dir "${RUN_DIR}" \
    --region-index "${SLURM_ARRAY_TASK_ID}" \
    ${EXTRA}

echo "=== done ($(date)) ==="

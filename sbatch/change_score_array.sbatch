#!/bin/bash
#SBATCH --job-name=mm_changescore
#SBATCH --partition=normal
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24GB
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
# is the mixture-corrected test on the random d_pain slope. Fewer than the
# stratified grid's three, so the 24 GB / 90 min envelope measured there (Insula,
# 502 contacts, 38 bins x 3 fits in ~80 s inside 16 GB) is the ceiling here too.
# Not padded further: an earlier job in this project sat unschedulable behind an
# 80k-core backlog purely because it asked for 48 GB it never used.
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

echo "=== change score | region ${SLURM_ARRAY_TASK_ID} | $(date) ==="
echo "    run: ${RUN_DIR}"

python -m ieeg_ehr.analysis.run_change_score_grid \
    --stage fit \
    --run-dir "${RUN_DIR}" \
    --region-index "${SLURM_ARRAY_TASK_ID}" \
    ${EXTRA}

echo "=== done ($(date)) ==="

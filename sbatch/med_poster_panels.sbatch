#!/bin/bash
#SBATCH --job-name=med_poster_panels
#SBATCH --partition=owners
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --output=logs/med_poster_panels_%j.out
#
# Poster panels A-E: analgesic exposure around pain epochs.
#
# Unlike every other figure in med_analysis, the unit here is a PAIN EPOCH, so
# this reads the epoch definitions and obeys the cohort gate -- discovery only
# by default, and the hold-out is not reachable through --split.
#
# Output goes under the PAIN event (analysis/pain/medication_exposure/), not
# meds/, because the unit is an epoch rather than an administration.
#
#   sbatch sbatch/med_poster_panels.sbatch
#   SCRATCH_RUN=1 sbatch sbatch/med_poster_panels.sbatch   # throwaway
#   WINDOW=4 sbatch sbatch/med_poster_panels.sbatch        # 4 h exposure window

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

WINDOW=${WINDOW:-2}
SPLIT=${SPLIT:-discovery}
DEST=${SCRATCH_RUN:+--scratch}

echo "=== poster panels A-E | window=${WINDOW}h | split=${SPLIT} ($(date)) ==="

python -m ieeg_ehr.med_analysis.plot_poster_epoch_meds \
    --window-hours "$WINDOW" --split "$SPLIT" ${DEST}

echo "=== done ($(date)) ==="

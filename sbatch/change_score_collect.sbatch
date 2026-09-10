#!/bin/bash
#SBATCH --job-name=mm_changescore_collect
#SBATCH --partition=normal
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --output=logs/mm_changescore_collect_%j.out
#
# Paired change-score grid, stage `collect`: concatenate the per-region output,
# apply BH SEPARATELY to each coefficient family (d_pain, med_between, the
# d_pain x med interaction, and the med x gap time-structure term), and write
# METHODS.md with the carryover and regression-to-the-mean caveats.
#
#   sbatch --dependency=afterok:<array_jobid> --export=ALL,RUN_DIR="$RUN_DIR" \
#          sbatch/change_score_collect.sbatch
#
# Four families, not one pooled correction: `med_gap` is the term that asks
# whether the effect has TIME STRUCTURE, which is the empirical precursor to a
# PK model, and pooling it with the main effect would bury it.
#
# 4 GB because this concatenates 21 region tables totalling ~800 rows and runs BH
# over them. The 16 GB it originally asked for was copied from the stratified
# grid's collect and was never justified by anything this stage does; on
# `ckeller1`, where mem/core is capped at 8 GB, that padding is the difference
# between running and being rejected.

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

: "${RUN_DIR:?set RUN_DIR to the directory printed by --stage prepare}"

echo "=== change score collect | $(date) ==="
echo "    run: ${RUN_DIR}"

python -m ieeg_ehr.analysis.run_change_score_grid \
    --stage collect \
    --run-dir "${RUN_DIR}"

echo "=== done ($(date)) ==="

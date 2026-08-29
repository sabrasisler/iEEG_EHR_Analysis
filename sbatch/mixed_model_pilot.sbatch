#!/bin/bash
#SBATCH --job-name=mm_pilot
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --output=logs/mm_pilot_%j.out
#
# PHASE 1 PILOT, stage `fit`: per-cell mixed-effects models on ~20 hand-picked
# region x frequency cells.
#
#   sbatch sbatch/mixed_model_pilot.sbatch
#
# Echoes `RUN_DIR=<path>` on its own line, which is what the permutation array and
# the collect job take as --run-dir. The path is on stdout and every log line is
# on stderr, so $(...) captures the path and nothing else:
#
#   RUN_DIR=$(grep -m1 '^RUN_DIR=' logs/mm_pilot_<jobid>.out | cut -d= -f2-)
#   sbatch --export=ALL,RUN_DIR="$RUN_DIR" sbatch/mixed_model_pilot_perm_array.sbatch
#   sbatch --export=ALL,RUN_DIR="$RUN_DIR" sbatch/mixed_model_pilot_collect.sbatch
#
# WHAT THIS STAGE COSTS. Measured on synthetic cells of the same shape: a full
# fit runs 0.45 s at 20 subjects x 3 contacts and 15 s at 48 subjects x 25
# contacts, and each cell also pays a reduced fit and a channel-slope fit. ~20
# cells is single-digit minutes. 2 h is slack for the view read, not the fits.
#
# MEMORY. The pilot needs ~20 of 924 cells, so the loader filters each subject's
# per-channel table to the wanted frequency bins BEFORE the ROI join and never
# holds the ~15M-row cohort table. Peak is one subject's slice; 32 GB is well
# clear of that and leaves room for the model objects being saved.
#
# 2 CPUs purely so numpy/BLAS has a second thread for the fits. `normal` allows
# 8 GB/core, so 32 GB already charges 4 cores -- asking for 2 costs nothing extra.

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

QUESTION=${QUESTION:-psd_physiology}
RUN_NAME=${RUN_NAME:-pilot_test_mixedlm}
EXTRA=${EXTRA:-}

echo "=== mixed-model pilot | stage=fit | question=$QUESTION ($(date)) ==="

RUN_DIR=$(python -m ieeg_ehr.analysis.run_mixed_model_pilot \
    --stage fit \
    --question "$QUESTION" \
    --run-name "$RUN_NAME" \
    ${EXTRA})

echo "RUN_DIR=${RUN_DIR}"
N_CELLS=$(ls "${RUN_DIR}/frames"/cell_*.parquet 2>/dev/null | wc -l)
echo "fittable cells with a saved frame: ${N_CELLS}"
echo "next: sbatch --export=ALL,RUN_DIR=\"${RUN_DIR}\" sbatch/mixed_model_pilot_perm_array.sbatch"
echo "=== done ($(date)) ==="

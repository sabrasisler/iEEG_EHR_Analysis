#!/bin/bash
#SBATCH --job-name=med_celldiag
#SBATCH --partition=ckeller1
#SBATCH --qos=high_p
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=24GB
#SBATCH --output=logs/med_celldiag_%j.out
#
# Spaghetti / caterpillar / partial-residual diagnostics for a few named cells of
# a medication-stratified run.
#
#   sbatch --export=ALL,RUN_DIR="$RUN_DIR" sbatch/med_cell_diagnostics.sbatch
#   sbatch --export=ALL,RUN_DIR="$RUN_DIR",CELLS="M1:21;S1:21" \
#          sbatch/med_cell_diagnostics.sbatch
#
# Not a dev-node job despite being small: it REFITS each named cell, because the
# grid deliberately saves no model objects, and both the BLUPs and the partial
# residuals need a fitted model rather than a coefficient row.

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

: "${RUN_DIR:?set RUN_DIR to a medstrata run directory}"
CELLS=${CELLS:-}

echo "=== med cell diagnostics | $(date) ==="
echo "    run: ${RUN_DIR}"

if [ -n "${CELLS}" ]; then
  python -m ieeg_ehr.analysis.plot_med_cell_diagnostics \
      --run-dir "${RUN_DIR}" --cells "${CELLS}"
else
  python -m ieeg_ehr.analysis.plot_med_cell_diagnostics --run-dir "${RUN_DIR}"
fi

echo "=== done ($(date)) ==="

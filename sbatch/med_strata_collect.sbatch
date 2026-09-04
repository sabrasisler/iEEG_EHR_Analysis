#!/bin/bash
#SBATCH --job-name=mm_medstrata_collect
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --output=logs/mm_medstrata_collect_%j.out
#
# Medication-stratified grid, stage `collect`: concatenate the per-region output
# for all three fits, apply BH WITHIN each stratum, derive sign consistency from
# the unpooled slopes the fit stage already wrote, and write METHODS.md.
#
#   sbatch --dependency=afterok:<array_jobid> --export=ALL,RUN_DIR="$RUN_DIR" \
#          sbatch/med_strata_collect.sbatch
#
# Each stratum lands in its own subdirectory as `grid_cells.parquet`, so
# plot_mixed_model_grid.py runs on medpos/ and medneg/ with no changes.

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

: "${RUN_DIR:?set RUN_DIR to the directory printed by --stage prepare}"

echo "=== med strata collect | $(date) ==="
echo "    run: ${RUN_DIR}"

python -m ieeg_ehr.analysis.run_mixed_model_med_strata \
    --stage collect \
    --run-dir "${RUN_DIR}"

echo "=== done ($(date)) ==="

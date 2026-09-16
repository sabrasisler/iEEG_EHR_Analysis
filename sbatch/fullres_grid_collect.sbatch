#!/bin/bash
#SBATCH --job-name=fr_collect
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --output=logs/fr_collect_%j.out
#
# Native-resolution grid, stage `collect` + both overview figures.
#
#   sbatch --export=ALL,RUN_DIR="$RUN_DIR" sbatch/fullres_grid_collect.sbatch
#
# Concatenates the 21 region tables, derives sign consistency from the unpooled
# per-subject slopes the fit stage already wrote, writes grid_cells.parquet and
# METHODS.md, then renders the map and the per-region spectra. NO multiple-
# comparison correction happens anywhere in here -- see the run's METHODS.md for
# why a BH family over 9,723 cells at 0.5 Hz spacing would be answering a
# question nobody asked.
#
# The figures run in the same job as collect because they read only the tables
# it just wrote, and a separate job would queue for longer than they take.

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

: "${RUN_DIR:?set RUN_DIR to the directory printed by --stage prepare}"
PLOT_EXTRA=${PLOT_EXTRA:-}

echo "=== fullres grid collect | $(date) ==="
echo "    run: ${RUN_DIR}"

python -m ieeg_ehr.analysis.run_fullres_grid \
    --stage collect \
    --run-dir "${RUN_DIR}"

python -m ieeg_ehr.analysis.plot_fullres_grid \
    --run-dir "${RUN_DIR}" \
    ${PLOT_EXTRA}

echo "=== done ($(date)) ==="

#!/bin/bash
#SBATCH --job-name=mm_grid_collect
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --output=logs/mm_grid_collect_%j.out
#
# PHASE 2, stage `collect`: concatenate the per-region outputs, apply BH in both
# families, write grid_cells.parquet / grid_blups.parquet / METHODS.md.
#
#   sbatch --export=ALL,RUN_DIR="$RUN_DIR" sbatch/mixed_model_grid_collect.sbatch
#
# Submit with a dependency so it cannot run against a half-finished grid:
#
#   sbatch --dependency=afterok:<array_jobid> --export=ALL,RUN_DIR="$RUN_DIR" \
#          sbatch/mixed_model_grid_collect.sbatch
#
# It refuses nothing if a region file is missing -- it logs the missing regions
# loudly and records them in METHODS.md, because a silently short grid looks
# exactly like a real result with fewer significant cells.

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

: "${RUN_DIR:?set RUN_DIR to the directory printed by --stage prepare}"

echo "=== grid collect | $(date) ==="
echo "    run: ${RUN_DIR}"

python -m ieeg_ehr.analysis.run_mixed_model_grid \
    --stage collect \
    --run-dir "${RUN_DIR}"

echo "=== done ($(date)) ==="

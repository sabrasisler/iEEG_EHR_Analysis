#!/bin/bash
#SBATCH --job-name=mm_collect
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --output=logs/mm_collect_%j.out
#
# PHASE 1 PILOT, stage `collect`: merge the permutation shards, compute the
# calibration comparison, write the calibration figure and the final provenance.
#
#   sbatch --export=ALL,RUN_DIR="$RUN_DIR" sbatch/mixed_model_pilot_collect.sbatch
#
# Cheap -- it fits nothing. Run it only after the whole perm array has finished;
# it refuses if there are no shards, but it will happily collect a PARTIAL array
# and report the per-cell shard count, so check `squeue --me` first.
#
# This is also the stage that appends the run to docs/analyses_run.md, labelled as
# a PILOT/TEST pass so the index cannot be misread as a result.

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

: "${RUN_DIR:?set RUN_DIR to the pilot run directory}"

echo "=== collect | ${RUN_DIR} ($(date)) ==="
echo "    shards present: $(ls "${RUN_DIR}/perm"/perm_*.parquet 2>/dev/null | wc -l)"

python -m ieeg_ehr.analysis.run_mixed_model_pilot \
    --stage collect \
    --run-dir "${RUN_DIR}"

echo "=== done ($(date)) ==="

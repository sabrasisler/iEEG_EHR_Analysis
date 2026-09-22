#!/bin/bash
#SBATCH --job-name=bp_consist
#SBATCH --partition=ckeller1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --output=logs/bp_consist_%j.out
#
# Across-subject consistency figures for a band-power mixed-model run.
#
#   sbatch --export=ALL,RUN_DIR="$RUN_DIR" sbatch/bandpower_consistency.sbatch
#
# TIME/MEMORY. Nothing is refitted -- this reads `band_cells.parquet` and
# `subject_slopes.parquet` (both under 200 kB) and spends its time in the two
# permutation nulls: 114 cells x 10k sign flips x ~44 subjects, and 51 subjects
# x 10k cell-label shuffles. Both are vectorised; the whole run is a couple of
# minutes and never exceeds a few hundred MB. 8 GB and 30 min are slack.

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

: "${RUN_DIR:?set RUN_DIR to the mixed-model run directory}"

python -m ieeg_ehr.analysis.plot_bandpower_consistency \
    --run-dir "$RUN_DIR" \
    ${EXTRA:-}

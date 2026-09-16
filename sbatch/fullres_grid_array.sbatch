#!/bin/bash
#SBATCH --job-name=fr_grid
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24GB
#SBATCH --output=logs/fr_grid_%A_%a.out
#SBATCH --array=0-20
#
# Mass-univariate mixed models on the NATIVE 0.5 Hz PSD axis, stage `fit`.
# ONE ARRAY TASK PER REGION: 21 regions x 463 bins = 9,723 cells.
#
#   RUN_DIR=$(python -m ieeg_ehr.analysis.run_fullres_grid --stage prepare)
#   sbatch --export=ALL,RUN_DIR="$RUN_DIR" sbatch/fullres_grid_array.sbatch
#   sbatch --export=ALL,RUN_DIR="$RUN_DIR" sbatch/fullres_grid_collect.sbatch
#
# WHY ONE TASK PER REGION, and 8 CPUS INSIDE IT. A region's 463 bins share one
# Lustre read exactly -- the epoch-mean view is ~1.5 GB cohort-wide and the wide
# layout lets a task decode only the columns it fits -- so per-region tasks pay
# that read 21 times instead of 9,723. The fits inside a task are independent
# single-threaded problems, so they go across processes (joblib, n_jobs =
# SLURM_CPUS_PER_TASK, BLAS pinned to 1 thread each).
#
# TIME. The 50-bin grid measured a median 1.1 s per fit and 32 s at its worst
# cell, and its slowest REGION averaged 15.6 s (Lateral Temporal). 463 bins at
# that rate over 8 processes is ~16 min; the median region is under 2 min. 4 h is slack for Lustre, not for the fits. NO REDUCED
# MODEL is fitted here (no heterogeneity LRT -- its p-value would not be shown),
# which halves the per-cell cost against the 50-bin grid.
#
# MEMORY. One region's rows across all 463 bins: 38k x 463 float32 = ~70 MB for
# the largest region, plus one ~2 MB model frame per in-flight worker. 24 GB is
# already generous; the 50-bin grid asked for 48 GB and peaked under 1 GB.

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

# statsmodels' fits are small dense problems; letting BLAS grab every core on a
# shared node costs more in contention than it wins in the solve. joblib pins
# each worker too, but the serial path needs this as well.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

: "${RUN_DIR:?set RUN_DIR to the directory printed by --stage prepare}"
EXTRA=${EXTRA:-}

echo "=== fullres grid | region ${SLURM_ARRAY_TASK_ID} | $(date) ==="
echo "    run: ${RUN_DIR}"

python -m ieeg_ehr.analysis.run_fullres_grid \
    --stage fit \
    --run-dir "${RUN_DIR}" \
    --region-index "${SLURM_ARRAY_TASK_ID}" \
    ${EXTRA}

echo "=== done ($(date)) ==="

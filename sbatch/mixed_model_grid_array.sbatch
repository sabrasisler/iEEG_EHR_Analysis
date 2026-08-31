#!/bin/bash
#SBATCH --job-name=mm_grid
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48GB
#SBATCH --output=logs/mm_grid_%A_%a.out
#SBATCH --array=0-20
#
# PHASE 2, stage `fit`: the FULL region x frequency grid, ONE ARRAY TASK PER
# REGION. 21 regions x 44 bins after line-noise removal.
#
#   RUN_DIR=$(python -m ieeg_ehr.analysis.run_mixed_model_grid --stage prepare)
#   sbatch --export=ALL,RUN_DIR="$RUN_DIR" sbatch/mixed_model_grid_array.sbatch
#   sbatch --export=ALL,RUN_DIR="$RUN_DIR" sbatch/mixed_model_grid_collect.sbatch
#
# WHY ONE TASK PER REGION rather than per cell. The expensive part of a cell is
# not the fit (median 1.1 s in the pilot) but reading 51 subject tables off Lustre
# and joining channels to ROIs. A region's 44 bins share that read exactly, so
# per-region tasks pay it 21 times instead of 924.
#
# TIME. The pilot measured full+reduced at ~26 min single-core for all 924 cells,
# but that is an average and the array is sized by its WORST region, not its mean.
# Insula was the slowest pilot cell at 7.0 s a fit with 46 subjects and 502
# contacts; 44 bins x (full + reduced) at that rate is ~10 min. 4 h is deliberate
# slack for the Lustre read, not for the fits.
#
# MEMORY. A task holds one region across all 44 bins: for Insula that is ~1M rows
# of per-channel epoch power. 48 GB is generous; the pilot's 32 GB peaked at
# 821 MB holding 19 cells plus 19 saved model objects, and this stage saves none.
#
# NO CHANNEL RANDOM SLOPE. Phase 1 found it improved fit in 1 of 19 cells, so the
# intercept-only prior stands and the third fit per cell is skipped. Pass
# EXTRA="--with-channel-slope" to re-open that question.

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

# statsmodels' fits are small dense problems; letting BLAS grab every core on a
# shared node costs more in contention than it wins in the solve.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

: "${RUN_DIR:?set RUN_DIR to the directory printed by --stage prepare}"
EXTRA=${EXTRA:-}

echo "=== grid | region ${SLURM_ARRAY_TASK_ID} | $(date) ==="
echo "    run: ${RUN_DIR}"

python -m ieeg_ehr.analysis.run_mixed_model_grid \
    --stage fit \
    --run-dir "${RUN_DIR}" \
    --region-index "${SLURM_ARRAY_TASK_ID}" \
    ${EXTRA}

echo "=== done ($(date)) ==="

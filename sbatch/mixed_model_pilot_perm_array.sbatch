#!/bin/bash
#SBATCH --job-name=mm_perm
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --output=logs/mm_perm_%A_%a.out
#SBATCH --array=0-18
#
# PHASE 1 PILOT, stage `perm`: the within-subject NRS shuffle null, ONE ARRAY TASK
# PER PILOT CELL.
#
#   RUN_DIR=$(grep -m1 '^RUN_DIR=' logs/mm_pilot_<jobid>.out | cut -d= -f2-)
#   sbatch --export=ALL,RUN_DIR="$RUN_DIR" sbatch/mixed_model_pilot_perm_array.sbatch
#
# THIS IS THE POINT OF THE PILOT. statsmodels reports a Wald z, not a
# Satterthwaite t, and nothing yet says that z is calibrated for a design with
# three variance components, unbalanced contact counts and 10-90 epochs per
# subject. Comparing it against a shuffle null decides whether the FULL GRID can
# use parametric p-values at all -- and 924 cells x 1000 shuffles is not
# affordable, so this has to be settled on 19 cells rather than discovered later.
#
# WHY IT IS EXPENSIVE. Every shuffle is a full refit; there is no matmul shortcut
# of the kind pain_coef.regression_weights gives the two-stage design. Measured
# per-fit cost on synthetic cells of the same shape runs 0.45 s (20 subjects x 3
# contacts) to 15 s (48 x 25), so 1000 shuffles is 8 minutes to 4 hours SERIAL.
#
# Two things make that fit in the wall clock:
#   - 8 workers via joblib, with BLAS pinned to one thread each (threadpool_limits
#     in mixed_model.permutation_null). Without the pin, 8 processes each spawn a
#     full thread pool and the node thrashes -- the fits get SLOWER than serial.
#   - a warm start from the OBSERVED fit's parameters. The variance components
#     barely move under a permutation of the predictor, so most of the optimizer's
#     work is already done. It cannot bias the null: the starting point is the
#     same for every shuffle and does not depend on that shuffle's data.
#
# 8 h covers the worst cell (~4 h serial / 8 workers, plus slack for a cell that
# converges badly). Most tasks finish in minutes. Cells that were not fittable
# write an empty shard and exit immediately.
#
# ARRAY BOUND must match the number of pilot cells. 19 = 15 hand-picked + 4
# data-driven nulls. CHECK IT, do not trust this comment -- a short array drops
# cells silently:
#   python - <<'EOF'
#   from ieeg_ehr import io; import sys
#   print(len(io.read_table(sys.argv[1] + '/pilot_cell_manifest.parquet',
#                           on_stale='ignore')))
#   EOF

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Belt and braces alongside threadpool_limits: some BLAS builds read only the
# environment, and an over-subscribed node is the difference between 20 minutes
# and 3 hours.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

: "${RUN_DIR:?set RUN_DIR to the pilot run directory printed by mixed_model_pilot.sbatch}"
N_PERM=${N_PERM:-1000}
SEED=${SEED:-0}

N_CELLS=$(python -c "
from ieeg_ehr import io
print(len(io.read_table('${RUN_DIR}/pilot_cell_manifest.parquet', on_stale='ignore')))
")
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_CELLS}" ]; then
  echo "ERROR: array task ${SLURM_ARRAY_TASK_ID} >= ${N_CELLS} pilot cells."
  echo "Update --array to 0-$(( N_CELLS - 1 ))."
  exit 1
fi

echo "=== perm | cell ${SLURM_ARRAY_TASK_ID}/${N_CELLS} | n_perm=${N_PERM} | "\
"${SLURM_CPUS_PER_TASK} jobs | $(date) ==="
echo "    run: ${RUN_DIR}"

python -m ieeg_ehr.analysis.run_mixed_model_pilot \
    --stage perm \
    --run-dir "${RUN_DIR}" \
    --cell-index "${SLURM_ARRAY_TASK_ID}" \
    --n-perm "${N_PERM}" \
    --seed "${SEED}" \
    --n-jobs "${SLURM_CPUS_PER_TASK}"

echo "=== done ($(date)) ==="

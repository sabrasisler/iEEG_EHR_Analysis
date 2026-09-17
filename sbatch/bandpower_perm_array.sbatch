#!/bin/bash
#SBATCH --job-name=bp_perm
#SBATCH --partition=normal,owners
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --output=logs/bp_perm_%A_%a.out
#SBATCH --array=0-119
#
# The permutation null for the band-power models: ONE ARRAY TASK PER CELL.
#
#   sbatch --export=ALL,RUN_DIR="$RUN_DIR" sbatch/bandpower_perm_array.sbatch
#   python -m ieeg_ehr.analysis.run_bandpower_mixed --stage collect --run-dir "$RUN_DIR"
#
# WHY THIS IS AFFORDABLE HERE AND NOT ON THE 9,723-CELL GRID. 120 cells x 1,000
# shuffles x ~1.6 s is ~53 CPU-hours: 8 cores for ~30 min a task once joblib
# spreads the shuffles. The same null on the native-resolution grid would be
# ~4,300 CPU-hours, which is why that grid reports no p-value at all.
#
# WHAT IT BUYS. The Wald p assumes z ~ N(0,1), which needs the standard error to
# be right and therefore the random-effect structure approximately right with
# enough subjects for the asymptotics. This assumes only that the epoch -> pain
# score pairing is exchangeable within a subject. It does NOT fix the confound
# both share: nothing here excludes opioid administrations or post-ictal periods,
# and epochs are treated as exchangeable in time.
#
# Each task reloads only its own saved frame, so no task re-reads the view. A
# cell that was not fittable has no frame and writes an empty shard rather than
# failing, so a missing shard always means a task died.

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

: "${RUN_DIR:?set RUN_DIR to the directory printed by --stage fit}"
N_PERM=${N_PERM:-1000}

echo "=== bandpower perm | cell ${SLURM_ARRAY_TASK_ID} | $(date) ==="
python -m ieeg_ehr.analysis.run_bandpower_mixed \
    --stage perm --run-dir "${RUN_DIR}" \
    --cell-index "${SLURM_ARRAY_TASK_ID}" --n-perm "${N_PERM}"
echo "=== done ($(date)) ==="

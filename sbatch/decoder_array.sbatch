#!/bin/bash
#SBATCH --job-name=decoder
#SBATCH --partition=ckeller1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --output=logs/decoder_%A_%a.out
#SBATCH --array=0-44%12
#
# PER-SUBJECT PAIN DECODER -- one array task per subject-session.
# (PLANNING "Pain state decoding", DECISIONS 2026-09-08)
#
#   RUN_TIMESTAMP=$(date +%Y%m%d-%H%M%S) sbatch sbatch/decoder_array.sbatch
#   python -m ieeg_ehr.decoding.aggregate --run-timestamp $RUN_TIMESTAMP
#
# RUN_TIMESTAMP IS REQUIRED AND IS SET BY THE SUBMITTER, not per task. sbatch
# exports the submitting environment by default, so computing it once on the
# command line gives every task the same value. If each task called date(1)
# instead, 45 tasks starting at slightly different seconds would build 45
# different run directories and the run would shatter into siblings that
# aggregate.py could not gather. The script refuses to start without it rather
# than defaulting, because the failure is invisible until aggregation.
#
# ELIGIBILITY IS APPLIED HERE, NOT IN THE VIEW. The view was built for all 60
# runnable discovery subjects (eligibility is an analysis choice); this array
# runs the 45 subject-sessions with >=30 epochs, which is what the nested CV
# needs to keep >=5 observations per fold. A unit that fails the cascade exits 0
# with a message -- "not decodable" is a RESULT, not a crash -- so one thin
# subject does not fail the array.
#
# RESOURCES -- MEASURED, not guessed. Benchmarked on sub-183's shape (95 x 972,
# the LARGEST unit, k=10 outer folds):
#
#     regression       4.8 s/bootstrap  ->  0.54 h/unit   <- the default arm
#     ordinal         21.4 s/bootstrap  ->  2.38 h/unit
#     classification  19.1 s/bootstrap  ->  2.12 h/unit
#
# A MEDIAN unit (n~48 -> k=5, and half the rows) is roughly 4x cheaper, so most
# tasks finish far below those figures. 2h covers the worst REGRESSION unit with
# a wide margin; raise it to 8h if ARMS is widened.
#
# WHY REGRESSION IS ~4x CHEAPER, and it is the solver rather than the model:
# ElasticNetCV uses COORDINATE DESCENT WITH WARM STARTS -- it solves at alpha_max
# (all coefficients zero) and reuses each solution as the next alpha's starting
# point, so a 50-point path costs barely more than a few full fits. The logistic
# arms must use `saga` (sklearn has no coordinate-descent path for elastic-net
# logistic regression), which converges INDEPENDENTLY for every (C, l1_ratio)
# pair with no warm starting, and one saga run costs more than a whole
# coordinate-descent path.
#
# Two earlier designs were measured and rejected rather than estimated: a
# GridSearchCV over an explicit alpha path needed 3.6M fits/unit (5-20 h/unit),
# and per-threshold penalty tuning in the ordinal arm cost 189 s/bootstrap. See
# arms.py. Re-size from `seff` after this run rather than trusting these numbers
# forever.
#
# 8GB/1CPU is the ceiling on ckeller1 (8 GB/core) and the matrices are nowhere
# near it -- the memory is for sklearn's copies, not the data.
#
# PARTITION. ckeller1 by request, departing from CLAUDE.md's "long arrays on
# normal". Note the node is a single 24-core box shared with the lab: %12 leaves
# half of it, and tasks will trickle rather than start together when it is busy.
#
# Array bound must match the eligible-unit count. Recompute with:
#   python -m ieeg_ehr.decoding.eligible --view-dir <dir> --count

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [ -z "${RUN_TIMESTAMP:-}" ]; then
  echo "ERROR: RUN_TIMESTAMP is not set. Submit with:"
  echo "  RUN_TIMESTAMP=\$(date +%Y%m%d-%H%M%S) sbatch sbatch/decoder_array.sbatch"
  exit 1
fi

VIEW_ROOT=${VIEW_ROOT:-$(python -c "
from ieeg_ehr import config
print(config.pain_epoch_unit_dir() / 'views')")}
VIEW_DIR=${VIEW_DIR:-$(ls -d "${VIEW_ROOT}"/chan-paper6-raw_* 2>/dev/null | head -1)}

if [ -z "${VIEW_DIR}" ] || [ ! -d "${VIEW_DIR}" ]; then
  echo "ERROR: no chan-paper6-raw view found under ${VIEW_ROOT}."
  echo "Build it first: sbatch sbatch/build_paper6_view_array.sbatch"
  exit 1
fi

# ARMS defaults to regression alone -- the primary arm, and ~4x cheaper than
# either logistic arm (see above). The other two are implemented and tested;
# widen with e.g. ARMS="regression ordinal classification" and raise --time.
ARMS=${ARMS:-regression}
N_BOOTSTRAPS=${N_BOOTSTRAPS:-100}
MIN_EPOCHS=${MIN_EPOCHS:-30}

mapfile -t UNITS < <(python -m ieeg_ehr.decoding.eligible \
    --view-dir "${VIEW_DIR}" --min-epochs "${MIN_EPOCHS}")

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#UNITS[@]}" ]; then
  echo "ERROR: array task ${SLURM_ARRAY_TASK_ID} >= ${#UNITS[@]} eligible units."
  echo "Update --array to 0-$(( ${#UNITS[@]} - 1 ))."
  exit 1
fi

UNIT="${UNITS[$SLURM_ARRAY_TASK_ID]}"
SUBJECT="${UNIT%%_*}"
SESSION="${UNIT##*_}"

echo "=== task ${SLURM_ARRAY_TASK_ID}: sub-${SUBJECT} ses-${SESSION} ==="
echo "    view=${VIEW_DIR}"
echo "    run_timestamp=${RUN_TIMESTAMP}  arms=${ARMS}  bootstraps=${N_BOOTSTRAPS}  ($(date))"

python -m ieeg_ehr.decoding.run_decoder \
    --subject "${SUBJECT}" \
    --session "${SESSION}" \
    --view-dir "${VIEW_DIR}" \
    --run-timestamp "${RUN_TIMESTAMP}" \
    --arms ${ARMS} \
    --n-bootstraps "${N_BOOTSTRAPS}" \
    --min-epochs "${MIN_EPOCHS}"

echo "=== done ($(date)) ==="

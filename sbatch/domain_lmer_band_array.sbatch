#!/bin/bash
#SBATCH -J lmer_band
#SBATCH -p ckeller1
#SBATCH -t 04:00:00
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH --array=0-5
#SBATCH -o logs/lmer_band_%A_%a.out
#SBATCH -e logs/lmer_band_%A_%a.err
#
# ONE ARRAY TASK PER BAND for any lme4 domain run. Generic over the moderator:
# the FRAME decides the model (dx_state -> diagnosis strata, med_state ->
# medication, neither -> all subjects) and run_domain_lmer asserts that choice
# to R over --expect-dx / --expect-med, so a dropped column fails loudly rather
# than silently selecting a different model.
#
# Supersedes domain_lmer_mdd_array.sbatch, which hard-coded the MDD view scheme.
#
# Required from the submit wrapper:
#   RUN_DIR      the SHARED run every task writes into. Without it each task
#                timestamps its own directory and there is nothing to collect.
#   FRAMES_GLOB  a glob, not a path -- analysis_run_dir appends its own stamp.
#   VIEW_SCHEME  level-4 folder name.
# Optional:
#   DROP_DOMAIN  comma-separated domains removed BEFORE the fit, so excluding a
#                domain is a genuinely different model, not a masked figure.
#   BANDS        space-separated override of the six-band default.

set -euo pipefail
export PATH="$HOME/bin:$PATH"

module load math R/4.4.2
module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export R_LIBS_USER="$GROUP_HOME/R/4.4.2"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

: "${RUN_DIR:?RUN_DIR must be exported by the submit wrapper}"
: "${FRAMES_GLOB:?FRAMES_GLOB must be exported by the submit wrapper}"
: "${VIEW_SCHEME:?VIEW_SCHEME must be exported by the submit wrapper}"

# `|| true` IS LOad-BEARING. Under `set -euo pipefail` a failing `ls` inside the
# substitution aborts the script AT THIS LINE, so the guard below never runs and
# the job dies with nothing in the log but module noise -- which is exactly how
# this failed the first time. Let the pipeline fail softly and report it here.
FRAMES_DIR=$(ls -1d ${FRAMES_GLOB} 2>/dev/null | sort | tail -1 || true)
if [ -z "${FRAMES_DIR}" ]; then
    echo "FATAL: no frames matched ${FRAMES_GLOB}" >&2
    echo "candidates under that parent:" >&2
    ls -1d "$(dirname "$(dirname "${FRAMES_GLOB}")")"/*/ 2>/dev/null >&2 || true
    exit 1
fi

# THE BAND LIST IS FIXED AND NEVER PASSED IN.
#
# It used to arrive as BANDS="delta,theta,..." over --export, which does not
# work: `--export` is ITSELF comma-separated, so Slurm split the value and set
# BANDS=delta, leaving tasks 1-5 to die on `BAND_ARR[i]: unbound variable`.
# Passing a subset is also unnecessary -- the idempotency check below means a
# full 0-5 array costs nothing for bands that already landed, so the reaper can
# just resubmit the whole array and let finished tasks exit immediately.
BAND_ARR=(delta theta alpha beta gamma high_gamma)

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#BAND_ARR[@]}" ]; then
    echo "task ${SLURM_ARRAY_TASK_ID} exceeds the ${#BAND_ARR[@]} bands" >&2
    exit 1
fi
BAND=${BAND_ARR[$SLURM_ARRAY_TASK_ID]}

# IDEMPOTENT. A requeued or re-reaped task must not redo work that already
# landed: on `owners` a task can be preempted after writing its output, and the
# reaper resubmits by "which band files are missing", so a surviving result is
# the signal to stop.
if [ -f "${RUN_DIR}/bands/${BAND}_fitinfo.csv" ]; then
    echo "[${BAND}] already fitted in ${RUN_DIR}; nothing to do"
    exit 0
fi

DROP_ARG=()
if [ -n "${DROP_DOMAIN:-}" ]; then
    DROP_ARG=(--drop-domain ${DROP_DOMAIN//,/ })
fi

echo "=== lmer | band=${BAND} | task=${SLURM_ARRAY_TASK_ID} | $(date) ==="
echo "run dir: ${RUN_DIR}"
echo "frames:  ${FRAMES_DIR}"
echo "drop:    ${DROP_DOMAIN:-<none>}"

python -m ieeg_ehr.analysis.run_domain_lmer \
    --frames-dir "${FRAMES_DIR}" \
    --run-dir "${RUN_DIR}" \
    --bands "${BAND}" \
    --view-scheme "${VIEW_SCHEME}" \
    "${DROP_ARG[@]}"

echo "=== done ${BAND} ($(date)) ==="

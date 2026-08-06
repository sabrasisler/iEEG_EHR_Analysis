#!/bin/bash
#SBATCH --job-name=slope_discovery
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --output=logs/slope_discovery_%A_%a.out
#SBATCH --array=0-3
#
# BG.2: build the per-epoch 1/f slope tables for every DISCOVERY subject.
#
#   sbatch sbatch/slope_discovery_array.sbatch
#   FIT_LO_HZ=30 sbatch sbatch/slope_discovery_array.sbatch     # a narrower fit
#
# CHUNKED: each task takes a SLICE of $CHUNK_SIZE subjects, not one subject.
# Unlike its sibling view_discovery_array.sbatch, which is one-subject-per-task.
#
# The reason is measured, not stylistic. A slope task is ~20 SECONDS (one pass,
# no baseline), and `normal` was carrying 13,076 pending jobs on 2026-08-05: at
# one subject per task the job spent all of its time waiting for 60 separate
# scheduling decisions to buy 20 seconds of compute each, and got 16 tasks
# through in 17 minutes before being resubmitted like this. 15 subjects per task
# turns that into 4 scheduling decisions for ~5 minutes of work apiece.
#
# The one-subject-per-task rule still holds where it was written for -- the POWER
# view, whose tasks are minutes long and where a single bad subject taking down a
# batch is a real cost. Here a failing subject loses at most its chunk-mate's
# work, which is a few minutes, and `set -e` is deliberately NOT applied to the
# per-chunk python call for that reason (see below).
#
# ARRAY BOUND = ceil(runnable / CHUNK_SIZE). Recompute BOTH when the count moves:
#   python -m ieeg_ehr.config.cohorts --runnable discovery 2>/dev/null | wc -l
# 60 subjects / 15 = 4 tasks -> --array=0-3. A bound that is too SHORT drops
# subjects silently; the guard below catches only the opposite case.
#
# The sibling of view_discovery_array.sbatch and deliberately its twin: same
# subject resolution, same mask, same per-task-per-subject shape, same resources.
# Read that file's header for the reasoning behind each; only the differences are
# recorded here.
#
# NORMALIZATION IS FIXED AT `none` and is not a variable. The builder refuses
# anything else: a slope is fit on RAW log-power, because z-scoring rescales every
# frequency bin by its own baseline SD and therefore tilts the line by whatever
# frequency dependence that SD happens to have. One array, not two.
#
# CHEAPER THAN THE POWER VIEW, roughly half: with no baseline to accumulate the
# builder makes ONE streaming pass over the cache instead of two. Measured at
# ~15 s/GB and ~20 s for sub-019; 30 minutes covers a 15-subject chunk with room
# to spare.
#
# The four subjects with no Desikan_Killiany_anode labels (093 154 159 240) emit
# an EMPTY table and a loud error. Expected, not a failure.

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

# Run the code this job was submitted from -- ieeg_ehr is installed editable
# against the main checkout, whose state is not guaranteed.
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

MASK_LABEL=${MASK_LABEL:-std10_rv-gross-std3_satmargin15_sw_logz4}
SPLIT=${SPLIT:-discovery}
ROI_SCHEME=${ROI_SCHEME:-roi_v2}
# Defaults live in config/psd_params.py (SLOPE_FIT_*); overriding here is how a
# narrower fit range gets run, and it lands in its own directory because the
# range is part of the hashed config.
FIT_LO_HZ=${FIT_LO_HZ:-1}
FIT_HI_HZ=${FIT_HI_HZ:-250}

CHUNK_SIZE=${CHUNK_SIZE:-15}

mapfile -t SUBJECTS < <(python -m ieeg_ehr.config.cohorts --runnable "$SPLIT" 2>/dev/null)

START=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
N_TASKS_NEEDED=$(( (${#SUBJECTS[@]} + CHUNK_SIZE - 1) / CHUNK_SIZE ))
if [ "$START" -ge "${#SUBJECTS[@]}" ]; then
  echo "ERROR: task ${SLURM_ARRAY_TASK_ID} starts at subject index ${START} but only"
  echo "       ${#SUBJECTS[@]} subject(s) are runnable at CHUNK_SIZE=${CHUNK_SIZE}."
  echo "       Update --array to 0-$(( N_TASKS_NEEDED - 1 ))."
  exit 1
fi
CHUNK=( "${SUBJECTS[@]:$START:$CHUNK_SIZE}" )

echo "=== task ${SLURM_ARRAY_TASK_ID}: ${#CHUNK[@]} subject(s) from index ${START}"
echo "=== ${CHUNK[*]}"
echo "=== split=${SPLIT} | mask=${MASK_LABEL} | roi=${ROI_SCHEME}"
echo "=== fit ${FIT_LO_HZ}-${FIT_HI_HZ} Hz ($(date)) ==="

# ONE PYTHON CALL PER SUBJECT, not one for the whole chunk. The builder happily
# takes a list, but then a single subject that raises aborts the rest of the
# chunk -- which is the failure mode chunking is supposed to make cheap, not
# reintroduce. `|| true` plus a tally keeps a bad subject to its own line, and
# the exit code below still fails the task if any subject failed, so a partial
# chunk cannot be mistaken for a clean one.
failed=()
for SUBJECT in "${CHUNK[@]}"; do
  echo "--- sub-${SUBJECT}"
  # No --out-dir: the builder resolves
  #   features/pain/psd_epochs/epoch-5min-pre/views/slope-<scheme_code>_<hash>/
  # itself. Naming it here would mean spelling the config_hash in bash, where it
  # could drift from the one the sidecar records.
  python -m ieeg_ehr.views.build_pain_epoch_slope \
      --subjects "$SUBJECT" \
      --split "$SPLIT" \
      --mask-label "$MASK_LABEL" \
      --roi-scheme "$ROI_SCHEME" \
      --normalization none \
      --fit-lo-hz "$FIT_LO_HZ" \
      --fit-hi-hz "$FIT_HI_HZ" || failed+=( "$SUBJECT" )
done

if [ "${#failed[@]}" -gt 0 ]; then
  echo "=== FAILED (${#failed[@]}/${#CHUNK[@]}): ${failed[*]} ($(date)) ==="
  exit 1
fi
echo "=== done, ${#CHUNK[@]} subject(s) ($(date)) ==="

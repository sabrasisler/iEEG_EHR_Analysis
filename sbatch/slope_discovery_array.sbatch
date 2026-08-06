#!/bin/bash
#SBATCH --job-name=slope_discovery
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --output=logs/slope_discovery_%A_%a.out
#SBATCH --array=0-59%16
#
# BG.2: build the per-epoch 1/f slope tables for every DISCOVERY subject.
#
#   sbatch sbatch/slope_discovery_array.sbatch
#   FIT_LO_HZ=30 sbatch sbatch/slope_discovery_array.sbatch     # a narrower fit
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
# builder makes ONE streaming pass over the cache instead of two. 30 minutes is
# already ~10x the measured per-subject time for the power view; kept the same
# because short jobs backfill well and there is nothing to gain by trimming it.
#
# CHECK THE ARRAY BOUND, do not trust this comment -- an array that is too SHORT
# drops subjects silently, and the in-script guard only catches a task id ABOVE
# the count, never a count above the array:
#   python -m ieeg_ehr.config.cohorts --runnable discovery 2>/dev/null | wc -l
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

mapfile -t SUBJECTS < <(python -m ieeg_ehr.config.cohorts --runnable "$SPLIT" 2>/dev/null)

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#SUBJECTS[@]}" ]; then
  echo "ERROR: array task ${SLURM_ARRAY_TASK_ID} >= ${#SUBJECTS[@]} runnable subjects."
  echo "Update --array to 0-$(( ${#SUBJECTS[@]} - 1 ))."
  exit 1
fi

SUBJECT="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
echo "=== task ${SLURM_ARRAY_TASK_ID}: sub-${SUBJECT} | split=${SPLIT} | mask=${MASK_LABEL}"
echo "=== fit ${FIT_LO_HZ}-${FIT_HI_HZ} Hz | roi=${ROI_SCHEME} ($(date)) ==="

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
    --fit-hi-hz "$FIT_HI_HZ"

echo "=== done ($(date)) ==="

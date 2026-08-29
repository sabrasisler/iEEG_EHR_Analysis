#!/bin/bash
#SBATCH --job-name=chan_view
#SBATCH --partition=normal
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --output=logs/chan_view_%A_%a.out
#SBATCH --array=0-59%16
#
# PER-CHANNEL view for the mixed-effects analysis: same axes as the
# raw-relpain-roiv2 view the continuous-pain heatmap reads, but with region
# aggregation TURNED OFF.
#
#   sbatch sbatch/build_chan_view_array.sbatch
#
# WHY A SEPARATE VIEW AND NOT THE EXISTING ONE. The mixed model's row is one
# CHANNEL x one epoch, because the whole point of the channel random intercept is
# to absorb per-contact amplitude differences inside the model rather than
# collapsing them beforehand. The existing raw-relpain-roiv2 view has already
# averaged channels within an ROI (linear-then-log), so the per-contact structure
# the model needs is gone by the time it is read.
#
# NO NEW CODE. `--region none` is registry AXIS 6's existing "don't aggregate"
# setting: build_pain_epoch_view emits one row per channel, with the channel NAME
# in the `region` column and n_channels=1. The channel -> ROI v2 join happens at
# analysis time from the cached channel_meta table.
#
# --view-label IS REQUIRED, not cosmetic. Without it the label falls back to
# view.scheme_code, which is 'raw-relpain-roiv2' -- the same label the EXISTING
# region-aggregated view already uses. The two would then sit side by side in
# views/ distinguishable only by their config_hash, and picking the wrong one
# would silently give a different quantity. --roi-scheme is still passed so the
# scheme is recorded in provenance even though --region none does not consult it.
#
# BUILT FOR THE WHOLE DISCOVERY SPLIT, not just the 51 subjects the reference
# analysis kept. Eligibility (min_epochs / min_range / min_non_modal) is an
# ANALYSIS choice; baking it into a feature artifact would force a rebuild the
# first time anyone changes it.
#
# RESOURCES. Same shape as view_discovery_array.sbatch, whose measured peak was
# 4.19 GB for two normalizations in one task. This runs ONE normalization but
# emits ~7x the rows (150 channels vs 21 regions), so the long-format row list
# grows to ~350k tuples -- tens of MB, not GB. 8 GB with 1 CPU keeps us at the
# 8 GB/core `normal` limit so Slurm charges one core for one core of work.
# 45 min rather than 30: the extra rows are all in the Python append loop and
# the parquet write.
#
# Array bound must match the runnable count. Recompute with:
#   python -m ieeg_ehr.config.cohorts --runnable discovery 2>/dev/null | wc -l

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
VIEW_LABEL=${VIEW_LABEL:-chan-raw-relpain-roiv2}

mapfile -t SUBJECTS < <(python -m ieeg_ehr.config.cohorts --runnable "$SPLIT" 2>/dev/null)

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#SUBJECTS[@]}" ]; then
  echo "ERROR: array task ${SLURM_ARRAY_TASK_ID} >= ${#SUBJECTS[@]} runnable subjects."
  echo "Update --array to 0-$(( ${#SUBJECTS[@]} - 1 ))."
  exit 1
fi

SUBJECT="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
echo "=== task ${SLURM_ARRAY_TASK_ID}: sub-${SUBJECT} | split=${SPLIT} | mask=${MASK_LABEL} ($(date)) ==="

python -m ieeg_ehr.views.build_pain_epoch_view \
    --subjects "$SUBJECT" \
    --split "$SPLIT" \
    --mask-label "$MASK_LABEL" \
    --roi-scheme "$ROI_SCHEME" \
    --normalization none \
    --region none \
    --view-label "$VIEW_LABEL"

echo "=== done ($(date)) ==="

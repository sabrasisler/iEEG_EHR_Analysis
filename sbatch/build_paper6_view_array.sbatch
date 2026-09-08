#!/bin/bash
#SBATCH --job-name=paper6_view
#SBATCH --partition=ckeller1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --output=logs/paper6_view_%A_%a.out
#SBATCH --array=0-59%12
#
# PER-CHANNEL x SIX-BAND view -- the feature substrate for the per-subject pain
# decoder (PLANNING "Pain state decoding", DECISIONS 2026-09-08).
#
#   sbatch sbatch/build_paper6_view_array.sbatch
#   SUBJECTS_OVERRIDE=183 sbatch --array=0 sbatch/build_paper6_view_array.sbatch
#
# WHY A SEPARATE VIEW FROM chan-raw-relpain-roiv2. That one is the same axes but
# `--freq log_bins_50`: per-channel x 50 log bins, which is what the mixed model
# reads. The decoder wants the target paper's six bands aggregated BEFORE the
# model, so the frequency axis differs -- and the frequency axis is part of the
# config hash, so this is a sibling directory, not a rebuild of that one.
#
# --drop-line-noise-bins IS REQUIRED, NOT COSMETIC. The paper's gamma (25-70)
# and high_gamma (70-170) STRADDLE the 60 and 120 Hz harmonics -- exactly what
# this project's own CANONICAL_BANDS_HZ is shaped to avoid. And the two band
# aggregators do not behave the same way: preprocessing.bipolar_bands.
# aggregate_to_bands masks flagged bins inline, but views.axes.aggregate_bands
# -- the one the view layer actually uses -- does NOT. It relies on the flag
# having already set them to NaN and on nanmean skipping them. Without the flag
# gamma silently absorbs the 58-62 Hz notch residue (measured at >1 log unit on
# synthetic input) and the output still looks like a perfectly plausible table.
# Pinned by tests/test_views.py::
#   test_line_noise_bins_must_be_nan_before_band_aggregation
#
# --normalization none is the decoder's deliberate choice, NOT a default. A
# baseline computed over the subject's 0-pain epochs would be fitted using the
# labels, which leaks straight into any cross-validated performance estimate.
# The decoder standardizes per feature INSIDE each training fold instead.
#
# --roi-scheme is still passed so the scheme is recorded in provenance, even
# though --region none does not consult it. The channel -> ROI join happens at
# analysis time from the cached channel_meta table, and in v1 the ROI is
# metadata for interpretation only -- never averaged into the features.
#
# BUILT FOR THE WHOLE RUNNABLE DISCOVERY SPLIT, not just the 45 subject-sessions
# the decoder will keep. Same rule as build_chan_view_array.sbatch: eligibility
# (>=30 epochs, nonzero median) is an ANALYSIS choice, and baking it into a
# feature artifact would force a rebuild the first time anyone changes it.
#
# RESOURCES. Same shape as build_chan_view_array.sbatch (measured: per-channel
# long format, ~350k tuples) but emitting SIX columns per channel instead of 50
# bins, so ~8x fewer rows and a smaller parquet. 8 GB / 1 CPU is inherited
# rather than re-measured, and is the ceiling on `ckeller1` anyway (8 GB/core).
#
# PARTITION. ckeller1 by request. It is a single 24-core node, so %12 leaves
# half of it for interactive work rather than saturating the lab's own node.
# Note this departs from CLAUDE.md's "long arrays on normal" -- deliberate.
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
VIEW_LABEL=${VIEW_LABEL:-chan-paper6-raw}

# SUBJECTS_OVERRIDE exists for the one-unit smoke run, so a single subject does
# not need a separate script that could drift from this one.
if [ -n "${SUBJECTS_OVERRIDE:-}" ]; then
  read -r -a SUBJECTS <<< "${SUBJECTS_OVERRIDE}"
else
  mapfile -t SUBJECTS < <(python -m ieeg_ehr.config.cohorts --runnable "$SPLIT" 2>/dev/null)
fi

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
    --freq paper_bands_6 \
    --drop-line-noise-bins \
    --view-label "$VIEW_LABEL"

echo "=== done ($(date)) ==="

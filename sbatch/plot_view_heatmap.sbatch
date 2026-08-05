#!/bin/bash
#SBATCH --job-name=view_heatmap
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=24GB
#SBATCH --output=logs/view_heatmap_%j.out
#
# ONE region x frequency heatmap run: one normalization, one ROI scheme, with the
# cluster-based permutation test and its outlines.
#
# ONE JOB PER FIGURE, deliberately. This used to be a single job looping over both
# normalizations and both figure types, which meant one traceback took down
# everything and one log interleaved four runs. Splitting costs nothing: the only
# expensive shared step was channel_meta, and that is cached on disk per
# subject/session (views/channel_meta.build) independently of normalization and ROI
# scheme, so it is read, not rebuilt.
#
# Reads the already-built view TABLES rather than recomputing, so the figures and
# the numbers behind them provably come from the same values.
#
# The view directory is ASKED FOR, never spelled here: --print-out-dir resolves the
# same <scheme_code>_<config_hash> the builder wrote to, so this job cannot plot a
# directory the builder did not fill.
#
# REFUSES an un-normalized view. The cluster test's one-sample null is "equal to
# zero" and raw log power sits near -10, so the map would come back uniformly
# outlined and mean nothing (docs/cluster_permutation.md). Use
# plot_view_spectra.sbatch for `none`.
#
#   NORM=zscore_vs_baseline ROI_SCHEME=roi_v2 sbatch sbatch/plot_view_heatmap.sbatch

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

NORM=${NORM:-zscore_vs_baseline}
ROI_SCHEME=${ROI_SCHEME:-default}
MASK_LABEL=${MASK_LABEL:-std10_rv-gross-std3_satmargin15_sw_logz4}
SPLIT=${SPLIT:-discovery}
RUN_PREFIX=${RUN_PREFIX:-discovery_std10}
QUESTION=${QUESTION:-psd_physiology}
MIN_SUBJECTS=${MIN_SUBJECTS:-8}
N_PERM=${N_PERM:-10000}
DEST=${SCRATCH_RUN:+--scratch}

if [ "$NORM" = "none" ]; then
  echo "ERROR: NORM=none has no meaningful cluster test against zero, and a raw-power"
  echo "       heatmap is a 1/f gradient with no contrast. Use plot_view_spectra.sbatch."
  exit 1
fi

echo "=== heatmap | norm=$NORM | roi=$ROI_SCHEME ($(date)) ==="

VIEW_DIR=$(python -m ieeg_ehr.views.build_pain_epoch_view \
    --print-out-dir --split "$SPLIT" --mask-label "$MASK_LABEL" \
    --roi-scheme "$ROI_SCHEME" --normalization "$NORM")
echo "    view dir: $VIEW_DIR"

n=$(ls "$VIEW_DIR"/view_subject_sub-*.parquet 2>/dev/null | wc -l)
echo "    view tables present: $n"
if [ "$n" -eq 0 ]; then
  echo "    ERROR: no view tables in $VIEW_DIR -- did the view array run for this"
  echo "           normalization AND this ROI scheme?"
  exit 1
fi

# Line-noise bins are dropped from the DISPLAY here; the cluster test drops them
# from the statistics unconditionally (docs/cluster_permutation.md section 7).
python -m ieeg_ehr.analysis.plot_pain_view_heatmaps \
    --view-dir "$VIEW_DIR" \
    --question "$QUESTION" \
    --exclude-line-noise-bins \
    --cluster-test --n-perm "$N_PERM" --min-subjects "$MIN_SUBJECTS" \
    --run-name "${RUN_PREFIX}_${NORM}" ${DEST}

echo "=== done ($(date)) ==="

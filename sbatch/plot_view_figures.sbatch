#!/bin/bash
#SBATCH --job-name=view_figures
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=24GB
#SBATCH --output=logs/view_figures_%j.out
#
# Every figure built from the already-built P1.3 view tables, for each
# normalization: region x frequency-bin HEATMAPS (where an effect is) and
# per-region SPECTRA (what shape it has). One job rather than two because both
# read the same tables and each takes seconds -- two jobs would spend more time
# queueing than plotting.
#
# Reads the TABLES rather than recomputing, so the figures and the numbers behind
# them provably come from the same values. Intended to run with --dependency on
# the view array.
#
# Outputs land in the deliberate analysis tree, NOT scratch:
#   analysis/pain/<question>/{region_freq_heatmap,region_spectrum}/<view-code>/<run>_<ts>/
# Pass --scratch (via SCRATCH=1) while iterating on a figure.
#
# The view directory is ASKED FOR, never spelled here: --print-out-dir resolves
# the same <scheme_code>_<config_hash> the builder writes to, so this job cannot
# plot a directory the builder did not fill.
#
# Region row order in the heatmaps is the FIXED anatomical config.ROI_REGIONS order
# by default, so every figure -- across normalizations, runs and subjects -- shares
# one ordering and can be compared directly.

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

MASK_LABEL=${MASK_LABEL:-std10_rv-gross-std3_satmargin15_sw_logz4}
SPLIT=${SPLIT:-discovery}
RUN_PREFIX=${RUN_PREFIX:-discovery_std10}
QUESTION=${QUESTION:-psd_physiology}
MIN_SUBJECTS=${MIN_SUBJECTS:-8}
N_PERM=${N_PERM:-10000}
DEST=${SCRATCH_RUN:+--scratch}

for NORM in ${NORMALIZATIONS:-zscore_vs_baseline baseline_subtract}; do
  echo "=== $NORM ($(date)) ==="

  VIEW_DIR=$(python -m ieeg_ehr.views.build_pain_epoch_view \
      --print-out-dir --split "$SPLIT" --mask-label "$MASK_LABEL" \
      --normalization "$NORM")
  echo "    view dir: $VIEW_DIR"

  n=$(ls "$VIEW_DIR"/view_subject_sub-*.parquet 2>/dev/null | wc -l)
  echo "    view tables present: $n"
  if [ "$n" -eq 0 ]; then
    echo "    ERROR: no view tables in $VIEW_DIR -- did the view array run?"
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

  python -m ieeg_ehr.analysis.plot_pain_view_spectra \
      --view-dir "$VIEW_DIR" \
      --question "$QUESTION" \
      --min-subjects "$MIN_SUBJECTS" \
      --run-name "${RUN_PREFIX}_${NORM}" ${DEST}
done

echo "=== done ($(date)) ==="

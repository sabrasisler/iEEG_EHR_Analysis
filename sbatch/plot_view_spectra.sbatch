#!/bin/bash
#SBATCH --job-name=view_spectra
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --output=logs/view_spectra_%j.out
#
# ONE per-region spectra run (the ribbon figure): one normalization, one ROI scheme.
# Mean +- SEM across subjects, one subplot per region, one line per pain bin.
#
# ONE JOB PER FIGURE -- see plot_view_heatmap.sbatch for why splitting is free.
#
# WORKS FOR ALL THREE NORMALIZATIONS, and the script adapts to which one:
#   zscore_vs_baseline / baseline_subtract -> two lines (low, high), y=0 reference
#       drawn, because the 0-pain bin IS the reference and is 0 by construction.
#   none                                   -> THREE lines including 0-pain, no y=0
#       line. Un-normalized, the 0-pain bin is a real power spectrum and is the
#       curve the other two are read against. Expect the between-subject SEM to be
#       wide and the lines to overlap: raw power varies enormously across
#       electrodes, impedances and gains, which is exactly why the normalized
#       figures exist.
#
#   NORM=none ROI_SCHEME=roi_v2 sbatch sbatch/plot_view_spectra.sbatch

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
DEST=${SCRATCH_RUN:+--scratch}

echo "=== spectra | norm=$NORM | roi=$ROI_SCHEME ($(date)) ==="

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

python -m ieeg_ehr.analysis.plot_pain_view_spectra \
    --view-dir "$VIEW_DIR" \
    --question "$QUESTION" \
    --min-subjects "$MIN_SUBJECTS" \
    --run-name "${RUN_PREFIX}_${NORM}" ${DEST}

echo "=== done ($(date)) ==="

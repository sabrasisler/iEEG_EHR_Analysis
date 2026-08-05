#!/bin/bash
#SBATCH --job-name=band_violin
#SBATCH --partition=ckeller1
#SBATCH --qos=high_p
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=24GB
#SBATCH --output=logs/band_violin_%j.out
#
# One band, three pain levels, one dot per subject, one panel per region.
#
# READS THE UN-NORMALIZED VIEW ON PURPOSE. This figure draws the 0-pain level as one
# of its violins, so it cannot use a view that is already referenced to 0-pain --
# that level would be exactly 0 by construction. The script does its own
# within-subject standardization instead, and REFUSES a baseline-referenced view
# rather than producing a plausible-looking figure with a degenerate first violin.
#
#   BAND=beta ROI_SCHEME=roi_v2 sbatch sbatch/plot_band_violin.sbatch

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

BAND=${BAND:-beta}
ROI_SCHEME=${ROI_SCHEME:-roi_v2}
MASK_LABEL=${MASK_LABEL:-std10_rv-gross-std3_satmargin15_sw_logz4}
SPLIT=${SPLIT:-discovery}
QUESTION=${QUESTION:-psd_physiology}
MIN_SUBJECTS=${MIN_SUBJECTS:-8}
DEST=${SCRATCH_RUN:+--scratch}

# normalization=none: the raw view. See the header.
VIEW_DIR=$(python -m ieeg_ehr.views.build_pain_epoch_view \
    --print-out-dir --split "$SPLIT" --mask-label "$MASK_LABEL" \
    --roi-scheme "$ROI_SCHEME" --normalization none)
echo "=== $BAND violins | roi=$ROI_SCHEME | view=$VIEW_DIR ($(date)) ==="

n=$(ls "$VIEW_DIR"/view_epochs_sub-*.parquet 2>/dev/null | wc -l)
echo "    epoch tables present: $n"
if [ "$n" -eq 0 ]; then
  echo "    ERROR: no epoch tables in $VIEW_DIR -- build the un-normalized view first:"
  echo "           NORMALIZATIONS=none ROI_SCHEME=$ROI_SCHEME sbatch sbatch/view_discovery_array.sbatch"
  exit 1
fi

python -m ieeg_ehr.analysis.plot_band_violin_view \
    --view-dir "$VIEW_DIR" \
    --band "$BAND" \
    --question "$QUESTION" \
    --min-subjects "$MIN_SUBJECTS" ${DEST}

echo "=== done ($(date)) ==="

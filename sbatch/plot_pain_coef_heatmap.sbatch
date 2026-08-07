#!/bin/bash
#SBATCH --job-name=pain_coef
#SBATCH --partition=ckeller1
#SBATCH --qos=high_p
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --output=logs/pain_coef_%j.out
#
# Continuous-pain heatmap: group-mean regression of log power on pain score, with
# cluster outlines. Lands beside the binned heatmaps as contpain-<roi>.
#
# READS THE UN-NORMALIZED VIEW, and the script refuses anything else. The whole
# point is to have no 0-pain baseline; regressing an already-baselined quantity on
# pain score would be a different and largely meaningless number.
#
# More memory and time than the binned heatmaps: the pain-score shuffle null
# recomputes a per-subject matmul per permutation rather than flipping a sign.
#
#   ROI_SCHEME=roi_v2 sbatch sbatch/plot_pain_coef_heatmap.sbatch

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

ROI_SCHEME=${ROI_SCHEME:-roi_v2}
MASK_LABEL=${MASK_LABEL:-std10_rv-gross-std3_satmargin15_sw_logz4}
SPLIT=${SPLIT:-discovery}
QUESTION=${QUESTION:-psd_physiology}
N_PERM=${N_PERM:-10000}
SHUFFLE_N_PERM=${SHUFFLE_N_PERM:-2000}
DEST=${SCRATCH_RUN:+--scratch}

VIEW_DIR=$(python -m ieeg_ehr.views.build_pain_epoch_view \
    --print-out-dir --split "$SPLIT" --mask-label "$MASK_LABEL" \
    --roi-scheme "$ROI_SCHEME" --normalization none)
echo "=== pain_coef heatmap | roi=$ROI_SCHEME | view=$VIEW_DIR ($(date)) ==="

n=$(ls "$VIEW_DIR"/view_epochs_sub-*.parquet 2>/dev/null | wc -l)
echo "    epoch tables present: $n"
if [ "$n" -eq 0 ]; then
  echo "    ERROR: no epoch tables in $VIEW_DIR -- build the un-normalized view first."
  exit 1
fi

python -m ieeg_ehr.analysis.plot_pain_coef_heatmap \
    --view-dir "$VIEW_DIR" \
    --question "$QUESTION" \
    --n-perm "$N_PERM" \
    --shuffle-n-perm "$SHUFFLE_N_PERM" ${DEST}

echo "=== done ($(date)) ==="

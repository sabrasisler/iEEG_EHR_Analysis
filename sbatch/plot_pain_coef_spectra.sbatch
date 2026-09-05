#!/bin/bash
#SBATCH --job-name=coef_spectra
#SBATCH --partition=owners
#SBATCH --qos=normal
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=24GB
#SBATCH --output=logs/coef_spectra_%j.out
#
# The line-plot companion to the contpain heatmap: one subplot per region, ONE line
# (the group-mean regression of log power on pain score) with a +-SEM ribbon.
#
# Reads the UN-NORMALIZED view and refuses anything else, for the same reason the
# heatmap does: the regression must run on raw log power.
#
# No permutation test here, so it is seconds rather than minutes -- significance
# lives on the heatmap, shape lives here.
#
#   ROI_SCHEME=roi_v2 sbatch sbatch/plot_pain_coef_spectra.sbatch

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

ROI_SCHEME=${ROI_SCHEME:-roi_v2}
MASK_LABEL=${MASK_LABEL:-std10_rv-gross-std3_satmargin15_sw_logz4}
SPLIT=${SPLIT:-discovery}
QUESTION=${QUESTION:-psd_physiology}
MIN_SUBJECTS=${MIN_SUBJECTS:-8}
DEST=${SCRATCH_RUN:+--scratch}

VIEW_DIR=$(python -m ieeg_ehr.views.build_pain_epoch_view \
    --print-out-dir --split "$SPLIT" --mask-label "$MASK_LABEL" \
    --roi-scheme "$ROI_SCHEME" --normalization none)
echo "=== pain_coef spectra | roi=$ROI_SCHEME | view=$VIEW_DIR ($(date)) ==="

python -m ieeg_ehr.analysis.plot_pain_coef_spectra \
    --view-dir "$VIEW_DIR" \
    --question "$QUESTION" \
    --min-subjects "$MIN_SUBJECTS" ${DEST}

echo "=== done ($(date)) ==="

#!/bin/bash
#SBATCH --job-name=elec_loc
#SBATCH --partition=ckeller1
#SBATCH --qos=high_p
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --output=logs/elec_loc_%j.out
#
# Static glass brain of electrode coverage, coloured by ROI.
#
# Takes the subject list AND the ROI scheme from a VIEW RUN by default, so the
# coverage figure describes exactly the analysis it illustrates rather than some
# other cohort. Set VIEW_DIR='' to fall back to the whole split.
#
# Needs nilearn/nibabel/scikit-learn/colorcet, installed into the shared venv
# 2026-08-05 with --no-deps --only-binary=:all: (io_conventions.md 8) so
# numpy/scipy/pandas stayed pinned. The glass brain outlines ship with nilearn, so
# this renders with NO network -- verified, which matters on a compute node.
#
#   NORM=zscore_vs_baseline ROI_SCHEME=roi_v2 sbatch sbatch/plot_electrode_locations.sbatch

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

NORM=${NORM:-zscore_vs_baseline}
ROI_SCHEME=${ROI_SCHEME:-roi_v2}
MASK_LABEL=${MASK_LABEL:-std10_rv-gross-std3_satmargin15_sw_logz4}
SPLIT=${SPLIT:-discovery}
QUESTION=${QUESTION:-psd_physiology}
HEMISPHERE=${HEMISPHERE:-both}
DEST=${SCRATCH_RUN:+--scratch}

VIEW_DIR=$(python -m ieeg_ehr.views.build_pain_epoch_view \
    --print-out-dir --split "$SPLIT" --mask-label "$MASK_LABEL" \
    --roi-scheme "$ROI_SCHEME" --normalization "$NORM")
echo "=== electrode coverage | roi=$ROI_SCHEME | view=$VIEW_DIR ($(date)) ==="

python -m ieeg_ehr.analysis.plot_electrode_locations \
    --view-dir "$VIEW_DIR" \
    --question "$QUESTION" \
    --hemisphere "$HEMISPHERE" \
    --split "$SPLIT" ${DEST}

echo "=== done ($(date)) ==="

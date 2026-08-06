#!/bin/bash
#SBATCH --job-name=slope_violin
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --output=logs/slope_violin_%j.out
#
# 1/f slope violins: zero / low / high, one dot per subject, one panel per region.
# Writes TWO figures from one table -- within-subject z (primary) and native
# slope units.
#
#   ROI_SCHEME=roi_v2 sbatch sbatch/plot_slope_violin.sbatch
#   MIN_R2=0.9 sbatch sbatch/plot_slope_violin.sbatch     # drop poor fits
#
# THE VIEW DIRECTORY IS ASKED FOR, NOT SPELLED. --print-out-dir resolves the
# config_hash through the same code that wrote it, so this script cannot drift
# from the builder the way a hard-coded hash in bash would. The fit range must
# therefore match whatever the array was run with -- the range is part of the
# hash, so a mismatch resolves to an empty directory and fails loudly below
# rather than silently plotting the wrong fit.

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

ROI_SCHEME=${ROI_SCHEME:-roi_v2}
MASK_LABEL=${MASK_LABEL:-std10_rv-gross-std3_satmargin15_sw_logz4}
SPLIT=${SPLIT:-discovery}
RUN_PREFIX=${RUN_PREFIX:-discovery_std10}
QUESTION=${QUESTION:-psd_physiology}
MIN_SUBJECTS=${MIN_SUBJECTS:-8}
FIT_LO_HZ=${FIT_LO_HZ:-1}
FIT_HI_HZ=${FIT_HI_HZ:-250}
NCOLS=${NCOLS:-4}
DEST=${SCRATCH_RUN:+--scratch}
MIN_R2_ARG=${MIN_R2:+--min-r2 $MIN_R2}

echo "=== slope violins | roi=$ROI_SCHEME | fit ${FIT_LO_HZ}-${FIT_HI_HZ} Hz ($(date)) ==="

VIEW_DIR=$(python -m ieeg_ehr.views.build_pain_epoch_slope \
    --print-out-dir --split "$SPLIT" --mask-label "$MASK_LABEL" \
    --roi-scheme "$ROI_SCHEME" --normalization none \
    --fit-lo-hz "$FIT_LO_HZ" --fit-hi-hz "$FIT_HI_HZ")
echo "    view dir: $VIEW_DIR"

n=$(ls "$VIEW_DIR"/view_epochs_sub-*.parquet 2>/dev/null | wc -l)
echo "    slope tables present: $n"
if [ "$n" -eq 0 ]; then
  echo "    ERROR: no slope tables in $VIEW_DIR -- run"
  echo "           sbatch/slope_discovery_array.sbatch with the SAME fit range"
  echo "           and ROI scheme first (both are part of the config hash)."
  exit 1
fi

python -m ieeg_ehr.analysis.plot_slope_violin \
    --view-dir "$VIEW_DIR" \
    --question "$QUESTION" \
    --min-subjects "$MIN_SUBJECTS" \
    --ncols "$NCOLS" \
    --run-name "${RUN_PREFIX}_slope" ${MIN_R2_ARG} ${DEST}

echo "=== done ($(date)) ==="

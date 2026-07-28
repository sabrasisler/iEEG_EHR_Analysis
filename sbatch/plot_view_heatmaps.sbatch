#!/bin/bash
#SBATCH --job-name=view_heatmaps
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --output=logs/view_heatmaps_%j.out
#
# Region x frequency-bin heatmaps from already-built view tables: one group figure
# plus one per subject, for each normalization.
#
# Reads the TABLES rather than recomputing, so the figures and the numbers behind
# them provably come from the same values, and re-plotting is seconds instead of
# minutes. Intended to run with --dependency on the view array.
#
# The region row order is clustered ONCE from each normalization's own group table.
# That means the z-score and delta figures may order rows differently; pass
# --region-order explicitly to force a shared order if you want them side by side.

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

DERIV=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler
OUT_ROOT=${OUT_ROOT:-$DERIV/analysis/scratch/p13_discovery}
RUN_PREFIX=${RUN_PREFIX:-discovery_std10}

for NORM in ${NORMALIZATIONS:-zscore_vs_baseline baseline_subtract}; do
  echo "=== $NORM ($(date)) ==="
  n=$(ls "$OUT_ROOT/$NORM"/view_subject_sub-*.parquet 2>/dev/null | wc -l)
  echo "    view tables present: $n"
  if [ "$n" -eq 0 ]; then
    echo "    ERROR: no view tables in $OUT_ROOT/$NORM -- did the view array run?"
    exit 1
  fi
  python -m ieeg_ehr.analysis.plot_pain_view_heatmaps \
      --view-dir "$OUT_ROOT/$NORM" \
      --run-name "${RUN_PREFIX}_${NORM}"
done

echo "=== done ($(date)) ==="

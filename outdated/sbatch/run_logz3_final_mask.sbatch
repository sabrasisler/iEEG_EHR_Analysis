#!/bin/bash
#SBATCH -p ckeller1
#SBATCH --qos=high_p
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH -J logz3_final_mask
#SBATCH -o /home/groups/ckeller1/sisler/iEEG_EHR_Analysis/logs/logz3_final_mask_%j.out

set -euo pipefail

module load python/3.12
source /home/groups/ckeller1/venvs/ieeg_ehr_analysis/bin/activate
cd /home/groups/ckeller1/sisler/iEEG_EHR_Analysis

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

LEVEL=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/analysis/qc/raw_voltage
MASK3_LABEL=gross-std3_satmargin15_sw
LOGZ3=logz3_masked-${MASK3_LABEL}
FINAL_LABEL=gross-std3_satmargin15_sw_logz3

echo "=== 1. flatline rollup: logz3 masked by $MASK3_LABEL ==="
python -m qc_scripts.build_exclusions --level-root "$LEVEL" --artifact-type flatline \
    --std-thresh 3 --mask-from-label "$MASK3_LABEL"

echo "=== 2. final combined mask: gross_artifact=std3, saturation=pct0_marginfrac0.15, square_wave=frac0.9, flatline=$LOGZ3 ==="
python -m qc_scripts.build_mask --level-root "$LEVEL" --label "$FINAL_LABEL" \
    --gross_artifact std3 --saturation pct0_marginfrac0.15 --square_wave frac0.9 --flatline "$LOGZ3"

echo "=== 3. summarize_exclusions on the final mask ==="
python -m qc_scripts.summarize_exclusions --mask-dir "$LEVEL/masks/$FINAL_LABEL"

echo "=== 4. flagged example plots (from flagged_for_review.csv) ==="
python -m qc_scripts.plot_flagged_runs --level-root "$LEVEL" \
    --review-csv "$LEVEL/masks/$FINAL_LABEL/summary/flagged_for_review.csv" \
    --mask-label "$FINAL_LABEL" --n-runs 1 \
    --plots-dir "$LEVEL/masks/$FINAL_LABEL/plots" --examples-subdir flagged_examples

echo "=== 5. random example plots (~50, no exclusion requirement) ==="
python -m qc_scripts.plot_flagged_runs --level-root "$LEVEL" \
    --random-any 50 --seed 42 \
    --mask-label "$FINAL_LABEL" \
    --plots-dir "$LEVEL/masks/$FINAL_LABEL/plots" --examples-subdir random_examples

echo "=== done ==="

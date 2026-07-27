#!/bin/bash
#SBATCH -p ckeller1
#SBATCH --qos=high_p
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH -J flatline_mask_sweep
#SBATCH -o /home/groups/ckeller1/sisler/iEEG_EHR_Analysis/logs/flatline_mask_sweep_%j.out

set -euo pipefail

module load python/3.12
source /home/groups/ckeller1/venvs/ieeg_ehr_analysis/bin/activate
cd /home/groups/ckeller1/sisler/iEEG_EHR_Analysis

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

LEVEL=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/analysis/qc/raw_voltage
MASK3_LABEL=gross-std3_satmargin15_sw

# Steps 1-2 (saturation exclusion + 3-type mask) already completed in job
# 34075695 -- masks/$MASK3_LABEL exists (17 subject/sessions; sub-236 dropped,
# known gap re: missing square_wave exclusions, see CONTEXT.md). Resuming at
# step 3, now that build_exclusions.py skips (not crashes on) subjects missing
# from the mask dir.

echo "=== 3. flatline rollups masked by $MASK3_LABEL: linz4, logz4, linz5, logz5 ==="
python -m qc_scripts.build_exclusions --level-root "$LEVEL" --artifact-type flatline \
    --std-thresh 4 --linear --mask-from-label "$MASK3_LABEL"
python -m qc_scripts.build_exclusions --level-root "$LEVEL" --artifact-type flatline \
    --std-thresh 4 --mask-from-label "$MASK3_LABEL"
python -m qc_scripts.build_exclusions --level-root "$LEVEL" --artifact-type flatline \
    --std-thresh 5 --linear --mask-from-label "$MASK3_LABEL"
python -m qc_scripts.build_exclusions --level-root "$LEVEL" --artifact-type flatline \
    --std-thresh 5 --mask-from-label "$MASK3_LABEL"

echo "=== 4. threshold_summary across the 4 new masked labels ==="
python -m qc_scripts.diagnostics.threshold_summary --level-root "$LEVEL" --artifact-type flatline \
    --labels "linz4_masked-${MASK3_LABEL},linz5_masked-${MASK3_LABEL},logz4_masked-${MASK3_LABEL},logz5_masked-${MASK3_LABEL}"

echo "=== 5a. threshold_diff: linz4 (baseline) vs linz5 (compare) ==="
python -m qc_scripts.diagnostics.threshold_diff --level-root "$LEVEL" --artifact-type flatline \
    --baseline-label "linz4_masked-${MASK3_LABEL}" --compare-label "linz5_masked-${MASK3_LABEL}" \
    --n-examples 12 --max-per-subject 2

echo "=== 5b. threshold_diff: linz4 (baseline) vs logz4 (compare) ==="
python -m qc_scripts.diagnostics.threshold_diff --level-root "$LEVEL" --artifact-type flatline \
    --baseline-label "linz4_masked-${MASK3_LABEL}" --compare-label "logz4_masked-${MASK3_LABEL}" \
    --n-examples 12 --max-per-subject 2

echo "=== done ==="

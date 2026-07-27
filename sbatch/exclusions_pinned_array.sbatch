#!/bin/bash
# Stage 1 of the pinned-mask chain: the three UNMASKED per-type exclusions, at
# the thresholds the pinned mask is built from. One array task per subject.
#
# NOT `--artifact-type all` — that uses the config defaults (gross std5,
# saturation default, flatline var5e-13), which are NOT this mask's thresholds.
#
# Full chain (see docs/qc_context.md):
#   1. THIS JOB          gross std3 + saturation marginfrac0.15 + square_wave frac0.9
#   2. build_pinned_masks.sbatch STAGE=intermediate   -> gross-std3_satmargin15_sw
#   3. exclusions_pinned_flatline_array.sbatch        -> logz4_masked-<intermediate>
#   4. build_pinned_masks.sbatch STAGE=final          -> gross-std3_satmargin15_sw_logz4
#
# Usage (submit FROM THE REPO ROOT; size the array to the file's line count):
#   sbatch --array=0-4 --export=ALL,SUBJECTS_FILE=<path> sbatch/exclusions_pinned_array.sbatch
#
#SBATCH -p normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH -J excl_pinned
#SBATCH -o logs/excl_pinned_%A_%a.out

set -euo pipefail
module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate

OAK=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler
LEVEL=$OAK/qc/raw_voltage
: "${SUBJECTS_FILE:=$OAK/cohorts/subjects_qc_raw_voltage_normal.txt}"

SUBJECT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SUBJECTS_FILE")
[ -n "$SUBJECT" ] || { echo "No subject at array index $SLURM_ARRAY_TASK_ID in $SUBJECTS_FILE"; exit 1; }
echo "Array task $SLURM_ARRAY_TASK_ID -> subject $SUBJECT (from $SUBJECTS_FILE)"

# 16GB, not the usual 8GB: these are the largest subjects in the cohort
# (sub-256's saturation metric CSV alone is 8.2GB). build_exclusions streams at
# chunksize=500_000 so memory is bounded, but headroom is cheap here.

python -m ieeg_ehr.qc.build_exclusions --level-root "$LEVEL" \
    --artifact-type gross_artifact --std-thresh 3 --subjects "$SUBJECT"

python -m ieeg_ehr.qc.build_exclusions --level-root "$LEVEL" \
    --artifact-type saturation --rail-margin-frac 0.15 --subjects "$SUBJECT"

python -m ieeg_ehr.qc.build_exclusions --level-root "$LEVEL" \
    --artifact-type square_wave --frac-thresh 0.9 --subjects "$SUBJECT"

echo "DONE stage 1 for $SUBJECT"

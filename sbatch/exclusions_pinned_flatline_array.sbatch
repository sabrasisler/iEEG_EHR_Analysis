#!/bin/bash
# Stage 3 of the pinned-mask chain: the MASK-AWARE flatline exclusion.
#
# Flatline is last and separate because it is the only per-type exclusion that
# depends on a mask: --mask-from-label reads the intermediate mask so the
# per-channel log-variance baseline is computed over unmasked windows only.
# That is why stage 2 (intermediate mask) has to run in between.
#
# Produces: exclusions/flatline/logz4_masked-gross-std3_satmargin15_sw/
#
# Usage (submit FROM THE REPO ROOT):
#   sbatch --array=0-4 --export=ALL,SUBJECTS_FILE=<path> \
#          sbatch/exclusions_pinned_flatline_array.sbatch
#
#SBATCH -p normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH -J excl_flatline
#SBATCH -o logs/excl_flatline_%A_%a.out

set -euo pipefail
module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate

OAK=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler
LEVEL=$OAK/qc/raw_voltage
INTERMEDIATE=gross-std3_satmargin15_sw
: "${SUBJECTS_FILE:=$OAK/cohorts/subjects_qc_raw_voltage_normal.txt}"

SUBJECT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SUBJECTS_FILE")
[ -n "$SUBJECT" ] || { echo "No subject at array index $SLURM_ARRAY_TASK_ID in $SUBJECTS_FILE"; exit 1; }
echo "Array task $SLURM_ARRAY_TASK_ID -> subject $SUBJECT (from $SUBJECTS_FILE)"

python -m ieeg_ehr.qc.build_exclusions --level-root "$LEVEL" \
    --artifact-type flatline --std-thresh 4 --mask-from-label "$INTERMEDIATE" \
    --subjects "$SUBJECT"

echo "DONE stage 3 for $SUBJECT"

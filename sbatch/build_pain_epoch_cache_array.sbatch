#!/bin/bash
# P1.1: build the per-window pain epoch cache. One array task per subject.
#
# Emits into features/pain/psd_epochs/epoch-5min-pre/:
#   cache/sub-XXX_ses-YY_epochs.parquet      raw per-window log-power (long)
#   epoch_defs/sub-XXX_ses-YY_defs.parquet   one row per epoch
#
# The cache is RAW: no QC mask, no line-noise filtering, no averaging, no
# normalization. All of those are view-layer concerns now.
#
# --session all so subjects with two sessions (sub-255, sub-209, sub-197) get
# both. Task 0 additionally writes the base-unit manifest.json, which describes
# the unit rather than the subject.
#
# Sized from a measured sub-071 run (12 epochs / 6.3M rows / 25s) and the
# largest subject (sub-256: 199 pairs x 137 events ~ 409M rows). Streaming
# writes keep peak memory to one run's PSD plus one epoch's frame, so memory is
# driven by run length and channel count, not by total epochs.
#
# Usage (submit FROM THE REPO ROOT):
#   sbatch --array=0-86%12 --export=ALL,SUBJECTS_FILE=<path> \
#          sbatch/build_pain_epoch_cache_array.sbatch
#
#SBATCH -p normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=24GB
#SBATCH -J pain_epoch_cache
#SBATCH -o logs/pain_epoch_cache_%A_%a.out

set -euo pipefail
module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate

OAK=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler
: "${SUBJECTS_FILE:=$OAK/cohorts/subjects_pain_epoch_cache.txt}"

SUBJECT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SUBJECTS_FILE")
[ -n "$SUBJECT" ] || { echo "No subject at array index $SLURM_ARRAY_TASK_ID in $SUBJECTS_FILE"; exit 1; }
echo "Array task $SLURM_ARRAY_TASK_ID -> subject $SUBJECT (from $SUBJECTS_FILE)"

MANIFEST_FLAG=""
[ "$SLURM_ARRAY_TASK_ID" -eq 0 ] && MANIFEST_FLAG="--write-manifest"

python -m ieeg_ehr.features.build_pain_epoch_power \
    --subjects "$SUBJECT" --session all --overwrite $MANIFEST_FLAG

echo "DONE $SUBJECT"

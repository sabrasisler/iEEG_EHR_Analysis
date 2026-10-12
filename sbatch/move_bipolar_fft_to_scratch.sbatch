#!/bin/bash
# Move the 50-log-bin continuous PSD tree (preprocessed/bipolar_fft, 835 GB) off
# Oak to $GROUP_SCRATCH, as a temporary backup. One-off (DECISIONS 2026-10-12).
#
# The continuous 0.5 Hz family (preprocessed/bipolar_psd) supersedes it: the
# 50 log bins become a view of the native grid. The scratch copy stays readable
# for the bit-exact gate (preprocessing.bipolar_psd_fullres_audit).
#
# No -t: the copies carry the copy date, so the 90-day purge removes them around
# 2027-01-10. Nothing refreshes them. Scratch is the ONLY copy after this job.
#
# The Oak original is deleted only if every file arrived with the same size.
#
# Usage (from the repo root -- the log path is relative):
#   sbatch sbatch/move_bipolar_fft_to_scratch.sbatch
#
#SBATCH -p normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH -J move_bipolar_fft
#SBATCH -o logs/move_bipolar_fft_%j.out

set -euo pipefail

SRC=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/preprocessed/bipolar_fft
DST=$GROUP_SCRATCH/sisler/bipolar_fft

mkdir -p "$DST"
echo "$(date -Is) copy $SRC -> $DST"
rsync -rl --size-only "$SRC/" "$DST/"

echo "$(date -Is) compare file lists and sizes"
list() { (cd "$1" && find . -type f -printf '%P %s\n' | sort); }
diff <(list "$SRC") <(list "$DST")
echo "$(date -Is) $(list "$DST" | wc -l) files match"

rm -rf "$SRC"
echo "$(date -Is) removed $SRC"

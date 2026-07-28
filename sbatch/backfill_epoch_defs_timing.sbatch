#!/bin/bash
#SBATCH --job-name=defs_timing
#SBATCH --partition=normal
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --output=logs/defs_timing_%A_%a.out
#SBATCH --array=0-82
#
# P1.3 / Step 2b: add epoch_start_sec + hop_sec to every epoch_defs file, and
# audit the (starting_time, rate) assumption while doing it.
#
# Metadata-only NWB reads (~0.3 s/run) -- this NEVER touches the 34 GB cache, so
# it is cheap enough to re-run. One array task per subject-session, per the
# arrays-not-agents rule in CLAUDE.md.
#
# The array size must match the number of defs files on disk (83 as of
# 2026-07-28 => 0-82). The task below hard-fails rather than silently no-op-ing
# if the index is out of range, so a stale --array bound is visible.

set -euo pipefail

# ~/bin/git ahead of /usr/bin so git provenance is recordable ON THE NODE.
# Without it the login node's git 1.8.3.1 cannot read a worktree gitfile and
# every sidecar records commit "no-git" (SCRATCHPAD, 2026-07-27).
export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

# RUN THE CODE THIS JOB WAS SUBMITTED FROM.
# `ieeg_ehr` is pip-installed editable against the MAIN checkout, so a bare
# `python -m ieeg_ehr...` executes /…/iEEG_EHR_Analysis/src regardless of where
# sbatch was invoked. Submitting from a worktree would then silently run the main
# checkout's code -- old code, under a commit hash that claims otherwise, which is
# exactly what the commit-before-you-run rule exists to prevent. Prepending the
# submit directory's src/ makes "what ran" and "what I submitted" the same thing;
# it is a no-op when submitting from the main checkout.
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
python -c "import ieeg_ehr; print('ieeg_ehr:', ieeg_ehr.__file__)"

DEFS_DIR="/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/features/pain/psd_epochs/epoch-5min-pre/epoch_defs"
mapfile -t STEMS < <(ls "$DEFS_DIR"/sub-*_ses-*_defs.parquet | xargs -n1 basename | sed 's/_defs\.parquet$//')

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#STEMS[@]}" ]; then
  echo "ERROR: array task ${SLURM_ARRAY_TASK_ID} >= ${#STEMS[@]} defs files present."
  echo "Update --array in this file to 0-$(( ${#STEMS[@]} - 1 ))."
  exit 1
fi

STEM="${STEMS[$SLURM_ARRAY_TASK_ID]}"
SUBJECT="$(echo "$STEM" | sed 's/^sub-\([^_]*\)_ses-.*/\1/')"
SESSION="$(echo "$STEM" | sed 's/^sub-[^_]*_ses-\(.*\)$/\1/')"

echo "=== task ${SLURM_ARRAY_TASK_ID}: sub-${SUBJECT} ses-${SESSION} ($(date)) ==="

python -m ieeg_ehr.features.backfill_epoch_defs_timing \
  --subjects "$SUBJECT" --session "$SESSION" --overwrite

echo "=== done ($(date)) ==="

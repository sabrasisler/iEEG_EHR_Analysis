#!/bin/bash
# Heart rate at every second, from the EKG leads (preprocessing/ekg_hr.py), into
# preprocessed/ekg_hr/. One array task per line of SUBJECTS_FILE; every session
# of that subject.
#
# Usage (from the repo root -- the log path is relative):
#   C=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/cohorts
#   SUBJECTS_FILE=$C/subjects_ekg_hr.txt \
#       sbatch --array=0-<n_subjects - 1>%<cap> sbatch/ekg_hr_array.sbatch
#
# Re-submitting is safe: sessions with an hr table are skipped. OVERWRITE=1
# redoes them.
#
# RESOURCES -- measured 2026-10-19 on sub-019 (5 h runs @ 500 Hz) and sub-256
# (2 h runs @ 2000 Hz), jobs 48579609 / 48579615:
#   ~80 s per 5 h run @ 500 Hz, ~65 s per 2 h run @ 2000 Hz, so sub-256's 121
#   runs take ~2.2 h. Serial, so one core.
#   MaxRSS 4.9 GB / 6.7 GB mid-run (page cache included). Memory scales with
#   samples per run, and a 5 h run @ 1000 Hz holds more than either.
#
#SBATCH -p normal
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=12GB
#SBATCH -J ekg_hr
#SBATCH -o logs/ekg_hr_%A_%a.out

set -euo pipefail
module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$HOME/bin:$PATH"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"

: "${SUBJECTS_FILE:?set SUBJECTS_FILE to a one-subject-per-line list}"
SUBJECT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SUBJECTS_FILE")
[ -n "$SUBJECT" ] || { echo "No subject at array index $SLURM_ARRAY_TASK_ID in $SUBJECTS_FILE"; exit 1; }
echo "Array task $SLURM_ARRAY_TASK_ID -> subject $SUBJECT (from $SUBJECTS_FILE)"

OVERWRITE_FLAG=""
[ "${OVERWRITE:-0}" = 1 ] && OVERWRITE_FLAG="--overwrite"

python -m ieeg_ehr.preprocessing.ekg_hr --subjects "$SUBJECT" $OVERWRITE_FLAG

echo "DONE $SUBJECT"

#!/bin/bash
# Per-subject array rollup: one task per subject, each running build_exclusions
# --artifact-type all for a single subject. Cheap (CSV-only) but parallelizes the
# big metric-CSV reads across tasks.
#
# Set SUBJECTS_FILE + the array range to match its line count. Example:
#   sbatch --array=0-16%8 --export=ALL,SUBJECTS_FILE=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/cohorts/subjects_qc_raw_voltage_normal.txt \
#          sbatch/build_exclusions_array.sbatch
#SBATCH -p normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH -J build_excl
#SBATCH -o logs/build_excl_%A_%a.out

module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate

: "${SUBJECTS_FILE:=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/cohorts/subjects_qc_raw_voltage_normal.txt}"
LEVEL=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/qc/raw_voltage

SUBJECT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SUBJECTS_FILE")
echo "Array task $SLURM_ARRAY_TASK_ID -> subject $SUBJECT (from $SUBJECTS_FILE)"
python -m ieeg_ehr.qc.build_exclusions --level-root "$LEVEL" --artifact-type all --subjects "$SUBJECT"

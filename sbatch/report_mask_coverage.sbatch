#!/bin/bash
# Audit raw-voltage mask coverage for a QC level: which subject/sessions had a
# mask when their metrics were computed, and which had a mask that was missing
# runs (the silent case -- see the module docstring).
#
# Read-only. Reads one column from every mask CSV (largest ~190 MB) plus the file
# registry and the level's run_info. Nothing is recomputed, so this can be run
# any time and re-run freely.
#
# Usage (submit FROM THE REPO ROOT):
#   sbatch sbatch/report_mask_coverage.sbatch                          # feature_level
#   sbatch --export=ALL,LEVEL=bipolar sbatch/report_mask_coverage.sbatch
#   sbatch --export=ALL,MASK_LABEL=gross-std3_satmargin15_sw sbatch/report_mask_coverage.sbatch
#
#SBATCH -p normal
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH -J mask_coverage
#SBATCH -o logs/mask_coverage_%j.out

# --ntasks=1 explicitly: an srun/sbatch that defaults to >1 task would run this
# whole script concurrently and race on the output CSVs (docs/qc_context.md).
set -euo pipefail
module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

OAK=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler
: "${LEVEL:=feature_level}"      # feature_level | bipolar | raw_voltage
: "${MASK_LABEL:=}"              # empty -> each subject's own recorded label
: "${SUBJECTS:=}"                # empty -> every subject with a run_info record

MASK_FLAG=""
[ -n "$MASK_LABEL" ] && MASK_FLAG="--mask-label $MASK_LABEL"
SUBJ_FLAG=""
[ -n "$SUBJECTS" ] && SUBJ_FLAG="--subjects $SUBJECTS"

echo "=== mask coverage audit: level=$LEVEL mask=${MASK_LABEL:-<per-subject recorded>}" \
     "subjects=${SUBJECTS:-<all>}"
python -W ignore -m ieeg_ehr.qc.report_mask_coverage \
    --level-root "$OAK/qc/$LEVEL" $MASK_FLAG $SUBJ_FLAG

echo "DONE $LEVEL"

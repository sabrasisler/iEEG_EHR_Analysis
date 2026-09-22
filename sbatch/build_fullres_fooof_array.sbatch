#!/bin/bash
# FOOOF (BG.3) across the cohort, one array task per subject, one ARM per submit.
#
# Reads the epoch-MEAN view (~1.4 GB), never the per-window cache (~206 GB) and
# never raw. That is what makes a parameter sweep affordable: every arm re-reads
# 1.4 GB instead of re-extracting.
#
#   sbatch --array=0-80%16 --export=ALL,ARM=fixed sbatch/build_fullres_fooof_array.sbatch
#   sbatch --array=0-80%16 --export=ALL,ARM=knee  sbatch/build_fullres_fooof_array.sbatch
#
# fixed 1-45 Hz is PRIMARY (below the first 60 Hz harmonic, so it never meets
# line noise); knee 1-150 Hz is the broadband arm. Neither reaches 250 Hz: at
# 500 Hz sampling that IS Nyquist and the top of the range is anti-alias rolloff.
#
# Sized from a measured sub-256 fixed-arm run: 24,238 channel-epochs in 453 s at
# well under 8 GB. The knee arm fits 299 frequencies rather than 89, so ~3x that.
#
#SBATCH -p normal,owners
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH -J fullres_fooof
#SBATCH -o logs/fullres_fooof_%A_%a.out
set -euo pipefail
module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$HOME/bin:$PATH"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
: "${ARM:=fixed}"
OAK=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler
SUBJ=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$OAK/cohorts/subjects_fullres_built.txt")
[ -n "$SUBJ" ] || { echo "no subject at index $SLURM_ARRAY_TASK_ID"; exit 1; }
echo "task $SLURM_ARRAY_TASK_ID -> sub-$SUBJ arm=$ARM"
python -m ieeg_ehr.views.build_pain_epoch_fooof \
  --subjects "$SUBJ" --session all --arm "$ARM" \
  --normalization none --mask-level bipolar \
  --mask-label "std10_rv-gross-std3_satmargin15_sw_logz4"
echo "DONE $SUBJ $ARM"

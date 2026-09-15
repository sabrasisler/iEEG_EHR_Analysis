#!/bin/bash
#SBATCH -p normal,owners
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24GB
#SBATCH -J fullres_mean
#SBATCH -o logs/fullres_mean_%A_%a.out
set -euo pipefail
module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$HOME/bin:$PATH"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
OAK=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler
SUBJ=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$OAK/cohorts/subjects_fullres_built.txt")
[ -n "$SUBJ" ] || { echo "no subject at index $SLURM_ARRAY_TASK_ID"; exit 1; }
echo "task $SLURM_ARRAY_TASK_ID -> sub-$SUBJ"
# Bipolar mask: the pair-level mask, applied at load time as a join. Raw log
# power, NO normalization -- this artifact is what FOOOF and the slope fit read.
python -m ieeg_ehr.views.build_pain_epoch_fullres_mean \
  --subjects "$SUBJ" --session all \
  --normalization none --mask-level bipolar \
  --mask-label "std10_rv-gross-std3_satmargin15_sw_logz4"
echo "DONE $SUBJ"

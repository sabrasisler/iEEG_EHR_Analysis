#!/bin/bash
#SBATCH -p ckeller1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH -J pc_trajectory
#SBATCH -o logs/pc_trajectory_%A_%a.out
# One task per subject. SUBJECTS is a space-separated list, or "all" for the
# reference run's subjects that are still discovery (53):
#   sbatch --array=0-3 --export=ALL,SUBJECTS="183 196 210 143" sbatch/pain_change_trajectory_array.sbatch
#   sbatch --array=0-52 --export=ALL,SUBJECTS=all sbatch/pain_change_trajectory_array.sbatch
set -euo pipefail
module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$HOME/bin:$PATH"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
OAK=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler
REFERENCE_RUN=${REFERENCE_RUN:-$OAK/analysis/pain/pain_change/bandpower/painchange_20261006-211218}
python -m ieeg_ehr.features.build_pain_change_trajectory \
  --subjects ${SUBJECTS:-all} --task-index "$SLURM_ARRAY_TASK_ID" \
  --reference-run "$REFERENCE_RUN" --roi-scheme roi_v3_ins \
  --mask-level bipolar --mask-label "std10_rv-gross-std3_satmargin15_sw_logz4"

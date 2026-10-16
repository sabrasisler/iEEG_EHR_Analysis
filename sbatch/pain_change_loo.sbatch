#!/bin/bash
#SBATCH -p ckeller1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=32GB
#SBATCH -J pc_loo
#SBATCH -o logs/pc_loo_%j.out
# Leave-one-subject-out refits of a pain_change run, exported as prediction weights:
#   sbatch sbatch/pain_change_loo.sbatch
# Score with them afterwards:
#   sbatch --dependency=afterok:<this job> --export=ALL,TRAJECTORY_DIR=<...>,WEIGHTS=loo_painchange_20261006-211218 \
#     sbatch/pain_change_prediction.sbatch
set -euo pipefail
module load math R/4.4.2
module load python/3.12
export R_LIBS_USER="$GROUP_HOME/R/4.4.2"
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$HOME/bin:$PATH"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
OAK=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler
REFERENCE_RUN=${REFERENCE_RUN:-$OAK/analysis/pain/pain_change/bandpower/painchange_20261006-211218}
# pain_change_loo prints its run directory as its last stdout line.
RUN_DIR=$(python -m ieeg_ehr.analysis.pain_change_loo --reference-run "$REFERENCE_RUN" \
  --roi-scheme roi_v3_ins --mask-level bipolar \
  --mask-label "std10_rv-gross-std3_satmargin15_sw_logz4" | tail -1)
echo "RUN_DIR=$RUN_DIR"
python -m ieeg_ehr.analysis.pain_change_prediction weights --source-run "$RUN_DIR"

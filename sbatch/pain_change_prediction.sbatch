#!/bin/bash
#SBATCH -p ckeller1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH -J pc_prediction
#SBATCH -o logs/pc_prediction_%j.out
# Score an extraction with one weights file, then draw every figure:
#   sbatch --export=ALL,TRAJECTORY_DIR=<features/pain/pain_change_trajectory/...>,WEIGHTS=insample_painchange_20261006-211218 \
#     sbatch/pain_change_prediction.sbatch
# After the extraction array:  sbatch --dependency=afterok:<array_job_id> --export=... sbatch/pain_change_prediction.sbatch
set -euo pipefail
module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$HOME/bin:$PATH"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
python -m ieeg_ehr.analysis.pain_change_prediction score \
  --trajectory-dir "$TRAJECTORY_DIR" --weights "$WEIGHTS" --run-name "${RUN_NAME:-painpred}"

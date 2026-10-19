#!/bin/bash
#SBATCH -p ckeller1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH -J pc_shifted_z
#SBATCH -o logs/pc_shifted_z_%A_%a.out
# Per-assessment z tables over [t - SHIFT - 5, t - SHIFT) min, one task per discovery
# subject with stored z tables (60), validated against the stored [t - 5, t) tables.
# Measured 2026-10-19: 43 s for sub-019's one session.
#   sbatch --array=0-59 --export=ALL,SHIFT=5 sbatch/build_pain_change_shifted_z_array.sbatch
set -euo pipefail
module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$HOME/bin:$PATH"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
python -m ieeg_ehr.features.build_pain_change_shifted_z \
  --subjects ${SUBJECTS:-all} --task-index "$SLURM_ARRAY_TASK_ID" --shift-min "${SHIFT:-5}" \
  --mask-level bipolar --mask-label "std10_rv-gross-std3_satmargin15_sw_logz4"

#!/bin/bash
#SBATCH -p ckeller1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH -J pain_change
#SBATCH -o logs/pain_change_%j.out
# After build_fullres_zscore_array.sbatch finishes:
#   sbatch --dependency=afterok:<array_job_id> sbatch/pain_change.sbatch
set -euo pipefail
module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$HOME/bin:$PATH"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
# pain_change prints its run directory as its last stdout line.
RUN_DIR=$(python -m ieeg_ehr.analysis.pain_change --roi-scheme roi_v3_ins \
  --mask-level bipolar --mask-label "std10_rv-gross-std3_satmargin15_sw_logz4" | tail -1)
echo "RUN_DIR=$RUN_DIR"
python -m ieeg_ehr.analysis.plot_pain_change_heatmap --run-dir "$RUN_DIR"

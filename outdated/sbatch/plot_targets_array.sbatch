#!/bin/bash
#SBATCH -p ckeller1
#SBATCH --qos=high_p
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --array=0-37
#SBATCH -J plot_targets_padded30
#SBATCH -o logs/plot_targets_padded30_%A_%a.out

module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -m ieeg_ehr.qc.plot_flagged_runs \
    --exact-csv qc_scripts/plot_targets_padded30.csv \
    --exact-index "$SLURM_ARRAY_TASK_ID" \
    --output-dir /oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/analysis/qc_variance_padded30

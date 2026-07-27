#!/bin/bash
#SBATCH -p normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --array=0-4%5
#SBATCH -J qc_raw_voltage_oom_retry5
#SBATCH -o /home/groups/ckeller1/sisler/iEEG_EHR_Analysis/logs/qc_raw_voltage_oom_retry5_%A_%a.out

module load python/3.12
source /home/groups/ckeller1/venvs/ieeg_ehr_analysis/bin/activate
cd /home/groups/ckeller1/sisler/iEEG_EHR_Analysis

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

SUBJECT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" qc_scripts/subjects_oom_retry5.txt)
echo "Array task $SLURM_ARRAY_TASK_ID -> subject $SUBJECT"
python -m qc_scripts.run_pipeline --subjects "$SUBJECT" \
    --level-root /oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/analysis/qc/raw_voltage

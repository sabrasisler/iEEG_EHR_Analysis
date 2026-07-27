#!/bin/bash
#SBATCH -p ckeller1
#SBATCH --qos=high_p
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --array=0-19%6
#SBATCH -J qc_pipeline_n20
#SBATCH -o logs/run_pipeline_n20_%A_%a.out

module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate

: "${SUBJECTS_FILE:=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/cohorts/subjects_qc_raw_voltage_normal.txt}"
SUBJECT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SUBJECTS_FILE")
echo "Array task $SLURM_ARRAY_TASK_ID -> subject $SUBJECT"
python -m ieeg_ehr.qc.run_pipeline --subjects "$SUBJECT"

#!/bin/bash
#SBATCH -p ckeller1
#SBATCH --qos=high_p
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --array=0-19%6
#SBATCH -J qc_session_rail_n20
#SBATCH -o /home/groups/ckeller1/sisler/iEEG_EHR_Analysis/qc_scripts/logs/session_rail_n20_%A_%a.out

module load python/3.12
source /home/groups/ckeller1/venvs/ieeg_ehr_analysis/bin/activate
cd /home/groups/ckeller1/sisler/iEEG_EHR_Analysis

SUBJECT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" qc_scripts/subjects_n20.txt)
echo "Array task $SLURM_ARRAY_TASK_ID -> subject $SUBJECT"
python -m qc_scripts.run_pipeline --subjects "$SUBJECT" \
    --output-dir /oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/analysis/qc_session_rail

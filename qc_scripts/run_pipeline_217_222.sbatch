#!/bin/bash
#SBATCH -p normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH -J qc_pipeline_217_222
#SBATCH -o /home/groups/ckeller1/sisler/iEEG_EHR_Analysis/qc_scripts/logs/run_pipeline_217_222_%j.out

module load python/3.12
source /home/groups/ckeller1/venvs/ieeg_ehr_analysis/bin/activate
cd /home/groups/ckeller1/sisler/iEEG_EHR_Analysis
python -m qc_scripts.run_pipeline --subjects 217,222

#!/bin/bash
#SBATCH -p normal
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH -J qc_raw_voltage_sub256
#SBATCH -o /home/groups/ckeller1/sisler/iEEG_EHR_Analysis/logs/qc_raw_voltage_sub256_retry_%j.out

module load python/3.12
source /home/groups/ckeller1/venvs/ieeg_ehr_analysis/bin/activate
cd /home/groups/ckeller1/sisler/iEEG_EHR_Analysis

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python -m qc_scripts.run_pipeline --subjects 256 \
    --level-root /oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/analysis/qc/raw_voltage

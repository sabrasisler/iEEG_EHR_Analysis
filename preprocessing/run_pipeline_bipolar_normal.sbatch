#!/bin/bash
#SBATCH -p normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --array=0-19%6
#SBATCH -J bipolar_reref_psd
#SBATCH -o /home/groups/ckeller1/sisler/iEEG_EHR_Analysis/logs/bipolar_reref_psd_%A_%a.out

# Mem note (see plan): this pass ALSO holds a transient bipolar_v array + the
# PSD output array in memory beyond what run_pipeline.py needs. Start at 32GB
# like the raw_voltage pass and watch seff/sacct on the first array tasks;
# bump to 48GB if OOMs appear -- don't preemptively over-provision.

module load python/3.12
source /home/groups/ckeller1/venvs/ieeg_ehr_analysis/bin/activate
cd /home/groups/ckeller1/sisler/iEEG_EHR_Analysis

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

SUBJECT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" preprocessing/subjects_bipolar_normal.txt)
echo "Array task $SLURM_ARRAY_TASK_ID -> subject $SUBJECT"
python -m preprocessing.run_pipeline_bipolar --subjects "$SUBJECT" \
    --level-root /oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/analysis/qc/bipolar \
    --psd-out-root /oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/preprocessed/bipolar_fft

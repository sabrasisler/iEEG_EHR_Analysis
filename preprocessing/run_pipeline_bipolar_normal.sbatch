#!/bin/bash
#SBATCH -p normal
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --array=0-4%6
#SBATCH -J bipolar_reref_psd
#SBATCH -o /home/groups/ckeller1/sisler/iEEG_EHR_Analysis/logs/bipolar_reref_psd_%A_%a.out

# Batches BATCH_SIZE subjects per array task (fewer, longer jobs -- cuts Slurm
# scheduling/env-load overhead per subject vs. one-array-task-per-subject).
# --array's upper bound = ceil(n_subjects / BATCH_SIZE) - 1; recompute it
# whenever preprocessing/subjects_bipolar_normal.txt or BATCH_SIZE changes:
#   python3 -c "import math; print(math.ceil(N_SUBJECTS/BATCH_SIZE)-1)"
# --time bumped to 24h (was 12h) since each task now processes multiple
# subjects sequentially; watch seff/sacct on the first few tasks and adjust.

# Mem note (see plan): this pass ALSO holds a transient bipolar_v array + the
# PSD output array in memory beyond what run_pipeline.py needs. Start at 32GB
# like the raw_voltage pass and watch seff/sacct on the first array tasks;
# bump to 48GB if OOMs appear -- don't preemptively over-provision.

module load python/3.12
source /home/groups/ckeller1/venvs/ieeg_ehr_analysis/bin/activate
cd /home/groups/ckeller1/sisler/iEEG_EHR_Analysis

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

BATCH_SIZE=${BATCH_SIZE:-4}
mapfile -t ALL_SUBJECTS < preprocessing/subjects_bipolar_normal.txt
START=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE))
BATCH=("${ALL_SUBJECTS[@]:START:BATCH_SIZE}")
SUBJECTS_CSV=$(IFS=,; echo "${BATCH[*]}")

echo "Array task $SLURM_ARRAY_TASK_ID -> subjects $SUBJECTS_CSV"
python -m preprocessing.run_pipeline_bipolar --subjects "$SUBJECTS_CSV" \
    --level-root /oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/analysis/qc/bipolar \
    --psd-out-root /oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/preprocessed/bipolar_fft

#!/bin/bash
#SBATCH -p normal
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --array=0-25%6
#SBATCH -J bipolar_reref_psd
#SBATCH -o /home/groups/ckeller1/sisler/iEEG_EHR_Analysis/logs/bipolar_reref_psd_%A_%a.out

# Batches BATCH_SIZE subjects per array task (fewer, longer jobs -- cuts Slurm
# scheduling/env-load overhead per subject vs. one-array-task-per-subject).
# preprocessing/subjects_bipolar_normal.txt has 104 subjects (every sub-*/
# folder under $RAW_DIR that actually contains >=1 raw .nwb file -- 3 folders,
# 204/243/246, exist but are empty and were excluded). At BATCH_SIZE=4 that's
# ceil(104/4)-1 = 25, hence --array=0-25%6. Recompute this bound whenever the
# subject list or BATCH_SIZE changes:
#   python3 -c "import math; print(math.ceil(N_SUBJECTS/BATCH_SIZE)-1)"
# --time bumped to 24h (was 12h) since each task now processes multiple
# subjects sequentially; watch seff/sacct on the first few tasks and adjust.

# --cpus-per-task=8 + --n-workers 8: the Welch PSD step parallelizes across
# channels (ProcessPoolExecutor, see bipolar_reref.compute_welch_log_bins).
# MEASURED on real data (sub-085, 83 runs, 8 workers): MaxRSS ~48GB (vs
# ~16-17GB single-threaded on the same subject) -- the process pool has real
# fixed overhead (each worker loads the full scipy/numpy/BLAS stack). --mem
# bumped 48GB->64GB for a safety margin above that measured peak, since 48GB
# was uncomfortably close to the observed usage. Keep OMP/OPENBLAS/
# MKL_NUM_THREADS=1 per process to avoid oversubscribing -- outer-level
# parallelism is the process pool, not BLAS-level threading. Watch
# seff/sacct on the first real array tasks and adjust further if needed.

module load python/3.12
source /home/groups/ckeller1/venvs/ieeg_ehr_analysis/bin/activate
cd /home/groups/ckeller1/sisler/iEEG_EHR_Analysis

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

BATCH_SIZE=${BATCH_SIZE:-4}
N_WORKERS=${N_WORKERS:-8}
SUBJECTS_FILE=${SUBJECTS_FILE:-preprocessing/subjects_bipolar_normal.txt}
mapfile -t ALL_SUBJECTS < "$SUBJECTS_FILE"
START=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE))
BATCH=("${ALL_SUBJECTS[@]:START:BATCH_SIZE}")
SUBJECTS_CSV=$(IFS=,; echo "${BATCH[*]}")

echo "Array task $SLURM_ARRAY_TASK_ID -> subjects $SUBJECTS_CSV (n_workers=$N_WORKERS)"
python -m preprocessing.run_pipeline_bipolar --subjects "$SUBJECTS_CSV" \
    --level-root /oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/analysis/qc/bipolar \
    --psd-out-root /oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/preprocessed/bipolar_fft \
    --n-workers "$N_WORKERS"

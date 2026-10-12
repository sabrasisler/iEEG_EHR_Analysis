#!/bin/bash
# Continuous bipolar PSD on the native 0.5 Hz grid, one NWB per run, into
# preprocessed/bipolar_psd/ (DECISIONS 2026-10-12). One array task per line of
# TASK_FILE, which `run_pipeline_bipolar plan` writes.
#
# Usage (from the repo root -- the log path is relative):
#   C=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/cohorts
#   TASK_FILE=$C/bipolar_psd_fullres_tasks_discovery.txt \
#       sbatch --array=0-<n_tasks - 1>%<cap> sbatch/run_pipeline_bipolar_normal.sbatch
#
# VARIANCE=1 also rebuilds the bipolar variance QC metric for every subject in a
# task. Plan with --whole-subjects for that, or the task refuses to start.
#
# Re-submitting the same TASK_FILE is safe: finished runs are skipped.
#
# RESOURCES -- one CPU, because the work is one serial spectrogram per channel
# and the parallelism is the array. Memory is one 300-window block, independent
# of run length. Replace these with the pilot's measured values.
#
#SBATCH -p normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6GB
#SBATCH -J bipolar_psd
#SBATCH -o logs/bipolar_psd_%A_%a.out

set -euo pipefail
module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate

# The parallelism is the array. BLAS threading would oversubscribe the one core.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# So git provenance is recordable on the node: the login node's git is 1.8.3.1
# and cannot read a worktree gitfile.
export PATH="$HOME/bin:$PATH"
# So the SUBMITTING checkout runs, not whatever state the editable install
# points at.
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"

: "${TASK_FILE:?set TASK_FILE to a task list written by run_pipeline_bipolar plan}"
VARIANCE_FLAG=""
[ "${VARIANCE:-0}" = "1" ] && VARIANCE_FLAG="--variance-metrics"

python -m ieeg_ehr.preprocessing.run_pipeline_bipolar run \
    --task-file "$TASK_FILE" --task-index "$SLURM_ARRAY_TASK_ID" $VARIANCE_FLAG

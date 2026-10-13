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
# RESOURCES -- MEASURED on the 2026-10-12 pilot (job 47574242 + a re-run of
# sub-208 for getrusage).
#   --cpus-per-task=1  one serial spectrogram per channel. The parallelism is
#       the array.
#   --mem=3GB  process peak 1.27 GB on the worst discovery run (sub-208, 231
#       pairs @ 1000 Hz). Memory is one 300-window block, so it scales with
#       channels x sfreq and not with run length. 3 GB covers 231 pairs @ 2000 Hz
#       (~2.3 GB). Slurm's MaxRSS reads ~8 GB because it counts page cache from
#       the raw reads and NWB writes, which the kernel reclaims.
#   --time=01:30:00  ~25 MB/s of raw NWB per task (sub-208: 5.0 GB in 199-209 s),
#       so a 45 GB `plan --budget-gb 45` task is ~30 min. A timeout only costs a
#       resubmit, since finished runs are skipped.
#
#SBATCH -p normal
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=3GB
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

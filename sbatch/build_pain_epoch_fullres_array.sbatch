#!/bin/bash
# Build the FULL-RESOLUTION pain epoch PSD cache. One array task per subject.
#
# Emits into features/pain/psd_epochs_fullres/epoch-5min-pre/:
#   cache/sub-XXX_ses-YY_epochs.parquet      per-window log-power, WIDE
#                                            (one row per window x channel, 499
#                                            float32 frequency columns)
#   epoch_defs/sub-XXX_ses-YY_defs.parquet   copied verbatim from psd_epochs
#   channel_meta/sub-XXX_ses-YY_channels.parquet   ditto
#
# Same epochs, same 2s/1s grid, same bipolar pairs as psd_epochs; the frequency
# axis is the native 0.5 Hz FFT grid over 1-250 Hz instead of 50 log bins. The
# cache is RAW: no QC mask, no line-noise notch, no averaging, no normalization,
# no frequency binning. All of those are view-layer concerns -- which is the
# entire point (DECISIONS 2026-09-15).
#
# READS RAW NWB, not preprocessed/bipolar_fft: the full-resolution PSD is
# computed inside bipolar_reref._welch_one_channel and immediately discarded by
# the binning step, so it exists nowhere on disk. Epoch-restricted, so this reads
# ~301 s per pain rating rather than whole runs -- 3,710 epochs instead of 7,902
# runs, which is why this is a sub-hour job where the continuous extraction was
# 26 tasks x 24 h.
#
# RUN THE GATE FIRST, and do not submit this until it passes:
#   python -m ieeg_ehr.features.fullres_psd_audit --subjects 019 --n-epochs 3
# Check A must be BIT-EXACT against the existing 50-bin cache. That single
# comparison proves the (n_windows+1)-second read length, the epoch offset, the
# pair ordering, detrend, scaling and the log transform.
#
# --session all so subjects with two sessions (sub-255, sub-209, sub-197) get
# both. Task 0 additionally writes the base-unit manifest.json, whose freqs_hz is
# the SINGLE SOURCE OF TRUTH for the frequency axis -- the cache columns are
# positional (f000..f498) and nothing else on disk says what f000 is in Hz.
#
# RESOURCES -- replace these with measured values from the smoke test's seff
# before a full submission. Reasoning for the starting point:
#   --cpus-per-task=2  NO process pool, deliberately. The spectrogram is ~2 s per
#       epoch serially, and run_pipeline_bipolar_normal.sbatch measured the pool
#       costing 48 GB RSS against 16 GB single-threaded for no useful gain here.
#   --mem=16GB  MEASURED flat at 1.76 GB across epochs of sub-256 (199 pairs @
#       2000 Hz, the worst case in the cohort), so this is ~9x headroom. The first
#       attempt sat at 16.00 GB of 16.00 GB -- that was pyarrow's DEFAULT
#       dictionary encoding building ~499 float-column hash tables per row group,
#       not the spectrogram and not an accumulation. Fixed by the encoding in
#       `parquet_options` (DECISIONS 2026-09-15), which also cut the cohort
#       footprint from 465 GB to 226 GB. Writes are streamed one row group per
#       epoch, so total epoch count does not enter memory.
#   --time=04:00:00  sub-019 (35 pairs, 49 epochs) is minutes; sub-256 (199
#       pairs, 137 epochs) is the worst case. Generous because the cost is
#       dominated by Lustre read latency, which varies with cluster load.
#
# SUBJECT LIST: cohorts/subjects_pain_epoch_fullres.txt -- the 81 subjects that
# actually have a 50-bin epoch_defs file (83 subject-sessions; sub-255/209/197
# have two each). Derived from what is ON DISK rather than reused from
# subjects_pain_epoch_cache.txt, which lists 87 candidates: 120, 122, 138, 212,
# 235 and 259 never produced a 50-bin unit, so those tasks would have nothing to
# read. The extractor skips a missing defs file with a warning rather than
# failing, so a stale list degrades to wasted tasks, not a broken array.
#
# Usage (submit FROM THE REPO ROOT -- the log path is relative):
#   sbatch --array=0-80%12 sbatch/build_pain_epoch_fullres_array.sbatch
#
#SBATCH -p normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH -J pain_epoch_fullres
#SBATCH -o logs/pain_epoch_fullres_%A_%a.out

set -euo pipefail
module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate

# The outer parallelism is the array; BLAS threading would oversubscribe the
# 2 cores requested above.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# So git provenance is recordable on the node: the login node's git is 1.8.3.1
# and cannot read a worktree gitfile, which records the commit as "no-git".
export PATH="$HOME/bin:$PATH"
# So the SUBMITTING checkout runs, not whatever state the editable install
# points at.
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"

OAK=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler
: "${SUBJECTS_FILE:=$OAK/cohorts/subjects_pain_epoch_fullres.txt}"

SUBJECT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SUBJECTS_FILE")
[ -n "$SUBJECT" ] || { echo "No subject at array index $SLURM_ARRAY_TASK_ID in $SUBJECTS_FILE"; exit 1; }
echo "Array task $SLURM_ARRAY_TASK_ID -> subject $SUBJECT (from $SUBJECTS_FILE)"

MANIFEST_FLAG=""
[ "$SLURM_ARRAY_TASK_ID" -eq 0 ] && MANIFEST_FLAG="--write-manifest"

python -m ieeg_ehr.features.build_pain_epoch_fullres_psd \
    --subjects "$SUBJECT" --session all $MANIFEST_FLAG

echo "DONE $SUBJECT"

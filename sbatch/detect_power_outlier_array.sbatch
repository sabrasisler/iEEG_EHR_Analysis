#!/bin/bash
# P2.1 metric stage: feature-level power-outlier metrics. One array task per subject.
#
# Emits into qc/feature_level/metrics/, scoped by the raw-voltage mask whose
# excluded windows are dropped from the baseline (rv-<MASK_LABEL>/):
#   baseline/    per (channel, bin): mean/std of log-power + counts + degenerate
#   per_window/  per (run, channel, window): z order statistics, SPARSE
#   summary/     per (run, channel): denominators + flagged counts at (K, B)
#   zhist/       per (run, channel, rv_excluded): z histogram on a fixed grid
#   run_info/    per subject/session: params + git provenance
#
# Reads only preprocessed/bipolar_fft + the raw-voltage mask CSVs, so this has NO
# dependency on the pain epoch cache and can run alongside it (it does read the
# same NWBs, so expect some Lustre contention if both arrays are live).
#
# K and B are NOT set here -- they live in config/feature_qc_params.py
# (FEATURE_Z_THRESH=5.0, FEATURE_BIN_FRAC=0.20) and are recorded in every
# sidecar. This stage stores order statistics, so K and B can both be re-swept
# later WITHOUT re-running this job; only a B outside FEATURE_BIN_FRAC_GRID or a
# K below FEATURE_METRIC_STORE_FLOOR needs a re-run.
#
# Sized from measured dev runs (mask=gross-std3_satmargin15_sw_logz4):
#   sub-039  85 ch /  0.79M channel-windows   ~8s     2.7 MB per_window
#   sub-071  35 ch /  6.4M  channel-windows  ~54s    14 MB
#   sub-085  63 ch / 27.3M  channel-windows  ~3m35s  76 MB
# The largest subjects (~199 pairs) should be well inside the walltime. Memory is
# driven by CHUNK_WINDOWS x n_pairs x n_bins for the float64 working copy (~330 MB
# at 4096x200x50) plus the accumulating sparse rows, hence 32GB for headroom on an
# unusually artifacty subject.
#
# Usage (submit FROM THE REPO ROOT):
#   sbatch --array=0-86%12 sbatch/detect_power_outlier_array.sbatch
#   sbatch --array=0-86%12 --export=ALL,MASK_LABEL=gross-std3_satmargin15_sw \
#          sbatch/detect_power_outlier_array.sbatch     # the other P0.1 candidate
#
#SBATCH -p normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH -J feat_qc_power
#SBATCH -o logs/feat_qc_power_%A_%a.out

set -euo pipefail
module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate

# Single-threaded numpy: the parallelism here is the Slurm array, not BLAS. The
# cpus-per-task=4 is to buy the 32GB (normal caps mem at 8GB/core), not threads.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

OAK=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler
# Same cohort file the P1.1 cache array uses, deliberately: the feature-level QC
# and the epoch cache must cover the same subjects or the mask cannot be applied
# to part of the cache.
: "${SUBJECTS_FILE:=$OAK/cohorts/subjects_pain_epoch_cache.txt}"
# Empty -> the config defaults, i.e. mask level 'bipolar' and the bipolar mask
# label built from FEATURE_BASELINE_BIPOLAR_VARIANCE_LABEL. The bipolar mask is
# the superset ((raw_voltage[anode] | raw_voltage[cathode]) | bipolar_variance)
# and is already keyed on pairs, which is how the PSD is keyed.
: "${MASK_LEVEL:=}"    # 'bipolar' | 'raw_voltage'
: "${MASK_LABEL:=}"    # 'none' -> unmasked baseline (see --allow-unmasked below)

SUBJECT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SUBJECTS_FILE")
[ -n "$SUBJECT" ] || { echo "No subject at array index $SLURM_ARRAY_TASK_ID in $SUBJECTS_FILE"; exit 1; }

MASK_FLAG=""
[ -n "$MASK_LEVEL" ] && MASK_FLAG="$MASK_FLAG --mask-level $MASK_LEVEL"
[ -n "$MASK_LABEL" ] && MASK_FLAG="$MASK_FLAG --mask-label $MASK_LABEL"
echo "Array task $SLURM_ARRAY_TASK_ID -> subject $SUBJECT" \
     "(mask=${MASK_LEVEL:-<config default>}/${MASK_LABEL:-<config default>})"

# NOT --allow-unmasked: a subject/session whose mask is missing is SKIPPED rather
# than given a weaker unmasked baseline that would sit in the same scope directory
# as the properly masked ones. The cohort file has more subjects than the bipolar
# mask covers, so this is load-bearing, not theoretical -- check the logs for
# SKIPPING lines and reconcile before treating the cohort as complete.
python -W ignore -m ieeg_ehr.qc.feature_level.detect_power_outlier \
    --subjects "$SUBJECT" --session all --overwrite $MASK_FLAG

echo "DONE $SUBJECT"

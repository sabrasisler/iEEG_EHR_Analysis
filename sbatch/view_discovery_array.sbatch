#!/bin/bash
#SBATCH --job-name=view_discovery
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --output=logs/view_discovery_%A_%a.out
#SBATCH --array=0-59%16
#
# Build the P1.3 view for every DISCOVERY subject that can actually produce one,
# in BOTH normalizations (z-score and baseline-subtract / "delta power").
#
# Both normalizations in the SAME task, on purpose: they share the expensive part.
# The first pass builds this subject's channel_meta from NWB metadata (~0.3 s x
# n_runs, and some subjects have 100+ runs); the second normalization then reuses
# that cached table and only re-streams the cache. Two separate arrays would pay
# the NWB cost twice.
#
# The subject list is COMPUTED, never a checked-in file:
#     python -m ieeg_ehr.config.cohorts --runnable discovery
# = discovery INTERSECT (has epoch cache) INTERSECT (has bipolar mask) INTERSECT
#   (PSD written by the current windowing design). 60 as of 2026-07-29: 65 discovery,
#   minus 5 with no epoch cache (122 138 212 235 259). Every exclusion is logged with
#   its reason by that command.
#
#   It was 59 on 2026-07-28, the 60th being sub-247, held out because its PSD was
#   being re-extracted off the superseded 60 s design. The timing audit was re-run
#   2026-07-29 09:44 and now finds ZERO stale runs (qc/psd_timing/psd_rerun_runs.csv
#   is header-only), so that re-extraction has landed and sub-247 rejoins.
#
# Four of the 60 (093 154 159 240) have NO Desikan_Killiany_anode labels, so they
# emit an EMPTY region table and a loud error. That is expected, not a failure --
# region-level n is 55 (60 - 4 unlabeled - 1 with no region rows), and the plot step
# simply has no rows from them.
#
# ONE SUBJECT PER TASK so a single bad subject cannot take down the batch; the view
# builder refuses a stale-PSD subject by design, and that must fail one task rather
# than 60.
#
# CHECK THE BOUND, do not trust this comment: the count moves as PSD re-extractions
# and mask rollouts land, and an array that is too SHORT drops subjects silently --
# the in-script guard only catches a task id ABOVE the count, never a count above
# the array.
#
# RESOURCES, against a MEASURED MaxRSS -- not inferred from cache size.
# Job 36201027 ran 5 subjects x 2 normalizations sequentially and peaked at 4.19 GB
# (and 4:20 wall), so per-subject peak is at most that; 8 GB is ~2x headroom. The
# work is effectively single-threaded (only pyarrow decompression threads), so 1 CPU.
#
# 8 GB with 1 CPU matters on `normal`, which allows at most 8 GB/CORE: the previous
# 24 GB / 2 CPU ask was 12 GB/core, so Slurm charged ceil(24/8)=3 cores and at %12
# reserved 36 cores to do ~12 cores of work -- wasteful AND slower to schedule.
#
# --time=00:30:00 rather than 2h for the same reason: measured worst case is a couple
# of minutes (largest subject's channel_meta build dominates), and short jobs backfill
# far better on a partition with 42k cores queued.
#
# Array bound must match the runnable count. Recompute with:
#   python -m ieeg_ehr.config.cohorts --runnable discovery 2>/dev/null | wc -l

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

# Run the code this job was submitted from -- ieeg_ehr is installed editable
# against the main checkout, whose state is not guaranteed.
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

MASK_LABEL=${MASK_LABEL:-std10_rv-gross-std3_satmargin15_sw_logz4}
SPLIT=${SPLIT:-discovery}
ROI_SCHEME=${ROI_SCHEME:-default}
# Both/all normalizations in the SAME task, for the reason in the header: they
# share the expensive channel_meta build. Override to build a subset.
NORMALIZATIONS=${NORMALIZATIONS:-zscore_vs_baseline baseline_subtract}

mapfile -t SUBJECTS < <(python -m ieeg_ehr.config.cohorts --runnable "$SPLIT" 2>/dev/null)

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#SUBJECTS[@]}" ]; then
  echo "ERROR: array task ${SLURM_ARRAY_TASK_ID} >= ${#SUBJECTS[@]} runnable subjects."
  echo "Update --array to 0-$(( ${#SUBJECTS[@]} - 1 ))."
  exit 1
fi

SUBJECT="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
echo "=== task ${SLURM_ARRAY_TASK_ID}: sub-${SUBJECT} | split=${SPLIT} | mask=${MASK_LABEL} ($(date)) ==="

for NORM in $NORMALIZATIONS; do
  echo "--- $NORM"
  # No --out-dir: the builder resolves
  #   features/pain/psd_epochs/epoch-5min-pre/views/<scheme_code>_<config_hash>/
  # itself. Naming it here would mean spelling the config_hash in bash, where it
  # could drift from the one the sidecar records.
  python -m ieeg_ehr.views.build_pain_epoch_view \
      --subjects "$SUBJECT" \
      --split "$SPLIT" \
      --mask-label "$MASK_LABEL" \
      --roi-scheme "$ROI_SCHEME" \
      --normalization "$NORM"
done

echo "=== done ($(date)) ==="

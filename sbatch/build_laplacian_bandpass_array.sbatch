#!/bin/bash
#SBATCH --job-name=lap_bandpass
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --output=logs/lap_bandpass_%A_%a.out
#SBATCH --array=0-44%20
#
# LAPLACIAN BAND-RMS extraction on pain epochs -- one array task per subject.
# (PLANNING "Pain state decoding", DECISIONS 2026-09-09)
#
#   sbatch sbatch/build_laplacian_bandpass_array.sbatch
#   SUBJECTS_OVERRIDE=019 sbatch --array=0 sbatch/build_laplacian_bandpass_array.sbatch
#
# WHAT THIS IS FOR. It reproduces Prasad et al. 2025's feature pipeline off the
# RAW TIME-DOMAIN signal: notch -> Laplacian re-reference -> six 8th-order
# zero-phase Butterworth bandpass filters -> RMS over the 5-min window -> log10.
# Every other difference between our decoder and theirs turned out to be
# reproducible from the stored PSD cache; the RE-REFERENCE is not, because the
# bipolar trace was never persisted. So this isolates that one variable.
#
# EPOCHS ONLY, not a continuous feature family. That is a deliberate departure
# from architecture.md's layer model: the continuous route would cost ~40x the
# I/O for a question that only ever reads 5-min pre-report windows, and V1 raw
# NWB is chunked along TIME so a windowed read is cheap.
#
# RESOURCES -- MEASURED, not guessed (per epoch, then x48 epochs):
#
#      40 ch @  500 Hz :  1.7 s ->  1.3 min/subject
#     174 ch @ 1000 Hz : 15.1 s -> 12.1 min/subject
#     250 ch @ 1000 Hz : 22.2 s -> 17.8 min/subject
#     200 ch @ 2000 Hz : 34.3 s -> 27.5 min/subject
#
# plus ~2 min/subject of I/O. 2h is generous for the worst case. MEMORY is the
# binding constraint, not time: one 5-min window at 250 ch x 1000 Hz is 600 MB as
# float64, and sosfiltfilt holds several copies, so 16 GB (2 cores' worth on
# `normal`, whose ceiling is 8 GB/core) rather than the usual 8.
#
# PARTITION. `normal` because it had 513 idle cores while ckeller1's 24 were
# fully committed. Falls back by hand -- resubmit with -p ckeller1 if that flips.
#
# Array bound must match the eligible-subject count. Recompute with:
#   python -m ieeg_ehr.decoding.eligible --view-dir <psd view dir> --count

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

MASK_LABEL=${MASK_LABEL:-gross-std3_satmargin15_sw_logz4}
SPLIT=${SPLIT:-discovery}
REREF=${REREF:-laplacian}
SESSION=${SESSION:-01}

# The MONOPOLAR raw-voltage mask, not the bipolar one: a Laplacian channel is
# built from three CONTACTS and cannot be keyed by a pair name. The extractor
# ORs the mask across a channel's three contributing contacts.

if [ -n "${SUBJECTS_OVERRIDE:-}" ]; then
  read -r -a SUBJECTS <<< "${SUBJECTS_OVERRIDE}"
else
  # The same 45 subject-sessions the decoder runs on, so the two feature sources
  # cover exactly the same cohort and can be compared PAIRWISE.
  VIEW_ROOT=$(python -c "
from ieeg_ehr import config
print(config.pain_epoch_unit_dir() / 'views')")
  VIEW_DIR=$(ls -d "${VIEW_ROOT}"/chan-paper6-raw_* 2>/dev/null | head -1)
  mapfile -t UNITS < <(python -m ieeg_ehr.decoding.eligible --view-dir "${VIEW_DIR}")
  SUBJECTS=("${UNITS[@]%%_*}")
fi

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#SUBJECTS[@]}" ]; then
  echo "ERROR: array task ${SLURM_ARRAY_TASK_ID} >= ${#SUBJECTS[@]} subjects."
  echo "Update --array to 0-$(( ${#SUBJECTS[@]} - 1 ))."
  exit 1
fi

SUBJECT="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
echo "=== task ${SLURM_ARRAY_TASK_ID}: sub-${SUBJECT} | reref=${REREF} | mask=${MASK_LABEL} ($(date)) ==="

python -m ieeg_ehr.features.build_pain_epoch_bandpass \
    --subjects "${SUBJECT}" \
    --session "${SESSION}" \
    --split "${SPLIT}" \
    --reref "${REREF}" \
    --mask-label "${MASK_LABEL}"

echo "=== done ($(date)) ==="

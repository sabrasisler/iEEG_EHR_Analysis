#!/bin/bash
#SBATCH --job-name=lap_bandpass
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --output=logs/lap_bandpass_%A_%a.out
#SBATCH --array=0-44%20
#
# LAPLACIAN BAND-RMS extraction -- the paper's feature pipeline off the RAW
# time-domain signal, one array task per subject-session.
# (PLANNING "Pain state decoding", DECISIONS 2026-09-09)
#
#   sbatch sbatch/build_laplacian_bandpass_array.sbatch
#   SUBJECTS_OVERRIDE=019 sbatch --array=0 sbatch/build_laplacian_bandpass_array.sbatch
#
# WHY THIS EXISTS. Every other difference between our decoder and Prasad et al.
# 2025's is reproducible from the stored PSD cache -- band edges, the
# RMS-vs-geometric-mean epoch collapse, band-power weighting. The RE-REFERENCE is
# not: run_pipeline_bipolar.py never persisted the bipolar trace, so matching
# their Laplacian montage needs a fresh read of the raw signal. This isolates the
# one variable the cache cannot answer.
#
# PARTITION. `normal`, not ckeller1 -- ckeller1's single 24-core node has been at
# 0 idle cores all day while normal has had 500+. The instruction is to prefer
# normal when it has capacity and fall back to ckeller1 otherwise, so check
# `sh_part` before assuming this is still the right choice.
#
# RESOURCES -- MEASURED. Per-epoch cost of laplacian + notch + 6x 8th-order
# Butterworth + RMS, plus the windowed raw read:
#
#      40 ch @  500 Hz   1.7 s/epoch   ->  ~1.3 min/subject
#     174 ch @ 1000 Hz  15.1 s/epoch   ->  ~12 min/subject
#     250 ch @ 1000 Hz  22.2 s/epoch   ->  ~18 min/subject
#     200 ch @ 2000 Hz  34.3 s/epoch   ->  ~28 min/subject
#
# Smoke-verified on sub-019: 49 epochs x 30 channels x 6 bands = 8,778 rows in
# 167 s (3.4 s/epoch; the gap over the 1.7 s benchmark is NWB open overhead,
# which is paid per epoch and is not worth optimising at this scale). 2h covers
# the worst subject 4x over.
#
# 8GB/1CPU deliberately: `normal` caps at 8 GB/core, so asking for 16 GB with one
# CPU makes Slurm hunt for two cores and the job sits PENDING behind a 67k-core
# queue for no benefit. The largest 5-min slice materialised is 600 MB.
#
# READS V1 (`iEEG_EHR/iEEG_NWB`), and a V2 port is NOT a path swap -- V2 has 3
# subjects, no ehr/ folder (so no pain scores), and its chunking is
# (120 s, 1 channel), which makes the short full-width window this reads the SLOW
# direction. See docs/dataset_v2.md §1 and §4.
#
# Array bound must match the runnable count. Recompute with:
#   python -m ieeg_ehr.config.cohorts --runnable discovery 2>/dev/null | wc -l

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

SPLIT=${SPLIT:-discovery}
REREF=${REREF:-laplacian}
# The MONOPOLAR raw-voltage mask, not the bipolar one: a Laplacian channel is
# built from three contacts and cannot be keyed by a pair name. The extractor
# ORs the mask across those three, the same rule the bipolar path applies to
# anode|cathode.
MASK_LABEL=${MASK_LABEL:-gross-std3_satmargin15_sw_logz4}

if [ -n "${SUBJECTS_OVERRIDE:-}" ]; then
  read -r -a SUBJECTS <<< "${SUBJECTS_OVERRIDE}"
else
  mapfile -t SUBJECTS < <(python -m ieeg_ehr.config.cohorts --runnable "$SPLIT" 2>/dev/null)
fi

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#SUBJECTS[@]}" ]; then
  echo "ERROR: array task ${SLURM_ARRAY_TASK_ID} >= ${#SUBJECTS[@]} runnable subjects."
  echo "Update --array to 0-$(( ${#SUBJECTS[@]} - 1 ))."
  exit 1
fi

SUBJECT="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
echo "=== task ${SLURM_ARRAY_TASK_ID}: sub-${SUBJECT} | reref=${REREF} | mask=${MASK_LABEL} ($(date)) ==="

python -m ieeg_ehr.features.build_pain_epoch_bandpass \
    --subjects "${SUBJECT}" \
    --reref "${REREF}" \
    --mask-label "${MASK_LABEL}"

echo "=== done ($(date)) ==="

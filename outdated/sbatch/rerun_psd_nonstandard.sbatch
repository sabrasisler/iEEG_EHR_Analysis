#!/bin/bash
#SBATCH --job-name=psd_rerun
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=logs/psd_rerun_%A_%a.out
#SBATCH --array=0-2
#
# Re-extract the PSD for the subjects whose stored output came from the SUPERSEDED
# two-level windowing design (60 s outer window of ~59 Welch-averaged 2 s inner
# segments, one spectrum per minute) instead of the current single-level design
# (each 2 s window its own periodogram, stepped 1 s).
#
# Subject list is READ FROM THE AUDIT, not hardcoded, so this cannot drift from
# what qc/psd_timing/ actually found:
#     qc/psd_timing/psd_rerun_subjects.txt   -> 247, 257, 259  (2026-07-28)
#
# Resources copied from run_pipeline_bipolar_normal.sbatch, which measured them.
# ONE SUBJECT PER TASK (not the usual batch of 4): these are large re-extractions
# (84 / 58 / 80 runs) and there is no reason to serialise them behind each other.
#
# --skip-variance-metrics, and this is VERIFIED not assumed: the bipolar variance
# metric is computed on the TIME-DOMAIN signal, and all three subjects' existing
# metric CSVs are on a 2 s grid (window_start_time = 0, 2, 4, 6 ...) even where
# their PSD is 60 s. So the variance metric -- and therefore std10 and the bipolar
# masks derived from it -- are already correct and must NOT be recomputed.
#
# WHAT THIS INVALIDATES (rebuild after it lands):
#   - epoch cache for sub-247 and sub-257 (sub-259 has none yet)
#   - qc/feature_level/ power metrics for all three (they read psd_log_bins)
#   - NOT std10, NOT the bipolar masks (see above)
#
# It OVERWRITES the existing PSD NWBs (run_pipeline_bipolar has no skip-if-exists),
# which is the intent -- the stale files must not survive. ~19 GB is rewritten, and
# sub-259 will GROW substantially: at a 60 s hop its 80 runs occupy only 359 MB,
# versus 12 GB for sub-247's mostly-1 s runs.
#
# Submit from a CLEAN checkout -- see the PYTHONPATH note below.

set -euo pipefail

# ~/bin/git ahead of /usr/bin so git provenance is recordable on the node.
export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Run the code this job was submitted from. `ieeg_ehr` is pip-installed editable
# against the MAIN checkout, so a bare `python -m` would execute whatever state
# that tree happens to be in -- which for a 24 h run that overwrites 19 GB of
# derived data is not good enough.
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
python -c "import ieeg_ehr; print('ieeg_ehr:', ieeg_ehr.__file__)"

DERIV=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler
SUBJECTS_FILE=${SUBJECTS_FILE:-$DERIV/qc/psd_timing/psd_rerun_subjects.txt}
N_WORKERS=${N_WORKERS:-8}

mapfile -t SUBJECTS < "$SUBJECTS_FILE"

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#SUBJECTS[@]}" ]; then
  echo "ERROR: array task ${SLURM_ARRAY_TASK_ID} >= ${#SUBJECTS[@]} subjects in $SUBJECTS_FILE."
  echo "Update --array to 0-$(( ${#SUBJECTS[@]} - 1 ))."
  exit 1
fi

SUBJECT="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
echo "=== task ${SLURM_ARRAY_TASK_ID}: re-extracting PSD for sub-${SUBJECT} ($(date)) ==="
echo "=== subject list: $SUBJECTS_FILE ==="

python -m ieeg_ehr.preprocessing.run_pipeline_bipolar \
    --subjects "$SUBJECT" \
    --level-root "$DERIV/qc/bipolar" \
    --psd-out-root "$DERIV/preprocessed/bipolar_fft" \
    --n-workers "$N_WORKERS" \
    --skip-variance-metrics

echo "=== done ($(date)) ==="
echo "=== NEXT: re-audit to confirm, then rebuild the epoch cache + feature-level metrics ==="
echo "  python -m ieeg_ehr.qc.audit_psd_timing --subjects ${SUBJECT} && \\"
echo "  python -m ieeg_ehr.qc.audit_psd_timing --reduce"

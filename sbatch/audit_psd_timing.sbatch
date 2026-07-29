#!/bin/bash
#SBATCH --job-name=psd_hop_audit
#SBATCH --partition=ckeller1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --output=logs/psd_hop_audit_%A_%a.out
#SBATCH --array=0-96%8
#
# Which windowing design wrote each stored PSD run? One array task per subject,
# metadata-only NWB reads (~0.3 s/run), over all ~6236 files in
# preprocessed/bipolar_fft.
#
# READ-ONLY with respect to every existing artifact: it opens PSD NWBs for metadata
# and writes only into qc/psd_timing/shards/. Nothing touches preprocessed/, the
# 34 GB epoch cache, or any mask.
#
# %8 because ckeller1 is a single 24-core node shared with interactive work.
# --qos=high_p is deliberately NOT set, so this does not consume the 4-job
# high-priority cap CLAUDE.md asks to keep free.
#
# The array bound must cover the subject count (97 as of 2026-07-28 => 0-96). The
# task hard-fails rather than silently no-op-ing if the index is out of range, so a
# stale bound is visible instead of quietly skipping subjects. Get the current count
# with:  python -m ieeg_ehr.qc.audit_psd_timing --list-subjects | wc -l
#
# Then reduce (fast, one process):
#   python -m ieeg_ehr.qc.audit_psd_timing --reduce

set -euo pipefail

# ~/bin/git ahead of /usr/bin so git provenance is recordable ON THE NODE (the
# login node's git 1.8.3.1 cannot read a worktree gitfile -> commit "no-git").
export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

# Run the code this job was submitted from: ieeg_ehr is installed editable against
# the MAIN checkout, so a bare `python -m` would ignore the submitting worktree and
# decouple what ran from the recorded commit hash.
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

mapfile -t SUBJECTS < <(python -m ieeg_ehr.qc.audit_psd_timing --list-subjects)

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#SUBJECTS[@]}" ]; then
  echo "ERROR: array task ${SLURM_ARRAY_TASK_ID} >= ${#SUBJECTS[@]} subjects with PSD output."
  echo "Update --array in this file to 0-$(( ${#SUBJECTS[@]} - 1 ))."
  exit 1
fi

SUBJECT="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
echo "=== task ${SLURM_ARRAY_TASK_ID}: sub-${SUBJECT} ($(date)) ==="

python -m ieeg_ehr.qc.audit_psd_timing --subjects "$SUBJECT"

echo "=== done ($(date)) ==="

#!/bin/bash
#SBATCH --job-name=ins_split
#SBATCH --partition=ckeller1
#SBATCH --qos=high_p
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --output=logs/ins_split_%j.out
#
# Glass brain + split diagnostics for the COARSE anterior/posterior insula split.
# A review figure: nothing is fitted, and the domain model does not read it.
#
# SUBJECTS_FROM should be the domain-model run this split is destined for, so the
# median is taken over exactly the electrodes that model will see -- the
# threshold is the cohort's median, so a different subject list is a different
# split, not the same split drawn differently.
#
#   SUBJECTS_FROM=<domain run dir> sbatch sbatch/insula_split.sbatch
#
# Needs nilearn/nibabel/scikit-learn (already in the shared venv; the glass brain
# outlines ship with nilearn, so this renders with no network).

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${SCRATCH}/mplconfig"

DERIV=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler
SUBJECTS_FROM=${SUBJECTS_FROM:-$DERIV/analysis/pain/bandpower/domain_model/paperbands6hg200-paindomainsv2/domain_mixedlm_20260918-100249}
EXTRA=${EXTRA:-}
DEST=${SCRATCH_RUN:+--scratch}

echo "=== insula A/P split | cohort from $SUBJECTS_FROM | $(date) ==="
python -m ieeg_ehr.analysis.plot_insula_split \
    --subjects-from "$SUBJECTS_FROM" ${DEST} ${EXTRA}
echo "=== done ($(date)) ==="

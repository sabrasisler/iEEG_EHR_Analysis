#!/bin/bash
#SBATCH --job-name=med_pain_coupling
#SBATCH --partition=owners
#SBATCH --qos=normal
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --output=logs/med_pain_coupling_%j.out
#
# Administrations of the four most-administered analgesics, binned by the pain
# score charted in the 30 minutes before the dose.
#
# Separate from med_figures.sbatch because this is a different level-2 QUESTION
# (pain_coupling, not administration_patterns), and the question is a directory
# level rather than a flag on a run -- mixing them in one submission would put
# two questions' output under whichever --question happened to be set.
#
# Input is the 98 MAR exports plus their 99 pain-score siblings: small CSVs, so
# this is seconds. It is on Slurm only because of the standing rule against
# Python on the login node.
#
#   sbatch sbatch/med_pain_coupling.sbatch
#   SCRATCH_RUN=1 sbatch sbatch/med_pain_coupling.sbatch    # throwaway
#   WINDOW=60 sbatch sbatch/med_pain_coupling.sbatch        # wider window
#   STRICT=1 sbatch sbatch/med_pain_coupling.sbatch         # drop same-minute scores
#   NOCLUST=1 sbatch sbatch/med_pain_coupling.sbatch        # Fig 7 sensitivity: drop clustered assessments

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

WINDOW=${WINDOW:-30}
DEST=${SCRATCH_RUN:+--scratch}
STRICT_FLAG=${STRICT:+--strict-prior}
NOCLUST_FLAG=${NOCLUST:+--exclude-clustered}

echo "=== analgesic dosing vs preceding pain score | window=${WINDOW}min ($(date)) ==="

# Fig 5: given a dose, what was the score before it.
python -m ieeg_ehr.med_analysis.plot_pain_score_bars \
    --window-minutes "$WINDOW" ${STRICT_FLAG} ${DEST}

# Fig 7: the mirror -- given an assessment, was a dose given after it. Same
# window and the same underlying pairing, so the two cannot disagree.
python -m ieeg_ehr.med_analysis.plot_pain_score_response \
    --window-minutes "$WINDOW" ${NOCLUST_FLAG} ${DEST}

echo "=== done ($(date)) ==="

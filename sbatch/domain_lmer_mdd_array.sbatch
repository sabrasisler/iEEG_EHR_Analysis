#!/bin/bash
#SBATCH -J lmer_mdd
#SBATCH -p ckeller1
#SBATCH -t 04:00:00
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH --array=0-5
#SBATCH -o logs/lmer_mdd_%A_%a.out
#SBATCH -e logs/lmer_mdd_%A_%a.err
#
# ONE ARRAY TASK PER BAND for the MDD (diagnosis-stratum) domain model in lme4.
#
#   log10_power ~ 0 + domain:dx + domain:dx:NRS_within + domain:NRS_submean
#                 + (1 + NRS_within || ROI) + (1 + NRS_within || subject)
#                 + (1 + NRS_within || subj_roi) + (1 | chan_id)
#
# Band is the parallel axis: the six fits are independent, each ~250-280 s on
# 119,858 rows, and six tasks x 4 cores exactly fills the 24-core ckeller1 node.
# Sequentially this stage is ~28 min; as an array it is one fit's wall-clock.
#
# NO OMNIBUS. Removed at the analyst's instruction; see domain_lmer.R.
#
# EVERY TASK WRITES INTO THE SAME RUN, passed in as RUN_DIR -- without it each
# task timestamps its own directory and there is nothing to collect. Tasks write
# only bands/<band>_*.csv; the run-level tables are written once by the
# dependent --collect-only stage (sbatch/submit_domain_lmer_mdd.sh).

set -euo pipefail
export PATH="$HOME/bin:$PATH"

module load math R/4.4.2
module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export R_LIBS_USER="$GROUP_HOME/R/4.4.2"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

: "${RUN_DIR:?RUN_DIR must be exported by the submit wrapper}"
: "${FRAMES_DIR:?FRAMES_DIR must be exported by the submit wrapper}"

BANDS=(delta theta alpha beta gamma high_gamma)
BAND=${BANDS[$SLURM_ARRAY_TASK_ID]}

echo "=== lmer MDD | band=${BAND} | task=${SLURM_ARRAY_TASK_ID} | $(date) ==="
echo "run dir: ${RUN_DIR}"

python -m ieeg_ehr.analysis.run_domain_lmer \
    --frames-dir "${FRAMES_DIR}" \
    --run-dir "${RUN_DIR}" \
    --bands "${BAND}" \
    --view-scheme paperbands6hg200-paindomainsv3-roiunit-mdd0d-noparcel

echo "=== done ${BAND} ($(date)) ==="

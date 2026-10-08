#!/bin/bash
#
# The pain_change CHANGE-SCORE domain model with medication in lme4, FOUR
# DOMAINS (Control dropped before the fit), as three chained Slurm stages:
# frames -> per-band array -> collect.
#
#   d_z ~ 0 + domain:med + domain:med:d_pain + domain:pain_1_within
#         + domain:gap_h + med_submean
#         + (1 + d_pain || ROI) + (1 + d_pain + med_within || subject)
#         + (1 + d_pain || subj_roi)
#
# Rows are assessment pair x channel. `med` is a dose of the drug set in
# [t1, t2). The level-model counterpart is submit_domain_lmer_med.sh.
#
#   bash sbatch/submit_pain_change_domain_med.sh

set -euo pipefail
cd "$(dirname "$0")/.."

DRUG_SET=${DRUG_SET:-analgesics}
DROP_DOMAIN=${DROP_DOMAIN:-Control}
MASK_LABEL=std10_rv-gross-std3_satmargin15_sw_logz4

STAMP=$(date +%Y%m%d-%H%M%S)
BASE=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/analysis/pain/pain_change/domain_model

# Frames scheme is composed by pain_change_domain from its own arguments. The
# fit scheme carries `-4dom` because Control is dropped at FIT time in R.
FRAMES_SCHEME="paperbands6hg200-paindomainsv3-${DRUG_SET}"
OUT_SCHEME="paperbands6hg200-paindomainsv3-${DRUG_SET}-4dom"

RUN_DIR="${BASE}/${OUT_SCHEME}/domain_lmer_${STAMP}"
FRAMES_RUN="lmer_frames_${STAMP}"
# A glob: analysis_run_dir appends its own timestamp to --run-name.
FRAMES_GLOB="${BASE}/${FRAMES_SCHEME}/${FRAMES_RUN}_*/frames"

mkdir -p logs "${RUN_DIR}/bands"
echo "drug set: ${DRUG_SET} | dropping: ${DROP_DOMAIN}"
echo "run dir:  ${RUN_DIR}"

FRAMES_JOB=$(sbatch --parsable \
  -J pc_dom_frames -p ckeller1 -t 01:00:00 -c 2 --mem=32GB \
  -o logs/pc_dom_frames_%j.out -e logs/pc_dom_frames_%j.err \
  --wrap "set -euo pipefail; \
          export PATH=\$HOME/bin:\$PATH; \
          module load python/3.12; \
          source \$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate; \
          export PYTHONPATH=$(pwd)/src; \
          python -m ieeg_ehr.analysis.pain_change_domain \
            --drug-set ${DRUG_SET} --mask-level bipolar --mask-label ${MASK_LABEL} \
            --run-name ${FRAMES_RUN}")
echo "stage 1 frames:  ${FRAMES_JOB}"

ARRAY_JOB=$(sbatch --parsable \
  --dependency=afterok:"${FRAMES_JOB}" \
  --export=ALL,RUN_DIR="${RUN_DIR}",FRAMES_GLOB="${FRAMES_GLOB}",DROP_DOMAIN="${DROP_DOMAIN}",VIEW_SCHEME="${OUT_SCHEME}" \
  sbatch/domain_lmer_band_array.sbatch)
echo "stage 2 array:   ${ARRAY_JOB}"

COLLECT_JOB=$(sbatch --parsable \
  --dependency=afterok:"${ARRAY_JOB}" \
  -J pc_dom_collect -p ckeller1 -t 00:30:00 -c 2 --mem=16GB \
  -o logs/pc_dom_collect_%j.out -e logs/pc_dom_collect_%j.err \
  --wrap "set -euo pipefail; \
          export PATH=\$HOME/bin:\$PATH; \
          module load math R/4.4.2; module load python/3.12; \
          source \$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate; \
          export PYTHONPATH=$(pwd)/src; \
          export R_LIBS_USER=\$GROUP_HOME/R/4.4.2; \
          FD=\$(ls -1d ${FRAMES_GLOB} | sort | tail -1); \
          python -m ieeg_ehr.analysis.run_domain_lmer \
            --frames-dir \"\$FD\" --run-dir ${RUN_DIR} --collect-only \
            --drop-domain ${DROP_DOMAIN} --view-scheme ${OUT_SCHEME}")
echo "stage 3 collect: ${COLLECT_JOB}"

echo
echo "results: ${RUN_DIR}"

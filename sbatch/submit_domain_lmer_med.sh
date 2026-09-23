#!/bin/bash
#
# The MEDICATION domain model in lme4, FOUR DOMAINS (Control excluded), as three
# chained Slurm stages: frames -> per-band array -> collect.
#
#   log10_power ~ 0 + domain:med + domain:med:NRS_within
#                 + domain:NRS_submean + med_submean
#                 + (1 + NRS_within || ROI)
#                 + (1 + NRS_within + med_within || subject)
#                 + (1 + NRS_within || subj_roi) + (1 | chan_id)
#
# CONTROL IS EXCLUDED FROM THE MODEL, not just the figure: --drop-domain is
# passed through to R, which filters those rows BEFORE fitting, so Occipital and
# Auditory channels never enter. The BH family downstream is 24 cells.
#
# `med` is EPOCH-level, so on-minus-off is a WITHIN-subject contrast -- the
# reason this model is better powered than the dx one, which was between-subject
# at n=51.
#
#   bash sbatch/submit_domain_lmer_med.sh
#
# DRUG_SET defaults to the analgesic union (analyst's choice). Window is 2 h,
# the default every prior medication run used.

set -euo pipefail
cd "$(dirname "$0")/.."

DRUG_SET=${DRUG_SET:-analgesics}
MED_WINDOW_H=${MED_WINDOW_H:-2.0}
DROP_DOMAIN=${DROP_DOMAIN:-Control}

STAMP=$(date +%Y%m%d-%H%M%S)
SCHEME="paperbands6hg200-paindomainsv3-roiunit-${DRUG_SET}-4dom-noparcel"
BASE=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/analysis/pain/bandpower/domain_model/${SCHEME}
RUN_DIR="${BASE}/domain_lmer_${STAMP}"
FRAMES_RUN="lmer_frames_${STAMP}"

# A GLOB: analysis_run_dir appends its OWN timestamp to --run-name, so no path
# composed here can be right. Stages 2 and 3 resolve it.
FRAMES_GLOB="${BASE}/${FRAMES_RUN}_*/frames"

mkdir -p logs "${RUN_DIR}/bands"
echo "drug set: ${DRUG_SET} | window ${MED_WINDOW_H}h | dropping: ${DROP_DOMAIN}"
echo "run dir:  ${RUN_DIR}"

# ---- stage 1: frames -------------------------------------------------------
FRAMES_JOB=$(sbatch --parsable \
  -J lmer_med_frames -p ckeller1 -t 01:00:00 -c 4 --mem=32GB \
  -o logs/lmer_med_frames_%j.out -e logs/lmer_med_frames_%j.err \
  --wrap "set -euo pipefail; \
          export PATH=\$HOME/bin:\$PATH; \
          module load python/3.12; \
          source \$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate; \
          export PYTHONPATH=$(pwd)/src; \
          python -m ieeg_ehr.analysis.run_domain_model \
            --unit roi --roi-scheme pain_domains_v3 \
            --insula-threshold -2.245993821300736 \
            --med-model interaction --drug-set ${DRUG_SET} \
            --med-window-hours ${MED_WINDOW_H} \
            --frames-only --run-name ${FRAMES_RUN}")
echo "stage 1 frames:  ${FRAMES_JOB}"

# ---- stage 2: one fit per band --------------------------------------------
ARRAY_JOB=$(sbatch --parsable \
  --dependency=afterok:"${FRAMES_JOB}" \
  --export=ALL,RUN_DIR="${RUN_DIR}",FRAMES_GLOB="${FRAMES_GLOB}",DROP_DOMAIN="${DROP_DOMAIN}",VIEW_SCHEME="${SCHEME}" \
  sbatch/domain_lmer_band_array.sbatch)
echo "stage 2 array:   ${ARRAY_JOB}"

# ---- stage 3: collect ------------------------------------------------------
COLLECT_JOB=$(sbatch --parsable \
  --dependency=afterok:"${ARRAY_JOB}" \
  -J lmer_med_collect -p ckeller1 -t 00:30:00 -c 2 --mem=16GB \
  -o logs/lmer_med_collect_%j.out -e logs/lmer_med_collect_%j.err \
  --wrap "set -euo pipefail; \
          export PATH=\$HOME/bin:\$PATH; \
          module load math R/4.4.2; module load python/3.12; \
          source \$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate; \
          export PYTHONPATH=$(pwd)/src; \
          export R_LIBS_USER=\$GROUP_HOME/R/4.4.2; \
          FD=\$(ls -1d ${FRAMES_GLOB} | sort | tail -1); \
          python -m ieeg_ehr.analysis.run_domain_lmer \
            --frames-dir \"\$FD\" --run-dir ${RUN_DIR} --collect-only \
            --drop-domain ${DROP_DOMAIN} --view-scheme ${SCHEME}")
echo "stage 3 collect: ${COLLECT_JOB}"

echo
echo "results: ${RUN_DIR}"

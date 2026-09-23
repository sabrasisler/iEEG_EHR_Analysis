#!/bin/bash
#
# Refit the ALL-SUBJECTS and MDD domain models with FOUR domains -- Control
# excluded from the MODEL, not just the figure -- so all three lme4 runs
# (all-subjects, MDD, medication) share one domain set.
#
# Control is dropped by R BEFORE lmer() is called, so Occipital and Auditory
# channels never enter the design matrix and the variance components are
# estimated without them. That is the difference from masking the rows on a
# figure, where those channels still inform the shared random effects.
#
# NO STAGE 1. Both runs reuse frames that already exist: only the FIT changes,
# and frames are a pure function of (view, cohort, roi scheme, dx window) with
# their own provenance.
#
# RUNS ON `owners` WITH A SELF-HEALING TAIL. owners is where nodes are actually
# free and the only partition that preempts. Each array carries --requeue, and a
# reaper runs afterany to resubmit whatever did not land, up to MAX_ATTEMPTS,
# then submits collect. The reaper decides from WHICH OUTPUT FILES EXIST, not
# exit codes, so preemption, failure and death-after-writing all behave.
#
#   bash sbatch/submit_domain_lmer_4dom.sh
#
# PARTITION=normal to use normal instead (no preemption; the reaper is then
# only insurance against a genuine failure).

set -euo pipefail
cd "$(dirname "$0")/.."

PARTITION=${PARTITION:-owners}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-5}
DROP_DOMAIN=${DROP_DOMAIN:-Control}
STAMP=$(date +%Y%m%d-%H%M%S)

DERIV=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/analysis/pain

# label | frames dir (existing) | output scheme dir
JOBS=(
  "allsubj|${DERIV}/bandpower/domain_model/paperbands6hg200-paindomainsv3-roiunit-noparcel/lmer_frames_20260922-182423/frames|${DERIV}/bandpower/domain_model/paperbands6hg200-paindomainsv3-roiunit-4dom-noparcel"
  "mdd|${DERIV}/mdd/domain_model/paperbands6hg200-paindomainsv3-roiunit-mdd0d-noparcel/lmer_frames_20260922-184611_20260922-185005/frames|${DERIV}/mdd/domain_model/paperbands6hg200-paindomainsv3-roiunit-mdd0d-4dom-noparcel"
)

mkdir -p logs
echo "partition: ${PARTITION} | dropping: ${DROP_DOMAIN} | max attempts: ${MAX_ATTEMPTS}"
echo

for spec in "${JOBS[@]}"; do
    IFS='|' read -r LABEL FRAMES_DIR OUT_BASE <<< "${spec}"

    if [ ! -d "${FRAMES_DIR}" ]; then
        echo "FATAL: ${LABEL} frames missing: ${FRAMES_DIR}" >&2
        exit 1
    fi

    VIEW_SCHEME=$(basename "${OUT_BASE}")
    RUN_DIR="${OUT_BASE}/domain_lmer_${STAMP}"
    mkdir -p "${RUN_DIR}/bands"

    ARR=$(sbatch --parsable \
      -J "lmer4_${LABEL}" -p "${PARTITION}" --array=0-5 \
      --requeue --open-mode=append \
      -o "logs/lmer4_${LABEL}_%A_%a.out" -e "logs/lmer4_${LABEL}_%A_%a.err" \
      --export=ALL,RUN_DIR="${RUN_DIR}",FRAMES_GLOB="${FRAMES_DIR}",VIEW_SCHEME="${VIEW_SCHEME}",DROP_DOMAIN="${DROP_DOMAIN}",BANDS="delta,theta,alpha,beta,gamma,high_gamma" \
      sbatch/domain_lmer_band_array.sbatch)

    REAP=$(sbatch --parsable \
      --dependency=afterany:"${ARR}" \
      -J "lmer4_${LABEL}_reap" -p "${PARTITION}" -t 00:15:00 -c 1 --mem=4GB \
      --requeue --open-mode=append \
      -o "logs/lmer4_${LABEL}_reap_%j.out" -e "logs/lmer4_${LABEL}_reap_%j.err" \
      --export=ALL,RUN_DIR="${RUN_DIR}",FRAMES_GLOB="${FRAMES_DIR}",VIEW_SCHEME="${VIEW_SCHEME}",DROP_DOMAIN="${DROP_DOMAIN}",PARTITION="${PARTITION}",ATTEMPT=1,MAX_ATTEMPTS="${MAX_ATTEMPTS}",LABEL="lmer4_${LABEL}" \
      sbatch/domain_lmer_reaper.sh)

    echo "${LABEL}:"
    echo "  array  ${ARR}"
    echo "  reaper ${REAP}"
    echo "  out    ${RUN_DIR}"
done

echo
echo "watch: squeue --me"

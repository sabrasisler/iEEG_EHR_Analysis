#!/bin/bash
#
# Submit the MDD domain-model lme4 pipeline as three chained Slurm stages:
#
#   1. frames   run_domain_model.py --frames-only --dx-model interaction
#   2. array    one lme4 fit per band (6 tasks, parallel)
#   3. collect  concatenate bands/ into the run-level tables + provenance
#
# Stages 2 and 3 are dependency-gated on afterok, so a failed frame build never
# leaves a half-populated run directory behind.
#
# RUN_DIR is computed HERE, before anything is submitted. The array needs the
# directory to exist as a shared target, and a timestamp generated inside six
# parallel tasks would give six different answers.
#
#   bash sbatch/submit_domain_lmer_mdd.sh
#
# Window is 'ever' (--dx-window-days 0), the analyst's choice: 26 cases / 25
# controls, the better-balanced of the two available windows.

set -euo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date +%Y%m%d-%H%M%S)
BASE=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/analysis/pain/mdd/domain_model/paperbands6hg200-paindomainsv3-roiunit-mdd0d-noparcel
RUN_DIR="${BASE}/domain_lmer_${STAMP}"
FRAMES_RUN="lmer_frames_${STAMP}"

# A GLOB, not a path. `config.analysis_run_dir` appends its OWN timestamp to
# --run-name, so stage 1 lands in `${FRAMES_RUN}_<its own stamp>/frames` and any
# path composed here is wrong by exactly that suffix. Stages 2 and 3 resolve it.
FRAMES_GLOB="${BASE}/${FRAMES_RUN}_*/frames"

mkdir -p logs "${RUN_DIR}/bands"

echo "run dir:     ${RUN_DIR}"
echo "frames glob: ${FRAMES_GLOB}"

# ---- stage 1: frames -------------------------------------------------------
# REUSE. Frames are a pure function of (view, cohort, roi scheme, dx window);
# rebuilding them to re-run the FITS is wasted work, and the frames carry their
# own provenance so a reused set is still traceable. Point EXISTING_FRAMES at a
# previous `<run>/frames` to skip stage 1 entirely.
if [ -n "${EXISTING_FRAMES:-}" ]; then
    [ -d "${EXISTING_FRAMES}" ] || { echo "no such frames: ${EXISTING_FRAMES}" >&2; exit 1; }
    FRAMES_GLOB="${EXISTING_FRAMES}"
    echo "stage 1 frames:  SKIPPED, reusing ${EXISTING_FRAMES}"
    ARRAY_DEP=""
else
FRAMES_JOB=$(sbatch --parsable \
  -J lmer_mdd_frames -p ckeller1 -t 01:00:00 -c 4 --mem=32GB \
  -o logs/lmer_mdd_frames_%j.out -e logs/lmer_mdd_frames_%j.err \
  --wrap "set -euo pipefail; \
          export PATH=\$HOME/bin:\$PATH; \
          module load python/3.12; \
          source \$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate; \
          export PYTHONPATH=$(pwd)/src; \
          python -m ieeg_ehr.analysis.run_domain_model \
            --unit roi --roi-scheme pain_domains_v3 \
            --insula-threshold -2.245993821300736 \
            --dx-model interaction --dx mdd --dx-window-days 0 \
            --question mdd --frames-only --run-name ${FRAMES_RUN}")
echo "stage 1 frames:  ${FRAMES_JOB}"
ARRAY_DEP="--dependency=afterok:${FRAMES_JOB}"
fi

# ---- stage 2: one fit per band --------------------------------------------
ARRAY_JOB=$(sbatch --parsable \
  ${ARRAY_DEP} \
  --export=ALL,RUN_DIR="${RUN_DIR}",FRAMES_GLOB="${FRAMES_GLOB}" \
  sbatch/domain_lmer_mdd_array.sbatch)
echo "stage 2 array:   ${ARRAY_JOB}"

# ---- stage 3: collect ------------------------------------------------------
COLLECT_JOB=$(sbatch --parsable \
  --dependency=afterok:"${ARRAY_JOB}" \
  -J lmer_mdd_collect -p ckeller1 -t 00:30:00 -c 2 --mem=16GB \
  -o logs/lmer_mdd_collect_%j.out -e logs/lmer_mdd_collect_%j.err \
  --wrap "set -euo pipefail; \
          export PATH=\$HOME/bin:\$PATH; \
          module load math R/4.4.2; module load python/3.12; \
          source \$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate; \
          export PYTHONPATH=$(pwd)/src; \
          export R_LIBS_USER=\$GROUP_HOME/R/4.4.2; \
          FD=\$(ls -1d ${FRAMES_GLOB} | sort | tail -1); \
          python -m ieeg_ehr.analysis.run_domain_lmer \
            --frames-dir \"\$FD\" --run-dir ${RUN_DIR} --collect-only \
            --view-scheme paperbands6hg200-paindomainsv3-roiunit-mdd0d-noparcel")
echo "stage 3 collect: ${COLLECT_JOB}"

echo
echo "watch:   squeue --me"
echo "results: ${RUN_DIR}"

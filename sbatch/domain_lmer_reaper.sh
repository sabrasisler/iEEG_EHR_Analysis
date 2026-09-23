#!/bin/bash
#
# Self-healing tail for an lme4 band array. Runs AFTER the array (afterany, so
# it runs whether tasks succeeded, failed, or were preempted), works out which
# bands are still missing, and either:
#
#   - nothing missing  -> submit the collect stage and stop
#   - some missing     -> resubmit an array for JUST those, plus another reaper
#   - out of attempts  -> fail loudly, naming what never landed
#
# WHY THIS EXISTS. `owners` is the fastest place to get nodes and the only place
# jobs get PREEMPTED. `--requeue` handles the preemption case by itself, but not
# a task that fails for any other reason, and neither covers "the requeued task
# was preempted again". Deciding what to redo from WHICH OUTPUT FILES EXIST --
# rather than from exit codes -- handles all of those the same way and is
# correct even if a task died between writing its output and exiting.
#
# Env contract (exported by the submit wrapper and carried forward by each
# generation of this script):
#   RUN_DIR FRAMES_GLOB VIEW_SCHEME DROP_DOMAIN PARTITION ATTEMPT MAX_ATTEMPTS
#   LABEL   short name used in job names and logs

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

: "${RUN_DIR:?}" ; : "${FRAMES_GLOB:?}" ; : "${VIEW_SCHEME:?}"
PARTITION=${PARTITION:-owners}
DROP_DOMAIN=${DROP_DOMAIN:-}
ATTEMPT=${ATTEMPT:-1}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-5}
LABEL=${LABEL:-lmer}

ALL_BANDS=(delta theta alpha beta gamma high_gamma)

missing=()
for b in "${ALL_BANDS[@]}"; do
    [ -f "${RUN_DIR}/bands/${b}_fitinfo.csv" ] || missing+=("$b")
done

echo "=== reaper ${LABEL} attempt ${ATTEMPT}/${MAX_ATTEMPTS} | $(date) ==="
echo "run dir: ${RUN_DIR}"
echo "missing: ${missing[*]:-<none>}"

if [ ${#missing[@]} -eq 0 ]; then
    echo "all six bands present -- submitting collect"
    sbatch --parsable \
      -J "${LABEL}_collect" -p "${PARTITION}" -t 00:30:00 -c 2 --mem=16GB \
      --requeue --open-mode=append \
      -o "logs/${LABEL}_collect_%j.out" -e "logs/${LABEL}_collect_%j.err" \
      --wrap "set -euo pipefail; \
              export PATH=\$HOME/bin:\$PATH; \
              module load math R/4.4.2; module load python/3.12; \
              source \$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate; \
              export PYTHONPATH=$(pwd)/src; \
              export R_LIBS_USER=\$GROUP_HOME/R/4.4.2; \
              FD=\$(ls -1d ${FRAMES_GLOB} | sort | tail -1); \
              python -m ieeg_ehr.analysis.run_domain_lmer \
                --frames-dir \"\$FD\" --run-dir ${RUN_DIR} --collect-only \
                ${DROP_DOMAIN:+--drop-domain ${DROP_DOMAIN}} \
                --view-scheme ${VIEW_SCHEME}"
    exit 0
fi

if [ "${ATTEMPT}" -ge "${MAX_ATTEMPTS}" ]; then
    echo "FATAL: ${#missing[@]} band(s) never landed after ${ATTEMPT} attempts:" >&2
    echo "       ${missing[*]}" >&2
    echo "       Not resubmitting -- repeated failure is a bug, not bad luck," >&2
    echo "       and another round would just burn the queue. Read" >&2
    echo "       logs/${LABEL}_*_*.err for the last attempt." >&2
    exit 1
fi

NEXT=$((ATTEMPT + 1))
JOINED=$(IFS=,; echo "${missing[*]}")
LAST=$(( ${#missing[@]} - 1 ))

ARR=$(sbatch --parsable \
  -J "${LABEL}_a${NEXT}" -p "${PARTITION}" --array=0-${LAST} \
  --requeue --open-mode=append \
  -o "logs/${LABEL}_%A_%a.out" -e "logs/${LABEL}_%A_%a.err" \
  --export=ALL,RUN_DIR="${RUN_DIR}",FRAMES_GLOB="${FRAMES_GLOB}",VIEW_SCHEME="${VIEW_SCHEME}",DROP_DOMAIN="${DROP_DOMAIN}",BANDS="${JOINED}" \
  sbatch/domain_lmer_band_array.sbatch)
echo "resubmitted ${#missing[@]} band(s) as ${ARR}"

REAP=$(sbatch --parsable \
  --dependency=afterany:"${ARR}" \
  -J "${LABEL}_reap${NEXT}" -p "${PARTITION}" -t 00:15:00 -c 1 --mem=4GB \
  --requeue --open-mode=append \
  -o "logs/${LABEL}_reap_%j.out" -e "logs/${LABEL}_reap_%j.err" \
  --export=ALL,RUN_DIR="${RUN_DIR}",FRAMES_GLOB="${FRAMES_GLOB}",VIEW_SCHEME="${VIEW_SCHEME}",DROP_DOMAIN="${DROP_DOMAIN}",PARTITION="${PARTITION}",ATTEMPT="${NEXT}",MAX_ATTEMPTS="${MAX_ATTEMPTS}",LABEL="${LABEL}" \
  sbatch/domain_lmer_reaper.sh)
echo "next reaper: ${REAP}"

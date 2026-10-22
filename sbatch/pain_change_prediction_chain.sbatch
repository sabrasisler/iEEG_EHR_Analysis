#!/bin/bash
#SBATCH -p ckeller1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=32GB
#SBATCH -J pc_chain
#SBATCH -o logs/pc_chain_%j.out
# One pain_change model from fit to prediction figures, in one job:
#   1. pain_change: the reference fit and its heatmaps
#   2. pain_change_loo: one refit per held-out subject (the reference run's engine)
#   3. pain_change_prediction weights, then score: raw uniform predictions and figures
#   4. pain_change_tracking: per-subject tracking of the same predictions
# One job, not an array: every fold reuses the same loaded rows, and lme4 spreads
# each fold's cells over all cores (pain_change_lmer.R, mclapply).
#
#   sbatch --export=ALL,MODEL=opioid-dc-pairint,ENGINE=lme4,ROI=roi_v3_ins sbatch/pain_change_prediction_chain.sbatch
# EXCLUDE_FIRST_H=24 adds the first-hours exclusion. TRAJECTORY_DIR must carry the ROI scheme.
# FIT_DIR=<pain_change run> skips step 1 and continues from that fit (e.g. after building
# its trajectory). INSULA_RUN= (empty) drops --insula-from-run, for a ROI scheme with no
# insula split (an hcpex_paper coverage run). DOMAIN_SCHEME=none draws ungrouped heatmaps.
# STOP_AFTER_FIT=1 ends after step 1, when the trajectory must first be built from the new fit.
# SHIFT=5 takes each pair's later epoch from [t2 - 10, t2 - 5) min
# (pain_change --second-epoch-shift-min; build the shifted tables first).
# FIRST_POST=1 takes each pair's earlier epoch from [t1, t1 + 5) min, both epochs against
# the post-report baseline (pain_change --first-epoch-post; build the post tables first,
# sbatch/build_pain_change_post_z_array.sbatch). STOP_AFTER_LOO=1 ends after step 2.
# A heatmap that fails is logged and skipped, so it never ends the job.
# AUTONOMIC=1 adds dHR and dHRV, keeps only pairs with both in both epochs, and
# writes under <data token>/autonomics/ (pain_change --autonomic). A `-dhr` or
# `-dhrv` model needs it, and STOP_AFTER_FIT=1 with it: LOO refuses autonomic runs.
set -euo pipefail
module load math R/4.4.2
module load python/3.12
export R_LIBS_USER="$GROUP_HOME/R/4.4.2"
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$HOME/bin:$PATH"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
OAK=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler
INSULA_RUN=${INSULA_RUN-$OAK/analysis/pain/pain_change/bandpower/painchange_20261006-211218}
DOMAIN_SCHEME=${DOMAIN_SCHEME:-pain_domains_v4}
TRAJECTORY_DIR=${TRAJECTORY_DIR:-$OAK/features/pain/pain_change_trajectory/dz-w5m-s1m-h180m_5d6156e6a827}
VIEW=(--mask-level bipolar --mask-label "std10_rv-gross-std3_satmargin15_sw_logz4")

if [ -z "${FIT_DIR:-}" ]; then
  FIT_DIR=$(python -m ieeg_ehr.analysis.pain_change --model "$MODEL" --engine "$ENGINE" \
    --roi-scheme "$ROI" ${EXCLUDE_FIRST_H:+--exclude-first-h "$EXCLUDE_FIRST_H"} \
    ${INSULA_RUN:+--insula-from-run "$INSULA_RUN"} ${SHIFT:+--second-epoch-shift-min "$SHIFT"} \
    ${FIRST_POST:+--first-epoch-post} ${AUTONOMIC:+--autonomic} "${VIEW[@]}" | tail -1)
  for TERM in dpain dc dpain_x_dc dhr dhrv; do
    if grep -q "^${TERM}_beta\|,${TERM}_beta," "$FIT_DIR/cells.csv"; then
      python -m ieeg_ehr.analysis.plot_pain_change_heatmap --run-dir "$FIT_DIR" --term "$TERM" \
        --domain-scheme "$DOMAIN_SCHEME" || echo "HEATMAP FAILED: $TERM (exit $?), continuing"
    fi
  done
fi
echo "FIT_DIR=$FIT_DIR"
if [ -n "${STOP_AFTER_FIT:-}" ]; then exit 0; fi

LOO_DIR=$(python -m ieeg_ehr.analysis.pain_change_loo --reference-run "$FIT_DIR" \
  --roi-scheme "$ROI" "${VIEW[@]}" | tail -1)
echo "LOO_DIR=$LOO_DIR"
if [ -n "${STOP_AFTER_LOO:-}" ]; then exit 0; fi
WEIGHTS="$(basename "$(dirname "$LOO_DIR")")_$(basename "$LOO_DIR")"
python -m ieeg_ehr.analysis.pain_change_prediction weights --source-run "$LOO_DIR" --name "$WEIGHTS"

PRED_DIR=$(python -m ieeg_ehr.analysis.pain_change_prediction score \
  --trajectory-dir "$TRAJECTORY_DIR" --weights "$WEIGHTS" | tail -1)
echo "PRED_DIR=$PRED_DIR"
python -m ieeg_ehr.analysis.pain_change_tracking --run-dir "$PRED_DIR"

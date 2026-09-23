#!/bin/bash
#SBATCH -J domain_lmer
#SBATCH -p ckeller1
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH -o logs/domain_lmer_%j.out
#SBATCH -e logs/domain_lmer_%j.err
#
# The processing-domain model refitted in lme4, with the CROSSED region random
# effects statsmodels cannot express. Two stages in one job:
#
#   1. run_domain_model.py --frames-only   build the per-band model frames
#   2. run_domain_lmer.py                  fit them in R
#
# WHY --frames-only. The statsmodels results for these bands ALREADY EXIST
# (domain_mixedlm_20260921-153105). Re-running the fits just to obtain the
# frames as a by-product costs ~20 min per band for a result we have. The flag
# stops after the frame write, so stage 1 pays only the view load.
#
# THE R LIBRARY is at $GROUP_HOME/R/4.4.2 -- NOT $HOME (15 GB, NFS backed, and a
# compiled library tree is neither small nor worth backing up). Built with
# `devel cmake` loaded, because nloptr (an lme4 dependency) configures bundled
# NLopt through CMake and fails with a bare "CMAKE NOT FOUND" otherwise.
#
# RUNTIME. The beta band fitted in 248 s on 119,858 rows; 12 h is generous
# headroom for six bands plus the view load, not a measured need.

set -euo pipefail
export PATH="$HOME/bin:$PATH"

module load math R/4.4.2
module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export R_LIBS_USER="$GROUP_HOME/R/4.4.2"

# lme4 is single threaded; let the BLAS use the cores instead of oversubscribing.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

INSULA_THRESHOLD=${INSULA_THRESHOLD:--2.245993821300736}
FRAMES_RUN=${FRAMES_RUN:-lmer_frames}
EXTRA=${EXTRA:-}

echo "=== stage 1/2: model frames | insula cut y=${INSULA_THRESHOLD} | $(date) ==="
python -m ieeg_ehr.analysis.run_domain_model \
    --unit roi \
    --roi-scheme pain_domains_v3 \
    --insula-threshold "${INSULA_THRESHOLD}" \
    --frames-only \
    --run-name "${FRAMES_RUN}"

# The frames run is the newest `${FRAMES_RUN}_*` directory under the v3 ROI-unit
# scheme. Resolved here rather than hard-coded: the timestamp is only known once
# stage 1 has run.
BASE=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/analysis/pain/bandpower/domain_model/paperbands6hg200-paindomainsv3-roiunit-noparcel
FRAMES_DIR=$(ls -1d "${BASE}/${FRAMES_RUN}"_*/frames 2>/dev/null | sort | tail -1)
if [ -z "${FRAMES_DIR}" ]; then
    echo "FATAL: stage 1 wrote no frames under ${BASE}/${FRAMES_RUN}_*" >&2
    exit 1
fi
echo "frames: ${FRAMES_DIR}"

echo "=== stage 2/2: lme4 fits | $(date) ==="
python -m ieeg_ehr.analysis.run_domain_lmer \
    --frames-dir "${FRAMES_DIR}" \
    --view-scheme paperbands6hg200-paindomainsv3-roiunit-noparcel \
    ${EXTRA}

echo "=== done ($(date)) ==="

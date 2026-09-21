#!/bin/bash
#SBATCH --job-name=bp_ins
#SBATCH --partition=ckeller1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=logs/bp_ins_%j.out
#
# Region-level band-power mixed models WITH THE INSULA SPLIT, then the
# across-subject consistency figures on top of them. 21 regions x 6 bands.
#
#   sbatch sbatch/bandpower_mixed_ins.sbatch
#
# WHY THIS EXISTS BESIDE sbatch/bandpower_mixed.sbatch. Two changes, and the
# second is the reason the first is worth a separate template:
#
#   --roi-scheme roi_v2_ofc_ins   roi_v2_ofc with Insula split into aIns/pIns by
#                                 MNI y, so the two halves get their own rows
#                                 instead of one row that averages them.
#   --insula-threshold            PINNED to the cut reviewed on a glass brain
#                                 (insula_ap_20260921-144743), not re-derived
#                                 from whatever cohort this run resolves to.
#
# THIS IS THE COMPANION TO THE ROI-LEVEL DOMAIN MODEL, not an alternative to it.
# The domain model takes the region out of the fit entirely, which means nothing
# in it guards against one well-sampled ROI carrying its domain. This run is
# where that guard lives: every ROI fitted separately, and the consistency stage
# reports what percentage of subjects share each cell's group sign -- including
# the ROIs (PCC, Hippocampus, Basal Ganglia...) that no domain claims.
#
# TIME. ~1.6 s median per fit plus the reduced model, dominated by the Lustre
# read of the epoch-mean view: 21 regions x ~1.5 GB cohort-wide. The consistency
# stage adds a couple of minutes of vectorised permutation.

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

INSULA_THRESHOLD=${INSULA_THRESHOLD:--2.245993821300736}
EXTRA=${EXTRA:-}
COMMON=(--roi-scheme roi_v2_ofc_ins --insula-threshold "${INSULA_THRESHOLD}")

echo "=== bandpower mixed, insula split | cut y=${INSULA_THRESHOLD} | $(date) ==="
RUN_DIR=$(python -m ieeg_ehr.analysis.run_bandpower_mixed --stage fit \
    "${COMMON[@]}" ${EXTRA} | tail -1)
echo "    run: ${RUN_DIR}"

# The collect stage re-derives the region list from --roi-scheme, so it MUST be
# passed here too or it looks for roi_v2_ofc names that are not in the cells
# table and crashes on an empty pivot after the fits have already succeeded.
python -m ieeg_ehr.analysis.run_bandpower_mixed --stage collect \
    --run-dir "${RUN_DIR}" "${COMMON[@]}" ${EXTRA}

echo "=== consistency figures ==="
python -m ieeg_ehr.analysis.plot_bandpower_consistency --run-dir "${RUN_DIR}"

echo "=== done ($(date)) ==="
echo "RUN_DIR=${RUN_DIR}"

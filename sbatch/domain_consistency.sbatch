#!/bin/bash
#SBATCH --job-name=dom_consist
#SBATCH --partition=ckeller1
#SBATCH --qos=high_p
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --output=logs/dom_consist_%j.out
#
# Across-subject consistency for a DOMAIN-MODEL run: is a domain's pain slope
# something most patients show, or a group average over a heterogeneous crowd?
#
#   RUN_DIR=<domain run> sbatch sbatch/domain_consistency.sbatch
#
# NOTHING IS REFITTED. The per-subject slopes are unpooled OLS lines, which are
# microseconds each; the cost is one pass over the epoch-mean full-res view,
# which the domain run's own log measured at ~5 s. Memory is the whole cohort's
# rows across the kept bins in one array, same as the fit stage -- hence 32 GB
# rather than the 8 GB the region-level consistency job needs, which reads only
# two small tables.
#
# The run's ROI scheme, unit, band set and insula threshold come from its
# provenance, so this cannot describe a different model from the one it is
# drawn beside. Pass --view-dir/--mask-label only for a run that used a
# non-default view.

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${SCRATCH}/mplconfig"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

DERIV=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler
RUN_DIR=${RUN_DIR:-$DERIV/analysis/pain/bandpower/domain_model/paperbands6hg200-paindomainsv3-roiunit-noparcel/domain_mixedlm_20260921-153105}
EXTRA=${EXTRA:-}

echo "=== domain consistency | $RUN_DIR | $(date) ==="
python -m ieeg_ehr.analysis.plot_domain_consistency --run-dir "$RUN_DIR" ${EXTRA}
echo "=== done ($(date)) ==="

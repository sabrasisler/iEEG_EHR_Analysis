#!/bin/bash
#SBATCH --job-name=thal_glass
#SBATCH --partition=ckeller1
#SBATCH --qos=high_p
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --output=logs/thal_glass_%j.out
#
# Thalamic nuclei and the thalamic electrodes on a zoomed glass brain, grouped
# by the paper ROIs of an hcpex_paper coverage run.
#
#   sbatch sbatch/thalamus_glass_brain.sbatch
#   COVERAGE_RUN=<hcpex_paper run dir> sbatch sbatch/thalamus_glass_brain.sbatch

set -euo pipefail
export PATH="$HOME/bin:$PATH"
module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${SCRATCH}/mplconfig"
DERIV=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler
COVERAGE_RUN=${COVERAGE_RUN:-$DERIV/analysis/electrode_localization/hcpex_paper_coverage/label_counts/roimap_v1_20261016-160252}
python -m ieeg_ehr.analysis.plot_thalamus_glass_brain --coverage-run "$COVERAGE_RUN" ${EXTRA:-}

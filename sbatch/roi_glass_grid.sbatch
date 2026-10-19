#!/bin/bash
#SBATCH --job-name=roi_glass_grid
#SBATCH --partition=ckeller1
#SBATCH --qos=high_p
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --output=logs/roi_glass_grid_%j.out
#
# One glass brain per paper ROI (territory + electrode midpoints) and a summary
# table that flags ROIs under the subject floor, from an hcpex_paper coverage run.
#
#   COVERAGE_RUN=<hcpex_paper run dir> sbatch sbatch/roi_glass_grid.sbatch

set -euo pipefail
export PATH="$HOME/bin:$PATH"
module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${SCRATCH}/mplconfig"
python -m ieeg_ehr.analysis.plot_roi_glass_grid --coverage-run "${COVERAGE_RUN:?set COVERAGE_RUN}" ${EXTRA:-}

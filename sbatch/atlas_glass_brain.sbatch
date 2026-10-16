#!/bin/bash
#SBATCH --job-name=atlas_glass
#SBATCH --partition=ckeller1
#SBATCH --qos=high_p
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --output=logs/atlas_glass_%j.out
#
# Glass brain of the electrodes, coloured by Yeo network, from a
# plot_atlas_coverage yeo7/yeo17 run's pair_labels.csv.
#
#   sbatch sbatch/atlas_glass_brain.sbatch
#   COVERAGE_RUN=<yeo17 coverage run dir> sbatch sbatch/atlas_glass_brain.sbatch

set -euo pipefail
export PATH="$HOME/bin:$PATH"
module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${SCRATCH}/mplconfig"
DERIV=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler
COVERAGE_RUN=${COVERAGE_RUN:-$DERIV/analysis/electrode_localization/yeo7_coverage/label_counts/contpain_ref_coordqc_20261014-143926}
python -m ieeg_ehr.analysis.plot_atlas_glass_brain --coverage-run "$COVERAGE_RUN" ${EXTRA:-}

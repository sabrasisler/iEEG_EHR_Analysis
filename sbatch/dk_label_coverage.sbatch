#!/bin/bash
#SBATCH --job-name=dk_coverage
#SBATCH --partition=ckeller1
#SBATCH --qos=high_p
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --output=logs/dk_coverage_%j.out
#
# Subjects and bipolar pairs per Desikan-Killiany anode label, hemispheres
# merged, for the reference run's cohort. Two bar panels + a CSV under
# analysis/electrode_localization/dk_coverage/label_counts/.
#
#   sbatch sbatch/dk_label_coverage.sbatch
#   EXTRA="--reference-run <run dir>" sbatch sbatch/dk_label_coverage.sbatch

set -euo pipefail
export PATH="$HOME/bin:$PATH"
module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${SCRATCH}/mplconfig"
python -m ieeg_ehr.analysis.plot_dk_label_coverage ${EXTRA:-}

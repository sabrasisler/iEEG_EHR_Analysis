#!/bin/bash
#SBATCH -p ckeller1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH -J pc_anat
#SBATCH -o logs/pc_anat_%j.out
# Anatomy glass brain of a pain_change run, hue = lobe group, shade = ROI:
#   sbatch --export=ALL,RUN_DIR=<pain_change run> sbatch/pain_change_anatomy.sbatch
set -euo pipefail
module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate
export PATH="$HOME/bin:$PATH"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${SCRATCH}/mplconfig"
python -m ieeg_ehr.analysis.plot_pain_change_anatomy --run-dir "$RUN_DIR"

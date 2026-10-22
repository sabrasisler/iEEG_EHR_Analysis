#!/bin/bash
#SBATCH -p ckeller1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH -J pc_glass
#SBATCH -o logs/pc_glass_%j.out
# Glass brain of one pain_change term, BH-significant ROIs coloured by region:
#   sbatch --export=ALL,RUN_DIR=<pain_change run>,TERM=dpain sbatch/pain_change_glass.sbatch
set -euo pipefail
module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate
export PATH="$HOME/bin:$PATH"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${SCRATCH}/mplconfig"
python -m ieeg_ehr.analysis.plot_pain_change_glass --run-dir "$RUN_DIR" --term "${TERM:-dpain}"

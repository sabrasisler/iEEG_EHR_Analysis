#!/bin/bash
#SBATCH --job-name=v2_coverage
#SBATCH --partition=ckeller1
#SBATCH --qos=high_p
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16GB
#SBATCH --output=logs/v2_coverage_%j.out
#
# Per subject-session recording coverage of iEEG_EHR_V2, read from the NWB
# headers. Writes the figure, per-file and per-session tables, and the load
# errors under iEEG_EHR_V2/derivatives/sisler/recording_coverage/.
#
#   sbatch sbatch/v2_recording_coverage.sbatch

set -euo pipefail
export PATH="$HOME/bin:$PATH"
module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${SCRATCH}/mplconfig"
python -m ieeg_ehr.analysis.plot_v2_recording_coverage ${EXTRA:-}

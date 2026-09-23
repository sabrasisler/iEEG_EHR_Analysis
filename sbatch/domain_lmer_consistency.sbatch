#!/bin/bash
#SBATCH -J lmer_consist
#SBATCH -p normal
#SBATCH -t 01:00:00
#SBATCH -c 2
#SBATCH --mem=8GB
#SBATCH -o logs/lmer_consist_%j.out
#SBATCH -e logs/lmer_consist_%j.err
#
# Dependent stage after domain_lmer_loo_array.sbatch: collect the LOO refits,
# then draw every consistency figure into <RUN_DIR>/consistency/.
#
# Required: RUN_DIR.

set -euo pipefail
export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

: "${RUN_DIR:?RUN_DIR must be exported by the submit wrapper}"

python -m ieeg_ehr.analysis.run_domain_lmer_loo --run-dir "${RUN_DIR}" --collect
python -m ieeg_ehr.analysis.plot_domain_lmer_consistency --run-dir "${RUN_DIR}"

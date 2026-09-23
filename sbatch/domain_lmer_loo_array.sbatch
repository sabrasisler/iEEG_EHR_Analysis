#!/bin/bash
#SBATCH -J lmer_loo
#SBATCH -p normal
#SBATCH -t 00:45:00
#SBATCH -c 2
#SBATCH --mem=4GB
#SBATCH -o logs/lmer_loo_%A_%a.out
#SBATCH -e logs/lmer_loo_%A_%a.err
#
# ONE ARRAY TASK PER (band, excluded subject): leave-one-subject-out refits of an
# lme4 domain run, for the consistency profile (run_domain_lmer_loo.py). The
# task list -- bands carrying a BH-significant cell x (none + every subject) --
# is built by the Python side; `--n-tasks` gives the array size, so submit via
# sbatch/submit_domain_lmer_loo.sh rather than by hand.
#
# RESOURCES: one full fit measured 4.5 min / 0.9 GB MaxRSS (job 44751273).
#
# Required: RUN_DIR (the domain_lmer run).

set -euo pipefail
export PATH="$HOME/bin:$PATH"

module load math R/4.4.2
module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export R_LIBS_USER="$GROUP_HOME/R/4.4.2"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}

: "${RUN_DIR:?RUN_DIR must be exported by the submit wrapper}"

echo "=== lmer LOO | task=${SLURM_ARRAY_TASK_ID} | $(date) ==="
python -m ieeg_ehr.analysis.run_domain_lmer_loo --run-dir "${RUN_DIR}"
echo "=== done ($(date)) ==="

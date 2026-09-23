#!/bin/bash
# Submit the LOO refit array for an lme4 domain run, then the dependent
# collect + consistency-figure stage.
#
#   RUN_DIR=<domain_lmer run> bash sbatch/submit_domain_lmer_loo.sh [array-spec]
#
# array-spec defaults to every task (0-(n-1)); pass e.g. "0" to test one fit.
# The task count comes from Python (bands with a significant cell x
# (none + subjects)) -- computed on a dev node, never on the login node.
set -euo pipefail
: "${RUN_DIR:?export RUN_DIR=<domain_lmer run dir>}"
export RUN_DIR

N=${N_TASKS:-$(srun -p dev -c 1 --mem=2G -t 00:05:00 bash -c \
    'module load python/3.12 >/dev/null 2>&1; source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate; PYTHONPATH=$PWD/src python -m ieeg_ehr.analysis.run_domain_lmer_loo --run-dir "$RUN_DIR" --n-tasks' | tail -1)}
SPEC=${1:-0-$((N - 1))}
echo "tasks: ${N}; submitting array ${SPEC}"

AID=$(sbatch --parsable --export=ALL --array="${SPEC}" sbatch/domain_lmer_loo_array.sbatch)
echo "array: ${AID}"
if [ -z "${1:-}" ]; then
    CID=$(sbatch --parsable --export=ALL --dependency=afterok:${AID} \
          sbatch/domain_lmer_consistency.sbatch)
    echo "collect + figures: ${CID} (afterok:${AID})"
fi

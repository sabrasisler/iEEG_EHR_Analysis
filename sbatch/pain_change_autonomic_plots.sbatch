#!/bin/bash
#SBATCH -p ckeller1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH -J pc_autonomic_plots
#SBATCH -o logs/pc_autonomic_plots_%j.out
# The autonomic scatters (plot_pain_change_autonomic) for the newest -dhr and -dhrv
# runs of one autonomics/ folder, each against the newest base run there:
#   sbatch --dependency=afterok:<fit jobs> \
#     --export=ALL,AUTONOMIC_DIR=<data token dir>/autonomics,MAIN_RUN=<main run> \
#     sbatch/pain_change_autonomic_plots.sbatch
set -euo pipefail
module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate
export PATH="$HOME/bin:$PATH"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
: "${AUTONOMIC_DIR:?}" "${MAIN_RUN:?}"
BASE_MODEL=${BASE_MODEL:-opioid-dc-pairint-nosubjint}
newest() { ls -1d "$AUTONOMIC_DIR/$1"_mme2_lme4_2* | tail -1; }
BASE=$(newest "$BASE_MODEL")
for SUFFIX in dhr dhrv; do
  RUN=$(newest "$BASE_MODEL-$SUFFIX")
  echo "$SUFFIX: $RUN vs $BASE"
  python -m ieeg_ehr.analysis.plot_pain_change_autonomic --run-dir "$RUN" \
    --base-run "$BASE" --main-run "$MAIN_RUN"
done

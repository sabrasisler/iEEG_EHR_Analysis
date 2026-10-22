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
# EXPOSURE is the runs' exposure version (default mme2). Without MAIN_RUN, the newest
# base-model run of the data token dir is the main run. A failed scatter is logged
# and the next one still runs.
set -euo pipefail
module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate
export PATH="$HOME/bin:$PATH"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
: "${AUTONOMIC_DIR:?}"
BASE_MODEL=${BASE_MODEL:-opioid-dc-pairint-nosubjint}
EXPOSURE=${EXPOSURE:-mme2}
newest() { ls -1d "$1/$2_${EXPOSURE}_lme4_2"* | tail -1; }
MAIN_RUN=${MAIN_RUN:-$(newest "$(dirname "$AUTONOMIC_DIR")" "$BASE_MODEL")}
BASE=$(newest "$AUTONOMIC_DIR" "$BASE_MODEL")
echo "main run: $MAIN_RUN"
status=0
for SUFFIX in dhr dhrv; do
  RUN=$(newest "$AUTONOMIC_DIR" "$BASE_MODEL-$SUFFIX")
  echo "$SUFFIX: $RUN vs $BASE"
  python -m ieeg_ehr.analysis.plot_pain_change_autonomic --run-dir "$RUN" \
    --base-run "$BASE" --main-run "$MAIN_RUN" || { echo "SCATTER FAILED: $SUFFIX"; status=1; }
done
exit $status

#!/bin/bash
#SBATCH -p ckeller1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=12GB
# One worker per core: cells are independent fits. Measured 2026-10-06: 1 min 43 s,
# 2.4 GB on 8 cores for 132 cells.
#SBATCH -J pain_change
#SBATCH -o logs/pain_change_%j.out
# MODEL is a pain_change.MODELS ID and ENGINE a pain_change.ENGINES token (docs/run_tokens.md):
#   sbatch --export=ALL,MODEL=opioid,ENGINE=lme4 sbatch/pain_change.sbatch
set -euo pipefail
module load math R/4.4.2
module load python/3.12
export R_LIBS_USER="$GROUP_HOME/R/4.4.2"
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$HOME/bin:$PATH"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"
MODEL=${MODEL:-opioid}
ENGINE=${ENGINE:-sm}
# The insula cut of the 2026-10-06 grid, so every pain_change run splits the insula alike.
INSULA_RUN=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/analysis/pain/pain_change/bandpower/painchange_20261006-211218
# pain_change prints its run directory as its last stdout line.
RUN_DIR=$(python -m ieeg_ehr.analysis.pain_change --model "$MODEL" --engine "$ENGINE" \
  --roi-scheme roi_v2_ofc_ins --exclude-first-h 24 --insula-from-run "$INSULA_RUN" \
  --mask-level bipolar --mask-label "std10_rv-gross-std3_satmargin15_sw_logz4" | tail -1)
echo "RUN_DIR=$RUN_DIR"
python -m ieeg_ehr.analysis.plot_pain_change_heatmap --run-dir "$RUN_DIR" --term dpain
if [ "$MODEL" = opioid ]; then
  python -m ieeg_ehr.analysis.plot_pain_change_heatmap --run-dir "$RUN_DIR" --term dpain_x_cbar
fi

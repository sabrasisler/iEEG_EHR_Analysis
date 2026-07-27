#!/bin/bash
# Rebuilds the final combined mask over all subjects now present in all 4 per-type
# exclusion label dirs, then re-summarizes it.
#SBATCH -p normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH -J build_final_mask
#SBATCH -o /home/groups/ckeller1/sisler/iEEG_EHR_Analysis/logs/build_final_mask_%j.out

set -euo pipefail
module load python/3.12
source /home/groups/ckeller1/venvs/ieeg_ehr_analysis/bin/activate
cd /home/groups/ckeller1/sisler/iEEG_EHR_Analysis

LEVEL=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/analysis/qc/raw_voltage
FINAL_LABEL=gross-std3_satmargin15_sw_logz4
LOGZ4=logz4_masked-gross-std3_satmargin15_sw

python -m qc_scripts.build_mask --level-root "$LEVEL" --label "$FINAL_LABEL" \
    --gross_artifact std3 --saturation pct0_marginfrac0.15 --square_wave frac0.9 --flatline "$LOGZ4"

python -m qc_scripts.summarize_exclusions --mask-dir "$LEVEL/masks/$FINAL_LABEL"

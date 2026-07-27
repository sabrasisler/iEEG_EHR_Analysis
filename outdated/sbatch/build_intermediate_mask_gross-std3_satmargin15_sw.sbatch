#!/bin/bash
# Rebuilds the intermediate (non-flatline) mask over all subjects now present in
# exclusions/{gross_artifact/std3,saturation/pct0_marginfrac0.15,square_wave/frac0.9}/.
# This mask feeds flatline's --mask-from-label baseline exclusion.
#SBATCH -p normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH -J build_mask_intermediate
#SBATCH -o /home/groups/ckeller1/sisler/iEEG_EHR_Analysis/logs/build_mask_intermediate_%j.out

set -euo pipefail
module load python/3.12
source /home/groups/ckeller1/venvs/ieeg_ehr_analysis/bin/activate
cd /home/groups/ckeller1/sisler/iEEG_EHR_Analysis

LEVEL=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/analysis/qc/raw_voltage
python -m qc_scripts.build_mask --level-root "$LEVEL" --label gross-std3_satmargin15_sw \
    --types gross_artifact,saturation,square_wave \
    --gross_artifact std3 --saturation pct0_marginfrac0.15 --square_wave frac0.9

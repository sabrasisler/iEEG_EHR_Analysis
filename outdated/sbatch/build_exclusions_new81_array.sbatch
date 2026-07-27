#!/bin/bash
# Per-subject rollup for the 81 newly-metriced subjects: builds the three
# non-flatline per-type exclusion labels needed for gross-std3_satmargin15_sw_logz4
# (flatline is a separate later step since it depends on this intermediate mask).
#SBATCH -p normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --array=0-80%16
#SBATCH -J build_excl_new81
#SBATCH -o /home/groups/ckeller1/sisler/iEEG_EHR_Analysis/logs/build_excl_new81_%A_%a.out

module load python/3.12
source /home/groups/ckeller1/venvs/ieeg_ehr_analysis/bin/activate
cd /home/groups/ckeller1/sisler/iEEG_EHR_Analysis

LEVEL=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/analysis/qc/raw_voltage
SUBJECT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" qc_scripts/subjects_new81_for_rollup.txt)
echo "Array task $SLURM_ARRAY_TASK_ID -> subject $SUBJECT"

python -m qc_scripts.build_exclusions --level-root "$LEVEL" --artifact-type gross_artifact --std-thresh 3 --subjects "$SUBJECT"
python -m qc_scripts.build_exclusions --level-root "$LEVEL" --artifact-type saturation --rail-margin-frac 0.15 --subjects "$SUBJECT"
python -m qc_scripts.build_exclusions --level-root "$LEVEL" --artifact-type square_wave --frac-thresh 0.9 --subjects "$SUBJECT"

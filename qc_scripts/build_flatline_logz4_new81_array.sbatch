#!/bin/bash
# Per-subject masked flatline (logz4) exclusions for the 81 newly-metriced subjects,
# using the just-rebuilt gross-std3_satmargin15_sw intermediate mask as the baseline mask.
#SBATCH -p normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --array=0-80%16
#SBATCH -J build_flatline_logz4_new81
#SBATCH -o /home/groups/ckeller1/sisler/iEEG_EHR_Analysis/logs/build_flatline_logz4_new81_%A_%a.out

module load python/3.12
source /home/groups/ckeller1/venvs/ieeg_ehr_analysis/bin/activate
cd /home/groups/ckeller1/sisler/iEEG_EHR_Analysis

LEVEL=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/analysis/qc/raw_voltage
SUBJECT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" qc_scripts/subjects_new81_for_rollup.txt)
echo "Array task $SLURM_ARRAY_TASK_ID -> subject $SUBJECT"

python -m qc_scripts.build_exclusions --level-root "$LEVEL" --artifact-type flatline \
    --std-thresh 4 --mask-from-label gross-std3_satmargin15_sw --subjects "$SUBJECT"

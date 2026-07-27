#!/bin/bash
#SBATCH -p normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --array=0-16%8
#SBATCH -J bipolar_excl
#SBATCH -o /home/groups/ckeller1/sisler/iEEG_EHR_Analysis/logs/bipolar_excl_%A_%a.out

# Per-array-task batch of subjects (BATCH_SIZE each) running
# build_bipolar_exclusions for one label/mask, parallelizing the big metric-CSV
# reads across tasks (some subjects' bipolar_variance.csv are 7+GB) --
# mirrors build_exclusions_array.sbatch's pattern for raw_voltage.
# 82 subjects / BATCH_SIZE=5 -> ceil(82/5)-1 = 16, hence --array=0-16.
# Recompute if SUBJECTS_FILE/BATCH_SIZE changes:
#   python3 -c "import math; print(math.ceil(N_SUBJECTS/BATCH_SIZE)-1)"

module load python/3.12
source /home/groups/ckeller1/venvs/ieeg_ehr_analysis/bin/activate
cd /home/groups/ckeller1/sisler/iEEG_EHR_Analysis

: "${SUBJECTS_FILE:=qc_scripts/subjects_bipolar_satmargin15_sw_logz4.txt}"
: "${BATCH_SIZE:=5}"
: "${MASK_DIR:=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/analysis/qc/raw_voltage/masks/gross-std3_satmargin15_sw_logz4}"
: "${LABEL:=std10}"
: "${STD_THRESH:=10.0}"

mapfile -t ALL_SUBJECTS < "$SUBJECTS_FILE"
START=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE))
BATCH=("${ALL_SUBJECTS[@]:START:BATCH_SIZE}")
SUBJECTS_CSV=$(IFS=,; echo "${BATCH[*]}")

echo "Array task $SLURM_ARRAY_TASK_ID -> subjects $SUBJECTS_CSV (label=$LABEL, std_thresh=$STD_THRESH)"
python -m qc_scripts.build_bipolar_exclusions \
    --raw-voltage-mask "$MASK_DIR" --label "$LABEL" --std-thresh "$STD_THRESH" \
    --subjects "$SUBJECTS_CSV"

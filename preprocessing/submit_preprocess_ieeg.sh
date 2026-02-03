#!/bin/bash
#SBATCH --job-name=ieeg_preprocess
#SBATCH --output=logs/preprocess_%A_%a.out
#SBATCH --error=logs/preprocess_%A_%a.err
#SBATCH --time=01:00:00           # Changed from 03:00:00
#SBATCH --mem=4G                  # Changed from 60G
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --array=1-154%20          # Changed from %2 to %20


# ============================================================================
# iEEG Preprocessing - SLURM Array Job
# ============================================================================
#
# BEFORE SUBMITTING:
#   1. Run: python preprocess_ieeg.py --discover
#   2. Check file_list.txt to see how many files
#   3. Edit line 6 above: change --array=1-100 to --array=1-N
#      where N is the number of files
#
# MEMORY TUNING:
#   - Default: 8GB per job, max 8 parallel (%8)
#   - Total RAM: 8GB * 8 = 64GB
#   - To use more memory: increase --mem, decrease %8
#     Example: --mem=16G with %4 = 64GB total
#
# SUBMIT: sbatch submit_preprocessing.sh
# ============================================================================

# Paths
SCRIPT_DIR="/home/groups/ckeller1/sisler/iEEG_EHR_Analysis/preprocessing"
cd ${SCRIPT_DIR}

# Create logs directory
mkdir -p logs

# Load environment
module load python/3.9
source /home/groups/ckeller1/venvs/ieeg_analysis/bin/activate

# Processing parameters
NPERSEG=500
OVERLAP=0.5

# Run preprocessing
python preprocess_ieeg_chunked.py \
    --file-list file_list.txt \
    --task-id ${SLURM_ARRAY_TASK_ID} \
    --nperseg ${NPERSEG} \
    --overlap ${OVERLAP}

echo "Exit code: $?"
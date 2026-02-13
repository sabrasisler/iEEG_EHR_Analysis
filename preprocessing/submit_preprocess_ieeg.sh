#!/bin/bash
#SBATCH --job-name=ieeg_preprocess
#SBATCH --output=logs/preprocess_%A_%a.out
#SBATCH --error=logs/preprocess_%A_%a.err
#SBATCH --time=02:00:00         # 2 hours for batch of 20 files (5-6 min each)
#SBATCH --mem=6G                # 8GB is safe (4GB per file + buffer)
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --array=1-39%10           # CHANGE THIS based on calculation
                                # Example: 60 files ÷ 15 per batch = 4 jobs

# ============================================================================
# iEEG Preprocessing - SLURM Array Job with BATCHING
# ============================================================================
#
# BEFORE SUBMITTING:
#   1. Run: python preprocess_ieeg_chunked.py --discover
#   2. Check file_list.txt to see how many files (e.g., 60 files)
#   3. Choose BATCH_SIZE (recommended: 15-20)
#   4. Calculate number of jobs: N_JOBS = ceil(N_FILES / BATCH_SIZE)
#      Example: 60 files ÷ 15 per batch = 4 jobs
#   5. Edit line 9 above: --array=1-N_JOBS%N_JOBS
#
# EXAMPLE CALCULATIONS:
#   60 files, batch_size=15:  --array=1-4%4   (4 jobs × 15 files)
#   60 files, batch_size=20:  --array=1-3%3   (3 jobs × 20 files)
#   100 files, batch_size=20: --array=1-5%5   (5 jobs × 20 files)
#
# MEMORY & TIME:
#   - Each file takes ~3-5 min, uses ~4GB RAM
#   - Sequential processing = only ONE file in memory at a time
#   - 8GB is enough with safety margin (even for larger files)
#   - Batch of 15: ~75 min
#   - Batch of 20: ~100 min
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
BATCH_SIZE=20          # Number of files per job (processed sequentially)
NPERSEG=1000           # CHANGED: 2000 for 0.5 Hz frequency resolution (was 500)
OVERLAP=0.5
CHUNK_DURATION=60
MAX_FREQ=170          # NEW: Limit frequency computation to 200 Hz

echo "=========================================="
echo "Job Array ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Batch size: ${BATCH_SIZE} (sequential)"
echo "nperseg: ${NPERSEG} (freq resolution: 0.5 Hz)"
echo "max_freq: ${MAX_FREQ} Hz"
echo "=========================================="

# Run preprocessing
python preprocess_ieeg_chunked.py \
    --file-list file_list.txt \
    --task-id ${SLURM_ARRAY_TASK_ID} \
    --batch-size ${BATCH_SIZE} \
    --nperseg ${NPERSEG} \
    --overlap ${OVERLAP} \
    --chunk-duration ${CHUNK_DURATION} \
    --max-freq ${MAX_FREQ}

EXIT_CODE=$?

echo "=========================================="
echo "Job finished with exit code: ${EXIT_CODE}"
echo "=========================================="

exit ${EXIT_CODE}
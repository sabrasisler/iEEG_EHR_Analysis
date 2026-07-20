#!/bin/bash
#SBATCH -p normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH -J bipolar_trace_cache
#SBATCH -o /home/groups/ckeller1/sisler/iEEG_EHR_Analysis/logs/bipolar_trace_cache_%j.out

# Single task, not an array -- only ~40 runs total (17-subject masked cohort,
# --runs-per-subject default 3), and each run is a single sequential raw-NWB
# read + re-reference (~2.8GB average raw file size per registry). --mem=16GB
# comfortably covers one run's raw array + its bipolar-referenced copy in
# memory at once (only one run resident at a time, not all 40).

module load python/3.12
source /home/groups/ckeller1/venvs/ieeg_ehr_analysis/bin/activate
cd /home/groups/ckeller1/sisler/iEEG_EHR_Analysis

SUBJECTS=${SUBJECTS:-039,071,085,088,099,150,176,191,193,198,205,207,211,217,227,244,248}
RUNS_PER_SUBJECT=${RUNS_PER_SUBJECT:-3}
MAX_RUNS=${MAX_RUNS:-40}

python -m preprocessing.save_bipolar_sample_traces \
    --subjects "$SUBJECTS" --runs-per-subject "$RUNS_PER_SUBJECT" --max-runs "$MAX_RUNS"

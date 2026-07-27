#!/bin/bash
#SBATCH -p ckeller1
#SBATCH --qos=high_p
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --array=0-18%4
#SBATCH -J qc_variance_n19
#SBATCH -o logs/qc_variance_n19_%A_%a.out

module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate

# single-threaded pipeline (numpy ops on small per-channel arrays, h5py reads) --
# avoid BLAS oversubscription beyond the 1 requested CPU
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

SUBJECT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" /oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/cohorts/subjects_qc_variance_remaining.txt)
echo "Array task $SLURM_ARRAY_TASK_ID -> subject $SUBJECT"
python -m ieeg_ehr.qc.run_pipeline --subjects "$SUBJECT" \
    --output-dir /oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/analysis/qc_variance

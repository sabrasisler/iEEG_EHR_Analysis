#!/bin/bash
#SBATCH --job-name=med_mme_curves
#SBATCH --partition=owners
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --output=logs/med_mme_curves_%j.out
#
# Estimated active-MME curves, one panel per session with an opioid dose, plus
# the unit-dose, totals and superposition checks. Input is the 98 MAR CSVs.
# Writes analysis/meds/pk_curves/mme_curves/<run>_<stamp>/ on Oak.
#
#   sbatch sbatch/med_mme_curves.sbatch
#   SCRATCH_RUN=1 sbatch sbatch/med_mme_curves.sbatch    # throwaway, to scratch/

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m pytest -q "${SLURM_SUBMIT_DIR}/tests/test_med_mme.py"
python -m ieeg_ehr.med_analysis.plot_mme_curves ${SCRATCH_RUN:+--scratch}

#!/bin/bash
#SBATCH --job-name=bp_checks
#SBATCH --partition=ckeller1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --output=logs/bp_checks_%j.out
#
# The seven check/diagnostic figures for a band-power run.
#
#   sbatch --export=ALL,RUN_DIR="$RUN_DIR" sbatch/bandpower_checks.sbatch
#   # one figure only, or a different cell:
#   sbatch --export=ALL,RUN_DIR="$RUN_DIR",EXTRA="--figure residuals --region M1 --band beta" \
#       sbatch/bandpower_checks.sbatch
#
# Cheap: the per-cell figures refit ONE cell from its saved frame (~1-15 s) and
# everything else reads the run's tables. The coverage figure is the slow part --
# it loads channel_meta for every subject and renders a nilearn glass brain.
#
# NOT srun: the owners partition refuses interactive jobs, and this is small
# enough that queueing on ckeller1 costs less than the render.

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

: "${RUN_DIR:?set RUN_DIR to a band-power run directory}"
EXTRA=${EXTRA:-}

echo "=== bandpower checks | $(date) ==="
echo "    run: ${RUN_DIR}"
python -m ieeg_ehr.analysis.plot_bandpower_checks --run-dir "${RUN_DIR}" ${EXTRA}
echo "=== done ($(date)) ==="

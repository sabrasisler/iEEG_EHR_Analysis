#!/bin/bash
#SBATCH --job-name=fr_clust
#SBATCH --partition=ckeller1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --output=logs/fr_clust_%j.out
#
# Cluster-based permutation test on the NATIVE 0.5 Hz region x frequency map.
# Read docs/cluster_permutation.md first; the run's METHODS.md records the four
# places this differs from the 50-log-bin version.
#
#   sbatch sbatch/fullres_cluster.sbatch
#
# ONE JOB, NOT AN ARRAY, and it fits on the lab node. The expensive part of a
# cluster test is the null, and both nulls here are cheap because neither refits
# a mixed model: the sign-flip null multiplies an in-memory (51, 21, 499) matrix
# by +/-1, and the predictor-shuffle null recomputes per-subject OLS slopes as a
# handful of matmuls (fullres_cells.coef_blocks groups cells by missingness so
# the per-column fallback never runs). Measured on the smoke run: 200 sign-flips
# and 20 shuffles in ~1 s each, 578 MB peak -- so 10,000 + 2,000 over two arms is
# minutes, not hours.
#
# WHY NOT ON THE MIXED-MODEL z. A cluster null needs the whole map per
# permutation; the grid cost 415 CPU-minutes of mixedlm fitting, and a fixed
# effect cannot be sign-flipped because the exchangeable unit is the epoch -> NRS
# pairing. This tests the two-stage statistic instead, which is stated on the
# figure so no reader takes the outlines for the grid's own inference.

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

# The permutation loops are one process doing many small matmuls; a full BLAS
# thread pool per call costs more in contention than it wins.
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2

EXTRA=${EXTRA:-}

echo "=== fullres cluster permutation | $(date) ==="
python -m ieeg_ehr.analysis.run_fullres_cluster ${EXTRA}
echo "=== done ($(date)) ==="

#!/bin/bash
#SBATCH --job-name=dom_model
#SBATCH --partition=ckeller1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --output=logs/dom_model_%j.out
#
# ONE mixed model per band with PROCESSING DOMAIN as a fixed effect and the atlas
# parcel as a nested random slope.
#
#   sbatch sbatch/domain_model.sbatch
#   # timing test on one band first:
#   sbatch --export=ALL,EXTRA="--bands delta" sbatch/domain_model.sbatch
#
# SIZE. This is the largest fit in the project: every parcel in every domain in
# one model, so ~120k rows against the band-power runs' 1-38k, with ~2,500
# channel random intercepts spread over ~50 subject groups. The per-group design
# stays small (a subject's own channels and parcels), which is what makes it
# tractable at all -- but budget minutes per band, not seconds, and run a single
# band first to measure it.
#
# WHY ONE MODEL AND NOT A MAP. "Do domains differ" is a contrast BETWEEN domains;
# two stratified estimates cannot support a claim that they differ. The omnibus is
# a joint Wald on the interaction block (an LRT would be invalid -- REML
# likelihoods are not comparable across fixed-effects designs).

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

EXTRA=${EXTRA:-}

echo "=== domain model | $(date) ==="
python -m ieeg_ehr.analysis.run_domain_model ${EXTRA}
echo "=== done ($(date)) ==="

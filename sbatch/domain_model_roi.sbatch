#!/bin/bash
#SBATCH --job-name=dom_roi
#SBATCH --partition=ckeller1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --output=logs/dom_roi_%j.out
#
# The domain model at the ROI LEVEL: one mixed model per band, domain as the
# fixed effect, and the region NOT in the model at all.
#
#   sbatch sbatch/domain_model_roi.sbatch
#   # timing test on one band first:
#   sbatch --export=ALL,EXTRA="--bands delta" sbatch/domain_model_roi.sbatch
#
# HOW THIS DIFFERS FROM sbatch/domain_model.sbatch
#   --unit roi            a channel's region is the hand-built ROI (S1, dACC,
#                         aIns...) instead of the DK parcel, and it enters the
#                         model ONLY as the domain lookup. The parcel-nested
#                         random slope is therefore gone -- it was the smallest
#                         variance component in every band and the documented
#                         cause of the non-positive-definite Hessians.
#   --roi-scheme pain_domains_v3
#                         v2 plus the insula, split by coordinate: aIns joins
#                         Affective, pIns joins Sensory.
#   --insula-threshold    PINNED, not re-derived. This is the cut from the
#                         plot_insula_split run that was reviewed on a glass
#                         brain before this model was allowed to use it; letting
#                         the job recompute a median would mean fitting a split
#                         nobody looked at.
#
# ROWS. Still one channel x one epoch -- nothing is averaged. Fewer rows than
# the parcel run only because the ROIs no domain claims (PCC, Hippocampus, Basal
# Ganglia, Parietal/MTL/Lateral Temporal) are dropped, and more because insula
# is back. Budget minutes per band; run one band first to measure it.

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# The reviewed cut: pooled median MNI y over 502 insular pairs in 46 of the 51
# cohort subjects (insula_ap_20260921-144743).
INSULA_THRESHOLD=${INSULA_THRESHOLD:--2.245993821300736}
EXTRA=${EXTRA:-}

echo "=== domain model, ROI unit | insula cut y=${INSULA_THRESHOLD} | $(date) ==="
python -m ieeg_ehr.analysis.run_domain_model \
    --unit roi \
    --roi-scheme pain_domains_v3 \
    --insula-threshold "${INSULA_THRESHOLD}" \
    ${EXTRA}
echo "=== done ($(date)) ==="

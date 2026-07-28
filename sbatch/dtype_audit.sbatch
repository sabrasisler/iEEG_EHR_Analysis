#!/bin/bash
# P0.6 dtype audit — one job, four legs (see ieeg_ehr/features/dtype_audit.py).
#
# Sized for leg D, the only leg that touches raw data: it loads ONE run (the
# subject's smallest, per the registry) at float32 and re-references a handful
# of pairs, so it needs far less than a run_pipeline task's 24-32 GB. 32 GB is
# comfortable headroom for a ~2 GB raw run; raise it only if leg D OOMs on a
# subject whose smallest run is unusually large.
#
#   sbatch sbatch/dtype_audit.sbatch
#   sbatch --export=ALL,SUBJECTS="085 071" sbatch/dtype_audit.sbatch
#
#SBATCH -p normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH -J dtype_audit
#SBATCH -o logs/dtype_audit_%j.out

module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate

# Keep caches off $HOME (15 GB, NFS) per the cluster policy.
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$SCRATCH/.cache}
export MPLCONFIGDIR=${MPLCONFIGDIR:-$SCRATCH/.cache/matplotlib}

: "${LABEL:=p0.6}"

if [ -n "${SUBJECTS:-}" ]; then
    python -m ieeg_ehr.features.dtype_audit --label "$LABEL" --subjects $SUBJECTS
else
    python -m ieeg_ehr.features.dtype_audit --label "$LABEL"
fi

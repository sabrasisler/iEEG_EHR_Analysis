#!/bin/bash
#SBATCH --job-name=bipolar_mask
#SBATCH --partition=ckeller1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --output=logs/bipolar_mask_%j.out
#
# Roll raw_voltage (projected to pairs) OR bipolar_variance into the pair-keyed
# bipolar MASK, for every subject that has BOTH inputs.
#
# A SINGLE job, not an array, on purpose: each subject is a couple of seconds of
# CSV reading and a vectorised merge (7 subjects took 23 s), so 82 array tasks
# would spend more time being scheduled than doing work. The module already walks
# the subject list internally and NOTEs anything it skips.
#
# Idempotent and re-runnable. Subjects missing either input are skipped and named,
# never written with the missing half zero-filled -- a label must not claim a
# rollup it did not have. So re-running after the pending bipolar_excl array
# (36181325) lands simply adds the five OOM-retry subjects.
#
# Safe to run before the PSD hop audit resolves: the bipolar variance metric is
# computed on the TIME-DOMAIN signal on a 2 s grid, independent of the PSD hop, so
# this level is unaffected by the 60s-hop problem (sub-247's variance CSV is 2 s
# even though its PSD is 60 s).

set -euo pipefail

# ~/bin/git ahead of /usr/bin so git provenance is recordable ON THE NODE. Without
# it the login node's git 1.8.3.1 cannot read a worktree gitfile and every sidecar
# records commit "no-git" (SCRATCHPAD, 2026-07-27).
export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

# RUN THE CODE THIS JOB WAS SUBMITTED FROM. `ieeg_ehr` is pip-installed editable
# against the MAIN checkout, so a bare `python -m ieeg_ehr...` executes that tree
# regardless of where sbatch was invoked -- silently decoupling "what ran" from the
# recorded commit hash. No-op when submitting from the main checkout.
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
python -c "import ieeg_ehr; print('ieeg_ehr:', ieeg_ehr.__file__)"

BIPOLAR_LABEL="${BIPOLAR_LABEL:-std10}"

# Optional subject restriction, space-separated:
#   sbatch --export=ALL,SUBJECTS="236" sbatch/build_bipolar_mask.sbatch
# Default (unset) sweeps EVERY subject with both inputs, which is usually what you
# want: the rollup is idempotent and takes ~5 min for the whole cohort, so a full
# sweep also picks up any subject that gained std10 since the last run. Restrict
# when you want a fast targeted rebuild, or to keep one subject's provenance
# separable from a cohort-wide pass.
SUBJECTS="${SUBJECTS:-}"

echo "=== build_bipolar_mask: bipolar_variance/${BIPOLAR_LABEL} + pinned raw-voltage mask ($(date)) ==="
echo "=== subjects: ${SUBJECTS:-<all with both inputs>} ==="

python -m ieeg_ehr.qc.build_bipolar_mask \
    --bipolar-variance-label "$BIPOLAR_LABEL" \
    ${SUBJECTS:+--subjects $SUBJECTS}

echo "=== done ($(date)) ==="

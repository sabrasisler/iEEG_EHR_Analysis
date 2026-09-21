#!/bin/bash
#SBATCH --job-name=bp_mixed
#SBATCH --partition=ckeller1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=logs/bp_mixed_%j.out
#
# Band-power mixed-effects models: 20 regions x 6 bands = 120 fits, then BH and
# the two figures. ONE JOB, not an array -- the whole thing is minutes.
#
#   sbatch sbatch/bandpower_mixed.sbatch
#   # then, if the permutation null is wanted:
#   sbatch --export=ALL,RUN_DIR="$RUN_DIR" sbatch/bandpower_perm_array.sbatch
#   sbatch --export=ALL,RUN_DIR="$RUN_DIR" sbatch/bandpower_collect.sbatch
#
# TIME. Measured on 6 test cells: 0.4-6.7 s per fit, median ~1.6 s, and the
# reduced model (fitted here, unlike the 9,723-cell grid) doubles that. The
# Lustre read of the epoch-mean view dominates -- 20 regions x ~1.5 GB
# cohort-wide, decoding only the 463 kept bins.
#
# MEMORY. One region's rows across the 463 kept bins is ~70 MB at the largest
# region, and the band aggregation exponentiates it in float64, so ~600 MB
# transient for Lateral Temporal. 32 GB is generous.
#
# REGIONS. roi_v2_ofc = roi_v2 with mOFC and lOFC fused into one OFC. Pass
# EXTRA="--roi-scheme roi_v2" to keep them split.

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

EXTRA=${EXTRA:-}

# A region name and a reason both contain SPACES, and ${EXTRA} expands unquoted --
# so they get their own variables and are passed as array elements. Putting them
# in EXTRA silently split "Lateral Temporal" into two arguments and killed a run.
EXCLUDE_REGIONS=${EXCLUDE_REGIONS:-}
EXCLUDE_REASON=${EXCLUDE_REASON:-}
args=()
[ -n "${EXCLUDE_REGIONS}" ] && args+=(--exclude-regions "${EXCLUDE_REGIONS}")
[ -n "${EXCLUDE_REASON}" ] && args+=(--exclude-reason "${EXCLUDE_REASON}")
# `${args[@]+...}` and NOT a bare "${args[@]}": under `set -u` bash treats an
# EMPTY array's expansion as an unbound variable and kills the job before
# python starts. It only bites when neither exclusion is passed, which is why
# it survived every run that used them.

echo "=== bandpower mixed models | $(date) ==="
RUN_DIR=$(python -m ieeg_ehr.analysis.run_bandpower_mixed --stage fit \
    ${args[@]+"${args[@]}"} ${EXTRA} | tail -1)
echo "    run: ${RUN_DIR}"
# ${EXTRA} MUST be passed to collect as well, not just to fit. The collect
# stage re-derives the region list from --roi-scheme and re-reads --dx-model
# to decide which figures and METHODS sections to write, so omitting it
# silently falls back to the DEFAULT scheme -- which on a pain_domains_v2
# run means looking for roi_v2 region names that are not in the cells
# table, an empty pivot, and a numpy "zero-size array" crash after the
# fits have already succeeded.
python -m ieeg_ehr.analysis.run_bandpower_mixed --stage collect \
    --run-dir "${RUN_DIR}" ${EXTRA}
echo "=== done ($(date)) ==="

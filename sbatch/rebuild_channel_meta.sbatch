#!/bin/bash
#SBATCH --job-name=chan_meta
#SBATCH --partition=ckeller1
#SBATCH --qos=high_p
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=12GB
#SBATCH --output=logs/rebuild_channel_meta_%j.out
#
# Refresh every subject's channel_meta table after views/channel_meta.COLUMNS grows.
#
# WHY THIS IS A JOB AND NOT AUTOMATIC. build() rebuilds a stale table on demand, but
# only when something calls it -- and the plotting scripts READ these tables
# directly, skipping any subject whose table predates the columns they need. So a
# schema bump silently shrinks the cohort of a figure until every table is
# refreshed. This makes that explicit and countable.
#
# Serial, not an array: each subject is a handful of NWB metadata opens (~0.3 s per
# run) and the whole cohort is minutes. An array would spend longer scheduling.
#
#   sbatch sbatch/rebuild_channel_meta.sbatch

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

SPLIT=${SPLIT:-discovery}

echo "=== rebuilding channel_meta | split=$SPLIT ($(date)) ==="
python -m ieeg_ehr.views.channel_meta --split "$SPLIT" --overwrite
echo "=== done ($(date)) ==="

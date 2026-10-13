#!/bin/bash
# After a bipolar_psd array: list every planned run that has no finished NWB,
# with the last error its task logged, for re-conversion of the raw file.
#
# Reads the plan, not just the logs, so a run lost to a timeout or a killed
# task is listed too ("no error logged"). Also deletes that run's leftover
# .partial.nwb. Only runs in TASK_FILE are touched, so another array writing
# to the tree at the same time is safe.
#
# Output: OUT, tab-separated (errors contain commas):
#   subject  session  run  raw_file_path  error
#
# Usage (from the directory the array was submitted from -- logs/ is relative):
#   C=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/cohorts
#   P=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/preprocessed/bipolar_psd
#   TASK_FILE=$C/bipolar_psd_fullres_tasks_discovery.txt ARRAY_JOB=<array id> \
#   OUT=$P/failed_runs_discovery.tsv \
#       sbatch --dependency=afterany:<array id> sbatch/collect_bipolar_psd_failures.sbatch
#
#SBATCH -p normal
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB
#SBATCH -J bipolar_psd_failures
#SBATCH -o logs/bipolar_psd_failures_%j.out

set -euo pipefail
: "${TASK_FILE:?}" "${ARRAY_JOB:?}" "${OUT:?}"

# A running task's in-progress run has no NWB yet, so it would be listed and its
# .partial.nwb deleted from under it. Tasks only COMPLETING when the
# afterany dependency releases this job are already done writing.
if [ -n "$(squeue -h -t PENDING,RUNNING,SUSPENDED -j "$ARRAY_JOB" 2>/dev/null)" ]; then
    echo "array $ARRAY_JOB still has tasks in the queue; run after it ends" >&2
    exit 1
fi

ROOT=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler/preprocessed/bipolar_psd
REGISTRY=/oak/stanford/groups/ckeller1/data/iEEG_EHR/iEEG_NWB/sherlock_file_registry.csv

# token <TAB> last line of its traceback, from every task log of the array.
# A failure is "<date> <time> ERROR <token> failed:" followed by the traceback,
# which ends at the next timestamped line.
ERRORS=$(mktemp)
awk '
  / ERROR [0-9]+:[0-9]+:[A-Za-z0-9]+ failed:$/ { if (tok) print tok "\t" last; tok = $4; last = ""; next }
  /^20[0-9][0-9]-/                             { if (tok) print tok "\t" last; tok = ""; next }
  tok && NF                                    { last = $0 }
  END                                          { if (tok) print tok "\t" last }
' logs/bipolar_psd_"${ARRAY_JOB}"_*.out > "$ERRORS"

printf 'subject\tsession\trun\traw_file_path\terror\n' > "$OUT"
n_planned=0
n_missing=0
for token in $(cat "$TASK_FILE"); do
    n_planned=$((n_planned + 1))
    IFS=: read -r s ses r <<< "$token"
    nwb=$ROOT/sub-$s/ses-$ses/sub-${s}_ses-${ses}_run-${r}_bipolar_psd_fullres.nwb
    [ -f "$nwb" ] && continue
    n_missing=$((n_missing + 1))
    rm -f "${nwb%.nwb}.partial.nwb"
    err=$(awk -F'\t' -v t="$token" '$1 == t { e = $2 } END { print e }' "$ERRORS")
    raw=$(awk -F, -v s="sub-$s" -v ses="ses-$ses" -v r="run-$r" \
              '$1 == s && $2 == ses && $3 == r { print $4 }' "$REGISTRY")
    printf '%s\t%s\t%s\t%s\t%s\n' "$s" "$ses" "$r" "$raw" \
        "${err:-no error logged: the task timed out or was killed}" >> "$OUT"
done
rm -f "$ERRORS"
echo "$n_missing of $n_planned planned runs have no NWB -> $OUT"

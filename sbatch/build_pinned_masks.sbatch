#!/bin/bash
# Stages 2 and 4 of the pinned-mask chain: OR the per-type exclusions into a
# combined mask.
#
#   STAGE=intermediate -> gross-std3_satmargin15_sw        (3 types; feeds flatline)
#   STAGE=final        -> gross-std3_satmargin15_sw_logz4  (4 types) + summarize
#
# IMPORTANT: build_mask has NO --subjects flag. It globs every subject-session
# present in ALL of the chosen type dirs and rebuilds the whole mask, so this
# OVERWRITES the existing mask files as well as adding new ones. That is safe
# (same inputs -> same outputs, it is a deterministic OR) but it is a rewrite of
# real data, so this job md5s the mask dir before and after and reports exactly
# what changed. Expected result: existing files identical, new files added.
#
# Usage (submit FROM THE REPO ROOT):
#   sbatch --export=ALL,STAGE=intermediate sbatch/build_pinned_masks.sbatch
#   sbatch --export=ALL,STAGE=final        sbatch/build_pinned_masks.sbatch
#
#SBATCH -p normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH -J build_pinned_mask
#SBATCH -o logs/build_pinned_mask_%j.out

set -euo pipefail
module load python/3.12
source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate

OAK=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler
LEVEL=$OAK/qc/raw_voltage
INTERMEDIATE=gross-std3_satmargin15_sw
FINAL=gross-std3_satmargin15_sw_logz4
LOGZ4=logz4_masked-$INTERMEDIATE
: "${STAGE:?set STAGE=intermediate or STAGE=final}"

case "$STAGE" in
  intermediate) LABEL=$INTERMEDIATE ;;
  final)        LABEL=$FINAL ;;
  *) echo "STAGE must be 'intermediate' or 'final', got '$STAGE'"; exit 1 ;;
esac

OUT=$LEVEL/masks/$LABEL
BEFORE=$(mktemp); AFTER=$(mktemp)
trap 'rm -f "$BEFORE" "$AFTER"' EXIT

echo "=== $STAGE -> $LABEL"
echo "--- fingerprinting existing mask files before rebuild"
if [ -d "$OUT" ]; then (cd "$OUT" && md5sum *.csv 2>/dev/null | sort -k2) > "$BEFORE"; else : > "$BEFORE"; fi
echo "    $(wc -l < "$BEFORE") existing files"

if [ "$STAGE" = intermediate ]; then
  python -m ieeg_ehr.qc.build_mask --level-root "$LEVEL" --label "$LABEL" \
      --types gross_artifact,saturation,square_wave \
      --gross_artifact std3 --saturation pct0_marginfrac0.15 --square_wave frac0.9
else
  python -m ieeg_ehr.qc.build_mask --level-root "$LEVEL" --label "$LABEL" \
      --gross_artifact std3 --saturation pct0_marginfrac0.15 --square_wave frac0.9 \
      --flatline "$LOGZ4"
fi

echo "--- fingerprinting after rebuild"
(cd "$OUT" && md5sum *.csv 2>/dev/null | sort -k2) > "$AFTER"
echo "    $(wc -l < "$AFTER") files now"

echo "--- NEW files (expected: the previously-missing subjects)"
comm -13 <(awk '{print $2}' "$BEFORE") <(awk '{print $2}' "$AFTER") | sed 's/^/      + /'
echo "--- files that EXISTED BEFORE and CHANGED (expected: none)"
changed=$(comm -12 <(awk '{print $2}' "$BEFORE") <(awk '{print $2}' "$AFTER") | while read -r f; do
  b=$(grep -F "  $f" "$BEFORE" | awk '{print $1}')
  a=$(grep -F "  $f" "$AFTER"  | awk '{print $1}')
  [ "$b" = "$a" ] || echo "      ! $f"
done)
if [ -n "$changed" ]; then echo "$changed"; echo "    *** PRE-EXISTING MASKS CHANGED — INVESTIGATE ***"; else echo "      (none — rebuild was idempotent for existing subjects)"; fi

if [ "$STAGE" = final ]; then
  echo "--- summarizing $LABEL"
  python -m ieeg_ehr.qc.summarize_exclusions --mask-dir "$OUT"
fi

echo "DONE $STAGE"

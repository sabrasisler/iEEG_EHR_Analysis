#!/bin/bash
#SBATCH --job-name=med_figures
#SBATCH --partition=owners
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --output=logs/med_figures_%j.out
#
# The four analgesic administration-pattern figures, plus the tidy table they are
# all rendered from.
#
# Input is 98 small EHR CSVs, not the PSD cache, so this is minutes and megabytes
# rather than an array job -- the whole thing runs in one task. The only reason it
# is not interactive is the standing rule against Python on the login node.
#
# Figures are written to analysis/meds/<question>/<output_type>/<run>_<stamp>/ on
# Oak, each with its own provenance.json and CSV tables.
#
#   sbatch sbatch/med_figures.sbatch
#   SCRATCH_RUN=1 sbatch sbatch/med_figures.sbatch      # throwaway, to scratch/
#   FIGURES="burden timing" sbatch sbatch/med_figures.sbatch

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

QUESTION=${QUESTION:-administration_patterns}
MIN_ADMIN=${MIN_ADMIN:-20}
MAX_PANELS=${MAX_PANELS:-6}
MAX_DRUGS=${MAX_DRUGS:-4}
MAX_ROWS=${MAX_ROWS:-5}
FIGURES=${FIGURES:-table burden timing hospital peth}
DEST=${SCRATCH_RUN:+--scratch}

COMMON=(--question "$QUESTION" --min-admin "$MIN_ADMIN")

echo "=== analgesic med figures | question=$QUESTION | figures=$FIGURES ($(date)) ==="

for fig in $FIGURES; do
    case "$fig" in
        table)
            python -m ieeg_ehr.med_analysis.build_admin_table \
                "${COMMON[@]}" --with-hours ${DEST}
            ;;
        burden)
            python -m ieeg_ehr.med_analysis.plot_admin_burden \
                "${COMMON[@]}" ${DEST}
            ;;
        timing)
            python -m ieeg_ehr.med_analysis.plot_admin_timing \
                "${COMMON[@]}" --max-panels "$MAX_PANELS" ${DEST}
            ;;
        hospital)
            python -m ieeg_ehr.med_analysis.plot_hospital_day \
                "${COMMON[@]}" --max-drugs "$MAX_DRUGS" ${DEST}
            ;;
        peth)
            python -m ieeg_ehr.med_analysis.plot_coadmin_peth \
                "${COMMON[@]}" --max-rows "$MAX_ROWS" ${DEST}
            ;;
        *)
            echo "unknown figure '$fig' -- expected one of: table burden timing hospital peth" >&2
            exit 1
            ;;
    esac
done

echo "=== done ($(date)) ==="

#!/bin/bash
#SBATCH --job-name=dom_glass
#SBATCH --partition=ckeller1
#SBATCH --qos=high_p
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --output=logs/dom_glass_%j.out
#
# Glass brains: each electrode coloured by the pain slope of the domain it
# belongs to, on the SAME numeric scale as fig_domain_summary.png.
#
#   RUN_DIR=<domain run> CMAP=berlin sbatch sbatch/domain_glass.sbatch
#   CMAPS="berlin coolwarm RdBu_r" sbatch sbatch/domain_glass.sbatch   # compare
#
# VERSIONED, NEVER OVERWRITING. Every invocation writes a new
# <label>_<timestamp> folder under <run>/glass_brain/, each with its own
# provenance.json and colour_source.csv saying exactly which beta produced
# which colour. Iterating on appearance cannot replace a version already in a
# slide.
#
# COLORMAP. The default is `berlin`, not the heatmap's RdBu_r, on purpose: on a
# heatmap a near-zero cell renders white between gridlines and still reads as a
# cell, but the glass brain background IS white, so an RdBu_r domain with a
# slope near zero becomes invisible dots. berlin's midpoint is near-black with
# the same blue-negative / red-positive direction. RdBu_r stays available for
# putting a brain beside the heatmap.
#
# Needs nilearn + nibabel (already in the shared venv). The MNI152 brain mask
# ships with nilearn, so the out-of-brain filter works with NO network.

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${SCRATCH}/mplconfig"

DERIV=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler
RUN_DIR=${RUN_DIR:-$DERIV/analysis/pain/bandpower/domain_model/paperbands6hg200-paindomainsv3-roiunit-noparcel/domain_mixedlm_20260921-153105}
CMAPS=${CMAPS:-${CMAP:-berlin}}
BANDS=${BANDS:-"delta beta gamma"}
EXTRA=${EXTRA:-}

for cmap in ${CMAPS}; do
    echo "=== glass brains | cmap=${cmap} | $(date) ==="
    python -m ieeg_ehr.analysis.plot_domain_glass \
        --run-dir "$RUN_DIR" --cmap "$cmap" --bands ${BANDS} ${EXTRA}
done
echo "=== done ($(date)) ==="

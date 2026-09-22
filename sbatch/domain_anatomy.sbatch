#!/bin/bash
#SBATCH --job-name=dom_anat
#SBATCH --partition=ckeller1
#SBATCH --qos=high_p
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --output=logs/dom_anat_%j.out
#
# Electrode coverage by processing domain. Three figures, one versioned folder:
#   fig_anatomy_glass_xz.png    two views (sagittal + axial), for a slide
#   fig_anatomy_glass_lyrz.png  all four views, to crop from
#   fig_anatomy_bars.png        reach (% of patients) over depth (n electrodes)
#
#   RUN_DIR=<domain run> EXTRA="--label poster --brain-gap 0.0" \\
#       sbatch sbatch/domain_anatomy.sbatch
#   EXTRA="--display-modes xz" sbatch sbatch/domain_anatomy.sbatch
#   EXTRA=--domains-only sbatch sbatch/domain_anatomy.sbatch
#
# ANATOMY ONLY -- no slope, no p-value, no colour scale. The effect-coloured
# counterpart is sbatch/domain_glass.sbatch. Versioned under the run's
# glass_brain/ folder exactly as that one is, and never overwrites.

set -euo pipefail
export PATH="$HOME/bin:$PATH"
module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${SCRATCH}/mplconfig"
DERIV=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler
RUN_DIR=${RUN_DIR:-$DERIV/analysis/pain/bandpower/domain_model/paperbands6hg200-paindomainsv3-roiunit-noparcel/domain_mixedlm_20260921-153105}
python -m ieeg_ehr.analysis.plot_domain_anatomy --run-dir "$RUN_DIR" ${EXTRA:-}

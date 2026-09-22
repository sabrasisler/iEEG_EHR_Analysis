#!/bin/bash
#SBATCH --job-name=cohort_desc
#SBATCH --partition=ckeller1
#SBATCH --qos=high_p
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --output=logs/cohort_desc_%j.out
#
# Cohort description for a domain-model run: what the pain data looks like.
#   fig_subject_grid.png     a panel per subject -- the contact sheet the
#                            example is CHOSEN from, not an afterthought
#   fig_example_subject.png  one subject large, hours since admission vs score
#   fig_cohort_violins.png   assessments per patient, mean score, score range
#
#   RUN_DIR=<domain run> sbatch sbatch/cohort_description.sbatch
#   EXTRA="--example-subject sub-209" sbatch sbatch/cohort_description.sbatch
#   EXTRA="--source ehr" sbatch sbatch/cohort_description.sbatch
#
# Versioned under the run's cohort/ folder; never overwrites.

set -euo pipefail
export PATH="$HOME/bin:$PATH"
module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${SCRATCH}/mplconfig"
DERIV=/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler
RUN_DIR=${RUN_DIR:-$DERIV/analysis/pain/bandpower/domain_model/paperbands6hg200-paindomainsv3-roiunit-noparcel/domain_mixedlm_20260921-153105}
python -m ieeg_ehr.analysis.plot_cohort_description --run-dir "$RUN_DIR" ${EXTRA:-}

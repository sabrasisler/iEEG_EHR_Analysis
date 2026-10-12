#!/bin/bash
#SBATCH --job-name=atlas_coverage
#SBATCH --partition=ckeller1
#SBATCH --qos=high_p
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --output=logs/atlas_coverage_%j.out
#
# Subjects and bipolar pairs per atlas label, hemispheres merged, for the
# reference run's cohort. Two bar panels + CSVs under
# analysis/electrode_localization/<atlas>_coverage/label_counts/.
#
#   ATLAS=dk    labels each pair by its anode's Desikan-Killiany label
#   ATLAS=dk_midpoint labels each pair's anode-cathode MNI midpoint in
#               FreeSurfer's cvs_avg35_inMNI152 aparc+aseg (DK + aseg)
#   ATLAS=hcpex labels each pair's anode-cathode MNI midpoint in HCPex v1.1
#               (reads contact coordinates from the raw NWB of each run that adds pairs)
#   ATLAS=cabnp the same midpoint, labeled with a CAB-NP v1.1 network: cortex
#               through HCPex's Glasser areas, subcortex through CAB-NP's voxels
#
#   ATLAS=hcpex sbatch sbatch/atlas_coverage.sbatch
#   ATLAS=cabnp sbatch sbatch/atlas_coverage.sbatch
#   ATLAS=hcpex EXTRA="--reference-run <run dir>" sbatch sbatch/atlas_coverage.sbatch

set -euo pipefail
export PATH="$HOME/bin:$PATH"
module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${SCRATCH}/mplconfig"
python -m ieeg_ehr.analysis.plot_atlas_coverage --atlas "${ATLAS:?set ATLAS=dk or ATLAS=hcpex}" ${EXTRA:-}

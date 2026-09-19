#!/bin/bash
#SBATCH --job-name=band_surf
#SBATCH --partition=ckeller1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --output=logs/band_surf_%j.out
#
# The region x band heatmap re-projected onto the fsaverage cortical surface:
# one brain per band, several views each, plus a subcortical slice panel.
# Writes into <RUN_DIR>/surface/.
#
#   sbatch --export=ALL,RUN_DIR="/oak/.../paperbands6_mixedlm_20260917-115938" \
#          sbatch/band_surface.sbatch
#
# TIME. Rendering dominates and is single-threaded matplotlib: ~35 s per
# full-resolution fsaverage panel (163k vertices, drawn twice -- once plain, once
# coloured -- plus the significance contours). The default 3 views x 2
# hemispheres x 6 bands, plus an 18-panel contact sheet, is ~55 panels, so ~35
# min. EXTRA="--fs-subject fsaverage6" is a quarter of the vertices and about
# four times faster; use it while iterating on layout, not for a talk.
#
# MEMORY. One hemisphere of fsaverage is ~330k faces and each panel holds a few
# float64 arrays that size, plus the cropped RGBA images of every panel in the
# figure at once (~4 MB each). 16 GB is generous; measured peak is ~3 GB.
#
# CPU. --cpus-per-task=2 rather than 1 only because nibabel's mgz decompression
# and the PNG encode overlap the render a little. There is no parallelism to buy
# here; render more bands per job rather than asking for more cores.

set -euo pipefail

export PATH="$HOME/bin:$PATH"

module load python/3.12
source "$GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate"

export PYTHONPATH="${SLURM_SUBMIT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

: "${RUN_DIR:?set RUN_DIR to a mixed_effects run directory}"
EXTRA=${EXTRA:-}

echo "=== band-power surface projection | $(date) ==="
echo "    run: ${RUN_DIR}"
python -m ieeg_ehr.analysis.plot_band_surface --run-dir "${RUN_DIR}" ${EXTRA}
echo "=== done ($(date)) ==="

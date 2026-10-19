#!/usr/bin/env python3
"""
One small glass brain per paper ROI: where its HCPex territory is, which
electrodes fall in it, and whether it clears the subject floor.

    analysis/electrode_localization/hcpex_paper_coverage/glass_brain/<run>_<timestamp>/

Reads an `hcpex_paper` coverage run: the ROI mapping from its provenance
(`paper_roi_parcels`) and the pairs from its `pair_labels.csv`, so every panel
shows exactly the ROI and the pairs that run counted. Each panel has two views
(sagittal, both hemispheres overlaid, and axial): the ROI's parcels as a
translucent outline, and its bipolar pairs at their virtual electrode
(anode-cathode MNI midpoint). A panel title turns red when the ROI has fewer
subjects than --min-subjects (default 8, the reference run's `min_subjects`).

`roi_summary.csv` lists every ROI with its subject and pair counts, the flag,
and its parcels.

Run on Slurm, never the login node:
    sbatch sbatch/roi_glass_grid.sbatch
"""

import argparse
import json
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis.plot_atlas_coverage import EVENT, HCPEX_DIR, hcpex_names
from ieeg_ehr.analysis.plot_thalamus_glass_brain import draw_territory
from ieeg_ehr.config import hcpex_rois
from ieeg_ehr.med_analysis import style

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_roi_glass_grid.py'
QUESTION = 'hcpex_paper_coverage'
OUTPUT_TYPE = 'glass_brain'
N_COLUMNS = 4
#: Whole-brain window (MNI mm) the territory outlines are binned over.
WINDOW = {'x': (-75, 75), 'y': (-110, 75), 'z': (-55, 85)}
TERRITORY_COLOR = style.BAR_COLOR
PAIR_COLOR = style.NORM_COLOR
FLAG_COLOR = '#c0392b'


def roi_voxels(mapping):
    """{display ROI: (n, 3) MNI voxel centres of all its parcels, both hemispheres}."""
    short, _, _ = hcpex_names()
    img = nib.load(HCPEX_DIR / 'HCPex.nii.gz')
    vol = np.rint(np.asarray(img.dataobj)).astype(int)
    values = {}
    for v, name in short.items():
        values.setdefault(name, []).append(v)
    out = {}
    for key, parcels in mapping.items():
        if key == hcpex_rois.NO_ROI:
            continue
        codes = [v for p in parcels for v in values[p]]
        xyz = nib.affines.apply_affine(img.affine, np.argwhere(np.isin(vol, codes)))
        for roi in hcpex_rois.rois_of(key):
            out[roi] = xyz
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--coverage-run', required=True,
                    help='A plot_atlas_coverage hcpex_paper run directory.')
    ap.add_argument('--min-subjects', type=int, default=8)
    ap.add_argument('--run-name', default=None,
                    help="Default: the coverage run's folder name minus its stamp.")
    args = ap.parse_args()

    io.warn_if_dirty()
    from nilearn import plotting

    coverage_run = Path(args.coverage_run)
    prov = json.loads((coverage_run / 'provenance.json').read_text())
    if prov['params']['atlas'] != 'hcpex_paper':
        raise SystemExit(f'{coverage_run} is not an hcpex_paper run')
    mapping = prov['params']['paper_roi_parcels']
    subjects = prov['subjects']
    rois = hcpex_rois.roi_order(mapping)

    pairs_path = coverage_run / 'pair_labels.csv'
    pairs = io.read_table(pairs_path, on_stale='warn')
    voxels = roi_voxels(mapping)

    rows = []
    for roi in rois:
        hit = pairs[pairs['label'] == roi]
        key = next(k for k in mapping if roi in hcpex_rois.rois_of(k))
        rows.append({'roi': roi, 'n_subjects': hit['subject_id'].nunique(),
                     'n_electrodes': len(hit),
                     'below_min_subjects': hit['subject_id'].nunique() < args.min_subjects,
                     'parcels': ' '.join(mapping[key])})
    summary = pd.DataFrame(rows)
    logger.info('ROI summary:\n%s', summary.drop(columns='parcels').to_string())

    run_name = args.run_name or coverage_run.name.rsplit('_', 1)[0]
    run_dir = config.analysis_run_dir(question=QUESTION, output_type=OUTPUT_TYPE,
                                      run_name=run_name, event=EVENT)
    parents = [io.parent_ref(coverage_run / 'provenance.json', digest=False),
               io.parent_ref(pairs_path, digest=False),
               io.parent_ref(HCPEX_DIR / 'HCPex.nii.gz')]
    params = {**vars(args), 'window_mm': WINDOW}
    io.write_table(summary, run_dir / 'roi_summary.csv', script=SCRIPT, params=params,
                   parents=parents, subjects=subjects)
    io.write_run_provenance(run_dir, script=SCRIPT, params=params, parents=parents,
                            subjects=subjects,
                            extra={'rois_below_min_subjects':
                                   summary.loc[summary['below_min_subjects'], 'roi'].tolist()})

    n_rows = math.ceil(len(rois) / N_COLUMNS)
    panel_w, panel_h = 4.6, 2.3
    fig = plt.figure(figsize=(N_COLUMNS * panel_w, n_rows * panel_h + 0.6))
    for i, row in summary.iterrows():
        r, c = divmod(i, N_COLUMNS)
        left = c / N_COLUMNS
        bottom = 1 - (r + 1) * (panel_h / (n_rows * panel_h + 0.6))
        height = panel_h / (n_rows * panel_h + 0.6)
        display = plotting.plot_glass_brain(
            None, display_mode='xz', figure=fig, annotate=False,
            axes=(left, bottom, 1 / N_COLUMNS, height * 0.82))
        for direction, gax in display.axes.items():
            draw_territory(gax.ax, voxels[row.roi], direction, WINDOW, TERRITORY_COLOR,
                           alpha=0.18, linewidth=0.8)
        pts = pairs.loc[pairs['label'] == row.roi, ['mni_x', 'mni_y', 'mni_z']].to_numpy(float)
        if len(pts):
            display.add_markers(pts, marker_color=PAIR_COLOR, marker_size=6,
                                edgecolors='0.2', linewidths=0.3)
        flagged = row.below_min_subjects
        fig.text(left + 0.5 / N_COLUMNS, bottom + height * 0.84,
                 f'{row.roi}: {row.n_subjects} subj, {row.n_electrodes} pairs'
                 + (f'  (< {args.min_subjects} subj)' if flagged else ''),
                 ha='center', va='bottom', fontsize=10,
                 color=FLAG_COLOR if flagged else style.TEXT_PRIMARY,
                 fontweight='bold' if flagged else 'normal')
    fig.suptitle(f'Paper ROIs on HCPex v1.1: territory (blue outline) and bipolar pairs at '
                 f'the pair midpoint (orange), {len(subjects)} subjects. Red title = fewer '
                 f'than {args.min_subjects} subjects.', fontsize=13, x=0.01, ha='left',
                 va='bottom', y=1.0)
    out = run_dir / 'roi_glass_grid.png'
    fig.savefig(out, dpi=style.DPI, facecolor='white', bbox_inches='tight')
    plt.close(fig)

    io.log_analysis(f'paper ROI glass-brain grid, {len(rois)} ROIs, n={len(subjects)}', run_dir)
    logger.info('figure + table + provenance -> %s', run_dir)


if __name__ == '__main__':
    main()

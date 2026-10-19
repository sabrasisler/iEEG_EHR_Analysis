#!/usr/bin/env python3
"""
Where the thalamic nuclei are, how the paper ROIs group them, and where the
electrodes sit among them.

    analysis/electrode_localization/hcpex_paper_coverage/regions/thalamus/<run>_<timestamp>/

Two figures:

  thalamic_nuclei_slices.png
      HCPex v1.1's 21 thalamic nuclei on axial and coronal SLICES of the MNI152
      template, each nucleus filled in a shade of its paper-ROI colour, outlined,
      and labelled at its centroid in that slice (left hemisphere only, to keep
      the labels readable). Slices, not a projection, because a projection lays
      every nucleus over the ones behind it.
  thalamus_glass_brain.png
      Glass brains zoomed to the thalamus (sagittal, coronal, axial): each paper
      ROI as a translucent outline of its projected territory, and on top the
      bipolar pairs whose virtual electrode (anode-cathode MNI midpoint) lands in
      a thalamic nucleus, coloured by ROI. Outlines, not filled clouds, so a
      group behind another stays visible.

The grouping is NOT fixed here: it is read from the `hcpex_paper` coverage run's
own provenance (`paper_roi_parcels`), so the figures follow whatever ROI mapping
that run used, including an edited `--roi-map`. `thalamic_nuclei.csv` lists
every nucleus with its group, long name, per-hemisphere centroid and electrode
counts.

Run on Slurm, never the login node:
    sbatch sbatch/thalamus_glass_brain.sbatch
"""

import argparse
import json
import logging
import re
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis.plot_atlas_coverage import EVENT, HCPEX_DIR, hcpex_names
from ieeg_ehr.med_analysis import style

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_thalamus_glass_brain.py'
QUESTION = 'hcpex_paper_coverage'
OUTPUT_TYPE = 'regions'
REGION = 'thalamus'
EDGE_COLOR = '0.2'
#: Window (MNI mm) around the thalamus, per 3D axis.
WINDOW = {'x': (-28, 28), 'y': (-42, 8), 'z': (-14, 24)}
#: The 2D axes of each view: a view drops its own axis.
VIEW_AXES = {'x': ('y', 'z'), 'y': ('x', 'z'), 'z': ('x', 'y')}
#: Slice levels, as quantiles of the thalamus's own extent along the cut axis.
SLICE_QUANTILES = (0.15, 0.38, 0.62, 0.85)
#: A nucleus is labelled on a slice only if it has this many voxels there.
MIN_LABEL_VOXELS = 8


def thalamic_nuclei():
    """[(value, short name, hemisphere)] for every HCPex thalamic nucleus."""
    short, long, hemisphere = hcpex_names()
    return [(v, name, hemisphere[v]) for v, name in short.items()
            if v and long[name].startswith('Thal_')]


def zoom(axes_by_view):
    """Crop each (direction, matplotlib axes) to WINDOW, keeping its orientation."""
    for direction, ax in axes_by_view:
        h, v = VIEW_AXES[direction]
        for get, set_, axis in ((ax.get_xlim, ax.set_xlim, h), (ax.get_ylim, ax.set_ylim, v)):
            lo, hi = WINDOW[axis]
            a, b = get()
            set_((lo, hi) if a < b else (hi, lo))


def project(xyz, direction):
    """(horizontal, vertical) columns of an (n, 3) array for one view."""
    h, v = VIEW_AXES[direction]
    return xyz[:, 'xyz'.index(h)], xyz[:, 'xyz'.index(v)]


def draw_territory(ax, xyz, direction, window, colour, alpha=0.12, linewidth=1.3):
    """Translucent fill and outline of a voxel cloud's projection in one glass-brain view.

    An outline, not the voxels themselves, so a territory drawn behind another
    stays visible.
    """
    h_ax, v_ax = VIEW_AXES[direction]
    h_edges = np.arange(window[h_ax][0] - 1, window[h_ax][1] + 2)
    v_edges = np.arange(window[v_ax][0] - 1, window[v_ax][1] + 2)
    centres = ((h_edges[:-1] + h_edges[1:]) / 2, (v_edges[:-1] + v_edges[1:]) / 2)
    occupied, _, _ = np.histogram2d(*project(xyz, direction), bins=[h_edges, v_edges])
    occupied = (occupied > 0).astype(float).T
    ax.contourf(*centres, occupied, levels=[0.5, 1.5], colors=[colour], alpha=alpha,
                zorder=900)
    ax.contour(*centres, occupied, levels=[0.5], colors=[colour], linewidths=linewidth,
               zorder=901)


def shades(groups, members, colours):
    """{nucleus: colour}: each group's hue, lightened step by step across its nuclei."""
    out = {}
    for g in groups:
        base = np.array(mcolors.to_rgb(colours[g]))
        for i, n in enumerate(members[g]):
            t = 0.55 * i / max(len(members[g]) - 1, 1)
            out[n] = tuple(base + (1 - base) * t)
    return out


def nucleus_legend(fig, groups, members, nucleus_colour, colours, full_name, anchor):
    handles = []
    for g in groups:
        handles.append(mpatches.Patch(color='none', label=g))
        handles += [mpatches.Patch(facecolor=nucleus_colour[n], edgecolor=EDGE_COLOR,
                                   label=f'{n}  {full_name[n]}') for n in members[g]]
    legend = fig.legend(handles=handles, loc='center left', bbox_to_anchor=anchor,
                        fontsize=8, frameon=False, handlelength=1.2)
    for text in legend.get_texts():
        if text.get_text() in groups:
            text.set_fontweight('bold')
            text.set_color(colours[text.get_text()])


def slices_figure(img, codes, names, nucleus_colour, voxels, groups, members, colours,
                  full_name, out_path):
    from nilearn import plotting
    from nilearn.datasets import load_mni152_template

    label_img = nib.Nifti1Image(codes.astype(np.int16), img.affine)
    cmap = mcolors.ListedColormap([nucleus_colour[n] for n in names])
    allxyz = np.vstack(list(voxels.values()))
    bg = load_mni152_template(resolution=1)

    fig = plt.figure(figsize=(16, 8.5))
    for row, direction in enumerate(('z', 'y')):
        cuts = [round(float(np.quantile(allxyz[:, 'xyz'.index(direction)], q)))
                for q in SLICE_QUANTILES]
        display = plotting.plot_roi(
            label_img, bg_img=bg, display_mode=direction, cut_coords=cuts,
            cmap=cmap, vmin=0.5, vmax=len(names) + 0.5, alpha=0.9, figure=fig,
            axes=(0.0, 0.52 - 0.48 * row, 0.74, 0.44), annotate=True,
            draw_cross=False, black_bg=False, colorbar=False,
            title='Axial' if direction == 'z' else 'Coronal')
        for i, name in enumerate(names, start=1):
            display.add_contours(nib.Nifti1Image((codes == i).astype(np.int8), img.affine),
                                 levels=[0.5], colors=[EDGE_COLOR], linewidths=0.4)
        k = 'xyz'.index(direction)
        for cut, cut_ax in display.axes.items():
            for name in names:
                xyz = voxels[(name, 'L')]
                here = xyz[np.abs(xyz[:, k] - cut) < 0.5]
                if len(here) >= MIN_LABEL_VOXELS:
                    h, v = project(here, direction)
                    cut_ax.ax.text(h.mean(), v.mean(), name, fontsize=6, ha='center',
                                   va='center', color=style.TEXT_PRIMARY, zorder=2000)
        zoom([(direction, a.ax) for a in display.axes.values()])
    nucleus_legend(fig, groups, members, nucleus_colour, colours, full_name, (0.75, 0.5))
    fig.suptitle(f'HCPex v1.1 thalamic nuclei, coloured by paper ROI '
                 f'(labels on the left hemisphere)', fontsize=12, x=0.01, ha='left')
    fig.savefig(out_path, dpi=style.DPI, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    logger.info('Wrote %s', out_path)


def glass_figure(thal, voxels, groups, members, colours, n_subjects, out_path):
    from nilearn import plotting

    fig = plt.figure(figsize=(15, 5.6))
    display = plotting.plot_glass_brain(None, display_mode='ortho', figure=fig,
                                        axes=(0.0, 0.0, 0.72, 0.9), annotate=False)
    for direction, gax in display.axes.items():
        for g in groups:
            xyz = np.vstack([voxels[(n, hemi)] for n in members[g] for hemi in ('L', 'R')])
            draw_territory(gax.ax, xyz, direction, WINDOW, colours[g])
    for g in groups:
        pts = thal.loc[thal['group'] == g, ['mni_x', 'mni_y', 'mni_z']].to_numpy(float)
        if len(pts):
            display.add_markers(pts, marker_color=colours[g], marker_size=28,
                                edgecolors=EDGE_COLOR, linewidths=0.5)
    zoom([(d, gax.ax) for d, gax in display.axes.items()])

    handles = []
    for g in groups:
        hit = thal[thal['group'] == g]
        handles.append(mlines.Line2D(
            [], [], marker='o', linestyle='', markersize=8, markerfacecolor=colours[g],
            markeredgecolor=EDGE_COLOR,
            label=f'{g} ({len(hit)} pairs, {hit["subject_id"].nunique()} subj)\n'
                  f'  {", ".join(members[g])}'))
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(0.73, 0.5), fontsize=9,
               title='Paper ROI: HCPex nuclei (outline = territory)', title_fontsize=10,
               frameon=False, labelspacing=1.2)
    fig.suptitle(f'Thalamic electrodes at the pair midpoint: {len(thal)} bipolar pairs '
                 f'from {thal["subject_id"].nunique()} of {n_subjects} subjects',
                 fontsize=12, x=0.01, ha='left')
    fig.savefig(out_path, dpi=style.DPI, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    logger.info('Wrote %s', out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--coverage-run', required=True,
                    help='A plot_atlas_coverage hcpex_paper run directory.')
    ap.add_argument('--run-name', default=None,
                    help="Default: the coverage run's folder name minus its stamp.")
    args = ap.parse_args()

    io.warn_if_dirty()

    coverage_run = Path(args.coverage_run)
    prov = json.loads((coverage_run / 'provenance.json').read_text())
    if prov['params']['atlas'] != 'hcpex_paper':
        raise SystemExit(f'{coverage_run} is not an hcpex_paper run')
    mapping = prov['params']['paper_roi_parcels']
    subjects = prov['subjects']

    nuclei = thalamic_nuclei()
    thalamic = {name for _, name, _ in nuclei}
    group_of = {p: roi for roi, parcels in mapping.items() for p in parcels}
    #: The nuclei in the order the ROI mapping lists them.
    names = [p for parcels in mapping.values() for p in parcels if p in thalamic]
    groups = list(dict.fromkeys(group_of[n] for n in names))
    members = {g: [n for n in names if group_of[n] == g] for g in groups}
    colours = style.categorical_colors(groups)
    nucleus_colour = shades(groups, members, colours)
    _, long, _ = hcpex_names()
    #: HCPex long names, readable: `Thal_AnteroventralNucleus` -> `Anteroventral`.
    full_name = {n: re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', long[n].removeprefix('Thal_'))
                 .replace('_', ' ').removesuffix(' Nucleus') for n in names}

    img = nib.load(HCPEX_DIR / 'HCPex.nii.gz')
    vol = np.rint(np.asarray(img.dataobj)).astype(int)
    code_of = {n: i for i, n in enumerate(names, start=1)}
    codes = np.zeros(vol.shape, dtype=int)
    voxels = {}
    for value, name, hemi in nuclei:
        mask = vol == value
        codes[mask] = code_of[name]
        voxels[(name, hemi)] = nib.affines.apply_affine(img.affine, np.argwhere(mask))

    pairs_path = coverage_run / 'pair_labels.csv'
    pairs = io.read_table(pairs_path, on_stale='warn')
    thal = pairs[pairs['hcpex_label'].isin(thalamic)].copy()
    thal['group'] = thal['hcpex_label'].map(group_of)
    logger.info('%d thalamic pair(s) from %d subject(s): %s', len(thal),
                thal['subject_id'].nunique(), thal['group'].value_counts().to_dict())

    rows = []
    for name in names:
        hit = thal[thal['hcpex_label'] == name]
        row = {'nucleus': name, 'name': full_name[name], 'group': group_of[name],
               'n_subjects': hit['subject_id'].nunique(), 'n_electrodes': len(hit)}
        for hemi in ('L', 'R'):
            c = voxels[(name, hemi)].mean(axis=0)
            row.update({f'centroid_{hemi}_{a}': round(float(v), 1) for a, v in zip('xyz', c)})
        rows.append(row)

    run_name = args.run_name or coverage_run.name.rsplit('_', 1)[0]
    run_dir = config.analysis_run_dir(question=QUESTION, output_type=OUTPUT_TYPE,
                                      run_name=run_name, view_scheme=REGION, event=EVENT)
    parents = [io.parent_ref(coverage_run / 'provenance.json', digest=False),
               io.parent_ref(pairs_path, digest=False),
               io.parent_ref(HCPEX_DIR / 'HCPex.nii.gz')]
    params = {**vars(args), 'groups': members, 'window_mm': WINDOW,
              'slice_quantiles': list(SLICE_QUANTILES)}
    io.write_table(pd.DataFrame(rows), run_dir / 'thalamic_nuclei.csv', script=SCRIPT,
                   params=params, parents=parents, subjects=subjects)
    io.write_run_provenance(run_dir, script=SCRIPT, params=params, parents=parents,
                            subjects=subjects,
                            extra={'n_thalamic_pairs': int(len(thal)),
                                   'n_subjects_with_thalamic_pairs':
                                       int(thal['subject_id'].nunique())})

    slices_figure(img, codes, names, nucleus_colour, voxels, groups, members, colours,
                  full_name, run_dir / 'thalamic_nuclei_slices.png')
    glass_figure(thal, voxels, groups, members, colours, len(subjects),
                 run_dir / 'thalamus_glass_brain.png')

    io.log_analysis(f'thalamic nuclei slices + {len(thal)} thalamic pairs glass brain, '
                    f'n={len(subjects)}', run_dir)
    logger.info('figures + table + provenance -> %s', run_dir)


if __name__ == '__main__':
    main()

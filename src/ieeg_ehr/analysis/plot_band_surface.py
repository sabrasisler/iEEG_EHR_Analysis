#!/usr/bin/env python3
"""
The region x band heatmap, re-projected onto a 3D cortical surface: one brain per
frequency band, several views each.

WHAT THIS IS
------------
`fig_band_map.png` is 19 region rows x 6 band columns of mixed-effects beta. It is
the right figure for reading a number off, and the wrong one for seeing SHAPE -- a
reader has to hold 19 region names and their anatomy in their head to notice that
the beta-band positives are a contiguous sensorimotor strip and the delta negatives
are the limbic/subcortical core. This script paints the SAME numbers onto fsaverage
so the anatomy does the remembering.

Nothing is recomputed. The betas, the BH-significance and the colour scale come
from the run's own `band_cells.parquet`; this is a re-rendering, not an analysis.

HOW A REGION BECOMES A PATCH OF CORTEX
--------------------------------------
Backwards through the assignment the analysis already made. A contact got its ROI
from its anode's Desikan-Killiany parcel via `roi_schemes.region_for_dk_label`;
here every DK parcel of the fsaverage `aparc` annotation is pushed through THE SAME
FUNCTION, and each parcel is painted with its region's beta. So the surface cannot
disagree with the heatmap about which region a piece of cortex is in -- there is no
second mapping table to drift.

Two consequences worth stating out loud, both drawn on the figure:

  - A PARCEL IS PAINTED UNIFORMLY. The model estimated one beta per region, not a
    per-vertex map, so the colour is flat within a parcel and the boundaries are
    the atlas's, not the effect's. This is a categorical map wearing a continuous
    colormap; it must not be read as a spatial gradient.
  - COVERAGE IS NOT SHOWN BY COLOUR. Every vertex of a region is painted whether
    that region held 11 subjects or 55. `fig_check_coverage_counts.png` in the same
    run directory is the figure that shows the denominator, and
    `plot_electrode_locations.py` shows where the contacts actually were. A painted
    surface always looks like whole-brain sampling and never is.

GREY MEANS ONE OF THREE THINGS, and the caption says which regions are in which:
outside the ROI scheme (white matter, `unknown`, `frontalpole`, corpus callosum),
excluded from this run for coverage or data-validation reasons, or subcortical --
the four subcortical ROIs (Hippocampus, Amygdala, Thalamus, Basal Ganglia) have no
cortical surface to live on. They get a panel of their own instead -- two axial
cuts through fsaverage's aseg, on the same colour scale -- rather than being
silently dropped, because the delta column in particular is mostly subcortical and
a cortex-only delta brain is a misleading figure on its own. `--no-subcortical`
turns it off.

HEMISPHERES ARE POOLED, SO BOTH SIDES ARE THE SAME
--------------------------------------------------
The model has no hemisphere term -- a left and a right insula contact are both
`Insula` -- so the left and right surfaces here are IDENTICAL BY CONSTRUCTION, not
by a measured symmetry. Both are drawn because a one-hemisphere brain reads as a
claim about that hemisphere. `--hemi left` drops the redundancy when space is
tight.

SIGNIFICANCE
------------
Black outlines around a parcel mean the cell was BH-significant at the run's q,
exactly as in the heatmap, and are drawn from the same `p_bh_reject` column.
Non-significant regions keep their colour rather than being greyed: the heatmap
shows the whole field of estimates and so should this, and thresholding a map is
how a nomination starts looking like a finding.

Run on Slurm, never the login node:

    python -m ieeg_ehr.analysis.plot_band_surface --run-dir <mixed_effects run dir>
"""

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.config import roi_schemes

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_band_surface.py'

#: Subfolder of the parent run directory. The run already owns a provenance.json
#: and a band_cells.parquet; these figures are derived from it and belong beside
#: it, not in a run directory of their own with a duplicate cohort record.
SUBDIR = 'surface'

#: Where to look for a FreeSurfer subject, in preference order. SUBJECTS_DIR first
#: so a caller can point at their own copy; the hardcoded module path last so the
#: script works on a compute node that never loaded the freesurfer module.
#: fsaverage, fsaverage5 and fsaverage6 all verified present there 2026-09-19.
SUBJECTS_DIR_CANDIDATES = (
    lambda: Path(os.environ['SUBJECTS_DIR']),
    lambda: Path(os.environ['FREESURFER_HOME']) / 'subjects',
    lambda: Path('/share/software/user/open/freesurfer/7.4.1/subjects'),
)

#: Views in nilearn's naming. `ventral` earns its place here rather than `dorsal`:
#: OFC and the MTL parcels are invisible from every other angle, and they are two
#: of the regions this project cares most about.
DEFAULT_VIEWS = ('lateral', 'medial', 'ventral')

#: aseg label ids -> the ROI they belong to under roi_v2/roi_v2_ofc, per
#: hemisphere. Derived from the same grouping as the cortex (`Basal Ganglia` is
#: caudate+putamen+pallidum+accumbens there and here), but written out explicitly
#: because aseg ids are not derivable from a substring match on a DK label.
ASEG_STRUCTURES = {
    'Hippocampus': {'left': [17], 'right': [53]},
    'Amygdala': {'left': [18], 'right': [54]},
    'Thalamus': {'left': [10], 'right': [49]},
    'Basal Ganglia': {'left': [11, 12, 13, 26], 'right': [50, 51, 52, 58]},
}

CAPTION_COLOR = '0.35'


# ============================================================================
# INPUTS
# ============================================================================

def find_fsaverage(subject='fsaverage', explicit=None):
    """The subject's FreeSurfer directory, or a SystemExit naming what was tried."""
    if explicit:
        path = Path(explicit)
        if not (path / 'surf' / 'lh.inflated').exists():
            raise SystemExit(f'--fsaverage {path} has no surf/lh.inflated')
        return path
    tried = []
    for candidate in SUBJECTS_DIR_CANDIDATES:
        try:
            path = candidate() / subject
        except KeyError:
            continue
        tried.append(path)
        if (path / 'surf' / 'lh.inflated').exists():
            return path
    raise SystemExit(
        f'{subject} not found. Tried: ' + ', '.join(str(p) for p in tried) +
        '. Pass --fsaverage, or `ml biology freesurfer/7.4.1` before running.')


def load_run(run_dir):
    """The run's cells table and the parameters the figure has to inherit.

    Band edges, ROI scheme and FDR q are read from the run's provenance rather
    than re-specified on the command line, because a surface drawn with different
    band edges than the model was fitted with is a wrong figure that looks right.
    """
    run_dir = Path(run_dir)
    cells_path = run_dir / 'band_cells.parquet'
    if not cells_path.exists():
        raise SystemExit(f'no band_cells.parquet in {run_dir}')

    cells = io.read_table(cells_path, on_stale='warn')
    if 'model' in cells.columns:
        cells = cells[cells['model'] == 'pain']

    with open(run_dir / 'provenance.json') as fh:
        prov = json.load(fh)
    params = prov.get('params', {})

    bands = params.get('bands') or {}
    if not bands:
        raise SystemExit(f'{run_dir}/provenance.json records no band edges')

    return cells, {
        'bands': {k: tuple(v) for k, v in bands.items()},
        'band_set': params.get('band_set'),
        'roi_scheme': params.get('roi_scheme', 'roi_v2_ofc'),
        'excluded_regions': list(params.get('excluded_regions') or []),
        'fdr_q': float(params.get('fdr_q', prov.get('params', {}).get('fdr_q', 0.05))),
        'status': prov.get('status', ''),
        'cells_path': cells_path,
    }


def band_order(cells, bands, only=None):
    """Bands in ascending frequency, restricted to the ones actually fitted.

    `only` subsets the figures without subsetting the COLOUR SCALE, which is still
    computed over every fitted cell -- a one-band render has to come out the same
    colour it would in a full run, or iterating on one band would silently produce
    a figure that cannot be compared with the others.
    """
    present = set(cells['band'].dropna())
    order = [b for b in sorted(bands, key=lambda b: bands[b][0]) if b in present]
    if not only:
        return order
    unknown = set(only) - set(order)
    if unknown:
        raise SystemExit(f'--bands: not fitted in this run: {sorted(unknown)}')
    return [b for b in order if b in set(only)]


def cell_tables(cells, bands):
    """(beta, significant) keyed [region][band], plus the symmetric colour cap.

    The cap is ONE number across all six bands, not per band. A per-band scale
    would make delta and high_gamma look equally strong when their betas differ by
    an order of magnitude, which is the whole thing the heatmap's single colorbar
    exists to prevent.
    """
    beta = cells.pivot_table(index='region', columns='band', values='beta_nrs_within')
    sig = (cells.assign(rej=cells['p_bh_reject'].fillna(False).astype(bool))
           .pivot_table(index='region', columns='band', values='rej')
           .fillna(0).astype(bool))
    beta = beta.reindex(columns=bands)
    sig = sig.reindex(columns=bands).fillna(False).astype(bool)
    cap = float(np.nanmax(np.abs(beta.to_numpy(dtype=float))))
    return beta, sig, cap


# ============================================================================
# DK PARCEL -> ROI
# ============================================================================

def parcel_regions(names, roi_scheme):
    """Parcel index -> ROI name (or None), through the analysis's own function.

    `names` is the annotation's parcel list in annot index order, so the returned
    list can be indexed straight by the per-vertex label array.
    """
    return [roi_schemes.region_for_dk_label(n, roi_scheme) for n in names]


def vertex_values(labels, regions_by_parcel, beta_for_region, fill=0.0):
    """Per-vertex beta: every vertex of a parcel gets its region's single estimate.

    `fill` for anything with no region or no fitted cell. It is 0.0 and NOT NaN
    because nilearn's matplotlib surface engine averages vertex values per FACE,
    and one NaN vertex poisons all the faces touching it -- a ragged bite out of
    every parcel border. 0.0 plus a threshold below the smallest real |beta|
    (see `unassigned_threshold`) masks exactly the unassigned vertices instead.
    """
    per_parcel = np.full(len(regions_by_parcel), fill, dtype=np.float64)
    for i, region in enumerate(regions_by_parcel):
        if region is None:
            continue
        value = beta_for_region.get(region, np.nan)
        if np.isfinite(value):
            per_parcel[i] = float(value)
    # A -1 label means "no parcel" in a FreeSurfer annot; send it to the fill slot.
    safe = np.where(labels < 0, 0, labels)
    out = per_parcel[safe]
    return np.where(labels < 0, fill, out)


def unassigned_threshold(beta):
    """A threshold that hides the 0.0 fill and nothing else.

    Half the smallest non-zero |beta| in the whole table. Picking a round constant
    instead would silently grey out a genuinely tiny estimate in some future run
    and make it look like missing coverage.
    """
    values = np.abs(beta.to_numpy(dtype=float))
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        return 1e-12
    return float(values.min()) * 0.5


def significance_map(labels, regions_by_parcel, significant_regions):
    """Per-vertex int map, one distinct id per significant region, 0 elsewhere.

    One id PER REGION rather than a single 1/0 mask, so `plot_surf_contours` draws
    the border between two adjacent significant regions instead of merging them
    into one blob -- S1 and M1 are adjacent and both light up in beta.

    Only regions with vertices ON THIS SURFACE get an id. A significant subcortical
    region has none, and nilearn's contour code raises `Vertices in parcellation do
    not form region` on an empty level rather than skipping it.
    """
    on_surface = {r for r in regions_by_parcel if r is not None}
    ids = {r: i + 1
           for i, r in enumerate(sorted(significant_regions & on_surface))}
    per_parcel = np.zeros(len(regions_by_parcel), dtype=np.int32)
    for i, region in enumerate(regions_by_parcel):
        per_parcel[i] = ids.get(region, 0)
    safe = np.where(labels < 0, 0, labels)
    out = per_parcel[safe]
    return np.where(labels < 0, 0, out), sorted(ids.values())


# ============================================================================
# SURFACE GEOMETRY
# ============================================================================

def load_hemisphere(fsaverage, hemi, surface_name, roi_scheme):
    """Everything one hemisphere needs, read once and reused for all six bands."""
    import nibabel as nib

    tag = {'left': 'lh', 'right': 'rh'}[hemi]
    coords, faces = nib.freesurfer.read_geometry(
        str(fsaverage / 'surf' / f'{tag}.{surface_name}'))
    labels, _, names = nib.freesurfer.read_annot(
        str(fsaverage / 'label' / f'{tag}.aparc.annot'))
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    sulc = nib.freesurfer.read_morph_data(str(fsaverage / 'surf' / f'{tag}.sulc'))

    return {
        'mesh': (coords, faces),
        'labels': labels,
        'names': names,
        'sulc': sulc,
        'regions_by_parcel': parcel_regions(names, roi_scheme),
    }


def as_mesh(mesh):
    """nilearn 0.14 takes an InMemoryMesh; older ones took the bare tuple."""
    coords, faces = mesh
    try:
        from nilearn.surface import InMemoryMesh
        return InMemoryMesh(coordinates=coords, faces=faces)
    except ImportError:
        return coords, faces


# ============================================================================
# THE CORTICAL FIGURE
# ============================================================================

# RENDER EACH PANEL, CROP IT, THEN COMPOSE
# ----------------------------------------
# A 3D axes cannot be packed. It sizes itself from a cube that bounds the mesh in
# every rotation, reports a bounding box that ignores where the brain actually
# landed inside that cube, and so defeats tight_layout, constrained_layout and
# hand-tuned gridspec fractions alike -- the first version of this figure was
# two-thirds white space with the panels drifting into the title. So each panel is
# rendered ALONE onto a transparent canvas, cropped to its own ink, and placed as
# an IMAGE in an ordinary 2D grid whose row heights and column widths come from the
# cropped sizes. The layout is then exact by construction rather than by tuning,
# and it stays exact when the view list or the hemisphere count changes.

PANEL_INCHES = 4.5
PANEL_DPI = 220


def render_panel(draw, transparent=True):
    """Run `draw(ax)` on a lone 3D axes and return the cropped RGBA image."""
    from io import BytesIO

    fig = plt.figure(figsize=(PANEL_INCHES, PANEL_INCHES), dpi=PANEL_DPI)
    ax = fig.add_axes((0, 0, 1, 1), projection='3d')
    draw(ax)
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=PANEL_DPI, transparent=transparent)
    plt.close(fig)
    buf.seek(0)
    return crop_to_ink(plt.imread(buf))


def crop_to_ink(img, pad=4):
    """Trim the transparent border. Alpha is the mask; the brain is opaque."""
    if img.ndim != 3 or img.shape[2] != 4:
        return img
    opaque = img[..., 3] > 0.01
    rows = np.flatnonzero(opaque.any(axis=1))
    cols = np.flatnonzero(opaque.any(axis=0))
    if not rows.size or not cols.size:
        return img
    r0, r1 = max(int(rows[0]) - pad, 0), min(int(rows[-1]) + 1 + pad, img.shape[0])
    c0, c1 = max(int(cols[0]) - pad, 0), min(int(cols[-1]) + 1 + pad, img.shape[1])
    return img[r0:r1, c0:c1]


def draw_surface(ax, hemi_data, values, threshold, cap, hemi, view, contour):
    """One 3D panel: the beta map, sulcal shading underneath, contours on top."""
    from nilearn import plotting

    mesh = as_mesh(hemi_data['mesh'])
    plotting.plot_surf_stat_map(
        mesh, values, bg_map=hemi_data['sulc'], hemi=hemi, view=view,
        engine='matplotlib', cmap='RdBu_r', colorbar=False, threshold=threshold,
        vmin=-cap, vmax=cap, symmetric_cbar=True, avg_method='median',
        axes=ax, figure=ax.figure,
    )
    sig_map, levels = contour
    if levels:
        plotting.plot_surf_contours(
            mesh, sig_map, hemi=hemi, levels=levels,
            colors=['black'] * len(levels), axes=ax, figure=ax.figure,
            linewidths=1.4,
        )


def cortex_panel(hemi_data, beta_for_region, significant, threshold, cap, hemi,
                 view):
    """The cropped image for one (hemisphere, view) at one band."""
    values = vertex_values(hemi_data['labels'], hemi_data['regions_by_parcel'],
                           beta_for_region)
    contour = significance_map(hemi_data['labels'], hemi_data['regions_by_parcel'],
                               significant)
    return render_panel(lambda ax: draw_surface(
        ax, hemi_data, values, threshold, cap, hemi, view, contour))


def compose(panels, *, col_labels=None, row_labels=None, title, caption_text, cap,
            out_path, width_inches=None):
    """Lay a grid of cropped panel images out with no slack, and save.

    `panels` is a list of rows, each a list of RGBA arrays or None. Row heights and
    column widths are taken from the images themselves, so every panel keeps its
    true aspect and no cell is padded to match a neighbour.
    """
    n_rows, n_cols = len(panels), max(len(r) for r in panels)
    widths = [max((p.shape[1] for p in (row[j] for row in panels)
                   if p is not None), default=1) for j in range(n_cols)]
    heights = [max((p.shape[0] for p in row if p is not None), default=1)
               for row in panels]

    # Physical size from the pixel grid, then margins in inches converted to the
    # figure fractions add_gridspec wants.
    grid_w, grid_h = sum(widths), sum(heights)
    width_inches = width_inches or min(16.0, max(7.0, grid_w / PANEL_DPI))
    grid_inches_h = width_inches * grid_h / grid_w
    left_in, right_in = (0.42 if row_labels else 0.12), 1.35
    top_in = 0.95 + (0.24 if col_labels else 0.0)
    bottom_in = 0.10 + 0.115 * (caption_text.count('\n') + 4)
    fig_w = width_inches + left_in + right_in
    fig_h = grid_inches_h + top_in + bottom_in

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(n_rows, n_cols, wspace=0.0, hspace=0.0,
                          width_ratios=widths, height_ratios=heights,
                          left=left_in / fig_w, right=1 - right_in / fig_w,
                          bottom=bottom_in / fig_h, top=1 - top_in / fig_h)

    for i, row in enumerate(panels):
        first = None
        for j in range(n_cols):
            img = row[j] if j < len(row) else None
            ax = fig.add_subplot(gs[i, j])
            ax.set_axis_off()
            first = first or ax
            if img is not None:
                ax.imshow(img, interpolation='antialiased')
            if col_labels and i == 0 and col_labels[j]:
                ax.set_title(col_labels[j], fontsize=10, pad=3)
        if row_labels and row_labels[i]:
            # The axes already made above, not a second add_subplot on the same
            # cell -- matplotlib stopped returning the existing axes for that in
            # 3.6 and would silently draw a blank one on top of the panel.
            pos = first.get_position()
            fig.text(pos.x0 - 0.008, (pos.y0 + pos.y1) / 2, row_labels[i],
                     rotation=90, va='center', ha='right', fontsize=10,
                     color='0.3')

    add_colorbar(fig, cap, right_in / fig_w)
    fig.suptitle(title, fontsize=13, y=1 - 0.10 / fig_h, va='top', wrap=True)
    fig.text(0.008, 0.10 / fig_h, caption_text, fontsize=6.5, va='bottom',
             ha='left', color=CAPTION_COLOR, wrap=True)
    fig.savefig(out_path, dpi=200, facecolor='white')
    plt.close(fig)
    logger.info('wrote %s', out_path.name)


def add_colorbar(fig, cap, right_fraction):
    """One colorbar for the figure, on the same scale as every other band."""
    sm = plt.cm.ScalarMappable(cmap='RdBu_r',
                               norm=matplotlib.colors.Normalize(-cap, cap))
    cax = fig.add_axes((1 - right_fraction + 0.008, 0.30, 0.012, 0.40))
    fig.colorbar(sm, cax=cax).set_label('d log10(band power) per pain point',
                                        fontsize=9)
    cax.tick_params(labelsize=8)


def band_figure(out_path, band, edges, hemis, views, hemi_data, beta, sig, cap,
                threshold, meta, subcortical=None):
    """One band, one PNG: (views x hemispheres) cortical panels plus the caption."""
    beta_for_region = beta[band].to_dict()
    significant = {r for r in sig.index if bool(sig.loc[r, band])}

    panels = [[cortex_panel(hemi_data[h], beta_for_region, significant, threshold,
                            cap, h, view) for h in hemis]
              for view in views]
    col_labels = [f'{h} hemisphere' for h in hemis]

    if subcortical:
        # One subcortical panel, in the middle row of its own column, because it
        # is one structure group seen once and not a per-view series.
        sub = render_subcortical(subcortical, beta_for_region, significant, cap)
        middle = len(views) // 2
        for i, row in enumerate(panels):
            row.append(sub if i == middle else None)
        col_labels.append(f'subcortical ({subcortical["hemi"]}, aseg on T1)\n'
                          '* = BH-significant')

    n_sig = len(significant)
    compose(
        panels, col_labels=col_labels, row_labels=list(views), cap=cap,
        title=(f'{band}  {edges[0]}-{edges[1]} Hz\n'
               f'band power vs pain, mixed-effects beta on {meta["template"]} -- '
               f'{n_sig} of {len(beta.index)} regions BH-significant at '
               f'q={meta["fdr_q"]}'),
        caption_text=caption(meta, hemi_data, beta, bool(subcortical)),
        out_path=out_path)


def caption(meta, hemi_data, beta, has_subcortical):
    """The three greys, the two things colour does not encode, and the disclaimer."""
    scheme = roi_schemes.resolve_roi_scheme(meta['roi_scheme'])
    covered = {r for d in hemi_data.values() for r in d['regions_by_parcel'] if r}
    subcortical = [r for r in scheme['display']
                   if r not in covered and r in ASEG_STRUCTURES]
    # `excluded_regions` regions are also missing from the cells table, so they
    # would otherwise be named twice as two different kinds of grey.
    excluded = set(meta['excluded_regions'])
    no_cell = [r for r in scheme['display']
               if r in covered and r not in set(beta.index) and r not in excluded]

    lines = [
        'A parcel is painted UNIFORMLY with its region\'s single mixed-effects '
        'beta: the model estimated one number per region, not a per-vertex map, so '
        'boundaries are the Desikan-Killiany atlas\'s and the map must not be read '
        'as a spatial gradient. Colour does NOT encode coverage -- a region '
        'sampled in 11 subjects is painted as solidly as one sampled in 55; see '
        'fig_check_coverage_counts.png for the denominator. Hemispheres are POOLED '
        'in the model, so left and right are identical by construction, not by a '
        'measured symmetry. Black outlines mark BH-significant regions at '
        f'q={meta["fdr_q"]}, the same cells outlined in fig_band_map.png; the '
        'colour scale is shared across all bands.',
    ]
    greys = ['outside the ROI scheme (white matter, unknown, frontalpole, '
             'corpus callosum)']
    if meta['excluded_regions']:
        greys.append('excluded from this run: ' + ', '.join(meta['excluded_regions']))
    if no_cell:
        greys.append('no fitted cell: ' + ', '.join(no_cell))
    if subcortical:
        where = ('shown in the subcortical panel' if has_subcortical
                 else 'NOT SHOWN -- no cortical surface to project onto')
        greys.append(f'subcortical, {where}: ' + ', '.join(subcortical))
    lines.append('Grey cortex is one of: ' + '; '.join(greys) + '.')
    if meta['status']:
        lines.append(meta['status'])
    return '\n'.join(lines)


# ============================================================================
# SUBCORTICAL
# ============================================================================

def load_subcortical(fsaverage, hemi='left'):
    """fsaverage's aseg and T1, cropped around the four subcortical ROIs.

    SLICES, NOT A 3D BLOB, after trying the blob. The four structures nest inside
    each other -- pallidum inside putamen inside the shell the thalamus sits
    medial to -- so from any single viewpoint they occlude each other into one
    mass, and a mass is not worth a panel. Two axial cuts show all four in their
    usual radiological presentation, which anyone reading this figure can parse at
    a glance.

    A proper 3D rendering would want marching cubes (scikit-image: not installed,
    and its current wheels are manylinux_2_28 against this cluster's glibc 2.17)
    or FreeSurfer's `mri_tessellate` and a cached mesh artifact to keep fresh.
    Neither is worth it for four numbers.

    NO REGISTRATION IS INVOLVED, which is the reason the background is fsaverage's
    own T1 rather than a stock MNI152 template: aseg.mgz and T1.mgz are the same
    256^3 conformed grid by construction, so the overlay is exact rather than
    approximately right.

    ONE HEMISPHERE, for the same reason the cortical panels are per-hemisphere:
    the model pools sides, so the other carries no extra information.
    """
    import nibabel as nib

    seg_img = nib.as_closest_canonical(nib.load(str(fsaverage / 'mri' / 'aseg.mgz')))
    t1_img = nib.as_closest_canonical(nib.load(str(fsaverage / 'mri' / 'T1.mgz')))
    seg = np.asarray(seg_img.dataobj)
    t1 = np.asarray(t1_img.dataobj, dtype=np.float32)

    # One integer plane per ROI, so a slice can be coloured without re-testing
    # every aseg id. 0 = not one of ours.
    regions = [r for r in ASEG_STRUCTURES if np.isin(seg, ASEG_STRUCTURES[r][hemi]).any()]
    index = np.zeros(seg.shape, dtype=np.int16)
    for i, region in enumerate(regions, start=1):
        index[np.isin(seg, ASEG_STRUCTURES[region][hemi])] = i
    if not regions:
        raise SystemExit(f'aseg has no {hemi} subcortical voxels -- wrong template?')

    # Two axial cuts: one through the limbic pair, one through the diencephalic
    # pair. Taken from the voxels themselves rather than hardcoded in mm, so this
    # still lands correctly on a template with a different origin.
    def centre_slice(group):
        rows = [i for i, r in enumerate(regions, start=1) if r in group]
        return int(round(np.argwhere(np.isin(index, rows))[:, 2].mean()))

    cuts = sorted({centre_slice({'Hippocampus', 'Amygdala'}),
                   centre_slice({'Thalamus', 'Basal Ganglia'})})

    # Crop to the structures plus a margin of context, in-plane only.
    occupied = np.argwhere(index > 0)
    lo = occupied.min(axis=0) - SUBCORTICAL_MARGIN
    hi = occupied.max(axis=0) + SUBCORTICAL_MARGIN + 1
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, index.shape)

    return {
        'hemi': hemi,
        'regions': regions,
        'cuts': cuts,
        'index': index[lo[0]:hi[0], lo[1]:hi[1], :],
        't1': t1[lo[0]:hi[0], lo[1]:hi[1], :],
        'zooms': t1_img.header.get_zooms(),
    }


#: Voxels of surrounding anatomy kept around the structures, so a slice reads as
#: a brain rather than as four shapes floating in black.
SUBCORTICAL_MARGIN = 18


def render_subcortical(sub, beta_for_region, significant, cap):
    """The cropped image for the subcortical panel: axial cuts plus a value key."""
    cmap = plt.get_cmap('RdBu_r')
    norm = matplotlib.colors.Normalize(-cap, cap)
    n_cuts = len(sub['cuts'])

    fig = plt.figure(figsize=(2.3 * n_cuts, 3.4), dpi=PANEL_DPI)
    gs = fig.add_gridspec(2, n_cuts, height_ratios=[3.0, 1.0], hspace=0.04,
                          wspace=0.02, left=0, right=1, top=1, bottom=0)

    for j, k in enumerate(sub['cuts']):
        ax = fig.add_subplot(gs[0, j])
        ax.set_axis_off()
        # .T with origin='lower' puts +x right and +y up: the canonical
        # reorientation above guarantees the array is RAS, so this is neurological
        # convention (subject left on the viewer's left) and not a coin flip.
        bg = sub['t1'][:, :, k].T
        ax.imshow(bg, cmap='gray', origin='lower', interpolation='bilinear',
                  vmin=0, vmax=float(np.percentile(sub['t1'], 99.5)))

        labels = sub['index'][:, :, k].T
        overlay = np.zeros(labels.shape + (4,), dtype=float)
        for i, region in enumerate(sub['regions'], start=1):
            here = labels == i
            if not here.any():
                continue
            value = beta_for_region.get(region, np.nan)
            overlay[here] = ((0.82, 0.82, 0.82, 1.0) if not np.isfinite(value)
                             else cmap(norm(value)))
            if region in significant:
                ax.contour(here.astype(float), levels=[0.5], colors='black',
                           linewidths=1.1, origin='lower')
        ax.imshow(overlay, origin='lower', interpolation='nearest')
        ax.set_aspect('equal')

    key = fig.add_subplot(gs[1, :])
    key.set_axis_off()
    for row, region in enumerate(sub['regions']):
        value = beta_for_region.get(region, np.nan)
        colour = '0.82' if not np.isfinite(value) else cmap(norm(value))
        y = 1.0 - (row + 0.5) / len(sub['regions'])
        key.add_patch(matplotlib.patches.Rectangle(
            (0.01, y - 0.07), 0.05, 0.14, transform=key.transAxes,
            facecolor=colour, edgecolor='black',
            linewidth=1.1 if region in significant else 0.4))
        shown = 'no cell' if not np.isfinite(value) else f'{value:+.4f}'
        star = ' *' if region in significant else ''
        key.text(0.09, y, f'{region}  {shown}{star}', transform=key.transAxes,
                 fontsize=8, va='center', ha='left')

    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=PANEL_DPI, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return crop_to_ink(plt.imread(buf))


# ============================================================================
# CONTACT SHEET
# ============================================================================

def contact_sheet(out_path, bands, edges_for, views, hemi_data, beta, sig, cap,
                  threshold, meta, hemi='left'):
    """Bands across, views down -- the heatmap's own axes, drawn as brains.

    The one figure to put next to fig_band_map.png: same six columns in the same
    order on the same colour scale, so the two can be read against each other cell
    for cell.
    """
    data = hemi_data[hemi]
    panels = []
    for view in views:
        row = []
        for band in bands:
            significant = {r for r in sig.index if bool(sig.loc[r, band])}
            row.append(cortex_panel(data, beta[band].to_dict(), significant,
                                    threshold, cap, hemi, view))
        panels.append(row)

    compose(
        panels,
        col_labels=[f'{b}\n{edges_for[b][0]}-{edges_for[b][1]} Hz' for b in bands],
        row_labels=list(views), cap=cap,
        title=(f'Band power vs pain, mixed-effects beta -- {hemi} hemisphere, '
               'all bands on one scale'),
        caption_text=(
            'The same cells as fig_band_map.png, in the same column order, so the '
            'two can be read against each other. Black outlines mark '
            f'BH-significant regions at q={meta["fdr_q"]}. A parcel is painted '
            'uniformly with its region\'s single beta and colour does not encode '
            'coverage. The four subcortical ROIs have no cortical surface and '
            'appear only in the per-band figures (fig_surface_<band>.png).\n'
            f'{meta["status"]}'),
        out_path=out_path, width_inches=min(17.0, 2.6 * len(bands)))


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True,
                    help='a mixed_effects run directory holding band_cells.parquet')
    ap.add_argument('--surface', default='inflated',
                    choices=['inflated', 'pial', 'white'],
                    help='inflated by default: insula, ACC and OFC are buried in '
                         'sulci and simply invisible on a pial surface')
    ap.add_argument('--views', nargs='+', default=list(DEFAULT_VIEWS),
                    choices=['lateral', 'medial', 'dorsal', 'ventral', 'anterior',
                             'posterior'])
    ap.add_argument('--hemi', default='both', choices=['both', 'left', 'right'])
    ap.add_argument('--bands', nargs='+', default=None,
                    help='render only these bands (the colour scale still comes '
                         'from all of them). For iterating on layout.')
    ap.add_argument('--subcortical', action=argparse.BooleanOptionalAction,
                    default=True,
                    help='the four subcortical ROIs as an aseg blob panel. ON by '
                         'default: they are 4 of the 19 rows and most of the delta '
                         'column, so a cortex-only brain drops the strongest '
                         'effects without saying so')
    ap.add_argument('--fs-subject', default='fsaverage',
                    help='fsaverage6 is a quarter of the vertices and about four '
                         'times faster; use it while iterating on layout, and the '
                         'full fsaverage for anything that goes in a talk')
    ap.add_argument('--fsaverage', default=None,
                    help='an explicit FreeSurfer subject directory, overriding '
                         '--fs-subject and the SUBJECTS_DIR search')
    ap.add_argument('--out-dir', default=None,
                    help=f'defaults to <run-dir>/{SUBDIR}')
    args = ap.parse_args()

    io.warn_if_dirty()

    run_dir = Path(args.run_dir)
    cells, meta = load_run(run_dir)
    all_bands = band_order(cells, meta['bands'])
    if not all_bands:
        raise SystemExit('no bands in band_cells.parquet match the recorded edges')
    bands = band_order(cells, meta['bands'], args.bands)
    # Colour scale over ALL bands, figures over the requested ones.
    beta, sig, cap = cell_tables(cells, all_bands)
    threshold = unassigned_threshold(beta)
    logger.info('%d regions x %d bands, |beta| cap %.5g, fill threshold %.3g',
                len(beta.index), len(bands), cap, threshold)

    fsaverage = find_fsaverage(args.fs_subject, args.fsaverage)
    # The template belongs in the title: a figure rendered on fsaverage6 while
    # iterating must not claim to be the full-resolution one.
    meta['template'] = f'{fsaverage.name} {args.surface}'
    logger.info('surface template: %s (%s)', fsaverage, args.surface)

    hemis = ['left', 'right'] if args.hemi == 'both' else [args.hemi]
    hemi_data = {h: load_hemisphere(fsaverage, h, args.surface, meta['roi_scheme'])
                 for h in hemis}

    covered = {r for d in hemi_data.values() for r in d['regions_by_parcel'] if r}
    painted = sorted(set(beta.index) & covered)
    unpainted = sorted(set(beta.index) - covered)
    logger.info('%d of %d fitted regions have cortical parcels; not on the surface: '
                '%s', len(painted), len(beta.index), ', '.join(unpainted) or 'none')

    subcortical = load_subcortical(fsaverage, hemis[0]) if args.subcortical else None

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for band in bands:
        band_figure(out_dir / f'fig_surface_{band}.png', band, meta['bands'][band],
                    hemis, args.views, hemi_data, beta, sig, cap, threshold, meta,
                    subcortical)

    contact_sheet(out_dir / 'fig_surface_all_bands.png', bands, meta['bands'],
                  args.views, hemi_data, beta, sig, cap, threshold, meta,
                  hemi=hemis[0])

    io.write_run_provenance(
        out_dir, script=SCRIPT, params=vars(args) | {
            'bands': meta['bands'], 'roi_scheme': meta['roi_scheme'],
            'fsaverage': str(fsaverage), 'beta_cap': cap,
            'regions_painted': painted, 'regions_not_on_surface': unpainted,
        },
        parents=[io.parent_ref(meta['cells_path']),
                 io.parent_ref(run_dir / 'provenance.json')],
        extra={'status': meta['status'],
               'note': 'a re-rendering of band_cells.parquet onto fsaverage; '
                       'no statistic is recomputed here'})
    io.log_analysis(
        f'band-power betas projected onto the fsaverage cortical surface, '
        f'{len(bands)} bands (EXPLORATORY)', out_dir)
    print(out_dir)


if __name__ == '__main__':
    main()

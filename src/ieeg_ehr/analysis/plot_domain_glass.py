#!/usr/bin/env python3
"""The domain model's pain slopes, painted onto the electrodes they came from.

    <domain run>/glass_brain/<label>_<timestamp>/

One glass brain per band. Every electrode is drawn at its MNI midpoint and
coloured by THE PAIN SLOPE OF THE DOMAIN IT BELONGS TO in that band, on the
same numeric scale as `fig_domain_summary.png`.

WHAT THIS FIGURE IS, AND THE THING TO KEEP IN MIND WHILE READING IT
--------------------------------------------------------------------
The domain model fits ONE beta per (domain, band). So every electrode inside a
domain carries the SAME colour, and a panel has at most as many distinct
colours as there are domains -- five here, not two thousand. This is a map of
the DOMAIN ASSIGNMENT tinted by that domain's group effect; it is not a map of
per-electrode effects, and a gradient across a domain's territory is not
something this figure can show because the model never estimated one.

It is still worth drawing, for the thing the heatmap cannot do: it says WHERE
the domains are, so a reader can see that "Sensory" here means a thalamic
cluster plus a peri-central strip plus posterior insula, and can judge whether
the anatomy behind a row is what they assumed.

`--level roi` is the genuinely spatially varying counterpart: it colours each
electrode by ITS OWN REGION's slope, taken from a region-level
`run_bandpower_mixed` run, which fits all 21 ROIs separately. Same scale, same
colormap, ~21 colours instead of 5.

COLORMAP: THE MIDDLE MUST NOT BE WHITE, BUT IT MUST NOT BE DARK EITHER
-----------------------------------------------------------------------
`fig_domain_summary` uses RdBu_r, which is right for a heatmap -- the cells are
bounded by gridlines, so a near-zero cell rendering as white still reads as a
cell. On a glass brain the background IS white, so a domain with beta near zero
renders as invisible dots and simply vanishes, which is the one thing a
coverage-sensitive figure must not do. RdBu_r's midpoint sits 0.06 from white.

THE OBVIOUS FIX IS A DARK-CENTRED MAP, AND IT WAS TRIED AND REJECTED. `berlin`
(midpoint 1.63 from white, near-black, same blue-negative/red-positive
direction) solves visibility and creates a worse problem. Within any one band
the domains here mostly share a SIGN -- in beta all five are positive, +0.0040
to +0.0198 -- so on a symmetric scale they all land in one half of the map, and
with a dark centre the WEAKEST domain comes out near-black while the strongest
is pale pink. The eye reads the near-black one as the strong one. A colormap
that makes "no effect" the most salient value is actively misleading, and on
the beta panel it inverted the visual ranking of all five domains.

So the default is `coolwarm`: midpoint 0.23 from white -- a light warm grey,
pale but not invisible -- keeping the correct "pale = near zero" reading. Pale
markers are then kept legible by a thin dark EDGE (`--edge-width`), which
restores visibility without touching the fill, so the fill goes on meaning only
the number. `--cmap` takes any matplotlib name; see CMAP_NOTES.

THE NUMERIC SCALE IS SHARED WITH THE HEATMAP AND ACROSS BANDS. vmin/vmax are
-cap/+cap where cap is max|beta| over the WHOLE grid -- every band, every
domain -- exactly as `summary_figure` computes it. So a colour means the same
thing in the delta brain, the beta brain and the heatmap, and the three can be
compared by eye. A per-band scale would make delta and beta look equally strong.

ELECTRODES OUTSIDE THE BRAIN ARE DROPPED, not clipped: the MNI152 brain mask
that ships with nilearn is the test, so nothing is drawn floating outside the
glass brain outline. The count dropped is logged and recorded -- a localisation
failure leaving the figure is a fact about the data, not a rendering detail.

VERSIONED OUTPUT. Each run writes a NEW timestamped folder under
`glass_brain/`, never overwriting an earlier one, and each carries a
`provenance.json` naming the source run and a `colour_source` block giving the
exact beta, p and contact count behind every colour. Iterating on the
appearance therefore cannot silently replace the version someone already put in
a slide.

    python -m ieeg_ehr.analysis.plot_domain_glass --run-dir <domain run>

EXPLORATORY. Discovery cohort. Nominations, not findings.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import insula_ap
from ieeg_ehr.analysis.run_bandpower_mixed import BAND_SETS
from ieeg_ehr.analysis.run_domain_model import DOMAIN_COLOURS, domain_caveat
from ieeg_ehr.config import roi_schemes

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_domain_glass.py'
SUBDIR = 'glass_brain'

DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. '
              'Not confirmed out of sample.')

#: The bands worth a brain. Not all six: high_gamma has no significant domain
#: at all and theta/alpha only Modulatory, so they would be five near-identical
#: dark panels. Override with --bands.
DEFAULT_BANDS = ('delta', 'beta', 'gamma')

#: Diverging maps that survive a WHITE BACKGROUND, with how far their midpoint
#: sits from white in RGB (measured on matplotlib 3.10.8). Anything under ~0.2
#: makes a near-zero region invisible on a glass brain.
CMAP_NOTES = {
    'coolwarm': 'midpoint 0.23 from white (light warm grey). DEFAULT. Pale at '
                'zero but not invisible, and pale still MEANS near-zero. Use '
                'with a marker edge.',
    'Spectral_r': 'midpoint 0.25 (pale yellow). Visible, but not '
                  'colourblind-safe and yellow reads as "high" to many.',
    'RdBu_r': 'midpoint 0.06 -- EXACTLY the heatmap. A near-zero domain is '
              'invisible without an edge. Use to place a brain beside the '
              'heatmap.',
    'berlin': 'midpoint 1.63 (near-black). REJECTED AS THE DEFAULT: when the '
              'units in a band share a sign, the dark centre makes the '
              'WEAKEST one the most salient. Kept for values that genuinely '
              'straddle zero.',
    'vanimo': 'midpoint 1.58 (near-black), pink-neg/green-pos. Same caution.',
    'managua': 'midpoint 1.28 (dark plum), orange-neg/teal-pos. Same caution.',
}


# ============================================================================
# DATA
# ============================================================================

def load_run(run_dir):
    """(params, subjects, slopes with term=='pain', coverage) for a domain run."""
    run_dir = Path(run_dir)
    prov = json.loads((run_dir / 'provenance.json').read_text())
    params, subjects = prov.get('params', {}), prov.get('subjects', [])
    slopes = io.read_table(run_dir / 'domain_slopes.parquet', on_stale='warn')
    if 'term' in slopes.columns:
        slopes = slopes[slopes['term'] == 'pain']
    if 'beta' not in slopes.columns and 'beta_pain' in slopes.columns:
        slopes = slopes.rename(columns={'beta_pain': 'beta'})
    coverage = io.read_table(run_dir / 'parcel_coverage.parquet',
                             on_stale='ignore')
    return params, subjects, slopes, coverage


def electrode_coords(subjects, epoch_minutes=None):
    """One row per (subject, channel) with its MNI midpoint, every session.

    Same de-duplication as `insula_ap.load_insula_contacts` and for the same
    reason: channel_meta is keyed (run_id, pair_index), so one physical
    electrode appears once per run and a subject here can have 100+ runs.
    """
    meta_dir = config.pain_epoch_channel_meta_path('000', '01',
                                                   epoch_minutes).parent
    frames = []
    for sid in subjects:
        bare = str(sid).replace('sub-', '')
        paths = sorted(meta_dir.glob(f'sub-{bare}_ses-*_channels.parquet'))
        if not paths:
            logger.warning('sub-%s: no channel_meta, skipped', bare)
            continue
        m = pd.concat([io.read_table(p, on_stale='ignore') for p in paths],
                      ignore_index=True)
        if not {'mni_x', 'mni_y', 'mni_z'} <= set(m.columns):
            logger.warning('sub-%s: channel_meta has no MNI columns, skipped', bare)
            continue
        m = m.drop_duplicates('channel')
        m = m.assign(subject_id=f'sub-{bare}')
        frames.append(m[['subject_id', 'channel', 'dk_anode',
                         'mni_x', 'mni_y', 'mni_z']])
    if not frames:
        raise SystemExit('no channel_meta with coordinates for any subject')
    return pd.concat(frames, ignore_index=True)


def inside_brain(coords, dilation_mm=0):
    """Boolean mask: is each MNI point inside the MNI152 brain?

    THE TEST THE GLASS BRAIN ITSELF IMPLIES. A bounding box (what
    `plot_electrode_locations.MNI_BOUNDS` uses) only catches gross localisation
    failures -- it happily passes a contact 40 mm outside the skull in a corner
    of the box, which then renders as a dot floating beside the outline. The
    template mask is the actual boundary, and a point inside it necessarily
    projects inside the outline in every one of the glass brain's views.

    `dilation_mm` tolerates registration error at the surface, where a genuine
    contact can fall a millimetre or two outside a thresholded template. It is
    0 by default -- "outside the brain" taken literally -- and the caller logs
    what a small dilation would have kept, so the choice is visible.
    """
    from nilearn.datasets import load_mni152_brain_mask
    import nibabel as nib

    mask_img = load_mni152_brain_mask()
    data = np.asarray(mask_img.get_fdata()) > 0
    if dilation_mm and dilation_mm > 0:
        from scipy import ndimage
        # The template is 1 mm isotropic, so one iteration is ~1 mm.
        vox = float(np.abs(mask_img.affine[0, 0])) or 1.0
        data = ndimage.binary_dilation(
            data, iterations=max(1, int(round(dilation_mm / vox))))

    ijk = nib.affines.apply_affine(np.linalg.inv(mask_img.affine),
                                   np.asarray(coords, dtype=float))
    ijk = np.rint(ijk).astype(int)
    ok = np.isfinite(coords).all(axis=1)
    for a in range(3):
        ok &= (ijk[:, a] >= 0) & (ijk[:, a] < data.shape[a])
    out = np.zeros(len(ijk), dtype=bool)
    idx = np.where(ok)[0]
    out[idx] = data[ijk[idx, 0], ijk[idx, 1], ijk[idx, 2]]
    return out


def build_electrodes(run_dir, args):
    """(electrode frame with a `unit` column, slopes, params, drop counts).

    `unit` is the domain (or, at --level roi, the ROI) whose beta colours that
    electrode. One colour per unit, which is the whole shape of this figure.
    """
    params, subjects, slopes, coverage = load_run(run_dir)
    epoch_minutes = params.get('epoch_minutes')
    el = electrode_coords(subjects, epoch_minutes)
    n0 = len(el)
    counts = {'n_electrodes_in_channel_meta': n0}

    if args.level == 'domain':
        # The run's OWN assignment, read from the artifact rather than rebuilt,
        # so the brain cannot disagree with the model about which domain a
        # contact is in.
        el = el.merge(coverage[['subject_id', 'channel', 'domain']],
                      on=['subject_id', 'channel'], how='inner')
        el = el.rename(columns={'domain': 'unit'})
    else:
        scheme = args.roi_scheme
        el['unit'] = [roi_schemes.region_for_dk_label(
            lbl, scheme, include_coordinate_parents=True)
            for lbl in el['dk_anode']]
        coord = roi_schemes.coordinate_regions(scheme)
        if coord:
            contacts, _ = insula_ap.load_insula_contacts(subjects,
                                                         epoch_minutes=epoch_minutes)
            contacts, _ = insula_ap.median_split(
                contacts, threshold=params.get('insula_threshold'))
            look = insula_ap.channel_lookup(contacts)
            el['unit'] = [look.get(s, {}).get(c) if u in coord else u
                          for s, c, u in zip(el['subject_id'], el['channel'],
                                             el['unit'])]
        el = el[el['unit'].notna()]
    counts['n_with_a_unit'] = int(len(el))
    logger.info('%d of %d electrode(s) carry a %s', len(el), n0, args.level)

    coords = el[['mni_x', 'mni_y', 'mni_z']].to_numpy(dtype=float)
    keep = inside_brain(coords, args.brain_dilation_mm)
    counts['n_outside_brain_dropped'] = int((~keep).sum())
    if (~keep).any():
        loose = inside_brain(coords, max(args.brain_dilation_mm, 0) + 3)
        counts['n_that_3mm_dilation_would_keep'] = int((loose & ~keep).sum())
        logger.warning('dropped %d electrode(s) outside the MNI152 brain mask '
                       '(dilation %g mm); %d of them are within 3 mm of it',
                       int((~keep).sum()), args.brain_dilation_mm,
                       counts['n_that_3mm_dilation_would_keep'])
    el = el[keep].reset_index(drop=True)
    counts['n_plotted'] = int(len(el))
    return el, slopes, params, counts


# ============================================================================
# FIGURES
# ============================================================================

def band_label(band, bands_def):
    lo, hi = bands_def[band]
    return f'{band} ({lo:g}-{hi:g} Hz)'


def draw_band(el, beta_by_unit, sig_by_unit, band, cap, args, figure=None,
              axes=None, title=None):
    """One band's glass brain into `figure`/`axes`; returns the display.

    ONE `plot_markers` CALL with continuous values, not one per unit: the
    colour is a number on a shared scale, so handing nilearn the numbers is
    both simpler and the only way the colour cannot drift from the value.
    """
    from nilearn import plotting

    vals = el['unit'].map(beta_by_unit).to_numpy(dtype=float)
    coords = el[['mni_x', 'mni_y', 'mni_z']].to_numpy(dtype=float)
    ok = np.isfinite(vals)

    # A thin dark edge is what lets a LIGHT-centred diverging map work here: a
    # near-zero marker becomes a pale disc with a visible rim instead of
    # nothing, and the fill goes on meaning only the number. Without it the
    # choice is between an invisible zero and a dark-centred map that misranks
    # the units -- see the module docstring.
    kw = {'edgecolors': args.edge_color, 'linewidths': args.edge_width}
    if args.grey_nonsignificant:
        # Draw the non-significant units first and flat grey, so significance
        # is a HUE change (grey vs coloured) rather than a lightness change.
        # Lightness would collide with the colormap, where light already means
        # "near zero".
        ns = ok & ~el['unit'].map(sig_by_unit).fillna(False).to_numpy(dtype=bool)
        if ns.any():
            plotting.plot_markers(
                node_values=np.zeros(int(ns.sum())), node_coords=coords[ns],
                node_size=args.node_size, display_mode='lyrz', colorbar=False,
                figure=figure, axes=axes, alpha=args.alpha,
                node_cmap=matplotlib.colors.ListedColormap(['0.72']),
                node_vmin=-1, node_vmax=1, node_kwargs=kw, title=title)
            ok = ok & ~ns
            title = None       # already drawn on the first call

    if not ok.any():
        return None
    return plotting.plot_markers(
        node_values=vals[ok], node_coords=coords[ok], node_size=args.node_size,
        node_cmap=plt.get_cmap(args.cmap), node_vmin=-cap, node_vmax=cap,
        display_mode='lyrz', colorbar=False, figure=figure, axes=axes,
        alpha=args.alpha, node_kwargs=kw, title=title)


def unit_legend(ax, units, beta_by_unit, sig_by_unit, cap, args, colour_by_unit):
    """A row of swatches: every unit, its beta, and whether it is significant.

    The colorbar says what a colour MEANS; this says which colour each domain
    GOT. Both are needed, because the reader's question is "what is Sensory
    doing", and on a glass brain there is no axis label to answer it.
    """
    import matplotlib.patches as mpatches
    handles = []
    for u in units:
        b = beta_by_unit.get(u)
        if b is None or not np.isfinite(b):
            continue
        star = ' *' if sig_by_unit.get(u) else ''
        handles.append(mpatches.Patch(
            facecolor=colour_by_unit[u], edgecolor='0.3',
            label=f'{u}  β={b:+.4f}{star}'))
    if handles:
        ax.legend(handles=handles, loc='center', ncol=min(len(handles), 5),
                  frameon=False, fontsize=9, handlelength=1.3,
                  columnspacing=1.4)
    ax.axis('off')


def figure_for_band(el, slopes, band, cap, args, bands_def, out_path, caveat):
    """One PNG for one band: the brain, a colorbar, and the unit swatches."""
    sub = slopes[slopes['band'] == band]
    beta_by_unit = dict(zip(sub['unit'], sub['beta']))
    sig_by_unit = dict(zip(sub['unit'], sub['significant']))
    cmap = plt.get_cmap(args.cmap)
    norm = matplotlib.colors.Normalize(-cap, cap)
    colour_by_unit = {u: cmap(norm(b)) for u, b in beta_by_unit.items()
                      if np.isfinite(b)}
    if args.grey_nonsignificant:
        for u in colour_by_unit:
            if not sig_by_unit.get(u):
                colour_by_unit[u] = '0.72'

    fig = plt.figure(figsize=(17, 6.4))
    display = draw_band(el, beta_by_unit, sig_by_unit, band, cap, args,
                        figure=fig, axes=(0.02, 0.26, 0.84, 0.66))

    cax = fig.add_axes((0.89, 0.32, 0.013, 0.52))
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=cax)
    cb.set_label('Δ log10 band power per pain point', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    lax = fig.add_axes((0.02, 0.15, 0.84, 0.08))
    units = [u for u in args.unit_order if u in beta_by_unit]
    unit_legend(lax, units, beta_by_unit, sig_by_unit, cap, args, colour_by_unit)

    fig.suptitle(f'{band_label(band, bands_def)} — pain slope of each '
                 f'{args.level}, on its electrodes', fontsize=14, y=0.97)
    fig.text(0.02, 0.015, _caption(el, args, cap, caveat), fontsize=6.4,
             va='bottom', ha='left', color='0.35', wrap=True)
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    if display is not None:
        display.close()
    logger.info('wrote %s', out_path.name)


def figure_all_bands(el, slopes, bands, cap, args, bands_def, out_path, caveat):
    """One PNG, one row per band, a single shared colorbar.

    The point of the stacked version: on ONE scale, delta is blue everywhere
    and beta/gamma are red everywhere, and the sign flip across frequency is
    the thing to see. Three separate files cannot show it.
    """
    from nilearn import plotting            # noqa: F401  (import cost, once)

    cmap = plt.get_cmap(args.cmap)
    norm = matplotlib.colors.Normalize(-cap, cap)
    n = len(bands)
    fig = plt.figure(figsize=(16, 3.5 * n + 1.5))
    h = 0.86 / n
    displays = []
    for i, band in enumerate(bands):
        sub = slopes[slopes['band'] == band]
        beta_by_unit = dict(zip(sub['unit'], sub['beta']))
        sig_by_unit = dict(zip(sub['unit'], sub['significant']))
        bottom = 0.10 + (n - 1 - i) * h
        d = draw_band(el, beta_by_unit, sig_by_unit, band, cap, args,
                      figure=fig, axes=(0.03, bottom, 0.80, h * 0.92))
        displays.append(d)
        nsig = int(sum(bool(v) for v in sig_by_unit.values()))
        fig.text(0.015, bottom + h * 0.46, band_label(band, bands_def),
                 rotation=90, va='center', ha='center', fontsize=11)
        fig.text(0.84, bottom + h * 0.46,
                 '\n'.join(f'{u} {beta_by_unit[u]:+.4f}'
                           f'{" *" if sig_by_unit.get(u) else ""}'
                           for u in args.unit_order if u in beta_by_unit),
                 va='center', ha='left', fontsize=7.5, color='0.25')

    cax = fig.add_axes((0.945, 0.30, 0.010, 0.45))
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=cax)
    cb.set_label('Δ log10 band power per pain point', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(f'Pain slope by {args.level}, on the electrodes — one scale '
                 'across all bands', fontsize=14, y=0.985)
    fig.text(0.02, 0.012, _caption(el, args, cap, caveat), fontsize=6.4,
             va='bottom', ha='left', color='0.35', wrap=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    for d in displays:
        if d is not None:
            d.close()
    logger.info('wrote %s', out_path.name)


def figure_heatmap(slopes, bands, cap, args, bands_def, out_path, caveat):
    """The domain x band heatmap, in THIS run's colormap and cap.

    WHY A SECOND COPY OF A FIGURE THAT ALREADY EXISTS. The run's own
    `fig_domain_summary.png` is drawn in RdBu_r, which is right for a heatmap
    and wrong for a glass brain, so the brains here use `coolwarm`. That leaves
    the two figures showing the same numbers in different colours, and "the
    electrode is the colour of its cell" stops being literally true -- which is
    exactly the correspondence the brains are for.

    So the folder carries its own heatmap, built from the same `slopes` table,
    the same cap and the same colormap as the brains beside it. The cell a
    reader looks up and the dots they then find on the brain are the identical
    RGB. The run's original figure is NOT touched.

    EVERY band is drawn, not only the ones with a brain: the cap comes from the
    whole grid, so showing the whole grid is what makes the cap legible.
    """
    all_bands = [b for b in bands_def if b in set(slopes['band'])]
    units = [u for u in args.unit_order if u in set(slopes['unit'])]
    piv = (slopes.pivot_table(index='unit', columns='band', values='beta',
                              aggfunc='first')
           .reindex(index=units, columns=all_bands))
    sig = (slopes.pivot_table(index='unit', columns='band',
                              values='significant', aggfunc='first')
           .reindex(index=units, columns=all_bands).fillna(False).astype(bool))
    arr = piv.to_numpy(dtype=float)

    from ieeg_ehr.features import common
    cmap = plt.get_cmap(args.cmap)
    fig, ax = plt.subplots(figsize=(1.35 * len(all_bands) + 3.6,
                                    0.62 * len(units) + 2.6))
    im = ax.imshow(arr, aspect='auto', cmap=cmap, vmin=-cap, vmax=cap,
                   interpolation='nearest')
    common.draw_mask_outline(ax, sig.to_numpy())
    for i in range(len(units)):
        for j in range(len(all_bands)):
            v = arr[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, f'{v:+.4f}' + ('\n*' if sig.iat[i, j] else ''),
                    ha='center', va='center', fontsize=8,
                    color='white' if abs(v) > 0.62 * cap else '0.12')
    ax.set_xticks(range(len(all_bands)))
    ax.set_xticklabels([band_label(b, bands_def).replace(' (', '\n(')
                        for b in all_bands], fontsize=8)
    ax.set_yticks(range(len(units)))
    ax.set_yticklabels(units, fontsize=10)
    for t, u in zip(ax.get_yticklabels(), units):
        t.set_color(DOMAIN_COLOURS.get(u, '0.2'))
    drawn = ', '.join(args.bands_drawn)
    ax.set_title(f'Pain slope by {args.level} and band — the SAME colormap and '
                 f'scale as the glass brains here\n(brains drawn for {drawn})',
                 fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label('Δ log10 band power per pain point', fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.text(0.01, 0.005,
             f'Identical numbers to the run\'s own fig_domain_summary.png, redrawn in '
             f'{args.cmap} at ±{cap:.4f} so a cell here and the electrodes of that '
             f'{args.level} on the brains beside it are the SAME COLOUR. Outlines and * '
             'are BH-significant in the group fit. ' + caveat + '\n' + DISCLAIMER,
             fontsize=6.4, va='bottom', ha='left', color='0.35', wrap=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info('wrote %s', out_path.name)


def _caption(el, args, cap, caveat):
    return (
        f'{len(el)} bipolar pairs from {el["subject_id"].nunique()} subjects, at the MNI '
        f'MIDPOINT of each pair. EVERY ELECTRODE IN A {args.level.upper()} SHARES ONE '
        f'COLOUR: the model fits one slope per ({args.level}, band), so this is the '
        f'{args.level} assignment tinted by its group effect, NOT a per-electrode effect '
        'map, and no gradient across a territory is implied or estimable. Colour scale is '
        f'±{cap:.4f}, the same cap fig_domain_summary uses (max |β| over every band and '
        f'{args.level}), so colours mean the same thing here, there, and across bands. '
        f'Colormap {args.cmap}, with a thin marker edge so a near-zero electrode stays '
        'visible rather than vanishing into the white background. Within a band the '
        'domains often share a SIGN, so they occupy one half of the symmetric scale; that '
        "is the cost of sharing the heatmap's cap, and it is what makes the bands "
        'comparable to each other. '
        'Electrodes outside the MNI152 brain mask are dropped, not clipped. '
        + ('* = BH-significant in the group fit; grey = not. '
           if args.grey_nonsignificant else
           '* marks BH-significant units in the legend; colour does NOT encode '
           'significance here, every electrode is drawn. ')
        + caveat + '\n' + DISCLAIMER)


# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True,
                    help='A domain-model run written by run_domain_model.py.')
    ap.add_argument('--bands', nargs='*', default=list(DEFAULT_BANDS))
    ap.add_argument('--level', choices=['domain', 'roi'], default='domain',
                    help="'domain' colours by the domain's slope (5 colours). "
                         "'roi' colours by each region's OWN slope from a "
                         'region-level run (--region-run), which is the '
                         'spatially varying version.')
    ap.add_argument('--region-run', default=None,
                    help='A run_bandpower_mixed run directory, required for '
                         '--level roi.')
    ap.add_argument('--roi-scheme', default='roi_v2_ofc_ins',
                    help='Region scheme for --level roi.')
    ap.add_argument('--edge-color', default='0.25',
                    help='Marker edge, which is what keeps a near-zero '
                         'electrode visible under a light-centred colormap.')
    ap.add_argument('--edge-width', type=float, default=0.3,
                    help='0 to disable the edge.')
    ap.add_argument('--cmap', default='coolwarm',
                    help='Diverging colormap. THE MIDPOINT MUST NOT BE WHITE '
                         'on a glass brain. Recommended: '
                         + '; '.join(f'{k} ({v.split(",")[0]})'
                                     for k, v in CMAP_NOTES.items()))
    ap.add_argument('--node-size', type=float, default=16)
    ap.add_argument('--alpha', type=float, default=0.85)
    ap.add_argument('--brain-dilation-mm', type=float, default=0.0,
                    help='Tolerance on the MNI152 brain mask, for contacts a '
                         'millimetre or two outside a thresholded template. 0 '
                         'means strictly inside.')
    ap.add_argument('--grey-nonsignificant', action='store_true',
                    help='Draw units the group fit did not call significant in '
                         'flat grey instead of their colour. Off by default: '
                         'all electrodes are shown, because a blank area on a '
                         'glass brain is otherwise indistinguishable from an '
                         'area with no coverage.')
    ap.add_argument('--no-heatmap', action='store_true',
                    help='Skip the companion heatmap. It is on by default so '
                         'the folder is self-contained: the same numbers in '
                         'the same colormap and scale as the brains, which is '
                         'what makes "the electrode is the colour of its cell" '
                         'literally true.')
    ap.add_argument('--label', default=None,
                    help='Name of the versioned folder under glass_brain/. '
                         'Defaults to the colormap, so comparing colormaps '
                         'never overwrites.')
    args = ap.parse_args()

    io.warn_if_dirty()
    if args.cmap not in plt.colormaps():
        raise SystemExit(f'unknown colormap {args.cmap!r}')
    if args.cmap in ('RdBu_r', 'bwr', 'seismic', 'PuOr_r') and not args.edge_width:
        logger.warning("colormap %r has a near-WHITE midpoint and --edge-width "
                       'is 0, so a unit with a slope near zero will be '
                       'invisible on the white glass brain. %s', args.cmap,
                       CMAP_NOTES.get(args.cmap, ''))
    if args.cmap in ('berlin', 'vanimo', 'managua'):
        logger.warning("colormap %r has a DARK midpoint: where the units in a "
                       'band share a sign, the weakest will be the most '
                       'visually salient. %s', args.cmap,
                       CMAP_NOTES.get(args.cmap, ''))

    run_dir = Path(args.run_dir)
    el, slopes, params, counts = build_electrodes(run_dir, args)

    bands_def = BAND_SETS[params['band_set']]
    caveat = domain_caveat(params['roi_scheme'])

    if args.level == 'domain':
        slopes = slopes.rename(columns={'domain': 'unit'})
        slopes['significant'] = slopes.get(
            'p_bh_reject', pd.Series(False, slopes.index)).fillna(False)
        args.unit_order = roi_schemes.domain_scheme(
            params['roi_scheme'])['display']
        parents = [str(run_dir / 'domain_slopes.parquet')]
    else:
        if not args.region_run:
            raise SystemExit('--level roi needs --region-run <bandpower run>')
        cells = io.read_table(Path(args.region_run) / 'band_cells.parquet',
                              on_stale='warn')
        slopes = cells.rename(columns={'region': 'unit',
                                       'beta_nrs_within': 'beta'})
        slopes['significant'] = slopes['p_bh_reject'].fillna(False)
        args.unit_order = roi_schemes.roi_regions(args.roi_scheme)
        parents = [str(Path(args.region_run) / 'band_cells.parquet')]

    # THE CAP IS OVER THE WHOLE GRID, every band -- not over the bands drawn.
    # summary_figure computes it the same way, which is what makes the brains
    # and the heatmap share a scale.
    cap = float(np.nanmax(np.abs(slopes['beta'].to_numpy(dtype=float))))
    logger.info('colour scale ±%.5f (max |beta| over all %d bands x %d %ss)',
                cap, slopes['band'].nunique(), slopes['unit'].nunique(),
                args.level)

    want = [b for b in args.bands if b in set(slopes['band'])]
    missing = [b for b in args.bands if b not in set(slopes['band'])]
    if missing:
        raise SystemExit(f'bands {missing} are not in this run')

    label = args.label or args.cmap
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = run_dir / SUBDIR / f'{label}_{stamp}'
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info('versioned output dir (never overwrites): %s', out_dir)

    args.bands_drawn = want
    if not args.no_heatmap:
        figure_heatmap(slopes, want, cap, args, bands_def,
                       out_dir / 'fig_domain_summary_matched.png', caveat)
    for band in want:
        figure_for_band(el, slopes, band, cap, args, bands_def,
                        out_dir / f'fig_glass_{band}.png', caveat)
    if len(want) > 1:
        figure_all_bands(el, slopes, want, cap, args, bands_def,
                         out_dir / 'fig_glass_all_bands.png', caveat)

    # WHAT COLOURED WHAT, as data rather than as a figure caption: one row per
    # (band, unit) with the beta, its p, and how many electrodes took that
    # colour. This is the link back from a dot to the number behind it.
    src = (slopes[slopes['band'].isin(want)]
           [['band', 'unit', 'beta', 'significant']
            + [c for c in ('p', 'p_bh', 'se') if c in slopes.columns]]
           .copy())
    n_el = el.groupby('unit').size().rename('n_electrodes')
    src = src.merge(n_el, left_on='unit', right_index=True, how='left')
    io.write_table(src, out_dir / 'colour_source.csv', script=SCRIPT,
                   parents=parents,
                   params={'cmap': args.cmap, 'vmin': -cap, 'vmax': cap,
                           'level': args.level, 'bands': want},
                   extra={'reading': 'the beta that coloured every electrode of '
                                     'that unit in that band; n_electrodes is '
                                     'how many dots took the colour'})
    io.write_table(el, out_dir / 'glass_electrodes.csv', script=SCRIPT,
                   parents=parents,
                   params={'level': args.level,
                           'brain_dilation_mm': args.brain_dilation_mm},
                   subjects=sorted(el['subject_id'].unique()),
                   extra={'coordinate_basis':
                              'MNI midpoint of the bipolar pair; unit comes '
                              "from the ANODE's DK label, matching the model",
                          'electrode_counts': counts})

    io.write_run_provenance(
        out_dir, script=SCRIPT,
        params={**vars(args), 'cap': cap, 'bands_drawn': want,
                'source_run': str(run_dir), 'electrode_counts': counts,
                'band_set': params.get('band_set'),
                'roi_scheme': params.get('roi_scheme'),
                'unit': params.get('unit')},
        parents=[str(run_dir / 'provenance.json')] + parents,
        subjects=sorted(el['subject_id'].unique()),
        extra={
            'status': DISCLAIMER, 'domain_caveat': caveat,
            'colour_source': {
                band: {r.unit: {'beta': float(r.beta),
                                'significant': bool(r.significant),
                                'n_electrodes': int(r.n_electrodes)
                                if np.isfinite(r.n_electrodes) else 0}
                       for r in src[src['band'] == band].itertuples()}
                for band in want},
            'colour_scale': f'diverging {args.cmap}, symmetric at ±{cap:.6f} = '
                            'max |beta| over EVERY band and unit, the same cap '
                            'fig_domain_summary uses',
            'cmap_note': CMAP_NOTES.get(args.cmap, 'not one of the vetted maps'),
            'what_one_colour_means':
                f'every electrode in a {args.level} shares one colour, because '
                f'the model fits one slope per ({args.level}, band). This is '
                'the assignment tinted by the group effect, not a '
                'per-electrode effect map.',
            'brain_mask': 'nilearn load_mni152_brain_mask(), dilation '
                          f'{args.brain_dilation_mm} mm; electrodes outside it '
                          'are DROPPED so nothing floats outside the outline',
            'versioning': 'each invocation writes a new <label>_<timestamp> '
                          'folder under glass_brain/ and never overwrites, so '
                          'an earlier version stays citable',
        })

    io.log_analysis(
        f'glass brains: {args.level}-level pain slope painted on electrodes, '
        f'{", ".join(want)}, cmap {args.cmap} (EXPLORATORY)', out_dir)
    logger.info('done -> %s', out_dir)


if __name__ == '__main__':
    main()

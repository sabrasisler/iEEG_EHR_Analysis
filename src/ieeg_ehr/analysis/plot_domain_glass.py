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

LME4 RUNS (`run_domain_lmer.py`) are read too. They carry no
`domain_slopes.parquet` / `parcel_coverage.parquet`: the colours come from
`table_domain_lmer_heatmap.csv` -- the exact table behind the run's
`fig_domain_lmer_heatmap.png`, BH over its 24 cells, Control excluded -- and
each electrode's domain from the FRAMES the model was fitted on, so the brain
cannot disagree with the model about membership.

`--display-mode z --significant-only` gives the axial (XY) brain carrying ONLY
the BH-significant (domain, band) cells, one brain per band side by side.
`--colour-by signed_log10q` colours by significance (sign of the slope x
-log10 BH q) instead of the slope itself.

EXPLORATORY. Discovery cohort. Nominations, not findings.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches  # noqa: F401  (matplotlib.patches.Rectangle below)
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

LMER_TABLE = 'table_domain_lmer_heatmap.csv'


def load_lmer_run(run_dir, prov):
    """The same four things as `load_run`, for an lme4 domain run.

    params gain the frames run's band_set / roi_scheme / epoch_minutes (the
    lme4 run inherits them and does not restate them). Coverage is the unique
    (subject, channel, domain) over every band's frame -- the rows actually
    fitted.
    """
    params, subjects = dict(prov.get('params', {})), prov.get('subjects', [])
    frames_dir = Path(params['frames_dir'])
    fprov = json.loads((frames_dir.parent / 'provenance.json').read_text())
    for k in ('band_set', 'roi_scheme', 'epoch_minutes', 'unit'):
        params.setdefault(k, fprov.get('params', {}).get(k))
    slopes = io.read_table(run_dir / LMER_TABLE, on_stale='warn').rename(
        columns={'NRS_within.trend': 'beta', 'SE': 'se', 'p.value': 'p'})
    slopes['term'] = 'pain'
    cov = []
    for band in params.get('bands_fitted') or sorted(slopes['band'].unique()):
        f = io.read_table(frames_dir / f'{band}.parquet', on_stale='ignore',
                          columns=['subject', 'channel_uid', 'domain'])
        cov.append(f.drop_duplicates())
    cov = pd.concat(cov, ignore_index=True).drop_duplicates()
    cov['subject_id'] = cov['subject']
    cov['channel'] = cov['channel_uid'].str.split('|', n=1).str[1]
    cov = cov.drop_duplicates(['subject_id', 'channel'])
    return params, subjects, slopes, cov[['subject_id', 'channel', 'domain']]


def load_run(run_dir):
    """(params, subjects, slopes with term=='pain', coverage) for a domain run."""
    run_dir = Path(run_dir)
    prov = json.loads((run_dir / 'provenance.json').read_text())
    if (not (run_dir / 'domain_slopes.parquet').exists()
            and (run_dir / LMER_TABLE).exists()):
        return load_lmer_run(run_dir, prov)
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
    if args.significant_only:
        # Not drawn at all -- a band with no significant unit is an empty brain.
        ok &= el['unit'].map(sig_by_unit).eq(True).to_numpy()

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
                node_size=args.node_size, display_mode=args.display_mode, colorbar=False,
                figure=figure, axes=axes, alpha=args.alpha,
                node_cmap=matplotlib.colors.ListedColormap(['0.72']),
                node_vmin=-1, node_vmax=1, node_kwargs=kw, title=title)
            ok = ok & ~ns
            title = None       # already drawn on the first call

    if not ok.any():
        # Still draw the empty outline, so the band's panel is not a hole.
        return plotting.plot_glass_brain(None, display_mode=args.display_mode,
                                         figure=figure, axes=axes, title=title)
    return plotting.plot_markers(
        node_values=vals[ok], node_coords=coords[ok], node_size=args.node_size,
        node_cmap=plt.get_cmap(args.cmap), node_vmin=-cap, node_vmax=cap,
        display_mode=args.display_mode, colorbar=False, figure=figure, axes=axes,
        alpha=args.alpha, node_kwargs=kw, title=title)


def unit_legend(ax, units, beta_by_unit, sig_by_unit, cap, args, colour_by_unit,
                slope_by_unit=None):
    """A row of swatches: every unit, its beta, and whether it is significant.

    The colorbar says what a colour MEANS; this says which colour each domain
    GOT. Both are needed, because the reader's question is "what is Sensory
    doing", and on a glass brain there is no axis label to answer it.
    """
    import matplotlib.patches as mpatches
    handles = []
    slope_by_unit = slope_by_unit or beta_by_unit
    for u in units:
        b = beta_by_unit.get(u)
        if b is None or not np.isfinite(b):
            continue
        if args.significant_only and not sig_by_unit.get(u):
            continue
        star = ' *' if sig_by_unit.get(u) else ''
        handles.append(mpatches.Patch(
            facecolor=colour_by_unit[u], edgecolor='0.3',
            label=f'{u}  β={slope_by_unit[u]:+.4f}{star}'))
    if handles:
        ax.legend(handles=handles, loc='center', ncol=min(len(handles), 5),
                  frameon=False, fontsize=9, handlelength=1.3,
                  columnspacing=1.4)
    ax.axis('off')


def figure_for_band(el, slopes, band, cap, args, bands_def, out_path, caveat):
    """One PNG for one band: the brain, a colorbar, and the unit swatches."""
    sub = slopes[slopes['band'] == band]
    beta_by_unit = dict(zip(sub['unit'], sub['value']))
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
    cb.set_label(args.value_label, fontsize=9)
    cb.ax.tick_params(labelsize=8)

    lax = fig.add_axes((0.02, 0.15, 0.84, 0.08))
    units = [u for u in args.unit_order if u in beta_by_unit]
    unit_legend(lax, units, beta_by_unit, sig_by_unit, cap, args, colour_by_unit,
                slope_by_unit=dict(zip(sub['unit'], sub['beta'])))

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
        beta_by_unit = dict(zip(sub['unit'], sub['value']))
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
    cb.set_label(args.value_label, fontsize=9)
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


def figure_bands_row(el, slopes, bands, cap, args, bands_def, out_path, caveat):
    """One PNG, one SINGLE-VIEW brain per band in a row, one shared colorbar.

    The layout for `--display-mode z` (or any one-letter mode): the stacked
    `figure_all_bands` gives each band a full-width row, which for one axial
    brain is mostly empty space. Under each brain, the units that coloured it.
    """
    from nilearn import plotting            # noqa: F401  (import cost, once)

    cmap = plt.get_cmap(args.cmap)
    norm = matplotlib.colors.Normalize(-cap, cap)
    n = len(bands)
    fig = plt.figure(figsize=(3.6 * n + 1.4, 7.4))
    w = 0.86 / n
    displays = []
    for i, band in enumerate(bands):
        sub = slopes[slopes['band'] == band]
        beta_by_unit = dict(zip(sub['unit'], sub['value']))
        slope_by_unit = dict(zip(sub['unit'], sub['beta']))
        q_by_unit = dict(zip(sub['unit'], sub.get('p_bh', sub['beta'] * np.nan)))
        sig_by_unit = dict(zip(sub['unit'], sub['significant']))
        left = 0.01 + i * w
        d = draw_band(el, beta_by_unit, sig_by_unit, band, cap, args,
                      figure=fig, axes=(left, 0.36, w * 0.96, 0.54))
        displays.append(d)
        fig.text(left + w * 0.48, 0.92, band_label(band, bands_def),
                 ha='center', va='bottom', fontsize=13)
        shown = [u for u in args.unit_order if u in beta_by_unit
                 and (sig_by_unit.get(u) or not args.significant_only)]
        lines = [f'{u}  β={slope_by_unit[u]:+.4f}'
                 + (f', q={q_by_unit[u]:.3f}' if np.isfinite(q_by_unit[u]) else '')
                 for u in shown] or ['no significant domain']
        for k, (u, line) in enumerate(zip(shown or [None], lines)):
            y = 0.325 - k * 0.035
            if u is not None:
                fig.add_artist(matplotlib.patches.Rectangle(
                    (left + 0.02 * w, y - 0.012), 0.012, 0.024,
                    transform=fig.transFigure,
                    facecolor=cmap(norm(beta_by_unit[u])), edgecolor='0.3',
                    lw=0.5))
            fig.text(left + 0.02 * w + 0.018, y, line, va='center', ha='left',
                     fontsize=9.5, color='0.15' if u else '0.45')

    cax = fig.add_axes((0.905, 0.40, 0.012, 0.46))
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=cax)
    cb.set_label(args.value_label, fontsize=9)
    cb.ax.tick_params(labelsize=8)

    what = ('BH-significant domains only' if args.significant_only
            else 'every domain')
    fig.suptitle(f'Pain slope by {args.level}, {what} — axial view, one scale '
                 'across bands' if args.display_mode == 'z' else
                 f'Pain slope by {args.level}, {what} — one scale across bands',
                 fontsize=14, y=0.995)
    fig.text(0.01, 0.005, _caption(el, args, cap, caveat), fontsize=6.0,
             va='bottom', ha='left', color='0.35', wrap=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    for d in displays:
        if d is not None:
            d.close()
    logger.info('wrote %s', out_path.name)


#: --sig-category classes: significant in the LOW bands only, the HIGH bands
#: only, or both. Blue/red follow the heatmap (in this run every significant
#: low-band slope is negative and every high-band one positive); purple is the
#: mix. The sign is NOT what picks the class -- the legend states it per cell.
SIG_CATEGORY_COLOURS = {'low': '#2c7bb6', 'high': '#d7191c', 'both': '#8e44ad'}


def sig_categories(slopes, low_bands):
    """One row per unit significant in any band: its class and its cells.

    `class` is 'low' (significant only in `low_bands`), 'high' (only in the
    others) or 'both'. `cells` lists every significant (band, beta, q) so the
    legend can say what put the unit in its class.
    """
    sig = slopes[slopes['significant'].eq(True)]
    rows = []
    for unit, g in sig.groupby('unit', sort=False):
        low = g['band'].isin(low_bands)
        cls = 'both' if low.any() and (~low).any() else 'low' if low.any() else 'high'
        rows.append({'unit': unit, 'class': cls,
                     'bands': ','.join(g['band']),
                     'cells': '; '.join(
                         f'{b} {v:+.4f}' + (f' (q={q:.3f})' if np.isfinite(q) else '')
                         for b, v, q in zip(g['band'], g['beta'],
                                            g.get('p_bh', g['beta'] * np.nan)))})
    return pd.DataFrame(rows, columns=['unit', 'class', 'bands', 'cells'])


def figure_sig_category(el, cats, args, bands_def, out_path, caveat):
    """ONE brain: each significant unit's electrodes in blue / red / purple.

    The colour is a CATEGORY (which band group the unit is significant in),
    not a number, so there is no colorbar; the legend carries the numbers.
    Units significant nowhere are not drawn.
    """
    from nilearn import plotting
    import matplotlib.lines as mlines

    low_lbl = '/'.join(args.low_bands)
    cls_name = {'low': f'{low_lbl} only', 'high': f'higher bands only',
                'both': f'{low_lbl} AND higher bands'}
    fig = plt.figure(figsize=(13, 5.6))
    display = plotting.plot_glass_brain(None, display_mode=args.display_mode,
                                        figure=fig, axes=(0.0, 0.08, 0.64, 0.88))
    handles = []
    order = {u: i for i, u in enumerate(args.unit_order)}
    cats = cats.sort_values('unit', key=lambda c: c.map(order).fillna(99))
    for cls in ('low', 'both', 'high'):
        sub = cats[cats['class'] == cls]
        if sub.empty:
            continue
        handles.append(mlines.Line2D([], [], ls='none', label=cls_name[cls],
                                     marker=None))
        for r in sub.itertuples():
            pts = el.loc[el['unit'] == r.unit, ['mni_x', 'mni_y', 'mni_z']]
            display.add_markers(pts.to_numpy(dtype=float),
                                marker_color=SIG_CATEGORY_COLOURS[cls],
                                marker_size=args.node_size, alpha=args.alpha,
                                edgecolors=args.edge_color,
                                linewidths=args.edge_width)
            handles.append(mlines.Line2D(
                [], [], ls='none', marker='o', markersize=9,
                markerfacecolor=SIG_CATEGORY_COLOURS[cls],
                markeredgecolor=args.edge_color,
                label=f'{r.unit} (n={len(pts)}): {r.cells}'))
    leg = fig.legend(handles=handles, loc='center left',
                     bbox_to_anchor=(0.645, 0.52), frameon=False, fontsize=9.5,
                     handletextpad=0.5, labelspacing=0.7)
    for t, h in zip(leg.get_texts(), handles):
        if h.get_marker() in (None, 'None', ''):
            t.set_fontweight('bold')
            t.set_fontsize(11)
    fig.suptitle(f'Where the pain slope is BH-significant: {low_lbl} (blue), '
                 'higher bands (red), both (purple)', fontsize=13, y=0.99)
    fig.text(0.01, 0.005,
             f'{len(el)} bipolar pairs (MNI midpoint) from '
             f'{el["subject_id"].nunique()} subjects, but ONLY units '
             f'BH-significant in some band are drawn; every electrode of a '
             f'{args.level} shares its colour, because the model fits one slope '
             f'per ({args.level}, band). Class = which bands the unit is '
             f'significant in, NOT the sign; slopes are in the legend (Δ log10 '
             'power per pain point). An empty area is not absent coverage. '
             + caveat + '\n' + DISCLAIMER,
             fontsize=6.0, va='bottom', ha='left', color='0.35', wrap=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    display.close()
    logger.info('wrote %s', out_path.name)


#: --split-bands: ONE colour per band, so a unit significant in two bands can
#: show both. Blue for delta and red for beta keep the --sig-category reading;
#: gamma is orange so a beta+gamma unit is not split red/red.
BAND_COLOURS = {'delta': '#2c7bb6', 'theta': '#41b6c4', 'alpha': '#1b9e77',
                'beta': '#d7191c', 'gamma': '#f08c00', 'high_gamma': '#8c510a'}


def wedge_markers(n):
    """n equal pie-slice marker Paths, first slice starting at 12 o'clock.

    matplotlib rescales a custom marker by its max |vertex| and does NOT
    re-centre it, so every slice of a unit disc lands on the same centre at the
    same size -- the slices of one dot tile a full disc.
    """
    from matplotlib.path import Path as MPath
    if n == 1:
        return ['o']
    step = 360.0 / n
    return [MPath.wedge(90 + i * step, 90 + (i + 1) * step) for i in range(n)]


#: Band names as symbols, for the compact (--bare / --with-heatmap) labels.
BAND_SYMBOLS = {'delta': 'δ', 'theta': 'θ', 'alpha': 'α', 'beta': 'β',
                'gamma': 'γ', 'high_gamma': 'hγ'}

#: Domain LABEL colours for the compact figures (heatmap row labels, legend
#: text). Only the text takes them -- the dots and cells stay on the slope
#: colormap, so these can never be read as a value.
LABEL_DOMAIN_COLOURS = {'Sensory': '#d95f02', 'Affective': '#1b9e77',
                        'Cognitive': '#7570b3', 'Modulatory': '#447eae'}

SPLIT_LEGEND_TITLE = 'Processing Domain (significant cells)'

#: Type (pt) of the heatmap + brains panel (`figure_heatmap_and_split`).
#: PANEL_TYPE is what it has always drawn (None = leave the heatmap's own);
#: POSTER_TYPE (`--poster`) matches the other poster cuts
#: (`plot_domain_lmer_consistency.POSTER_FS`, `plot_domain_lmer_med.POSTER`):
#: title 12 bold, domain names 11, band symbols 11, axis label 10, cell text 9,
#: colorbar 9 / 8.5, legend 9.5 with an 11 pt bold title.
PANEL_TYPE = dict(title=None, domain=None, band=12, xlabel=10, cell=None,
                  cb_label=9, cb_tick=8, legend=9.5, legend_title=10)
POSTER_TYPE = dict(title=12, domain=11, band=11, xlabel=10, cell=9,
                   cb_label=9, cb_tick=8.5, legend=9.5, legend_title=11)
POSTER_TITLE = 'Pain–power slopes by pain processing domain'
#: Top margin (in) with the poster title in it, and the poster heatmap height.
POSTER_TITLE_IN = 0.50
POSTER_HEAT_IN = 2.5
POSTER_HEAT_X = (0.175, 0.665)


def draw_sig_split(fig, rect, el, slopes, args, bands_def, cap, node_size,
                   compact):
    """The split-dot brain into `fig` at `rect`; returns (display, handles, source).

    A unit significant in one band is a solid dot; in two bands a
    half-and-half dot; in k bands k equal wedges, band order starting at 12
    o'clock (for k=2: LEFT half = the lower band).

    `--split-colour value` (default): each slice is THAT CELL'S HEATMAP
    COLOUR -- `args.cmap` on +/-cap, the same cap `plot_domain_lmer.heatmap`
    uses -- so a slice and its heatmap cell are the same RGB. `--split-colour
    band`: one fixed colour per band (BAND_COLOURS).

    `compact` labels a unit by its significant bands' SYMBOLS only ("Cognitive:
    δ, β"); otherwise by n, slope and q.
    """
    from nilearn import plotting
    import matplotlib.lines as mlines

    band_order = [b for b in bands_def if b in set(slopes['band'])]
    sig = slopes[slopes['significant'].eq(True)].copy()
    sig['b_ord'] = sig['band'].map({b: i for i, b in enumerate(band_order)})
    order = {u: i for i, u in enumerate(args.unit_order)}
    units = sorted(sig['unit'].unique(), key=lambda u: order.get(u, 99))

    display = plotting.plot_glass_brain(None, display_mode=args.display_mode,
                                        figure=fig, axes=rect,
                                        annotate=not args.no_annotate)
    rim = args.edge_color if args.edge_width > 0 else 'none'
    by_value = args.split_colour == 'value'
    cmap = plt.get_cmap(args.cmap)
    norm = matplotlib.colors.Normalize(-cap, cap)

    def colour(band, value):
        return cmap(norm(value)) if by_value else BAND_COLOURS.get(band, '0.5')

    handles = []
    if not by_value:
        handles.append(mlines.Line2D([], [], ls='none', marker=None, label='band'))
        for b in [b for b in band_order if b in set(sig['band'])]:
            handles.append(mlines.Line2D(
                [], [], ls='none', marker='o', markersize=9,
                markerfacecolor=BAND_COLOURS.get(b, '0.5'), markeredgecolor=rim,
                label=BAND_SYMBOLS.get(b, b) if compact else band_label(b, bands_def)))
    handles.append(mlines.Line2D([], [], ls='none', marker=None,
                                 label=SPLIT_LEGEND_TITLE))
    source = []
    for u in units:
        g = sig[sig['unit'] == u].sort_values('b_ord')
        cols = [colour(b, v) for b, v in zip(g['band'], g['value'])]
        pts = el.loc[el['unit'] == u, ['mni_x', 'mni_y', 'mni_z']].to_numpy(dtype=float)
        for m, c in zip(wedge_markers(len(cols)), cols):
            display.add_markers(pts, marker_color=c, marker_size=node_size,
                                marker=m, alpha=args.alpha, linewidths=0)
        # One rim round the whole disc, not one per slice, so a split dot
        # still reads as ONE electrode. --edge-width 0 draws no rim at all.
        if args.edge_width > 0:
            display.add_markers(pts, marker_color='none', marker_size=node_size,
                                marker='o', edgecolors=args.edge_color,
                                linewidths=args.edge_width)
        q = g.get('p_bh', g['beta'] * np.nan)
        cells = '; '.join(f'{b} {v:+.4f}' + (f' (q={x:.3f})' if np.isfinite(x) else '')
                          for b, v, x in zip(g['band'], g['beta'], q))
        label = (f'{u}: ' + ', '.join(BAND_SYMBOLS.get(b, b) for b in g['band'])
                 if compact else f'{u} (n={len(pts)}): {cells}')
        # --poster: the count of electrodes DRAWN for the unit (inside-brain,
        # so it can sit below the unit's coverage n elsewhere).
        if compact and getattr(args, 'poster', False):
            label += f' (n = {len(pts)})'
        kw = ({'fillstyle': 'left', 'markerfacecoloralt': cols[1]}
              if len(cols) == 2 else {})
        handles.append(mlines.Line2D(
            [], [], ls='none', marker='o', markersize=10, markerfacecolor=cols[0],
            markeredgecolor=rim, label=label, **kw))
        source.append({'unit': u, 'bands': ','.join(g['band']),
                       'colours': ','.join(matplotlib.colors.to_hex(c)
                                           for c in cols), 'cells': cells,
                       'n_electrodes': len(pts)})
    return display, handles, pd.DataFrame(source)


def _split_legend(fig, handles, anchor, fontsize, title_size, title=None,
                  loc='center left', ncol=1, colour_text=False):
    """The split-dot legend. With `title`, the header handle is dropped and the
    title is the legend's own, CENTRED over the entries. `colour_text` tints
    each entry by LABEL_DOMAIN_COLOURS (off: the text colour competing with
    the marker colour, which means something else, read as confusing)."""
    if title is not None:
        handles = [h for h in handles if h.get_marker() not in (None, 'None', '')]
    leg = fig.legend(handles=handles, loc=loc, bbox_to_anchor=anchor,
                     frameon=False, fontsize=fontsize, handletextpad=0.5,
                     labelspacing=0.7, borderaxespad=0, title=title, ncol=ncol,
                     columnspacing=1.6,
                     title_fontproperties={'weight': 'bold', 'size': title_size},
                     alignment='center')
    if title is not None:
        leg.get_title().set_multialignment('center')
        for t in leg.get_texts():
            t.set_color(LABEL_DOMAIN_COLOURS.get(t.get_text().split(':')[0], '0.1')
                        if colour_text else '0.1')
    else:
        for t, h in zip(leg.get_texts(), handles):
            if h.get_marker() in (None, 'None', ''):
                t.set_fontweight('bold')
                t.set_fontsize(title_size)
    return leg


def figure_sig_split(el, slopes, args, bands_def, out_path, caveat, cap=None):
    """ONE brain: each significant unit's dots split into its cells' colours.

    `--bare`: no title, caption or colorbar, and a compact symbol legend -- the
    figure is meant to sit beside the heatmap and share ITS colorbar.
    """
    by_value = args.split_colour == 'value'
    fig = plt.figure(figsize=(13, 5.6))
    display, handles, source = draw_sig_split(
        fig, (0.0, 0.08, 0.64, 0.88), el, slopes, args, bands_def, cap,
        args.node_size, compact=args.bare)
    _split_legend(fig, handles,
                  (0.645, 0.52 if (args.bare or not by_value) else 0.62),
                  9.5, 11, title=SPLIT_LEGEND_TITLE if args.bare else None)
    if not args.bare:
        cmap = plt.get_cmap(args.cmap)
        norm = matplotlib.colors.Normalize(-cap, cap)
        if by_value:
            cax = fig.add_axes((0.665, 0.22, 0.20, 0.025))
            cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                              cax=cax, orientation='horizontal')
            cb.set_label(args.value_label + ' (same colours as the heatmap)',
                         fontsize=9)
            cb.ax.tick_params(labelsize=8)
        fig.suptitle('Where the pain slope is BH-significant — each dot split into '
                     'its significant cells\' ' + ('heatmap colours' if by_value
                                                   else 'band colours'),
                     fontsize=13, y=0.99)
        fig.text(0.01, 0.005,
                 f'{len(el)} bipolar pairs (MNI midpoint) from '
                 f'{el["subject_id"].nunique()} subjects, but ONLY {args.level}s '
                 'BH-significant in some band are drawn. Every electrode of a '
                 f'{args.level} shares its dot, because the model fits one slope per '
                 f'({args.level}, band). A dot significant in two bands is split in '
                 'half, LEFT half = the lower band. '
                 + (f'Each half is that cell\'s colour in the heatmap ({args.cmap}, '
                    f'±{cap:.4f} = max |value| over the whole grid). ' if by_value else
                    'Colour names the BAND, not the sign or size. ')
                 + 'Slopes (Δ log10 power per pain point) are in the legend. An empty '
                 'area is not absent coverage. '
                 + caveat + '\n' + DISCLAIMER,
                 fontsize=6.0, va='bottom', ha='left', color='0.35', wrap=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    display.close()
    logger.info('wrote %s', out_path.name)
    return source


def _figure_bbox(artists, fig):
    """Union of the artists' drawn extents, in figure fraction."""
    from matplotlib.transforms import Bbox
    r = fig.canvas.get_renderer()
    return Bbox.union([a.get_window_extent(r) for a in artists]).transformed(
        fig.transFigure.inverted())


def _ink_bbox(fig, y0, y1, hide=()):
    """Bounding box of NON-WHITE pixels between figure fractions y0..y1.

    Measured on the rendered raster because nothing else is reliable here:
    nilearn's axes fill the requested rect whatever the brain's aspect, and its
    outline patches do not enter the axes' dataLim, so neither extent is the
    brain. `hide` artists (e.g. the legend) are left out of the measurement.
    """
    from matplotlib.transforms import Bbox
    was = [a.get_visible() for a in hide]
    for a in hide:
        a.set_visible(False)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    for a, v in zip(hide, was):
        a.set_visible(v)
    H, W = img.shape[:2]
    r0, r1 = int((1 - y1) * H), int(np.ceil((1 - y0) * H))
    ink = (img[r0:r1] < 245).any(axis=2)
    rows, cols = np.where(ink)
    if not len(rows):
        raise RuntimeError('no ink found in the lower panel')
    return Bbox([[cols.min() / W, 1 - (r0 + rows.max() + 1) / H],
                 [(cols.max() + 1) / W, 1 - (r0 + rows.min()) / H]])


def figure_heatmap_and_split(run_dir, el, slopes, args, bands_def, out_path,
                             cap):
    """Heatmap on top, split-dot brains below, ONE colorbar -- a fixed-size panel.

    The heatmap is `plot_domain_lmer.heatmap` itself, fed the same table the
    run's `fig_domain_lmer_heatmap.png` was drawn from, so cells, BH outlines
    and stars are that figure's; its title goes, band labels become symbols,
    stars are enlarged, and row labels take LABEL_DOMAIN_COLOURS. Its colorbar
    is the brains' colorbar too: same cmap, same cap.

    Layout is in INCHES from the top: the heatmap keeps a fixed height and the
    lower panel takes the rest, so --panel-size's extra height all goes to the
    brains. The lower panel (brains + legend, as ONE group) is then centred on
    the heatmap by measuring what was actually drawn -- nilearn leaves uneven
    whitespace inside its axes, so centring the requested rect is not enough.
    Saved at EXACTLY --panel-size (no bbox_inches='tight').
    """
    from ieeg_ehr.analysis import plot_domain_lmer as pdl

    if args.cmap != 'RdBu_r' or args.split_colour != 'value':
        logger.warning('--with-heatmap: the heatmap is RdBu_r on slope; dots in '
                       '%s / %s do NOT share its colorbar', args.cmap,
                       args.split_colour)
    tab = io.read_table(Path(run_dir) / LMER_TABLE, on_stale='warn')
    domains = [d for d in args.unit_order if d in set(tab['domain'])]
    bands = [b for b in pdl.BAND_ORDER if b in set(tab['band'])]
    hcap = float(np.nanmax(np.abs(tab[pdl.SLOPE_COL].to_numpy(dtype=float))))

    w, h = args.panel_size
    # --poster: POSTER_TYPE sizes and a bold title, which takes the extra top
    # margin (the lower panel gives up that height, the heatmap keeps its own).
    fs = POSTER_TYPE if args.poster else PANEL_TYPE
    top_in = POSTER_TITLE_IN if args.poster else 0.165
    # The poster panel is 7 x 7 in: at the full 3.34 in heatmap the brains
    # would get under 2 in, so the poster heatmap is shorter.
    heat_in = POSTER_HEAT_IN if args.poster else 3.34
    gap_in, bottom_in = 0.62, 0.08
    fig = plt.figure(figsize=(w, h))
    # Heatmap left edge and width (figure fraction). The poster's 11 pt domain
    # names need more left margin on a 7 in page, or "Modulatory" is clipped.
    hx0, hw = POSTER_HEAT_X if args.poster else (0.125, 0.73)
    hax = fig.add_axes((hx0, 1 - (top_in + heat_in) / h, hw, heat_in / h))
    im = pdl.heatmap(hax, tab, domains, bands, hcap, '')
    hax.set_xticklabels([BAND_SYMBOLS.get(b, b) for b in bands],
                        fontsize=fs['band'])
    hax.set_xlabel('Frequency Band', fontsize=fs['xlabel'])
    for t in hax.get_yticklabels():
        t.set_color(LABEL_DOMAIN_COLOURS.get(t.get_text(), '0.1'))
        t.set_fontweight('bold')
        if fs['domain']:
            t.set_fontsize(fs['domain'])
    for t in hax.texts:
        if t.get_text() and set(t.get_text()) == {'*'}:
            t.set_fontsize(args.star_size)
            t.set_fontweight('bold')
        elif t.get_text() and fs['cell']:
            t.set_fontsize(fs['cell'])
    hpos = hax.get_position()
    cax = fig.add_axes((hpos.x1 + 0.02, hpos.y0, 0.018, hpos.height))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('Δ log10 power per pain point', fontsize=fs['cb_label'])
    cb.ax.tick_params(labelsize=fs['cb_tick'])

    target = (hpos.x0 + hpos.x1) / 2
    if args.poster:
        fig.text(target, 1 - 0.10 / h, POSTER_TITLE, ha='center', va='top',
                 fontsize=fs['title'], fontweight='bold', color='0.1')

    # Legend FIRST: one centred row along the bottom, title centred above it.
    # The brains then get every inch between it and the heatmap.
    low_top = (h - top_in - heat_in - gap_in) / h
    before = set(fig.axes)
    display, handles, source = draw_sig_split(
        fig, (0.0, bottom_in / h, 1.0, low_top - bottom_in / h), el, slopes,
        args, bands_def, hcap, args.panel_node_size, compact=True)
    entries = [x for x in handles if x.get_marker() not in (None, 'None', '')]
    right = getattr(args, 'legend_right', False)
    if right:
        # --legend-right: ONE narrow column right of the brains, entries and
        # title wrapped, so the brains take the full lower-panel height. The
        # two views are ~2.1:1, so they are WIDTH-bound here: every inch the
        # legend takes comes off the brains.
        for x in entries:
            x.set_label(x.get_label().replace(' (n = ', '\n(n = '))
        leg = _split_legend(fig, handles,
                            (0.995, (bottom_in / h + low_top) / 2),
                            fs['legend'], fs['legend_title'],
                            title=SPLIT_LEGEND_TITLE.replace(' (', '\n('),
                            loc='center right', ncol=1)
        fig.canvas.draw()
        lb = _figure_bbox([leg], fig)
        avail_y0, avail_y1 = bottom_in / h, low_top
        avail_x0, avail_x1 = 0.01, lb.x0 - 0.12 / w
    else:
        # With the n's, four entries no longer fit one row at 9 in: two per row.
        leg = _split_legend(fig, handles, (target, bottom_in / h), fs['legend'],
                            fs['legend_title'], title=SPLIT_LEGEND_TITLE,
                            loc='lower center',
                            ncol=2 if args.poster else len(entries))
        fig.canvas.draw()
        avail_y0 = _figure_bbox([leg], fig).y1 + 0.15 / h
        avail_y1 = low_top

    # Fit the brains into the space left by the legend, then centre what was
    # actually DRAWN -- on the heatmap, or (--legend-right) in the space left
    # of the legend. Scaling every brain axes about one point
    # scales the drawing uniformly (they are equal-aspect), so two passes
    # settle it; the second only mops up rounding.
    max_w = (avail_x1 - avail_x0) if right else args.panel_brain_width
    centre_x = (avail_x0 + avail_x1) / 2 if right else target
    # nilearn re-derives every view's position from `display.rect` on each
    # draw (an axes locator), so THAT is what gets scaled and shifted --
    # set_position on the view axes is silently overridden.
    for _ in range(3):
        bb = _ink_bbox(fig, 0.0, low_top, hide=[leg])
        scale = min((avail_y1 - avail_y0) / bb.height, max_w / bb.width)
        new_x0 = centre_x - bb.width * scale / 2
        new_y0 = avail_y0 + ((avail_y1 - avail_y0) - bb.height * scale) / 2
        rx0, ry0, rx1, ry1 = display.rect
        display.rect = (new_x0 + (rx0 - bb.x0) * scale,
                        new_y0 + (ry0 - bb.y0) * scale,
                        new_x0 + (rx1 - bb.x0) * scale,
                        new_y0 + (ry1 - bb.y0) * scale)
        fr = display.rect
        display.frame_axes.set_position((fr[0], fr[1], fr[2] - fr[0],
                                         fr[3] - fr[1]))
    bb = _ink_bbox(fig, 0.0, low_top, hide=[leg])
    lb = _figure_bbox([leg], fig)
    logger.info('brains centre %.4f, legend centre %.4f, heatmap centre %.4f; '
                'brains %.2f x %.2f in', (bb.x0 + bb.x1) / 2,
                (lb.x0 + lb.x1) / 2, target, bb.width * w, bb.height * h)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    display.close()
    logger.info('wrote %s (%.2f x %.2f in)', out_path.name, w, h)
    return source


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
    piv = (slopes.pivot_table(index='unit', columns='band', values='value',
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
            ax.text(j, i, f'{v:+.{args.value_decimals}f}'
                    + ('\n*' if sig.iat[i, j] else ''),
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
    cb.set_label(args.value_label, fontsize=9)
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
        f'Colour = {args.value_label}. '
        f'{len(el)} bipolar pairs from {el["subject_id"].nunique()} subjects, at the MNI '
        f'MIDPOINT of each pair. EVERY ELECTRODE IN A {args.level.upper()} SHARES ONE '
        f'COLOUR: the model fits one slope per ({args.level}, band), so this is the '
        f'{args.level} assignment tinted by its group effect, NOT a per-electrode effect '
        'map, and no gradient across a territory is implied or estimable. Colour scale is '
        f'±{cap:.{args.value_decimals}f}, the same cap fig_domain_summary uses (max |β| over every band and '
        f'{args.level}), so colours mean the same thing here, there, and across bands. '
        f'Colormap {args.cmap}, with a thin marker edge so a near-zero electrode stays '
        'visible rather than vanishing into the white background. Within a band the '
        'domains often share a SIGN, so they occupy one half of the symmetric scale; that '
        "is the cost of sharing the heatmap's cap, and it is what makes the bands "
        'comparable to each other. '
        'Electrodes outside the MNI152 brain mask are dropped, not clipped. '
        + ('ONLY BH-significant (domain, band) cells are drawn; a domain missing '
           'from a band was not significant there -- its electrodes exist but '
           'are not shown, so an empty area is NOT absent coverage. '
           if args.significant_only else
           '* = BH-significant in the group fit; grey = not. '
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
    ap.add_argument('--significant-only', action='store_true',
                    help='Draw ONLY the electrodes of units that are '
                         'BH-significant in that band; the rest are left off '
                         'entirely (not greyed).')
    ap.add_argument('--display-mode', default='lyrz',
                    help="nilearn display_mode. 'z' = axial only (the XY "
                         "plane). A one-letter mode also gets a side-by-side "
                         'figure, one brain per band.')
    ap.add_argument('--colour-by', choices=['beta', 'signed_log10q'],
                    default='beta',
                    help="'beta' = the pain slope (the heatmap cell's value). "
                         "'signed_log10q' = sign(beta) x -log10(BH q), i.e. "
                         'colour by significance with the direction kept.')
    ap.add_argument('--dpi', type=int, default=220)
    ap.add_argument('--sig-category', action='store_true',
                    help='ONE brain instead of one per band: every unit '
                         'significant in some band, blue if only in '
                         '--low-bands, red if only in the others, purple if '
                         'both. Use with --display-mode xz for sagittal + '
                         'axial.')
    ap.add_argument('--split-bands', action='store_true',
                    help='With --sig-category: colour by BAND instead of band '
                         'group, and split a dot into its bands\' colours '
                         'when the unit is significant in more than one.')
    ap.add_argument('--bare', action='store_true',
                    help='--split-bands figure without title, caption or '
                         'colorbar, with a symbol-only legend.')
    ap.add_argument('--with-heatmap', action='store_true',
                    help='--split-bands: ALSO write the heatmap-over-brains '
                         'panel at --panel-size (lme4 runs).')
    ap.add_argument('--panel-brain-width', type=float, default=0.94,
                    help='--with-heatmap: max width of the brains, as a '
                         'fraction of the figure width.')
    ap.add_argument('--star-size', type=float, default=14,
                    help='--with-heatmap: font size of the (bold) stars.')
    ap.add_argument('--panel-size', type=float, nargs=2, default=[9.0, 9.25],
                    metavar=('W_IN', 'H_IN'))
    ap.add_argument('--panel-node-size', type=float, default=14,
                    help='Marker size in the --with-heatmap panel, where the '
                         'brains are smaller than in the standalone figure.')
    ap.add_argument('--no-annotate', action='store_true',
                    help='--sig-category: drop the L/R labels from the glass '
                         'brain.')
    ap.add_argument('--split-colour', choices=['value', 'band'],
                    default='value',
                    help="--split-bands slice colour. 'value' = the cell's "
                         "heatmap colour (--cmap on the shared cap); 'band' = "
                         'one fixed colour per band.')
    ap.add_argument('--low-bands', nargs='*', default=['delta'],
                    help="The 'blue' band group for --sig-category; every "
                         'other band is the red group.')
    ap.add_argument('--no-heatmap', action='store_true',
                    help='Skip the companion heatmap. It is on by default so '
                         'the folder is self-contained: the same numbers in '
                         'the same colormap and scale as the brains, which is '
                         'what makes "the electrode is the colour of its cell" '
                         'literally true.')
    ap.add_argument('--poster', action='store_true',
                    help='with --with-heatmap: POSTER_TYPE sizes, the bold '
                         'POSTER_TITLE, n per unit in the legend, and output '
                         'into <run>/poster/<label>_<ts>/')
    ap.add_argument('--legend-right', action='store_true',
                    help='with --with-heatmap: the split-dot legend as one '
                         'wrapped column right of the brains, not a row below')
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
        parents = [str(run_dir / ('domain_slopes.parquet'
                                  if (run_dir / 'domain_slopes.parquet').exists()
                                  else LMER_TABLE))]
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

    if args.colour_by == 'signed_log10q':
        if 'p_bh' not in slopes.columns:
            raise SystemExit('--colour-by signed_log10q needs a p_bh column')
        slopes['value'] = (np.sign(slopes['beta'])
                           * -np.log10(slopes['p_bh'].astype(float)))
        args.value_label = 'sign(β) × −log10 BH q'
        args.value_decimals = 2
    else:
        slopes['value'] = slopes['beta']
        args.value_label = 'Δ log10 band power per pain point'
        args.value_decimals = 4

    # THE CAP IS OVER THE WHOLE GRID, every band -- not over the bands drawn.
    # summary_figure computes it the same way, which is what makes the brains
    # and the heatmap share a scale.
    cap = float(np.nanmax(np.abs(slopes['value'].to_numpy(dtype=float))))
    logger.info('colour scale ±%.5f (max |beta| over all %d bands x %d %ss)',
                cap, slopes['band'].nunique(), slopes['unit'].nunique(),
                args.level)

    want = [b for b in args.bands if b in set(slopes['band'])]
    missing = [b for b in args.bands if b not in set(slopes['band'])]
    if missing:
        raise SystemExit(f'bands {missing} are not in this run')

    label = args.label or args.cmap
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    # --poster cuts sit with the run's other poster figures, not in glass_brain/.
    out_dir = run_dir / ('poster' if args.poster else SUBDIR) / f'{label}_{stamp}'
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info('versioned output dir (never overwrites): %s', out_dir)

    args.bands_drawn = want
    if args.sig_category and args.split_bands:
        split = figure_sig_split(
            el, slopes[slopes['band'].isin(want)], args, bands_def,
            out_dir / f'fig_glass_sigsplit_{args.display_mode}.png', caveat,
            cap=cap)
        io.write_table(split, out_dir / 'category_source.csv', script=SCRIPT,
                       parents=parents, params={'bands': want},
                       extra={'reading': 'one row per unit significant in '
                                         'some band; colours are one per '
                                         'significant cell, in band order',
                              'split_colour': args.split_colour,
                              'cmap': args.cmap, 'vmin': -cap, 'vmax': cap})
        if args.with_heatmap:
            if not (run_dir / LMER_TABLE).exists():
                raise SystemExit(f'--with-heatmap needs {LMER_TABLE} in the run')
            figure_heatmap_and_split(
                run_dir, el, slopes[slopes['band'].isin(want)], args, bands_def,
                out_dir / f'fig_heatmap_glass_sigsplit_{args.display_mode}.png',
                cap)
    elif args.sig_category:
        cats = sig_categories(slopes[slopes['band'].isin(want)], args.low_bands)
        figure_sig_category(el, cats, args, bands_def,
                            out_dir / f'fig_glass_sigcategory_{args.display_mode}.png',
                            caveat)
        io.write_table(cats, out_dir / 'category_source.csv', script=SCRIPT,
                       parents=parents,
                       params={'low_bands': args.low_bands, 'bands': want},
                       extra={'reading': 'one row per unit significant in '
                                         'some band; class picks the colour '
                                         f'{SIG_CATEGORY_COLOURS}'})
    elif not args.no_heatmap:
        figure_heatmap(slopes, want, cap, args, bands_def,
                       out_dir / 'fig_domain_summary_matched.png', caveat)
    for band in ([] if args.sig_category else want):
        figure_for_band(el, slopes, band, cap, args, bands_def,
                        out_dir / f'fig_glass_{band}.png', caveat)
    if args.sig_category:
        pass
    elif len(want) > 1 and len(args.display_mode) == 1:
        figure_bands_row(el, slopes, want, cap, args, bands_def,
                         out_dir / f'fig_glass_{args.display_mode}_bands_row.png',
                         caveat)
    elif len(want) > 1:
        figure_all_bands(el, slopes, want, cap, args, bands_def,
                         out_dir / 'fig_glass_all_bands.png', caveat)

    # WHAT COLOURED WHAT, as data rather than as a figure caption: one row per
    # (band, unit) with the beta, its p, and how many electrodes took that
    # colour. This is the link back from a dot to the number behind it.
    src = (slopes[slopes['band'].isin(want)]
           [['band', 'unit', 'beta', 'value', 'significant']
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

"""Overview figures for a NATIVE-RESOLUTION mixed-model grid (no significance).

Two figures, the same two altitudes as `plot_mixed_model_grid`, rebuilt for a
frequency axis with 463 bins instead of 44:

`fig_fullres_map` -- the whole grid at once, region x frequency. Pain fixed
effect, heterogeneity effect size, and sign consistency, plus coverage bars.

`fig_fullres_spectra` -- one small panel per region, beta against frequency with
its 95% band and a smoothed overlay. At 0.5 Hz spacing this is the figure to
read: a heatmap makes you compare colours across a log axis, which is exactly
where a sign flip or a narrow peak hides.

THREE THINGS DONE DIFFERENTLY FROM THE 50-BIN VERSION, all forced by the axis:

  - NO SIGNIFICANCE, ANYWHERE. No BH panel, no outlines, no red dots. The run
    computes no corrected p-value (see its METHODS.md) and this module must not
    invent the appearance of one. What is on the page is effect sizes.
  - `pcolormesh` ON A REAL LOG-Hz AXIS, not `imshow` over bin indices. 463
    categorical ticks are unreadable, and worse, an index axis would draw the
    0.5 Hz bin at 1 Hz and the one at 250 Hz the same width, which is a lie
    about where the information is. Cell WIDTHS are the bins' actual widths.
  - THE NOTCHED BINS ARE DRAWN AS GAPS. The matrix is reindexed onto the full
    native axis from the unit's manifest, so the 36 line-noise bins come back as
    NaN and render in the `bad` colour. An axis that simply omitted them would
    silently close a 4 Hz hole around every harmonic.

    python -m ieeg_ehr.analysis.plot_fullres_grid --run-dir <run>
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import io
from ieeg_ehr.analysis import view_tables
from ieeg_ehr.features import common
from ieeg_ehr.views import fullres_reader

logger = logging.getLogger(__name__)

DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. No '
              'multiple-comparison correction and NO SIGNIFICANCE TESTING was '
              'performed: nothing on this figure marks a cell as real. Not '
              'confirmed out of sample.')

#: Ticks a reader can place on a log frequency axis without counting.
LOG_TICKS = (1, 2, 5, 10, 20, 50, 100, 200)


# ============================================================================
# SHARED PIECES
# ============================================================================

def run_params(run_dir):
    try:
        return json.loads((Path(run_dir) / 'provenance.json').read_text()).get(
            'params', {})
    except (OSError, ValueError):
        logger.warning('no readable provenance.json in %s', run_dir)
        return {}


def full_axis(epoch_minutes=None):
    """The COMPLETE native frequency axis, notched bins included.

    From the unit's manifest, which is the only source of truth for what f000
    means in Hz -- and from the manifest rather than from the run's tables
    precisely because the notched bins are missing from those.
    """
    return fullres_reader.freq_table(epoch_minutes).set_index('freq_bin_index')


def pivot(cells, value, regions, bins):
    """region x freq_bin table in a FIXED row/column order.

    Reindexed rather than pivoted alone: a region or bin missing from the data
    must appear as an empty ROW or COLUMN, not vanish and shift every label
    against its colours. This is also what turns the line-noise notch into a
    visible gap.
    """
    if value not in cells.columns:
        return pd.DataFrame(np.nan, index=regions, columns=bins)
    p = cells.pivot_table(index='region', columns='freq_bin_index', values=value,
                          aggfunc='first')
    return p.reindex(index=regions, columns=bins)


def order_regions(cells, regions):
    """The registry display order, restricted to regions actually present."""
    present = set(cells['region'])
    return [r for r in regions if r in present]


def _mesh(ax, mat, axis, regions, cmap, vmin, vmax):
    """One heat panel: log-Hz x, one row per region, NaN drawn as `bad`."""
    edges_x = np.append(axis['bin_low_hz'].to_numpy(),
                        axis['bin_high_hz'].to_numpy()[-1])
    edges_y = np.arange(len(regions) + 1)
    data = np.ma.masked_invalid(mat.to_numpy(dtype=float))
    im = ax.pcolormesh(edges_x, edges_y, data, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading='flat')
    ax.set_xscale('log')
    ax.set_xlim(edges_x[0], edges_x[-1])
    ax.set_xticks(LOG_TICKS)
    ax.set_xticklabels([str(t) for t in LOG_TICKS], fontsize=7)
    ax.minorticks_off()
    ax.set_yticks(np.arange(len(regions)) + 0.5)
    ax.set_ylim(len(regions), 0)          # first region on top
    ax.set_xlabel('frequency (Hz)', fontsize=8)
    common.add_band_boundary_lines(ax)
    return im


def _cmap(plt, name, bad='0.85'):
    cm = plt.get_cmap(name).copy()
    cm.set_bad(bad)
    return cm


def _clip(values, pct):
    """(cap, how many finite values exceed it). NaN-safe, positive-only input."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0, 0
    cap = float(np.percentile(finite, pct))
    if not np.isfinite(cap) or cap <= 0:
        cap = float(np.nanmax(finite))
    return cap, int((finite > cap).sum())


def _coverage_bars(ax, cells, regions, col, label, colour):
    """One number per region as a BAR, never a heat strip.

    Subject and electrode counts are constant across frequency, and a 463-wide
    strip would imply they vary with it -- a claim the data does not make.
    """
    per_region = (cells.groupby('region')[col].max().reindex(regions)
                  if col in cells.columns else pd.Series(np.nan, index=regions))
    vals = per_region.to_numpy(dtype=float)
    ax.barh(np.arange(len(regions)) + 0.5, vals, height=0.7, color=colour)
    ax.set_yticks(np.arange(len(regions)) + 0.5)
    ax.set_yticklabels([])
    ax.set_ylim(len(regions), 0)
    ax.set_title(f'n {label}\nper region', fontsize=10)
    ax.set_xlabel(f'n {label}', fontsize=8)
    ax.tick_params(labelsize=7)
    span = np.nanmax(vals) if np.isfinite(np.nanmax(vals)) else 1.0
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(v + 0.02 * span, i + 0.5, f'{int(v)}', va='center',
                    fontsize=6.2, color='0.3')


# ============================================================================
# FIGURE 1: THE MAP
# ============================================================================

def fig_fullres_map(cells, cons, regions, axis, out_path, beta_pct=99.0,
                    het_pct=95.0, notched=()):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    bins = list(axis.index)
    beta = pivot(cells, 'beta_nrs_within', regions, bins)
    beta_cap, n_sat = _clip(np.abs(beta.to_numpy(dtype=float)), beta_pct)

    # Between-subject slope SD over the residual SD. The RAW SD is not comparable
    # across frequency -- it falls with frequency mainly because residual
    # variability falls faster -- and on this axis that objection is stronger than
    # on log bins, not weaker, because every bin now averages exactly one FFT
    # frequency and the only thing left varying is the physiology.
    het = np.sqrt(pivot(cells, 'var_subj_slope', regions, bins).astype(float))
    het = het / np.sqrt(pivot(cells, 'var_resid', regions, bins).astype(float))
    het_cap, n_het_sat = _clip(het.to_numpy(dtype=float), het_pct)

    if cons is not None and len(cons):
        cmat = (cons.pivot_table(index='region', columns='freq_bin_index',
                                 values='frac_sign_consistent')
                .reindex(index=regions, columns=bins))
        cons_label = 'SIGN CONSISTENCY\nunpooled per-subject fits'
    else:
        cmat = pd.DataFrame(np.nan, index=regions, columns=bins)
        cons_label = 'SIGN CONSISTENCY\nNOT AVAILABLE (no per-subject slopes)'
    # Symmetric about 0.5 -- half the subjects agreeing IS chance -- but only as
    # wide as the data goes, floored so a uniformly consistent grid does not get
    # its noise amplified into structure.
    span = float(np.nanmax(np.abs(cmat.to_numpy(dtype=float) - 0.5)))
    span = max(span, 0.05) if np.isfinite(span) else 0.5

    panels = [
        (beta, _cmap(plt, 'RdBu_r'), -beta_cap, beta_cap,
         'PAIN FIXED EFFECT\nd log10 power per pain point'
         f'  (scale clipped at |{beta_cap:.4f}|, {n_sat} cells saturate)'),
        (het, _cmap(plt, 'viridis'), 0.0, het_cap,
         'HETEROGENEITY, normalized\nbetween-subject slope SD / residual SD'
         f'  (clipped at {het_cap:.3f}, {n_het_sat} saturate)'),
        (cmat, _cmap(plt, 'PuOr'), 0.5 - span, 0.5 + span, cons_label),
    ]

    n_heat = len(panels)
    fig, axes = plt.subplots(
        1, n_heat + 2, figsize=(6.4 * n_heat + 3.6, 0.42 * len(regions) + 3.8),
        gridspec_kw={'width_ratios': [1] * n_heat + [0.28, 0.28]})

    for i, (mat, cm, vlo, vhi, title) in enumerate(panels):
        ax = axes[i]
        im = _mesh(ax, mat, axis, regions, cm, vlo, vhi)
        ax.set_title(title, fontsize=9.5)
        ax.set_yticklabels(regions if i == 0 else [], fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).ax.tick_params(labelsize=7)

    _coverage_bars(axes[n_heat], cells, regions, 'n_subjects', 'subjects', '0.55')
    _coverage_bars(axes[n_heat + 1], cells, regions, 'n_channels', 'electrodes',
                   '#4a7ba7')

    fig.suptitle('Mixed-model pain encoding, region x NATIVE 0.5 Hz frequency '
                 f'({len(cells)} cells)', fontsize=13)
    fig.tight_layout(rect=(0, 0.085, 1, 0.945))
    fig.text(0.01, 0.005,
             'NOTHING HERE IS A SIGNIFICANCE CLAIM. No p-value was corrected and '
             'no cell is outlined; the panels are effect sizes. Colour scales are '
             'clipped at a PERCENTILE rather than the maximum, because a handful '
             'of thin, badly-estimated cells would otherwise spend the whole ramp '
             'and flatten every real one -- cells past the cap saturate, and both '
             'the cap and the count are in the panel titles. The x axis is real '
             'log frequency and each cell is drawn at its own 0.5 Hz width. GREY '
             f'gaps at {", ".join(f"{h:.0f}" for h in (60, 120, 180, 240))} Hz are '
             f'the line-noise notch ({len(notched)} bins removed before fitting); '
             'grey elsewhere is a cell with no data. Orange in the '
             'sign-consistency panel means FEWER than half of subjects share the '
             "group's direction -- the mean is carried by a minority with large "
             'slopes and does not describe a typical patient; its denominator is '
             'subjects whose slope is defined at all (>=2 distinct pain scores). '
             'Subject and electrode counts are constant across frequency, hence '
             'bars.\n' + DISCLAIMER,
             fontsize=7, va='bottom', ha='left', color='0.35', wrap=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# FIGURE 2: PER-REGION SPECTRA
# ============================================================================

def _smooth(y, window):
    """Centred rolling mean over BIN POSITIONS, NaN-tolerant.

    Over positions, not Hz, so the window is 0.5*window Hz wide everywhere
    except across the notch, where the gap makes it reach slightly further in
    frequency than it does in bins. Stated on the figure rather than special-
    cased: interpolating the notch would draw a line through frequencies that
    were deliberately not measured.
    """
    if window <= 1:
        return y
    s = pd.Series(y)
    return s.rolling(window, center=True, min_periods=max(2, window // 3)).mean().to_numpy()


def fig_fullres_spectra(cells, regions, out_path, ncol=5, smooth=9, ylim_pct=99.5):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    nrow = int(np.ceil(len(regions) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 2.8 * nrow),
                             squeeze=False, sharex=True)

    # Shared y so region amplitudes are comparable, but set from a PERCENTILE of
    # the CI envelope: on 463 bins the single widest interval is typically a
    # badly-estimated thin cell, and scaling to it would flatten all 21 panels.
    env = np.concatenate([(cells['beta_nrs_within'] + 1.96 * cells['se']).to_numpy(),
                          (cells['beta_nrs_within'] - 1.96 * cells['se']).to_numpy()])
    env = np.abs(env[np.isfinite(env)])
    ymax = float(np.percentile(env, ylim_pct)) if env.size else 1.0

    for i, region in enumerate(regions):
        ax = axes[i // ncol][i % ncol]
        d = cells[cells['region'] == region].sort_values('freq_hz')
        hz = d['freq_hz'].to_numpy(dtype=float)
        b = d['beta_nrs_within'].to_numpy(dtype=float)
        se = d['se'].to_numpy(dtype=float)

        ax.fill_between(hz, b - 1.96 * se, b + 1.96 * se, color='0.78', alpha=0.5,
                        lw=0)
        ax.plot(hz, b, color='0.45', lw=0.6, alpha=0.9)
        if smooth > 1:
            ax.plot(hz, _smooth(b, smooth), color='black', lw=1.5)
        ax.axhline(0, color='0.6', lw=0.8, ls='--')
        ax.set_xscale('log')
        ax.set_xlim(hz.min(), hz.max())
        ax.set_xticks(LOG_TICKS)
        ax.set_xticklabels([str(t) for t in LOG_TICKS], fontsize=6)
        ax.minorticks_off()
        ax.set_ylim(-ymax, ymax)
        common.add_band_boundary_lines(ax)
        n = int(d['n_subjects'].max()) if d['n_subjects'].notna().any() else 0
        ax.set_title(f'{region}  (n={n})', fontsize=9)
        # `sharex` hides inner tick labels, which on a 21-panel grid leaves 20
        # panels with no frequency axis at all -- the reader cannot tell 10 Hz
        # from 100 Hz on the panel they are looking at. Shared LIMITS are worth
        # keeping; shared labels are not.
        ax.tick_params(labelsize=7, labelbottom=True)
        if i % ncol == 0:
            ax.set_ylabel('beta (d log10 power / pain point)', fontsize=7.5)
        # Bottom-most panel IN ITS OWN COLUMN, not just the last row: the final
        # row is partly empty, so a row test leaves four columns unlabelled.
        if i + ncol >= len(regions):
            ax.set_xlabel('frequency (Hz)', fontsize=8)

    for j in range(len(regions), nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)

    fig.suptitle('Pain fixed effect vs frequency, per region -- native 0.5 Hz axis\n'
                 f'grey = per-bin estimate, band = 95% CI, black = {smooth}-bin '
                 f'({0.5 * smooth:.1f} Hz) rolling mean',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.text(0.01, 0.005,
             'NO SIGNIFICANCE IS MARKED and no p-value was corrected; the band is '
             'a 95% CI on one cell in isolation, not a test. All panels share a '
             'y-scale so region amplitudes are comparable, set to the '
             f'{ylim_pct:g}th percentile of the CI envelope rather than its '
             'maximum -- a few thin cells with huge intervals would otherwise '
             'flatten every panel, and those cells run off the top rather than '
             'disappearing. The smoothed line is a rolling mean over BIN '
             'positions, so it reaches slightly further in Hz across the '
             'line-noise notch, which is left as a gap rather than interpolated. '
             'The x axis is log-spaced so this figure can be laid beside the '
             '50-log-bin map it replaces.\n' + DISCLAIMER,
             fontsize=7, va='bottom', ha='left', color='0.35', wrap=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--roi-scheme', default=None,
                    help="Default: the run's own recorded scheme.")
    ap.add_argument('--beta-pct', type=float, default=99.0,
                    help='Percentile of |beta| the diverging colour scale is '
                         'clipped at (default 99). 100 = scale to the maximum.')
    ap.add_argument('--het-pct', type=float, default=95.0,
                    help='Percentile the heterogeneity colour bar is capped at.')
    ap.add_argument('--smooth', type=int, default=9,
                    help='Rolling-mean window, in bins, for the spectra overlay '
                         '(9 bins = 4.5 Hz). 1 disables it.')
    ap.add_argument('--suffix', default='',
                    help='Appended to figure filenames, so an alternative scaling '
                         'does not overwrite the first render.')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    run_dir = Path(args.run_dir)
    cells = io.read_table(run_dir / 'grid_cells.parquet', on_stale='warn')
    params = run_params(run_dir)

    cons_path = run_dir / 'sign_consistency.parquet'
    cons = (io.read_table(cons_path, on_stale='warn') if cons_path.exists()
            else None)
    if cons is None:
        logger.warning('no sign_consistency.parquet in %s; that panel will say so',
                       run_dir)

    roi_scheme = args.roi_scheme or params.get('roi_scheme', 'roi_v2')
    regions = order_regions(cells,
                            view_tables.roi_regions_for({'roi_scheme': roi_scheme}))
    axis = full_axis(params.get('epoch_minutes'))
    notched = params.get('notched_bins_removed', [])
    logger.info('%d cells | %d regions | %d fitted bins of %d on the native axis '
                '(%d notched)', len(cells), len(regions),
                cells['freq_bin_index'].nunique(), len(axis), len(notched))

    map_path = run_dir / f'fig_fullres_map{args.suffix}.png'
    spec_path = run_dir / f'fig_fullres_spectra{args.suffix}.png'
    fig_fullres_map(cells, cons, regions, axis, map_path,
                    beta_pct=args.beta_pct, het_pct=args.het_pct, notched=notched)
    logger.info('wrote %s', map_path)
    fig_fullres_spectra(cells, regions, spec_path, smooth=args.smooth)
    logger.info('wrote %s', spec_path)

    io.log_analysis('native-resolution mixed-model grid figures: region x '
                    'frequency map and per-region spectra, NO significance '
                    '(EXPLORATORY)', run_dir)


if __name__ == '__main__':
    main()

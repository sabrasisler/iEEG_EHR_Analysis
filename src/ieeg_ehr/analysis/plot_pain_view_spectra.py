#!/usr/bin/env python3
"""
Per-region spectra from a P1.3 view: one subplot per region, one line per pain
level, mean +- SEM across subjects.

THE QUESTION THIS ANSWERS, AND THE HEATMAP DOESN'T. The region x frequency-bin
heatmap says WHERE an effect is; it cannot say what SHAPE it has, because a
colour scale is not a quantitative axis. Shape is the physiology: a broadband
shift, a tilt in the 1/f slope, and a narrowband alpha/beta suppression all look
like "some blue, some red" on a heatmap and look like three different things on
a line plot. Same numbers, same view tables, read the other way.

Three deliberate differences from the heatmap figures:

1. REAL Hz ON A LOG AXIS, not the heatmap's categorical bin index. The heatmap
   has no choice -- an imshow column is a category. A line plot does, and using
   the true frequency is what makes a 1/f tilt look like a straight line and
   lets band boundaries be drawn where they actually are.

2. LINE-NOISE BINS ARE MASKED, so each line BREAKS across them and their span is
   shaded. This is a DISPLAY mask, independent of the view's
   drop_line_noise_bins -- the bins are still in the table. It matters because
   ~59 Hz and ~179 Hz are the largest-magnitude cells in the current group
   heatmap: left in, they would spike and set the y-limits of every panel, and
   the physiology would be a flat line at the bottom.

3. SPREAD IS SEM, not variance. Variance is in squared log-units and cannot
   share an axis with the mean it describes. SEM answers the question the figure
   is actually for -- do the pain levels separate -- and --spread sd is there for
   when the question is instead how variable subjects are.

The 'none' bin is not drawn: under any baseline normalization it is its own
reference and is 0 by construction, so it is the y=0 line rather than a series.
It is still reported in the log as a correctness check.

Run on Slurm, never the login node:
    python -m ieeg_ehr.analysis.plot_pain_view_spectra --view-dir <dir> \\
        --run-name discovery_std10_blsub
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import view_tables
from ieeg_ehr.analysis.view_tables import PANELS
from ieeg_ehr.features import common
from ieeg_ehr.views import cache_reader

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_TYPE = 'region_spectrum'

# Recessive, so the data is the only thing with weight on the page.
GRID_KW = dict(color='0.9', linewidth=0.6)
NOISE_SPAN_KW = dict(color='0.85', alpha=0.55, linewidth=0, zorder=0)

# Explicit decade+band ticks. A log axis left to itself labels only 10^0/10^1/10^2,
# which throws away the reason for plotting real Hz in the first place: you cannot
# see that a peak sits in beta rather than alpha. These are the canonical band
# edges (config.FREQ_BAND_BOUNDARIES_HZ) plus the ends of the stored range.
FREQ_TICKS_HZ = [1, 2, 4, 8, 12, 30, 65, 125, 250]


def spectra_table(stats, bin_labels, panels, line_noise_bins):
    """The plotted numbers, long-format and self-describing.

    Written beside the figure so the figure's claims are checkable without
    re-deriving them, and so per-bin n (which the panel titles compress to a
    single minimum) stays visible.
    """
    table = stats[stats['pain_bin'].isin(panels)].merge(
        bin_labels.reset_index()[['freq_bin_index', 'bin_low_hz', 'bin_high_hz']],
        on='freq_bin_index', how='left')
    table['is_line_noise_bin'] = table['freq_bin_index'].isin(line_noise_bins)
    return table.sort_values(['region', 'pain_bin', 'freq_bin_index'],
                             ignore_index=True)


def _isolated_points(y):
    """Finite entries of `y` whose neighbours are both absent (or off the end).

    These are the ones a line plot cannot render: matplotlib connects finite
    neighbours and skips NaN, so a lone finite value has nothing to connect to
    and disappears entirely.
    """
    finite = np.isfinite(y)
    left = np.r_[False, finite[:-1]]
    right = np.r_[finite[1:], False]
    return finite & ~left & ~right


def _series(stats, region, pain_bin, freq_bins, line_noise_bins, spread):
    """(x_hz, mean, lo, hi) for one line, NaN where there is nothing to draw.

    Reindexed onto the full bin axis so a region missing a bin produces a GAP
    rather than a line segment interpolated across it -- matplotlib skips NaN,
    which is exactly the behaviour wanted for both missing coverage and the
    masked line-noise bins.
    """
    rows = (stats[(stats['region'] == region) & (stats['pain_bin'] == pain_bin)]
            .set_index('freq_bin_index').reindex(freq_bins))
    mean = rows['mean'].to_numpy(dtype=float).copy()
    half = (rows[spread].to_numpy(dtype=float).copy()
            if spread else np.full(len(freq_bins), np.nan))

    masked = np.isin(freq_bins, line_noise_bins)
    mean[masked] = np.nan
    half[masked] = np.nan
    return mean, mean - half, mean + half


def plot_grid(stats, regions, region_n, panels, bin_labels, line_noise_bins,
              title, out_path, value_label, spread='sem', ncols=4, share_y=True):
    """One panel per region; one line per pain bin; optional +-spread ribbon."""
    freq_bins = bin_labels.index.tolist()
    x_hz = bin_labels['bin_low_hz'].to_numpy(dtype=float)

    nrows = int(np.ceil(len(regions) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.7 * nrows),
                             sharex=True, sharey=share_y, squeeze=False)
    flat = axes.ravel()

    for ax, region in zip(flat, regions):
        # Shade first, at zorder 0, so the spans sit under the data.
        for b in line_noise_bins:
            if b in bin_labels.index:
                ax.axvspan(bin_labels.loc[b, 'bin_low_hz'],
                           bin_labels.loc[b, 'bin_high_hz'], **NOISE_SPAN_KW)
        ax.axhline(0, color='0.35', linewidth=0.8, zorder=1)
        common.add_band_boundary_lines(ax)

        for pain_bin in panels:
            mean, lo, hi = _series(stats, region, pain_bin, freq_bins,
                                   line_noise_bins, spread)
            color = config.PAIN_BIN_COLORS[pain_bin]
            if spread:
                ax.fill_between(x_hz, lo, hi, color=color, alpha=0.22,
                                linewidth=0, zorder=2)
            # Solid, all levels: colour carries pain level (the cool->warm ramp in
            # config.PAIN_BIN_COLORS) and the figure legend is its only fallback,
            # so the legend is LOAD-BEARING -- do not drop it to save space.
            ax.plot(x_hz, mean, color=color, linewidth=1.8, label=pain_bin, zorder=3)
            # Above ~50 Hz a surviving bin is often ISOLATED between two masked
            # ones, and matplotlib draws NOTHING for a single non-NaN point with
            # NaN on both sides -- the high-gamma end of every panel would
            # silently go blank. Marked only where that actually happens, so the
            # dense low-frequency part stays a clean line rather than a dotted one.
            isolated = _isolated_points(mean)
            if isolated.any():
                ax.plot(x_hz[isolated], mean[isolated], linestyle='none',
                        marker='o', markersize=2.6, color=color, zorder=3)

        ax.set_xscale('log')
        ax.set_xticks(FREQ_TICKS_HZ)
        ax.set_xticklabels([str(t) for t in FREQ_TICKS_HZ])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f'{region}  (n={int(region_n.get(region, 0))} subj)', fontsize=9)
        ax.grid(True, which='major', **GRID_KW)
        ax.tick_params(labelsize=7)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    for ax in flat[len(regions):]:
        ax.set_visible(False)

    # Axis labels once per edge rather than per panel: 15 copies of the same
    # string is noise, and the shared axes make repeating it meaningless.
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel('Frequency (Hz)', fontsize=8)
    # With a partly-filled last row the panels above the gaps are also bottom
    # panels, so they need the label too.
    for col in range(ncols):
        for row in range(nrows - 1):
            if not axes[row + 1, col].get_visible():
                axes[row, col].set_xlabel('Frequency (Hz)', fontsize=8)
    # ONE y-label for the whole figure, not one per row: the units string is far
    # longer than a 2.7-inch panel is tall, so per-row labels collide with each
    # other into an unreadable stack.
    fig.supylabel(value_label, fontsize=9)

    handles, labels = flat[0].get_legend_handles_labels()
    spread_note = {'sem': ' (shaded: +-SEM across subjects)',
                   'sd': ' (shaded: +-SD across subjects)'}.get(spread, '')
    fig.legend(handles, labels, loc='lower center', ncol=len(panels) + 1,
               frameon=False, fontsize=9,
               title=f'Pain level{spread_note}', title_fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('Wrote %s', out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--view-dir', required=True,
                    help='Directory holding view_subject_*.parquet / view_epochs_*.parquet')
    ap.add_argument('--run-name', default=None)
    ap.add_argument('--pain-bin-scheme', choices=list(PANELS), default='subject_relative')
    ap.add_argument('--min-subjects', type=int, default=8,
                    help='Skip a region unless at least this many subjects back EVERY '
                         'plotted pain level (default: 8). A DISPLAY floor for this '
                         'figure, not a claim about what is analysable.')
    ap.add_argument('--spread', choices=['sem', 'sd', 'none'], default='sem',
                    help='Ribbon around each line. sem (default) = precision of the '
                         'group mean, which is what "do the levels separate" needs. '
                         'sd = how variable individual subjects are.')
    ap.add_argument('--ncols', type=int, default=4)
    ap.add_argument('--free-y', action='store_true',
                    help='Give each panel its own y-scale. OFF by default: a shared '
                         'scale is what makes regions comparable, which is the point '
                         'of small multiples.')
    ap.add_argument('--keep-line-noise-bins', action='store_true',
                    help='Draw the line-noise bins instead of breaking the line across '
                         'them. They are the largest-magnitude bins in this data, so '
                         'expect them to set the y-limits of every panel.')
    ap.add_argument('--per-subject', action='store_true',
                    help='Also write one figure per subject under by_subject/. No '
                         'ribbon there -- a single subject has no across-subject spread.')
    ap.add_argument('--value-label', default=None,
                    help='Default: read from the view sidecar so units cannot be mislabeled')
    view_tables.add_output_arguments(ap)
    args = ap.parse_args()

    view_dir = Path(args.view_dir)
    io.warn_if_dirty()

    subject_tables, subject_paths = view_tables.load_view_tables(view_dir, 'subject')
    epoch_tables, epoch_paths = view_tables.load_view_tables(view_dir, 'epochs')
    subjects = sorted(subject_tables['subject_id'].unique())
    logger.info('%d subject(s): %s', len(subjects), subjects)

    view_params, view = view_tables.view_params_from(subject_paths)
    value_label = args.value_label or (view.value_label if view else 'value')
    logger.info('view: %s', {k: view_params.get(k) for k in
                             ('normalization', 'domain', 'mask_label', 'pain_bins',
                              'roi_scheme')})

    # The figure's pain-bin scheme must match the view's, or the lines would be
    # labelled with bins the view never produced. Caught here rather than showing
    # up as two empty panels.
    if view is not None and view.pain_bins != args.pain_bin_scheme:
        raise SystemExit(
            f'--pain-bin-scheme {args.pain_bin_scheme!r} but the view in {view_dir} '
            f'was built with pain_bins={view.pain_bins!r}. Rebuild the view or plot '
            'the scheme it has.')

    panels = PANELS[args.pain_bin_scheme]
    epoch_minutes = view_params.get('epoch_minutes')
    bin_labels = cache_reader.bin_edges(epoch_minutes).set_index('freq_bin_index')
    line_noise_bins = (np.array([], dtype=int) if args.keep_line_noise_bins
                       else cache_reader.line_noise_bins(epoch_minutes))
    logger.info('masking %d line-noise bin(s) from display: %s Hz', len(line_noise_bins),
                [f'{bin_labels.loc[b, "bin_low_hz"]:.0f}' for b in line_noise_bins
                 if b in bin_labels.index])

    view_tables.log_baseline_check(subject_tables)

    stats = view_tables.subject_stats(subject_tables)
    regions, region_n = view_tables.regions_with_min_subjects(
        stats, panels, args.min_subjects)
    if not regions:
        raise SystemExit(f'no region has >= {args.min_subjects} subjects in every '
                         f'plotted bin {panels}; nothing to draw')
    logger.info('%d region(s) plotted: %s', len(regions),
                {r: int(region_n[r]) for r in regions})

    spread = None if args.spread == 'none' else args.spread
    run_dir = view_tables.resolve_run_dir(args, OUTPUT_TYPE, view)
    logger.info('run dir: %s', run_dir)

    io.write_run_provenance(
        run_dir, script='ieeg_ehr/analysis/plot_pain_view_spectra.py',
        params={**vars(args), 'view_params': view_params},
        parents=[io.parent_ref(p, digest=False) for p in subject_paths + epoch_paths],
        subjects=subjects,
        extra={'panels': panels, 'n_subjects': len(subjects),
               'regions_plotted': regions,
               'n_subjects_per_region': {r: int(n) for r, n in region_n.items()},
               'min_subjects': args.min_subjects,
               'line_noise_bins_masked': [int(b) for b in line_noise_bins],
               'roi_regions': config.ROI_REGIONS},
    )

    table = spectra_table(stats, bin_labels, panels, line_noise_bins)
    io.write_table(table, run_dir / 'spectra_table.parquet', kind='table',
                   script='ieeg_ehr/analysis/plot_pain_view_spectra.py',
                   params={**view_params, 'min_subjects': args.min_subjects,
                           'spread': args.spread},
                   parents=[io.parent_ref(p, digest=False) for p in subject_paths],
                   subjects=subjects)

    plot_grid(stats, regions, region_n, panels, bin_labels, line_noise_bins,
              f'Group region spectra (n={len(subjects)} subjects) — '
              f'{view_params.get("normalization")}, mask {view_params.get("mask_label")}',
              run_dir / 'group_region_spectra.png', value_label,
              spread=spread, ncols=args.ncols, share_y=not args.free_y)

    if args.per_subject:
        for subject_id, rows in subject_tables.groupby('subject_id'):
            # One subject: the "group mean" is that subject's own value and there
            # is no across-subject spread, so n=1 everywhere and no ribbon.
            one = view_tables.subject_stats(rows)
            present = [r for r in config.ROI_REGIONS
                       if r in set(one.loc[one['pain_bin'].isin(panels), 'region'])]
            if not present:
                logger.warning('%s: no region rows, skipping', subject_id)
                continue
            plot_grid(one, present, pd.Series(1, index=present), panels, bin_labels,
                      line_noise_bins, subject_id,
                      run_dir / 'by_subject' / f'{subject_id}_region_spectra.png',
                      value_label, spread=None, ncols=args.ncols,
                      share_y=not args.free_y)

    io.log_analysis(f'P1.3 region spectra ({view_params.get("normalization")}, '
                    f'mask {view_params.get("mask_label")}), {len(regions)} regions '
                    f'>= {args.min_subjects} subj, n={len(subjects)}', run_dir)
    logger.info('figures + provenance -> %s', run_dir)


if __name__ == '__main__':
    main()

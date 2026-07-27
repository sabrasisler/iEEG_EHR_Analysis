"""
Frequency-band z-score violin plots. Supports both pain-bin schemes
(--pain-bin-scheme, same as plot_pain_heatmaps.py / plot_epoch_count_violin.py,
recomputed at plot time from the cache's raw `pain_score` column):
- absolute (default): none/low/medium/high, fixed cutpoints (config.PAIN_BIN_EDGES).
- subject_relative: none/low/high, split at each subject's own mean non-zero
  pain score (see common.assign_relative_pain_bins) -- no 'medium'.
'none' is always excluded from the violins themselves (see
violin_pain_bins_for_scheme) -- it's the z-score baseline, trivially ~0.

Bands: config.VIOLIN_BANDS_HZ (delta/theta/alpha/beta/gamma/high_gamma) --
a coarser grouping than config.CANONICAL_BANDS_HZ used elsewhere in the
pipeline, merging low_gamma -> gamma and high_gamma1/2/3 -> one high_gamma,
per user instruction (fewer/simpler gamma categories for this plot).

Three interchangeable layouts over the exact same underlying data (--layout):
- by_band (default): one figure per band, one subplot per ROI region.
- by_region: one figure per ROI region, one subplot per band -- same
  z-score values, just transposed grouping.
- grid: ONE combined figure, region rows x band columns, every subplot a
  small violin -- the whole region-by-band picture at a glance.

Each subplot is a violin (seaborn-styled) of subject-level z-scores (vs.
that subject's own none-bin baseline, same convention as
plot_pain_zscore_heatmaps.py), with one colored dot per subject.

Band aggregation: average LINEAR power across the 50 log-spaced freq bins
that fall within a band's range, then log10 -- matches
preprocessing/bipolar_bands.py's aggregate_to_bands convention (avoids
Jensen's-inequality bias vs. averaging log-power directly). Every freq bin
belongs to exactly one band (no overlap, verified against this cohort's real
bin edges), so no bin's value is double-counted across bands.

Run on a dev/interactive Slurm shell (never the login node):
    module load python/3.12
    source /home/groups/ckeller1/venvs/ieeg_ehr_analysis/bin/activate
    python -m pain_analysis.plot_band_violin
"""

import argparse
import logging
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pain_analysis import common, config

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def _safe_filename(name):
    return re.sub(r'[^a-zA-Z0-9]+', '_', name).strip('_').lower()


def violin_pain_bins_for_scheme(scheme):
    """'none' is always excluded from these violins -- it's the z-score
    baseline itself, so it's trivially ~0 for every subject (no spread) and
    adds no information. Remaining bins depend on scheme (no 'medium' under
    subject_relative)."""
    return [b for b in config.pain_bin_order(scheme) if b != 'none']


def _subject_legend_handles(subject_color):
    return [plt.Line2D([0], [0], marker='o', linestyle='', markeredgecolor='black',
                       markeredgewidth=0.3, color=c, label=s) for s, c in subject_color.items()]


def plot_violin_grid(zscores, panel_col, panel_order, subject_color, title, out_path, violin_pain_bins,
                      n_cols=5):
    """One figure, one subplot per value of panel_col (in panel_order,
    skipping any not present). Used for by_band (panel_col='region') and
    by_region (panel_col='band')."""
    panels = [p for p in panel_order if p in zscores[panel_col].unique()]
    if not panels:
        logger.warning('No %s values present for %s, skipping', panel_col, title)
        return

    n_rows = -(-len(panels) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.8 * n_rows), sharey=True, squeeze=False)
    axes_flat = axes.ravel()

    for ax, panel in zip(axes_flat, panels):
        panel_df = zscores[zscores[panel_col] == panel].rename(columns={'zscore': 'value'})
        common.draw_seaborn_violin_with_subject_dots(ax, panel_df, subject_color, value_col='value',
                                                      pain_bins=violin_pain_bins)
        ax.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.5)
        ax.set_title(panel, fontsize=9)

    for ax in axes_flat[len(panels):]:
        ax.axis('off')
    for row in range(n_rows):
        axes[row, 0].set_ylabel('Z-score vs. own none-bin baseline')

    fig.legend(handles=_subject_legend_handles(subject_color), loc='upper center',
               ncol=min(len(subject_color), 10), bbox_to_anchor=(0.5, 1.06), title='Subject', fontsize=7)
    fig.suptitle(title, y=1.1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('Wrote %s', out_path)


def plot_full_grid(subject_zscores, regions, bands, subject_color, violin_pain_bins, title, out_path):
    """ONE combined figure: region rows x band columns, every cell a small
    violin. Rows share a y-scale (comparable within a region across bands);
    scale can differ freely row-to-row since raw z-score magnitude varies a
    lot region-to-region."""
    regions = [r for r in regions if r in subject_zscores['region'].unique()]
    bands = [b for b in bands if b in subject_zscores['band'].unique()]
    if not regions or not bands:
        logger.warning('No regions/bands present for %s, skipping', title)
        return

    fig, axes = plt.subplots(len(regions), len(bands), figsize=(2.1 * len(bands), 1.7 * len(regions)),
                              sharey='row', squeeze=False)

    for row, region in enumerate(regions):
        for col, band in enumerate(bands):
            ax = axes[row, col]
            cell_df = (subject_zscores[(subject_zscores['region'] == region) & (subject_zscores['band'] == band)]
                       .rename(columns={'zscore': 'value'}))
            common.draw_seaborn_violin_with_subject_dots(ax, cell_df, subject_color, value_col='value',
                                                          pain_bins=violin_pain_bins)
            ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
            ax.set_title(band if row == 0 else '', fontsize=9)
            ax.set_ylabel(region if col == 0 else '', fontsize=8)
            if row != len(regions) - 1:
                ax.set_xticklabels([])
                ax.set_xlabel('')

    fig.text(0.005, 0.5, 'Z-score vs. own none-bin baseline', rotation=90, va='center', fontsize=10)

    # Reserve a small, fixed top margin for suptitle + legend regardless of
    # figure height (previous version anchored them ABOVE y=1.0 scaled by
    # 1/n_regions, which for a tall 15-region figure left a large blank gap
    # once bbox_inches='tight' expanded the canvas to include them).
    fig.tight_layout(rect=[0.02, 0, 1, 0.93])
    fig.suptitle(title, y=0.995, fontsize=13)
    fig.legend(handles=_subject_legend_handles(subject_color), loc='upper center',
               ncol=min(len(subject_color), 12), bbox_to_anchor=(0.5, 0.965), title='Subject', fontsize=7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('Wrote %s', out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subjects', nargs='+', default=None,
                         help='Subject IDs without sub- prefix (default: all present in cache dir).')
    parser.add_argument('--run-name', default=None,
                         help='Name for this run\'s output subdirectory under '
                              f'{config.PLOTS_ROOT} (default: auto-generated from timestamp + subject count).')
    parser.add_argument('--min-baseline-epochs', type=int, default=config.ZSCORE_MIN_BASELINE_EPOCHS,
                         help='Minimum none-bin epochs required per subject/region/band to trust its '
                              'baseline std (default: %(default)s).')
    parser.add_argument('--n-cols', type=int, default=5,
                         help='Subplot grid columns for by_band/by_region layouts (default: %(default)s).')
    parser.add_argument('--layout', choices=['by_band', 'by_region', 'grid'], default='by_band',
                         help='"by_band" = one figure per band, subplot per region. '
                              '"by_region" = one figure per region, subplot per band. '
                              '"grid" = ONE combined figure, region rows x band columns. '
                              'All three plot the exact same z-score data. (default: %(default)s)')
    parser.add_argument('--pain-bin-scheme', choices=['absolute', 'subject_relative'], default='absolute',
                         help='"absolute" = fixed cutpoints shared across subjects (none/low/medium/high). '
                              '"subject_relative" = none is score==0, low/high split at that subject\'s own '
                              'mean non-zero pain score (no medium). (default: %(default)s)')
    args = parser.parse_args()

    df, cache_paths = common.load_cache(args.subjects)
    if args.pain_bin_scheme == 'subject_relative':
        df['pain_bin'] = common.assign_relative_pain_bins(df)
    df = common.add_region(df)
    bin_labels = common.bin_label_table(df)
    violin_pain_bins = violin_pain_bins_for_scheme(args.pain_bin_scheme)

    epoch_table = common.subject_region_epoch_table(df)
    band_epoch_table = common.aggregate_epoch_table_to_bands(epoch_table, bin_labels, bands=config.VIOLIN_BANDS_HZ)
    subject_zscores = common.compute_subject_zscores(
        band_epoch_table, value_col='band_log_power', group_col='band',
        min_baseline_epochs=args.min_baseline_epochs,
    )

    subjects = sorted(subject_zscores['subject'].unique())
    subject_color = common.subject_color_map(subjects)

    run_dir = common.make_run_dir(args.run_name, len(subjects),
                                   category=f'band_violin_{args.layout}/{args.pain_bin_scheme}')
    common.write_run_provenance(
        run_dir, 'pain_analysis/plot_band_violin.py', args, cache_paths,
        subjects=subjects,
        extra={'violin_bands_hz': config.VIOLIN_BANDS_HZ, 'min_baseline_epochs': args.min_baseline_epochs,
               'layout': args.layout, 'pain_bin_scheme': args.pain_bin_scheme},
    )

    band_order = list(config.VIOLIN_BANDS_HZ.keys())
    if args.layout == 'by_band':
        for band_name in band_order:
            plot_violin_grid(
                subject_zscores[subject_zscores['band'] == band_name], 'region', config.ROI_REGIONS,
                subject_color, f'{band_name} band', run_dir / f'{band_name}_violin.png',
                violin_pain_bins, n_cols=args.n_cols,
            )
    elif args.layout == 'by_region':
        for region in config.ROI_REGIONS:
            region_df = subject_zscores[subject_zscores['region'] == region]
            if region_df.empty:
                continue
            plot_violin_grid(
                region_df, 'band', band_order, subject_color, region,
                run_dir / f'{_safe_filename(region)}_violin.png', violin_pain_bins, n_cols=args.n_cols,
            )
    else:
        plot_full_grid(
            subject_zscores, config.ROI_REGIONS, band_order, subject_color, violin_pain_bins,
            f'Region x band (n={len(subjects)} subjects)', run_dir / 'region_x_band_violin_grid.png',
        )

    logger.info('Run outputs + provenance.json written to %s', run_dir)


if __name__ == '__main__':
    main()

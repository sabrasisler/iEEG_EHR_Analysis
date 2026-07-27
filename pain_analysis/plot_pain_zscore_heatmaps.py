"""
Z-score variant of plot_pain_heatmaps.py: instead of raw delta log-power
(low/medium/high minus none, in the same log10(V^2/Hz) units as the cache),
each epoch's region/freq_bin power is z-scored against that SAME subject's
own 'none'-bin distribution (mean/std across that subject's none-bin
epochs), then averaged across epochs and subjects. This puts every
region/freq_bin on a comparable scale regardless of its absolute power or
subject-to-subject variance, at the cost of being undefined wherever a
subject has too few 'none'-bin epochs to estimate a baseline std from (see
config.ZSCORE_MIN_BASELINE_EPOCHS).

This is a variant alongside plot_pain_heatmaps.py, not a replacement --
both are useful (raw delta preserves physical units, z-score standardizes
across regions/subjects).

Run on a dev/interactive Slurm shell (never the login node):
    module load python/3.12
    source /home/groups/ckeller1/venvs/ieeg_ehr_analysis/bin/activate
    python -m pain_analysis.plot_pain_zscore_heatmaps
"""

import argparse
import logging

import numpy as np

from pain_analysis import common, config

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def zscore_cols_for_scheme(scheme):
    """'none' is always excluded -- it's the z-score baseline itself,
    trivially ~0 for every subject. Remaining bins depend on scheme (no
    'medium' under subject_relative)."""
    return [b for b in config.pain_bin_order(scheme) if b != 'none']


def group_table(subject_zscores):
    """Average subject-level z-scores across subjects (equal-weighted),
    within each (pain_bin, region, freq_bin_index) -- pain_bin must stay a
    groupby key here since subject_zscores is still long-format (unlike
    plot_pain_heatmaps.py's group_table, which operates on already-wide
    delta columns)."""
    return (subject_zscores.groupby(['pain_bin', 'region', 'freq_bin_index'])['zscore']
            .agg(lambda s: np.nanmean(s.to_numpy()))
            .reset_index())


def _wide_by_bin(subject_zscores, index_cols, zscore_cols):
    wide = subject_zscores.pivot_table(index=index_cols, columns='pain_bin', values='zscore')
    for bin_name in zscore_cols:
        if bin_name not in wide.columns:
            wide[bin_name] = np.nan
    return wide[zscore_cols].reset_index()


def plot_zscore_heatmaps(table, zscore_cols, bin_labels, counts, count_bin_order, title, out_path, regions=None):
    regions = regions or config.ROI_REGIONS
    freq_bins = bin_labels.index.tolist()
    pivots = [common.pivot_for_plot(table, col, regions, freq_bins) for col in zscore_cols]
    common.plot_region_freq_heatmaps(
        pivots, zscore_cols, bin_labels, counts, title, out_path,
        cbar_label='Mean z-score vs. own none-bin baseline', regions=regions, count_bin_order=count_bin_order,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subjects', nargs='+', default=None,
                         help='Subject IDs without sub- prefix (default: all present in cache dir).')
    parser.add_argument('--run-name', default=None,
                         help='Name for this run\'s output subdirectory under '
                              f'{config.PLOTS_ROOT} (default: auto-generated from timestamp + subject count).')
    parser.add_argument('--min-baseline-epochs', type=int, default=config.ZSCORE_MIN_BASELINE_EPOCHS,
                         help='Minimum none-bin epochs required per subject/region/freq_bin to trust its '
                              'baseline std (default: %(default)s).')
    parser.add_argument('--row-order', choices=['default', 'cluster', 'effect_size'], default='cluster',
                         help='Region row order: "default" = fixed anatomical config.ROI_REGIONS order, '
                              '"cluster" = hierarchical clustering by spectral-response similarity (computed '
                              'once from the group table, then applied to all figures for comparability), '
                              '"effect_size" = descending mean |z-score| (default: %(default)s).')
    parser.add_argument('--pain-bin-scheme', choices=['absolute', 'subject_relative'], default='absolute',
                         help='"absolute" = fixed cutpoints shared across subjects (none/low/medium/high). '
                              '"subject_relative" = none is score==0, low/high split at that subject\'s own '
                              'mean non-zero pain score (no medium). (default: %(default)s)')
    parser.add_argument('--region-order', nargs='+', default=None,
                         help='Explicit region row order (space-separated region names), overriding '
                              '--row-order entirely -- e.g. to force the SAME row order across an absolute '
                              'and a subject_relative run for side-by-side comparison (each scheme\'s own '
                              'z-scores otherwise produce a different cluster/effect_size order). Regions not '
                              'present in this run\'s data are silently dropped, in the given order.')
    args = parser.parse_args()

    df, cache_paths = common.load_cache(args.subjects)
    if args.pain_bin_scheme == 'subject_relative':
        df['pain_bin'] = common.assign_relative_pain_bins(df)
    df = common.add_region(df)
    bin_labels = common.bin_label_table(df)
    zscore_cols = zscore_cols_for_scheme(args.pain_bin_scheme)
    count_bin_order = config.pain_bin_order(args.pain_bin_scheme)

    events_per_bin = df.groupby(['subject', 'pain_bin'])['pain_event_id'].nunique().unstack(fill_value=0)
    logger.info('Pain events per subject/bin (%s scheme):\n%s', args.pain_bin_scheme, events_per_bin)

    epoch_table = common.subject_region_epoch_table(df)
    subject_zscores = common.compute_subject_zscores(epoch_table, min_baseline_epochs=args.min_baseline_epochs)

    group = group_table(subject_zscores)
    group_wide = _wide_by_bin(group, ['region', 'freq_bin_index'], zscore_cols)

    group_counts = common.epoch_counts(df)
    subject_counts = common.epoch_counts(df, by_subject=True)

    # Row order computed ONCE from the group table and reused for every
    # figure (group + all per-subject) -- a clustered/effect-size order
    # recomputed per subject would make rows jump around between subjects,
    # defeating side-by-side comparison.
    regions_present = [r for r in config.ROI_REGIONS if r in group_wide['region'].unique()]
    freq_bins = bin_labels.index.tolist()
    if args.region_order:
        regions_order = [r for r in args.region_order if r in regions_present]
        logger.info('Region row order (explicit --region-order override): %s', regions_order)
    elif args.row_order == 'cluster':
        regions_order = common.cluster_region_order(group_wide, zscore_cols, freq_bins, regions=regions_present)
        logger.info('Region row order (%s): %s', args.row_order, regions_order)
    elif args.row_order == 'effect_size':
        regions_order = common.effect_size_region_order(group_wide, zscore_cols, regions=regions_present)
        logger.info('Region row order (%s): %s', args.row_order, regions_order)
    else:
        regions_order = regions_present
        logger.info('Region row order (%s): %s', args.row_order, regions_order)

    n_subjects = subject_zscores['subject'].nunique()
    # scheme baked into the run_name itself (not a parent folder) -- row_order
    # and pain_bin_scheme are just run parameters, recorded in provenance.json,
    # not separate directory levels.
    run_label = f'{args.run_name}_{args.pain_bin_scheme}' if args.run_name else None
    run_dir = common.make_run_dir(run_label, n_subjects, category='zscore_heatmap')
    common.write_run_provenance(
        run_dir, 'pain_analysis/plot_pain_zscore_heatmaps.py', args, cache_paths,
        subjects=df['subject'].unique().tolist(),
        extra={'zscore_cols': zscore_cols, 'min_baseline_epochs': args.min_baseline_epochs,
               'region_row_order': regions_order, 'pain_bin_scheme': args.pain_bin_scheme},
    )

    plot_zscore_heatmaps(
        group_wide, zscore_cols, bin_labels, group_counts, count_bin_order, f"Group (n={n_subjects} subjects)",
        run_dir / 'group_zscore_heatmap.png', regions=regions_order,
    )

    subject_wide = _wide_by_bin(subject_zscores, ['subject', 'region', 'freq_bin_index'], zscore_cols)
    for subject, sub_wide in subject_wide.groupby('subject'):
        plot_zscore_heatmaps(
            sub_wide, zscore_cols, bin_labels, subject_counts.xs(subject, level='subject'), count_bin_order,
            f'sub-{subject}', run_dir / 'by_subject' / f'sub-{subject}_zscore_heatmap.png', regions=regions_order,
        )

    logger.info('Run outputs + provenance.json written to %s', run_dir)


if __name__ == '__main__':
    main()

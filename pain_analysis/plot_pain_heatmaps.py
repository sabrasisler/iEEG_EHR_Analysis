"""
Cheap plotting step: load cached per-subject epoch/channel PSD CSVs (written
by build_pain_epoch_power.py), group channels into regions, average within
subject (equal-weighted across channels then epochs), compute delta power vs.
each subject's own no-pain baseline, then average across subjects
(equal-weighted) for the group figure.

Pure pandas/matplotlib -- does not touch NWB, so region definitions,
weighting, and delta choices can be iterated on for free.

Supports two pain-bin schemes (--pain-bin-scheme), recomputed at plot time
from the cache's raw `pain_score` column -- no need to re-run
build_pain_epoch_power.py to switch:
- absolute (default): fixed cutpoints shared across all subjects
  (config.PAIN_BIN_EDGES) -- none/low/medium/high.
- subject_relative: 'none' still means score == 0, but 'low'/'high' splits
  at that SAME subject's own mean pain score among their non-zero events --
  no 'medium' bin (see common.assign_relative_pain_bins).

Each run writes its own subdirectory under
config.PLOTS_ROOT/delta_heatmap/<scheme>/<run_name>_<timestamp>/ (see
pain_analysis/CONTEXT.md for the full naming convention -- a timestamp is
always appended so reruns never collide/overwrite a prior run). Contains the
PNGs plus a `provenance.json`: git commit/dirty state, script args, the
subject list included, region-grouping config, and the source cache CSVs
(+ their per-file provenance, when available) that went into it.

Run on a dev/interactive Slurm shell (never the login node):
    module load python/3.12
    source /home/groups/ckeller1/venvs/ieeg_ehr_analysis/bin/activate
    python -m pain_analysis.plot_pain_heatmaps
"""

import argparse
import logging

import numpy as np
import pandas as pd

from pain_analysis import common, config

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def delta_pairs_for_scheme(scheme):
    if scheme == 'subject_relative':
        return [('low', 'none'), ('high', 'none')]
    return [('low', 'none'), ('medium', 'none'), ('high', 'none')]


def compute_deltas(region_mean, bin_order, delta_pairs):
    """Delta vs. each subject's own 'none'-bin baseline. NaN (not imputed)
    where the subject has no 'none'-bin value for that region/freq_bin."""
    wide = region_mean.pivot_table(
        index=['subject', 'region', 'freq_bin_index'], columns='pain_bin', values='mean_log_power',
    )
    for bin_name in bin_order:
        if bin_name not in wide.columns:
            wide[bin_name] = np.nan
    deltas = pd.DataFrame(index=wide.index)
    for hi, lo in delta_pairs:
        deltas[f'{hi}-{lo}'] = wide[hi] - wide[lo]
    return deltas.reset_index()


def group_table(deltas, delta_cols):
    return (deltas.groupby(['region', 'freq_bin_index'])[delta_cols]
            .agg(lambda s: np.nanmean(s.to_numpy()))
            .reset_index())


def plot_delta_heatmaps(table, delta_cols, bin_labels, counts, count_bin_order, title, out_path):
    regions = config.ROI_REGIONS
    freq_bins = bin_labels.index.tolist()
    pivots = [common.pivot_for_plot(table, col, regions, freq_bins) for col in delta_cols]
    common.plot_region_freq_heatmaps(
        pivots, delta_cols, bin_labels, counts, title, out_path,
        cbar_label='Mean delta log10(V^2/Hz)', count_bin_order=count_bin_order,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subjects', nargs='+', default=None,
                         help='Subject IDs without sub- prefix (default: all present in cache dir).')
    parser.add_argument('--run-name', default=None,
                         help='Name for this run\'s output subdirectory under '
                              f'{config.PLOTS_ROOT} (default: auto-generated from timestamp + subject count).')
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

    bin_order = config.pain_bin_order(args.pain_bin_scheme)
    delta_pairs = delta_pairs_for_scheme(args.pain_bin_scheme)
    delta_cols = [f'{hi}-{lo}' for hi, lo in delta_pairs]

    events_per_bin = df.groupby(['subject', 'pain_bin'])['pain_event_id'].nunique().unstack(fill_value=0)
    logger.info('Pain events per subject/bin (%s scheme):\n%s', args.pain_bin_scheme, events_per_bin)

    region_mean = common.subject_region_table(df)
    deltas = compute_deltas(region_mean, bin_order, delta_pairs)
    group = group_table(deltas, delta_cols)

    group_counts = common.epoch_counts(df)
    subject_counts = common.epoch_counts(df, by_subject=True)

    run_dir = common.make_run_dir(args.run_name, deltas['subject'].nunique(),
                                   category=f'delta_heatmap/{args.pain_bin_scheme}')
    common.write_run_provenance(
        run_dir, 'pain_analysis/plot_pain_heatmaps.py', args, cache_paths,
        subjects=df['subject'].unique().tolist(),
        extra={'delta_pairs': delta_cols, 'pain_bin_scheme': args.pain_bin_scheme},
    )

    plot_delta_heatmaps(
        group, delta_cols, bin_labels, group_counts, bin_order,
        f"Group (n={deltas['subject'].nunique()} subjects)",
        run_dir / 'group_delta_heatmap.png',
    )

    for subject, sub_deltas in deltas.groupby('subject'):
        plot_delta_heatmaps(
            sub_deltas, delta_cols, bin_labels, subject_counts.xs(subject, level='subject'), bin_order,
            f'sub-{subject}', run_dir / 'by_subject' / f'sub-{subject}_delta_heatmap.png',
        )

    logger.info('Run outputs + provenance.json written to %s', run_dir)


if __name__ == '__main__':
    main()

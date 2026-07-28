#!/usr/bin/env python3
"""
Region x frequency-bin heatmaps from a P1.3 view (group + per subject).

Reads the view TABLES that build_pain_epoch_view wrote rather than recomputing
them, so plot iteration is decoupled from cache reads and so the figure and the
numbers behind it provably come from the same values.

Reuses the existing plotting stack unchanged -- features/common.py's
`plot_region_freq_heatmaps`, `pivot_for_plot`, `cluster_region_order`,
`epoch_count_labels` -- which is why these figures are comparable to the
pre-refactor ones despite coming through an entirely new path.

The 'none' pain bin is not plotted: under any baseline normalization it is its own
reference and sits at ~0 by construction, so a panel for it would be a band of
white that crushes the shared colour scale. It IS still reported in the log as a
correctness check -- a 'none' mean far from 0 means the baseline leaked.

Run on Slurm, never the login node:
    python -m ieeg_ehr.analysis.plot_pain_view_heatmaps --view-dir <dir> \\
        --run-name std10_zscore
"""

import argparse
import logging

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.features import common
from ieeg_ehr.views import cache_reader

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Panels, in order, per pain-bin scheme. 'none' deliberately absent (see docstring).
PANELS = {
    'subject_relative': ['low', 'high'],
    'absolute': ['low', 'medium', 'high'],
}


def load_view_tables(view_dir, kind):
    paths = sorted(view_dir.glob(f'view_{kind}_sub-*.parquet'))
    if not paths:
        raise FileNotFoundError(f'no view_{kind}_*.parquet in {view_dir}')
    frames = [io.read_table(p, on_stale='warn') for p in paths]
    return pd.concat(frames, ignore_index=True), paths


def group_table(subject_tables):
    """Average subject-level values across subjects, EQUAL-WEIGHTED.

    The subject is the unit of replication (not the electrode and not the epoch),
    so a subject with 200 contacts must not outvote one with 30. nanmean because a
    region/bin a subject has no coverage for is missing, not zero.
    """
    return (subject_tables
            .groupby(['pain_bin', 'region', 'freq_bin_index'], dropna=False)['value']
            .agg(lambda s: np.nanmean(s.to_numpy()))
            .reset_index())


def wide_by_bin(long_table, index_cols, panels):
    wide = long_table.pivot_table(index=index_cols, columns='pain_bin', values='value')
    for panel in panels:
        if panel not in wide.columns:
            wide[panel] = np.nan
    return wide[panels].reset_index()


def epoch_counts(epoch_tables, by_subject=False):
    """(region, pain_bin) -> distinct contributing epochs.

    De-duplicated to one row per (subject, epoch, region) first: a freq-bin row is
    not a distinct epoch, and counting rows would inflate n by 50x.
    """
    keys = ['subject_id', 'region', 'pain_bin'] if by_subject else ['region', 'pain_bin']
    deduped = epoch_tables.drop_duplicates(['subject_id', 'epoch_id', 'region', 'pain_bin'])
    counts = deduped.groupby(keys).size()
    if by_subject:
        counts.index = counts.index.set_names(['subject', 'region', 'pain_bin'])
    return counts


def plot_one(table, panels, bin_labels, counts, count_order, title, out_path,
             regions, cbar_label):
    freq_bins = bin_labels.index.tolist()
    pivots = [common.pivot_for_plot(table, panel, regions, freq_bins) for panel in panels]
    common.plot_region_freq_heatmaps(
        pivots, panels, bin_labels, counts, title, out_path,
        cbar_label=cbar_label, regions=regions, count_bin_order=count_order,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--view-dir', required=True,
                    help='Directory holding view_subject_*.parquet / view_epochs_*.parquet')
    ap.add_argument('--run-name', default=None)
    ap.add_argument('--out-root', default=None,
                    help=f'Default: {config.PLOTS_ROOT} (analysis/scratch on Oak)')
    ap.add_argument('--pain-bin-scheme', choices=list(PANELS), default='subject_relative')
    ap.add_argument('--row-order', choices=['default', 'cluster', 'effect_size'],
                    default='cluster')
    ap.add_argument('--region-order', nargs='+', default=None,
                    help='Explicit region row order, overriding --row-order. Use to force '
                         'the SAME rows across two runs for side-by-side comparison.')
    ap.add_argument('--cbar-label', default=None,
                    help='Default: read from the view sidecar so units cannot be mislabeled')
    args = ap.parse_args()

    from pathlib import Path
    view_dir = Path(args.view_dir)
    io.warn_if_dirty()

    subject_tables, subject_paths = load_view_tables(view_dir, 'subject')
    epoch_tables, epoch_paths = load_view_tables(view_dir, 'epochs')
    subjects = sorted(subject_tables['subject_id'].unique())
    logger.info('%d subject(s): %s', len(subjects), subjects)

    # Units and view config come from the artifact, not from a CLI flag, so a
    # figure cannot claim a normalization it was not built with.
    sidecar = io.read_sidecar(subject_paths[0]) or {}
    view_params = sidecar.get('params', {})
    cbar_label = args.cbar_label
    if cbar_label is None:
        from ieeg_ehr.views.view_config import ViewConfig
        keys = {f.name for f in ViewConfig.__dataclass_fields__.values()}
        cbar_label = ViewConfig(**{k: v for k, v in view_params.items()
                                   if k in keys}).value_label
    logger.info('view: %s', {k: view_params.get(k) for k in
                             ('normalization', 'domain', 'mask_label', 'pain_bins',
                              'roi_scheme')})

    panels = PANELS[args.pain_bin_scheme]
    count_order = config.pain_bin_order(args.pain_bin_scheme)
    bin_labels = (cache_reader.bin_edges(view_params.get('epoch_minutes'))
                  .set_index('freq_bin_index'))

    # Correctness check, logged rather than plotted (see module docstring).
    none_rows = subject_tables[subject_tables['pain_bin'] == 'none']['value']
    if not none_rows.empty:
        logger.info("baseline check -- 'none' bin mean %.2e, max |value| %.2e "
                    '(should be ~0: it is its own reference)',
                    float(none_rows.mean()), float(none_rows.abs().max()))

    group = wide_by_bin(group_table(subject_tables), ['region', 'freq_bin_index'], panels)
    regions_present = [r for r in config.ROI_REGIONS if r in set(group['region'])]
    freq_bins = bin_labels.index.tolist()

    # Row order computed ONCE from the group table and reused for every figure --
    # a per-subject clustering would make rows jump between panels and defeat the
    # side-by-side comparison the by_subject/ figures exist for.
    if args.region_order:
        regions_order = [r for r in args.region_order if r in regions_present]
    elif args.row_order == 'cluster':
        regions_order = common.cluster_region_order(group, panels, freq_bins,
                                                    regions=regions_present)
    elif args.row_order == 'effect_size':
        regions_order = common.effect_size_region_order(group, panels,
                                                        regions=regions_present)
    else:
        regions_order = regions_present
    logger.info('region row order (%s): %s', args.row_order, regions_order)

    out_root = Path(args.out_root) if args.out_root else config.PLOTS_ROOT
    run_dir = common.make_run_dir(args.run_name, len(subjects),
                                  category=f'view_heatmap/{args.pain_bin_scheme}')
    if args.out_root:
        run_dir = out_root / run_dir.name
        run_dir.mkdir(parents=True, exist_ok=True)

    io.write_run_provenance(
        run_dir, script='ieeg_ehr/analysis/plot_pain_view_heatmaps.py',
        params={**{k: v for k, v in vars(args).items()},
                'view_params': view_params},
        parents=[io.parent_ref(p, digest=False) for p in subject_paths + epoch_paths],
        subjects=subjects,
        extra={'panels': panels, 'region_row_order': regions_order,
               'roi_regions': config.ROI_REGIONS, 'n_subjects': len(subjects)},
    )

    plot_one(group, panels, bin_labels, epoch_counts(epoch_tables), count_order,
             f'Group (n={len(subjects)} subjects)', run_dir / 'group_view_heatmap.png',
             regions_order, cbar_label)

    per_subject_counts = epoch_counts(epoch_tables, by_subject=True)
    subject_wide = wide_by_bin(subject_tables,
                               ['subject_id', 'region', 'freq_bin_index'], panels)
    for subject_id, rows in subject_wide.groupby('subject_id'):
        short = subject_id.replace('sub-', '')
        try:
            counts = per_subject_counts.xs(subject_id, level='subject')
        except KeyError:
            counts = None
        plot_one(rows, panels, bin_labels, counts, count_order, subject_id,
                 run_dir / 'by_subject' / f'{subject_id}_view_heatmap.png',
                 regions_order, cbar_label)

    io.log_analysis(f'P1.3 view heatmaps ({view_params.get("normalization")}, '
                    f'mask {view_params.get("mask_label")}), n={len(subjects)}', run_dir)
    logger.info('figures + provenance -> %s', run_dir)


if __name__ == '__main__':
    main()

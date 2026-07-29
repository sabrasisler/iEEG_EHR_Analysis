#!/usr/bin/env python3
"""
Region x frequency-bin heatmaps from a P1.3 view (group + per subject).

Reads the view TABLES that build_pain_epoch_view wrote rather than recomputing
them, so plot iteration is decoupled from cache reads and so the figure and the
numbers behind it provably come from the same values.

Reuses the existing plotting stack unchanged -- features/common.py's
`plot_region_freq_heatmaps`, `pivot_for_plot`, `cluster_region_order`,
`epoch_count_labels` -- which is why these figures are comparable to the
pre-refactor ones despite coming through an entirely new path. Loading and group
aggregation come from analysis/view_tables.py, shared with the spectra figures so
a cell here and a line there are the same number.

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

from ieeg_ehr import config, io
from ieeg_ehr.analysis import view_tables
from ieeg_ehr.analysis.view_tables import (PANELS, epoch_counts, group_table,
                                           load_view_tables, wide_by_bin)
from ieeg_ehr.features import common
from ieeg_ehr.views import cache_reader

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_TYPE = 'region_freq_heatmap'


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
    ap.add_argument('--pain-bin-scheme', choices=list(PANELS), default='subject_relative')
    ap.add_argument('--row-order', choices=['default', 'cluster', 'effect_size'],
                    default='default',
                    help='Region row order. DEFAULT is the fixed anatomical '
                         'config.ROI_REGIONS order, so EVERY figure from every run '
                         'shares one ordering and can be compared directly. '
                         '"cluster"/"effect_size" reorder rows from THIS run\'s own '
                         'data, which makes two figures incomparable unless you also '
                         'pass --region-order; ask for them deliberately.')
    ap.add_argument('--region-order', nargs='+', default=None,
                    help='Explicit region row order, overriding --row-order. Use to force '
                         'the SAME rows across two runs for side-by-side comparison.')
    ap.add_argument('--cbar-label', default=None,
                    help='Default: read from the view sidecar so units cannot be mislabeled')
    view_tables.add_output_arguments(ap)
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
    view_params, view = view_tables.view_params_from(subject_paths)
    cbar_label = args.cbar_label or (view.value_label if view else 'value')
    logger.info('view: %s', {k: view_params.get(k) for k in
                             ('normalization', 'domain', 'mask_label', 'pain_bins',
                              'roi_scheme')})

    panels = PANELS[args.pain_bin_scheme]
    count_order = config.pain_bin_order(args.pain_bin_scheme)
    bin_labels = (cache_reader.bin_edges(view_params.get('epoch_minutes'))
                  .set_index('freq_bin_index'))

    view_tables.log_baseline_check(subject_tables)

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

    run_dir = view_tables.resolve_run_dir(args, OUTPUT_TYPE, view)
    logger.info('run dir: %s', run_dir)

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

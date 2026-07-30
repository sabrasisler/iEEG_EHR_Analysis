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

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import cluster_permutation, view_tables
from ieeg_ehr.analysis.view_tables import (PANELS, epoch_counts, group_table,
                                           load_view_tables, wide_by_bin)
from ieeg_ehr.features import common
from ieeg_ehr.views import cache_reader

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_TYPE = 'region_freq_heatmap'


def plot_one(table, panels, bin_labels, counts, count_order, title, out_path,
             regions, cbar_label, outline_masks=None, footnote=None):
    freq_bins = bin_labels.index.tolist()
    pivots = [common.pivot_for_plot(table, panel, regions, freq_bins) for panel in panels]
    common.plot_region_freq_heatmaps(
        pivots, panels, bin_labels, counts, title, out_path,
        cbar_label=cbar_label, regions=regions, count_bin_order=count_order,
        outline_masks=outline_masks, footnote=footnote,
    )


# ============================================================================
# CLUSTER-BASED PERMUTATION TEST  (docs/cluster_permutation.md)
# ============================================================================

def subject_matrix(subject_tables, panels, regions, freq_bins):
    """(n_subject, n_region, n_bin) per pain level, plus the subject order.

    One value per subject per region per bin -- which is what the view tables
    already are, channels and epochs having been averaged in the view layer. That
    is the exchangeability unit the permutation scheme requires: if channels
    entered as independent rows the test would run anticonservative.
    """
    per_subj = view_tables.per_subject(subject_tables)
    subjects = sorted(per_subj['subject_id'].unique())
    s_idx = {s: i for i, s in enumerate(subjects)}
    r_idx = {r: i for i, r in enumerate(regions)}
    b_idx = {b: i for i, b in enumerate(freq_bins)}

    out = {}
    for panel in panels:
        m = np.full((len(subjects), len(regions), len(freq_bins)), np.nan)
        rows = per_subj[per_subj['pain_bin'] == panel]
        rows = rows[rows['region'].isin(r_idx) & rows['freq_bin_index'].isin(b_idx)]
        m[rows['subject_id'].map(s_idx).to_numpy(),
          rows['region'].map(r_idx).to_numpy(),
          rows['freq_bin_index'].map(b_idx).to_numpy()] = rows['value'].to_numpy()
        out[panel] = m
    return out, subjects


def noise_floor(result):
    """The largest group-mean magnitude anywhere in a contrast's valid map.

    Taken from the 'none' control, this is the pipeline's MEASURED noise floor:
    'none' is circular (the same windows define the baseline and are tested
    against it) and equal-weights epochs of unequal surviving-window count, so it
    departs from 0 by a small mechanical amount rather than by leakage. That amount
    is the smallest effect this pipeline can claim to distinguish from bookkeeping.
    """
    m = np.abs(result['mean_map'])
    return float(np.nanmax(m)) if np.isfinite(m).any() else float('nan')


def cluster_rows(result, contrast, regions, bin_labels, freq_bins, floor=None):
    """Cluster records as table rows, with real region names and Hz edges.

    EFFECT SIZE TRAVELS WITH EVERY P-VALUE. A permutation test answers "is this
    larger than chance", which is not "is this large": a tiny mean over a tinier
    standard error is overwhelmingly significant and scientifically empty. That is
    not hypothetical here -- the 'none' control produces significant clusters at a
    mean of 0.004 z. `mean_abs_z` and `floor_ratio` are what make a cluster
    readable without re-deriving them.
    """
    rows = []
    for c in result['clusters']:
        lo_bin, hi_bin = freq_bins[c['bin_lo']], freq_bins[c['bin_hi']]
        cells = result['mean_map'][c['region_idx'], c['bin_lo']:c['bin_hi'] + 1]
        mean_abs = float(np.nanmean(np.abs(cells)))
        rows.append({
            'contrast': contrast, 'region': regions[c['region_idx']],
            'bin_lo_idx': int(lo_bin), 'bin_hi_idx': int(hi_bin),
            'bin_low_hz': float(bin_labels.loc[lo_bin, 'bin_low_hz']),
            'bin_high_hz': float(bin_labels.loc[hi_bin, 'bin_high_hz']),
            'n_bins': c['n_bins'], 'sign': c['sign'],
            'mass': c['mass'], 'peak_t': c['peak_t'],
            'mean_signed_z': float(np.nanmean(cells)), 'mean_abs_z': mean_abs,
            'peak_abs_z': float(np.nanmax(np.abs(cells))),
            'floor_ratio': (mean_abs / floor) if floor and np.isfinite(floor) else np.nan,
            'n_subjects_min': int(result['n_map'][c['region_idx'],
                                                  c['bin_lo']:c['bin_hi'] + 1].min()),
            'p_within_region': c['p_within_region'], 'p_global': c['p_global'],
            'region_p_bh': c['region_p_bh'],
            'sig_two_stage': c['sig_two_stage'], 'sig_global': c['sig_global'],
            'statistic': result['statistic'],
        })
    return rows


def run_cluster_tests(subject_tables, panels, regions, bin_labels, freq_bins, valid, args):
    """Every contrast through ONE code path. Returns (results, rows, subjects).

    low-vs-0 and high-vs-0 are the levels themselves; high-vs-low is the per-subject
    PAIRED difference handed to the same one-sample function -- not a second
    implementation. 'none' is included as a NEGATIVE CONTROL: it is its own
    reference under any baseline normalization, so it must come back with
    essentially no clusters. If it does not, the baseline is leaking and every other
    number here is suspect.
    """
    wanted = list(dict.fromkeys(list(panels) + ['none']))
    mats, subjects = subject_matrix(subject_tables, wanted, regions, freq_bins)

    contrasts = {p: mats[p] for p in panels}
    if 'low' in mats and 'high' in mats:
        contrasts['high_minus_low'] = mats['high'] - mats['low']
    contrasts['none_control'] = mats['none']

    # The control runs FIRST: its floor is needed to score every other contrast.
    ordered = ['none_control'] + [k for k in contrasts if k != 'none_control']

    results, rows, floor = {}, [], None
    for name in ordered:
        x = contrasts[name]
        if args.detrend_freq:
            x = cluster_permutation.detrend_over_frequency(x, valid=valid)
        res = cluster_permutation.cluster_test(
            x, valid=valid, alpha=args.alpha, min_extent=args.min_cluster_bins,
            n_perm=args.n_perm, q=args.fdr_q, seed=args.seed,
            statistic='t', min_subjects=args.min_subjects)
        results[name] = res
        if name == 'none_control':
            floor = noise_floor(res)
            logger.info('measured noise floor from the none control: max |group mean| '
                        '= %.4f. Clusters below ~3x this are bookkeeping, not effects.',
                        floor)
        rows.extend(cluster_rows(res, name, regions, bin_labels, freq_bins, floor))

        if args.robust:
            # Sensitivity check, NOT a second test: sign-flipping does not protect
            # against one subject driving a cluster, because t is still mean/SD.
            rob = cluster_permutation.cluster_test(
                x, valid=valid, alpha=args.alpha, min_extent=args.min_cluster_bins,
                n_perm=args.robust_n_perm, q=args.fdr_q, seed=args.seed,
                statistic='yuen', min_subjects=args.min_subjects)
            results[f'{name}__robust'] = rob
            rows.extend(cluster_rows(rob, f'{name}__robust', regions,
                                     bin_labels, freq_bins, floor))

        n_two = sum(c['sig_two_stage'] for c in res['clusters'])
        n_glob = sum(c['sig_global'] for c in res['clusters'])
        above = sum(1 for r in rows if r['contrast'] == name and r['sig_two_stage']
                    and np.isfinite(r['floor_ratio']) and r['floor_ratio'] >= 3.0)
        logger.info('%-16s %3d formed | %2d sig two-stage | %2d sig global | '
                    '%2d sig AND >=3x the noise floor',
                    name, len(res['clusters']), n_two, n_glob, above)

    # NOT a pass/fail gate. 'none' is CIRCULAR -- the same windows define the
    # baseline and are tested against it -- and it equal-weights epochs of unequal
    # surviving-window count, so it cannot be exactly 0 and significant clusters
    # there are expected. Its value is quantitative: it measures how small a
    # "significant" effect this pipeline will happily report.
    control = sum(c['sig_two_stage'] for c in results['none_control']['clusters'])
    logger.info("none control (CIRCULAR by construction, so this is a FLOOR, not a "
                'pass/fail): %d significant cluster(s), max |group mean| %.4f. Read '
                'every other cluster against that number.', control, floor)
    return results, rows, subjects, floor


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
    ap.add_argument('--exclude-line-noise-bins', action='store_true',
                    help='Blank the line-noise-flagged frequency bins, which also '
                         'RESCALES the shared colour bar -- those bins hold the '
                         'largest-magnitude cells in this data, so leaving them in '
                         'compresses everything else toward white. A DISPLAY-time '
                         'exclusion of an unchanged view, distinct from the view axis '
                         'of the same idea (ViewConfig.drop_line_noise_bins), which '
                         'bakes it into the tables. For freq=log_bins_50 the two give '
                         'identical values for every surviving bin: bins do not mix in '
                         'region-averaging or in the group mean.')
    g = ap.add_argument_group('cluster permutation test (docs/cluster_permutation.md)')
    g.add_argument('--cluster-test', action='store_true',
                   help='Run a cluster-based permutation test against the 0-pain '
                        'state and outline significant clusters on the heatmaps.')
    g.add_argument('--n-perm', type=int, default=10000)
    g.add_argument('--alpha', type=float, default=0.05,
                   help='Cluster-FORMING threshold (two-sided) and the cluster-level '
                        'significance cutoff.')
    g.add_argument('--min-cluster-bins', type=int, default=3,
                   help='Minimum cluster extent in frequency bins. Applied to the '
                        'observed map AND inside every permutation -- filtering only '
                        'the observed clusters would invalidate the test.')
    g.add_argument('--fdr-q', type=float, default=0.05)
    g.add_argument('--seed', type=int, default=0,
                   help='Permutation RNG seed, recorded in provenance so the result '
                        'is reproducible.')
    g.add_argument('--correction', choices=['two_stage', 'global'], default='two_stage',
                   help='Which scope decides the OUTLINES; both are always computed '
                        'and both land in clusters.parquet. two_stage = family-wise '
                        'within region across frequency, then BH-FDR across regions. '
                        'global = one max-stat null over the whole map, which lets '
                        'well-covered regions (Temporal n=51) swallow the power from '
                        'sparse ones (ACC n=21).')
    g.add_argument('--min-subjects', type=int, default=8,
                   help='Cells with fewer contributing subjects cannot enter a '
                        'cluster. Affects the TEST only -- the heatmap still displays '
                        'them.')
    g.add_argument('--detrend-freq', action='store_true',
                   help='Subtract each subject x region mean over valid bins first. '
                        'CHANGES THE HYPOTHESIS to "spectral shape is not flat", so '
                        'it is off by default. Use it when the broadband '
                        'low-frequency offset absorbs everything into one giant '
                        '1-40 Hz cluster that spans delta through beta and cannot be '
                        'attributed to a band.')
    g.add_argument('--robust', action='store_true',
                   help='Also run a 20%% trimmed-mean (Yuen) sensitivity check, since '
                        'sign-flipping alone does not stop one subject driving a '
                        'cluster. Reported in the table, never plotted.')
    g.add_argument('--robust-n-perm', type=int, default=2000)
    g.add_argument('--test-includes-line-noise', action='store_true',
                   help='Let line-noise bins enter clusters. OFF by default and you '
                        'almost certainly do not want it: line noise is consistent '
                        'across subjects, so its t can be large, and because the null '
                        'is a MAX statistic those masses inflate it and cost power for '
                        'every real cluster. Independent of '
                        '--exclude-line-noise-bins, which is about display.')
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
    epoch_minutes = view_params.get('epoch_minutes')
    bin_labels = cache_reader.bin_edges(epoch_minutes).set_index('freq_bin_index')

    # Dropped from the SUBJECT tables, once, before any aggregation, so the group
    # mean, the per-subject panels and the colour scale all agree about which bins
    # exist. Epoch counts are unaffected: they de-duplicate to one row per
    # (subject, epoch, region) and never involve a frequency bin.
    line_noise_bins = cache_reader.line_noise_bins(epoch_minutes)
    if args.exclude_line_noise_bins and len(line_noise_bins):
        keep = ~subject_tables['freq_bin_index'].isin(line_noise_bins)
        logger.info('excluding %d line-noise bin(s) from display (%s Hz): %d of %d '
                    'rows dropped', len(line_noise_bins),
                    ', '.join(f'{bin_labels.loc[b, "bin_low_hz"]:.0f}'
                              for b in line_noise_bins if b in bin_labels.index),
                    int((~keep).sum()), len(subject_tables))
        subject_tables = subject_tables[keep].copy()

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

    # ---------------- cluster permutation test ----------------
    outline_masks, cluster_extra, footnote = None, {}, None
    if args.cluster_test:
        # Cells eligible to enter a cluster. An invalid cell TERMINATES a run, so
        # nothing bridges the 60 Hz notch.
        #
        # LINE-NOISE BINS ARE EXCLUDED FROM THE TEST BY DEFAULT, independently of
        # whether they are excluded from the DISPLAY, and not because they would be
        # non-significant -- the opposite. Line noise is highly consistent across
        # subjects, so its across-subject variance is small and its t can be large;
        # these bins already hold the most extreme cells in the group map. Three
        # consequences, of which the second is the decisive one:
        #   1. A significant 59 Hz cluster is an artifact statement, not physiology,
        #      so it is an outline you would have to explain away.
        #   2. The null is a MAX statistic. Large artifact masses inflate it, which
        #      makes every REAL cluster harder to detect -- including them costs
        #      power for the physiology.
        #   3. Left valid, a run can bridge THROUGH the notch and merge a beta
        #      cluster with a gamma one into a single uninterpretable blob.
        valid = np.ones((len(regions_order), len(freq_bins)), dtype=bool)
        test_excluded = [] if args.test_includes_line_noise else [
            b for b in line_noise_bins if b in freq_bins]
        if test_excluded:
            valid[:, [freq_bins.index(b) for b in test_excluded]] = False
        logger.info('cluster test excludes %d line-noise bin(s) from clustering: %s',
                    len(test_excluded),
                    [f'{bin_labels.loc[b, "bin_low_hz"]:.0f} Hz' for b in test_excluded])

        logger.info('cluster test: %d permutations, alpha=%.3f, min extent=%d bins, '
                    'q=%.3f, seed=%d, correction=%s%s', args.n_perm, args.alpha,
                    args.min_cluster_bins, args.fdr_q, args.seed, args.correction,
                    ', DETRENDED over frequency' if args.detrend_freq else '')
        results, cluster_table_rows, _, floor = run_cluster_tests(
            subject_tables, panels, regions_order, bin_labels, freq_bins, valid, args)

        outline_masks = [cluster_permutation.significant_mask(
            results[p], len(regions_order), len(freq_bins), args.correction)
            for p in panels]
        footnote = (
            f'Outlines: cluster-based permutation test vs the 0-pain state, '
            f'{args.n_perm} sign-flip permutations (subject-wise, one sign vector '
            f'across all regions), cluster-forming p<{args.alpha}, min '
            f'{args.min_cluster_bins} bins, adjacency along frequency only within a '
            f'region, {args.correction} correction.\n'
            f'MEASURED NOISE FLOOR {floor:.4f} z, from the 0-pain bin tested against '
            f'itself -- circular by construction, so it is a floor and not a control. '
            f'A cluster whose mean |z| is not several times this is bookkeeping, not '
            f'an effect; see clusters.parquet columns mean_abs_z and floor_ratio.\n'
            f'{cluster_permutation.BOUNDARY_CAVEAT}\n'
            f'EXPLORATORY, discovery cohort -- NOMINATIONS, NOT FINDINGS.')

        clusters_df = pd.DataFrame(cluster_table_rows)
        io.write_table(clusters_df, run_dir / 'clusters.parquet', kind='table',
                       script='ieeg_ehr/analysis/plot_pain_view_heatmaps.py',
                       params={**view_params, 'n_perm': args.n_perm,
                               'alpha': args.alpha, 'fdr_q': args.fdr_q,
                               'min_cluster_bins': args.min_cluster_bins,
                               'seed': args.seed, 'detrend_freq': args.detrend_freq,
                               'min_subjects': args.min_subjects},
                       parents=[io.parent_ref(p, digest=False) for p in subject_paths],
                       subjects=subjects,
                       extra={'interpretation_caveat': cluster_permutation.BOUNDARY_CAVEAT,
                              'noise_floor_z': floor,
                              'noise_floor_note':
                                  "Max |group mean| of the 0-pain bin tested against "
                                  "itself. CIRCULAR by construction (same windows "
                                  "define the baseline) and equal-weights epochs of "
                                  "unequal surviving-window count, so it is a FLOOR on "
                                  "reportable effect size, not a pass/fail control. "
                                  "Compare via mean_abs_z / floor_ratio.",
                              'status': 'EXPLORATORY nomination, not a finding '
                                        '(CLAUDE.md; pending P2.6 FREEZE)'})

        cluster_extra = {
            'cluster_test': {
                'n_perm': args.n_perm, 'alpha': args.alpha, 'fdr_q': args.fdr_q,
                'min_cluster_bins': args.min_cluster_bins, 'seed': args.seed,
                'correction': args.correction, 'detrend_freq': args.detrend_freq,
                'min_subjects': args.min_subjects, 'robust': args.robust,
                'statistic': 'one-sample t, sign-flip permutation, cluster mass',
                'adjacency': 'frequency only, within region; invalid bins break runs',
                'n_clusters_formed': {k: len(v['clusters']) for k, v in results.items()},
                'n_significant_two_stage': {
                    k: int(sum(c['sig_two_stage'] for c in v['clusters']))
                    for k, v in results.items()},
                'n_significant_global': {
                    k: int(sum(c['sig_global'] for c in v['clusters']))
                    for k, v in results.items()},
                'none_control_significant': int(sum(
                    c['sig_two_stage'] for c in results['none_control']['clusters'])),
                'noise_floor_z': floor,
                'n_significant_above_3x_floor': {
                    k: int(sum(1 for r in cluster_table_rows
                               if r['contrast'] == k and r['sig_two_stage']
                               and np.isfinite(r['floor_ratio'])
                               and r['floor_ratio'] >= 3.0))
                    for k in results},
                'interpretation_caveat': cluster_permutation.BOUNDARY_CAVEAT,
                'status': 'EXPLORATORY nomination, not a finding',
            }
        }

    io.write_run_provenance(
        run_dir, script='ieeg_ehr/analysis/plot_pain_view_heatmaps.py',
        params={**{k: v for k, v in vars(args).items()},
                'view_params': view_params},
        parents=[io.parent_ref(p, digest=False) for p in subject_paths + epoch_paths],
        subjects=subjects,
        extra={'panels': panels, 'region_row_order': regions_order,
               'roi_regions': config.ROI_REGIONS, 'n_subjects': len(subjects),
               'line_noise_bins_excluded': ([int(b) for b in line_noise_bins]
                                            if args.exclude_line_noise_bins else []),
               **cluster_extra},
    )

    plot_one(group, panels, bin_labels, epoch_counts(epoch_tables), count_order,
             f'Group (n={len(subjects)} subjects)', run_dir / 'group_view_heatmap.png',
             regions_order, cbar_label, outline_masks=outline_masks, footnote=footnote)

    # high - low on its OWN colour scale, in its own figure. Forced onto the
    # low/high scale it would wash out to near-white: a difference of two
    # baseline-referenced quantities is much smaller than either.
    if args.cluster_test and 'high_minus_low' in results:
        diff = group[['region', 'freq_bin_index']].copy()
        diff['high_minus_low'] = group['high'] - group['low']
        plot_one(diff, ['high_minus_low'], bin_labels, None, count_order,
                 f'Group high - low (n={len(subjects)} subjects)',
                 run_dir / 'group_view_heatmap_high_minus_low.png',
                 regions_order, f'{cbar_label} (high - low)',
                 outline_masks=[cluster_permutation.significant_mask(
                     results['high_minus_low'], len(regions_order), len(freq_bins),
                     args.correction)],
                 footnote=footnote)

    per_subject_counts = epoch_counts(epoch_tables, by_subject=True)
    subject_wide = wide_by_bin(subject_tables,
                               ['subject_id', 'region', 'freq_bin_index'], panels)
    for subject_id, rows in subject_wide.groupby('subject_id'):
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

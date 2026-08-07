#!/usr/bin/env python3
"""
Continuous-pain heatmap: region x frequency, shaded by the group-mean regression
of log power on PAIN SCORE, with cluster-permutation outlines.

Red = power rises with pain, blue = falls. Units: log10(V^2/Hz) per pain point.

HOW THIS DIFFERS FROM THE BINNED HEATMAPS, and why it is worth having both
--------------------------------------------------------------------------
`zscore-relpain-roiv2` and `delta-relpain-roiv2` reference every subject to their
own 0-pain epochs and compare bins. That makes the precision of the result the
precision of the baseline -- and ten of 56 discovery subjects carry a 0-pain mean
whose SEM (0.083) EXCEEDS the effect being measured (0.052) (TASKS.md,
docs/labnotebook 2026-08-07).

This figure has no baseline. Each subject's coefficient is fitted across ALL of
their epochs against the full pain range, so:
  - the thin-baseline problem does not arise, and no view rebuild is needed;
  - there is no `none` bin, so none of the circularity that made the cluster test's
    0-pain control a floor rather than a control;
  - there is no epoch-weighting asymmetry, which was a consequence of having a
    pooled-window baseline at all;
  - between-subject scale cancels, because a gain difference is an additive shift
    in log space and does not change a slope.

It lives in the SAME output type as the binned heatmaps, under the level-4 folder
`contpain-<roi>`, so the three sit side by side and the folder says which pain axis
each used.

LINE-NOISE BINS ARE REMOVED, NOT INVALIDATED, and the axis closes over them. With
them merely invalidated they terminate a cluster, and measured on this 50-bin axis
that leaves the runs above 100 Hz at 2 bins (129-144 Hz) and 1 bin (200 Hz) --
shorter than min_extent, so those bins can never reach significance at ANY effect
size. Closing the gap costs an assertion that 48 Hz and 66 Hz are adjacent when
18 Hz between them was never measured, so every cluster carries `spans_removed_gap`.

Run on Slurm, never the login node:
    python -m ieeg_ehr.analysis.plot_pain_coef_heatmap --view-dir <RAW view>
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import cluster_permutation as cp
from ieeg_ehr.analysis import pain_coef, view_tables
from ieeg_ehr.features import common  # noqa: F401 (plot_region_freq_heatmaps)
from ieeg_ehr.views import cache_reader

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_TYPE = 'region_freq_heatmap'


def cluster_rows(result, regions, bin_labels, freq_bins, kept_indices, floor=None,
                 null_label='signflip'):
    """Cluster records as table rows, in ORIGINAL bin indices and real Hz."""
    rows = []
    for c in result['clusters']:
        lo = int(kept_indices[c['bin_lo']])
        hi = int(kept_indices[c['bin_hi']])
        cells = result['mean_map'][c['region_idx'], c['bin_lo']:c['bin_hi'] + 1]
        mean_abs = float(np.nanmean(np.abs(cells)))
        rows.append({
            'contrast': 'pain_coef', 'null': null_label,
            'region': regions[c['region_idx']],
            'bin_lo_idx': int(freq_bins[lo]), 'bin_hi_idx': int(freq_bins[hi]),
            'bin_low_hz': float(bin_labels.loc[freq_bins[lo], 'bin_low_hz']),
            'bin_high_hz': float(bin_labels.loc[freq_bins[hi], 'bin_high_hz']),
            'n_bins': c['n_bins'], 'sign': c['sign'],
            # TRUE when the cluster's original-index span contains a removed bin:
            # the outline will render as two boxes, but it is ONE cluster, and it
            # rests on an adjacency that was asserted rather than measured.
            'spans_removed_gap': cp.spans_removed_gap(lo, hi, kept_indices),
            'mass': c['mass'], 'peak_t': c['peak_t'],
            'mean_signed_coef': float(np.nanmean(cells)), 'mean_abs_coef': mean_abs,
            'floor_ratio': (mean_abs / floor) if floor else np.nan,
            'n_subjects_min': int(result['n_map'][c['region_idx'],
                                                  c['bin_lo']:c['bin_hi'] + 1].min()),
            'p_within_region': c['p_within_region'], 'p_global': c['p_global'],
            'region_p_bh': c['region_p_bh'],
            'sig_two_stage': c['sig_two_stage'], 'sig_global': c['sig_global'],
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--view-dir', required=True,
                    help='An UN-NORMALIZED view (--normalization none). The '
                         'regression must run on raw log power.')
    ap.add_argument('--run-name', default=None)
    ap.add_argument('--n-perm', type=int, default=10000)
    ap.add_argument('--shuffle-n-perm', type=int, default=2000,
                    help='Permutations for the pain-score shuffle null, which costs '
                         'a matmul per subject per permutation rather than a single '
                         'sign flip.')
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--min-cluster-bins', type=int, default=3)
    ap.add_argument('--fdr-q', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--correction', choices=['two_stage', 'global'], default='two_stage')
    ap.add_argument('--min-subjects', type=int, default=8)
    ap.add_argument('--min-epochs', type=int, default=pain_coef.MIN_EPOCHS)
    ap.add_argument('--min-range', type=float, default=pain_coef.MIN_RANGE)
    ap.add_argument('--min-non-modal', type=int, default=pain_coef.MIN_NON_MODAL)
    ap.add_argument('--keep-line-noise-bins', action='store_true',
                    help='Keep the line-noise bins in the map AND in the test. Off '
                         'by default; see the module docstring for why removing them '
                         'and closing the gap is the better trade.')
    ap.add_argument('--no-shuffle-null', action='store_true')
    view_tables.add_output_arguments(ap)
    args = ap.parse_args()

    view_dir = Path(args.view_dir)
    io.warn_if_dirty()

    epoch_tables, epoch_paths = view_tables.load_view_tables(view_dir, 'epochs')
    _, subject_paths = view_tables.load_view_tables(view_dir, 'subject')
    view_params, view = view_tables.view_params_from(subject_paths)
    logger.info('view: %s', {k: view_params.get(k) for k in
                             ('normalization', 'domain', 'mask_label', 'roi_scheme')})

    if view is not None and view.is_difference:
        raise SystemExit(
            f'--view-dir is a {view.normalization!r} view. Regressing an already '
            'baseline-referenced quantity on pain score is a different and largely '
            'meaningless number -- the baseline has already removed part of what the '
            'regression is meant to measure. Use --normalization none.')

    roi_regions = view_tables.roi_regions_for(view_params)
    epoch_minutes = view_params.get('epoch_minutes')
    bin_labels = cache_reader.bin_edges(epoch_minutes).set_index('freq_bin_index')
    freq_bins = bin_labels.index.tolist()
    logger.info('roi_scheme %r -> %d region(s)', view_params.get('roi_scheme'),
                len(roi_regions))

    # ---------------- per-subject coefficients ----------------
    coef, subjects, per_subject, diagnostics = pain_coef.subject_coef_matrix(
        epoch_tables, roi_regions, freq_bins,
        min_epochs=args.min_epochs, min_range=args.min_range,
        min_non_modal=args.min_non_modal)
    logger.info('pain_coef matrix: %d subject(s) x %d region(s) x %d bin(s)',
                *coef.shape)

    regions_present = [r for r in roi_regions
                       if np.isfinite(coef[:, roi_regions.index(r), :]).any()]
    keep_r = [roi_regions.index(r) for r in regions_present]
    coef = coef[:, keep_r, :]
    logger.info('%d region(s) with data: %s', len(regions_present), regions_present)

    # ---------------- compact the frequency axis ----------------
    drop = ([] if args.keep_line_noise_bins
            else [b for b in cache_reader.line_noise_bins(epoch_minutes)
                  if b in freq_bins])
    drop_pos = [freq_bins.index(b) for b in drop]
    coef_c, kept = cp.compact_bins(coef, drop_pos, len(freq_bins))
    logger.info('removed %d line-noise bin(s) (%s Hz) and CLOSED the gap: the test '
                'axis is %d contiguous bins', len(drop),
                [f'{bin_labels.loc[b, "bin_low_hz"]:.0f}' for b in drop], coef_c.shape[2])

    valid_c = np.ones((len(regions_present), coef_c.shape[2]), dtype=bool)

    # ---------------- the test ----------------
    logger.info('cluster test: %d sign-flip permutations, alpha=%.3f, min extent=%d, '
                'q=%.3f, seed=%d', args.n_perm, args.alpha, args.min_cluster_bins,
                args.fdr_q, args.seed)
    result = cp.cluster_test(coef_c, valid=valid_c, alpha=args.alpha,
                             min_extent=args.min_cluster_bins, n_perm=args.n_perm,
                             q=args.fdr_q, seed=args.seed,
                             min_subjects=args.min_subjects)
    floor = float(np.nanmax(np.abs(result['mean_map'])))

    rows = cluster_rows(result, regions_present, bin_labels, freq_bins, kept,
                        floor=None, null_label='signflip')
    n_sig = sum(r['sig_two_stage'] for r in rows)
    n_gap = sum(r['spans_removed_gap'] for r in rows)
    logger.info('sign-flip null: %d cluster(s), %d significant (two-stage), '
                '%d of which bridge a removed bin', len(rows), n_sig, n_gap)

    # ---------------- the pain-score shuffle null ----------------
    shuffle_summary = {}
    if not args.no_shuffle_null:
        logger.info('pain-score shuffle null: %d permutations (within subject, one '
                    'shuffle per subject applied across all regions)',
                    args.shuffle_n_perm)
        # Restrict the epoch matrices to the regions actually plotted, in order.
        n_bin_full = len(freq_bins)
        trimmed = {}
        for s, (Y, x) in per_subject.items():
            Y3 = Y.reshape(Y.shape[0], len(roi_regions), n_bin_full)[:, keep_r, :]
            trimmed[s] = (Y3[..., kept].reshape(Y.shape[0], -1), x)

        null_region, null_global = cp.predictor_shuffle_null(
            trimmed, (len(regions_present), coef_c.shape[2]), valid_c,
            args.alpha, args.min_cluster_bins, args.shuffle_n_perm,
            seed=args.seed, min_subjects=args.min_subjects,
            coef_fn=pain_coef.coef_from_predictor)

        for r, c in zip(rows, result['clusters']):
            m = abs(c['mass'])
            r['p_shuffle_within_region'] = cp.permutation_p(
                m, null_region[:, c['region_idx']])
            r['p_shuffle_global'] = cp.permutation_p(m, null_global)

        # THE SAME TWO-STAGE CORRECTION AS THE SIGN-FLIP NULL. Without this the two
        # numbers are not comparable -- one would be family-wise-within-region THEN
        # BH-across-regions, the other a bare within-region p -- and the shuffle
        # would look more permissive when it is merely less corrected. Same recipe:
        # each region contributes its minimum cluster p, BH runs over the regions
        # that were actually tested, and a cluster needs both its region to survive
        # and its own p < alpha.
        n_region = len(regions_present)
        region_p = np.ones(n_region)
        for r, c in zip(rows, result['clusters']):
            region_p[c['region_idx']] = min(region_p[c['region_idx']],
                                            r['p_shuffle_within_region'])
        tested = result['region_tested']
        rejected = np.zeros(n_region, dtype=bool)
        adj = np.ones(n_region)
        if tested.any():
            rej_t, adj_t = cp.bh_fdr(region_p[tested], args.fdr_q)
            rejected[tested] = rej_t
            adj[tested] = adj_t
        for r, c in zip(rows, result['clusters']):
            i = c['region_idx']
            r['region_p_bh_shuffle'] = float(adj[i])
            r['sig_two_stage_shuffle'] = bool(
                rejected[i] and r['p_shuffle_within_region'] < args.alpha)

        n_shuf_wr = sum(r['p_shuffle_within_region'] < args.alpha for r in rows)
        n_shuf = sum(r['sig_two_stage_shuffle'] for r in rows)
        agree = sum(r['sig_two_stage'] == r['sig_two_stage_shuffle'] for r in rows)
        shuffle_summary = {'n_perm': args.shuffle_n_perm,
                           'n_clusters_p_lt_alpha_within_region': int(n_shuf_wr),
                           'n_significant_two_stage': int(n_shuf),
                           'n_agreeing_with_signflip': int(agree)}
        logger.info('shuffle null: %d within-region p<%.3f, %d significant after the '
                    'SAME two-stage correction; agrees with the sign-flip verdict on '
                    '%d/%d cluster(s)', n_shuf_wr, args.alpha, n_shuf, agree, len(rows))

    # ---------------- outputs ----------------
    if not args.view_scheme:
        roi_code = (view.scheme_code.rsplit('-', 1)[-1]
                    if view is not None and '-' in view.scheme_code else 'roidefault')
        args.view_scheme = f'contpain-{roi_code}'
    run_dir = view_tables.resolve_run_dir(
        args, OUTPUT_TYPE, view, run_name=args.run_name or 'discovery_contpain')
    logger.info('run dir: %s', run_dir)

    mask_c = cp.significant_mask(result, len(regions_present), coef_c.shape[2],
                                 args.correction)
    outline = cp.expand_mask(mask_c, kept, len(freq_bins))

    group = pd.DataFrame({
        'region': np.repeat(regions_present, len(freq_bins)),
        'freq_bin_index': np.tile(freq_bins, len(regions_present)),
        'pain_coef': np.full(len(regions_present) * len(freq_bins), np.nan)})
    full_mean = np.full((len(regions_present), len(freq_bins)), np.nan)
    full_mean[:, kept] = result['mean_map']
    group['pain_coef'] = full_mean.ravel()

    params = {**view_params, 'n_perm': args.n_perm, 'alpha': args.alpha,
              'min_cluster_bins': args.min_cluster_bins, 'seed': args.seed,
              'min_epochs': args.min_epochs, 'min_range': args.min_range,
              'min_non_modal': args.min_non_modal,
              'line_noise_bins_removed': [int(b) for b in drop]}

    io.write_table(pd.DataFrame(rows), run_dir / 'clusters.parquet', kind='table',
                   script='ieeg_ehr/analysis/plot_pain_coef_heatmap.py',
                   params=params,
                   parents=[io.parent_ref(p, digest=False) for p in epoch_paths],
                   subjects=subjects,
                   extra={'interpretation_caveat': cp.BOUNDARY_CAVEAT,
                          'gap_caveat':
                              'Line-noise bins were REMOVED and the frequency axis '
                              'closed over them, so a cluster may assert adjacency '
                              'across frequencies that were never measured. '
                              'spans_removed_gap flags every such cluster.',
                          'status': 'EXPLORATORY nomination, not a finding'})
    io.write_table(group, run_dir / 'pain_coef.parquet', kind='table',
                   script='ieeg_ehr/analysis/plot_pain_coef_heatmap.py',
                   params=params, subjects=subjects)
    io.write_table(diagnostics, run_dir / 'subject_diagnostics.parquet', kind='table',
                   script='ieeg_ehr/analysis/plot_pain_coef_heatmap.py',
                   params=params)

    io.write_run_provenance(
        run_dir, script='ieeg_ehr/analysis/plot_pain_coef_heatmap.py',
        params={**vars(args), 'view_params': view_params},
        parents=[io.parent_ref(p, digest=False) for p in epoch_paths + subject_paths],
        subjects=subjects,
        extra={'quantity': 'pain_coef = OLS slope of log10(V^2/Hz) on pain score, '
                           'per subject per region per frequency bin',
               'n_subjects': len(subjects), 'regions_plotted': regions_present,
               'n_excluded': int((~diagnostics['included']).sum()),
               'exclusions': diagnostics.loc[~diagnostics['included'],
                                             ['subject_id', 'excluded_because']]
                             .to_dict('records'),
               'line_noise_bins_removed': [int(b) for b in drop],
               'gap_closed': not args.keep_line_noise_bins,
               'n_clusters': len(rows),
               'n_significant_two_stage': int(n_sig),
               'n_significant_spanning_a_removed_gap': int(n_gap),
               'shuffle_null': shuffle_summary,
               'interpretation_caveat': cp.BOUNDARY_CAVEAT,
               'status': 'EXPLORATORY nomination, not a finding '
                         '(CLAUDE.md; pending P2.6 FREEZE)'})

    footnote = (
        f'Group mean of a WITHIN-SUBJECT regression of log power on pain score '
        f'(n={len(subjects)}); red = power rises with pain. No 0-pain baseline, so no '
        f'thin-baseline noise and no circular 0-pain reference.\n'
        f'Outlines: cluster permutation, {args.n_perm} sign-flip permutations, '
        f'cluster-forming p<{args.alpha}, min {args.min_cluster_bins} bins, '
        f'{args.correction} correction'
        + ('' if args.no_shuffle_null else
           f'; a within-subject pain-score shuffle null ({args.shuffle_n_perm} perms) '
           f'is reported per cluster in clusters.parquet')
        + '.\n'
        f'Line-noise bins REMOVED and the axis closed over them, so clusters can span '
        f'them; any that does is flagged spans_removed_gap and rests on an adjacency '
        f'that was asserted, not measured.\n'
        f'{cp.BOUNDARY_CAVEAT}\n'
        f'EXPLORATORY, discovery cohort -- NOMINATIONS, NOT FINDINGS.')

    pivot = (group.pivot(index='region', columns='freq_bin_index', values='pain_coef')
             .reindex(index=regions_present, columns=freq_bins))
    common.plot_region_freq_heatmaps(
        [pivot], ['pain_coef'], bin_labels, None,
        f'Continuous pain: d(log power)/d(pain score) — n={len(subjects)} subjects',
        run_dir / 'group_pain_coef_heatmap.png',
        cbar_label='d log10(V^2/Hz) per pain point',
        regions=regions_present, outline_masks=[outline], footnote=footnote)

    io.log_analysis(f'continuous-pain regression heatmap (pain_coef), '
                    f'{len(rows)} clusters, {n_sig} significant, n={len(subjects)}',
                    run_dir)
    logger.info('figure + provenance -> %s', run_dir)


if __name__ == '__main__':
    main()

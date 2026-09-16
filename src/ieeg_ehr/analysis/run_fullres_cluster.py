"""Cluster-based permutation test on the NATIVE 0.5 Hz region x frequency map.

The inference `run_fullres_grid` deliberately does not do. Read
`docs/cluster_permutation.md` first -- this is that test, on a 10x finer
frequency axis, and four of its parameters mean something different here.

WHAT IS CLUSTERED, AND WHY IT IS NOT THE MIXED-MODEL z
-----------------------------------------------------
The statistic is the one-sample t, ACROSS SUBJECTS, of the UNPOOLED per-subject
OLS slope of ROI log power on pain score. Not the mixed model's Wald z, for two
independent reasons:

  - COST. A cluster null needs the whole map recomputed per permutation. The
    observed grid cost 415 CPU-minutes of `mixedlm` fitting; 10,000 permutations
    of that is ~2,900 CPU-days.
  - THERE IS NO CHEAP VALID NULL FOR IT. A fixed effect cannot be sign-flipped;
    the exchangeable unit is the epoch -> NRS pairing within a subject, and
    changing it requires a refit by construction.

So this is the TWO-STAGE question -- equal-weighted across subjects, channels
averaged within an ROI -- and it therefore under-uses the data relative to the
grid it supports. That is stated on the figure, because the alternative is a
reader assuming the outlines belong to the mixed-model betas they sit beside.

FOUR THINGS THAT DIFFER FROM THE 50-LOG-BIN TEST
------------------------------------------------
1. MINIMUM EXTENT IS SET IN HZ, NOT BINS. `min_cluster_bins=3` was ~a whole band
   on the log axis; three native bins is 1.5 Hz, which is a wiggle. The parameter
   here is `--min-cluster-hz` (default 4.0 -> 9 bins), converted through the
   manifest's df so it cannot drift from the axis.
2. THE NOTCH IS AN ADJACENCY CHOICE, `--notch-adjacency`, and it defaults to
   `bridge`. A physiological effect does not stop at 58 Hz and restart at 62; the
   notch is an instrumentation hole, and forcing a run to terminate there splits
   one broadband gamma effect into three clusters, each with less mass than the
   effect really has, which costs power in exactly the place the spectrum is
   continuous. So the notched bins are COMPACTED OUT and the axis closed over
   them, as `plot_pain_coef_heatmap` does on the log axis.
   THE COST IS REAL AND IS REPORTED PER CLUSTER: a bridged cluster asserts
   adjacency across 4.5 Hz that was never measured, and `spans_removed_gap` is
   True for every cluster whose span contains a removed bin -- 9 bins here versus
   1-2 on the log axis, so the flag matters more. `--notch-adjacency break` keeps
   the bins in the axis as INVALID cells so runs terminate at each harmonic
   (docs §5); nothing else about the test changes between the two.
3. BOTH ARMS ARE RUN. `raw` asks whether the pain slope is non-zero; `detrend`
   subtracts each subject x region map's mean over valid bins and asks whether
   the spectral SHAPE is non-flat (docs §9). The broadband low-frequency effect
   is large and continuous below ~5 Hz in nearly every region, so the raw arm is
   expected to return few enormous clusters spanning delta through beta --
   significant and unattributable to a band. They are reported side by side
   rather than one being chosen.
4. THERE IS NO `none`-BIN FLOOR (docs §6). A slope design has no baseline bin to
   test against itself, so the effect-size floor comes from the PREDICTOR-SHUFFLE
   null instead: the largest |mean slope| a cluster can reach when pain scores
   are randomly paired to epochs.

BOTH NULLS, ONE CORRECTION
--------------------------
  signflip  -- one +/-1 per subject per permutation, applied across all regions at
               once. Tests whether the subject-slope distribution is symmetric
               about zero. Free, because the slopes are already computed.
  shuffle   -- one permutation of the epoch -> pain-score pairing per subject per
               permutation, applied across all cells at once, slopes refitted.
               Tests whether PAIRING A PAIN SCORE TO AN EPOCH carries
               information, which is the scientific claim.

Both go through the SAME two-stage correction (family-wise across frequency
within a region, then BH over the regions that were tested), or the shuffle would
look more permissive merely for being less corrected.

    python -m ieeg_ehr.analysis.run_fullres_cluster

EXPLORATORY. Discovery cohort only. Nominations, not findings.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import cluster_permutation as cp
from ieeg_ehr.analysis import fullres_cells, mixed_model as mm
from ieeg_ehr.analysis import reference_run, view_tables
from ieeg_ehr.analysis.plot_fullres_grid import LOG_TICKS
from ieeg_ehr.analysis.run_fullres_grid import (CONFOUND_CAVEAT, ESTIMAND_CAVEAT,
                                                QUESTION, VIEW_SCHEME, resolve_cohort)
from ieeg_ehr.features import common
from ieeg_ehr.views import fullres_reader

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/run_fullres_cluster.py'
OUTPUT_TYPE = 'univariate_analysis'
RUN_NAME = 'fullres_cluster'

ARMS = ('raw', 'detrend')

DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. '
              'Two-stage cluster test on unpooled per-subject slopes, not on the '
              'mixed-model coefficients it is plotted beside.')

TWO_STAGE_NOTE = (
    'This test is EQUAL-WEIGHTED ACROSS SUBJECTS and averages channels within an '
    'ROI before fitting, so it does not use the precision weighting or the '
    'per-channel structure the mixed-model grid uses. It is the deliberately '
    'conservative counterpart of that map, not a test OF it: a cell can be large '
    'in the grid and uncovered here because one subject with many contacts drove '
    'it, which is information rather than a contradiction.')


# ============================================================================
# THE PREDICTOR-SHUFFLE NULL
# ============================================================================

def shuffle_null(per_subject, n_region_full, n_bin_full, select, test_cols, valid,
                 alpha, min_extent, n_perm, seed=0, min_subjects=1, transform=None):
    """(null_region, null_global) of max |cluster mass| under epoch relabelling.

    A local implementation rather than `cluster_permutation.predictor_shuffle_null`
    for two reasons, both load-bearing here:

      - SPEED. That function calls `pain_coef.coef_from_predictor`, whose
        per-column fallback is a Python loop over every cell with a missing epoch.
        At 10,479 cells it dominates everything. `fullres_cells.coef_from_blocks`
        groups columns by missingness pattern first and gets the same numbers from
        a handful of matmuls.
      - THE DETRENDED ARM NEEDS THE TRANSFORM APPLIED INSIDE THE LOOP. Detrending
        only the observed map would compare a detrended statistic against a
        non-detrended null, which is the same class of error as filtering cluster
        extent on the observation alone -- and it inflates significance the same
        way.

    EXCHANGEABILITY, unchanged from the shared implementation: ONE shuffle per
    subject per permutation, applied to every region and bin at once. An epoch is
    relabelled as a whole. Shuffling per cell would destroy the within-subject
    correlation across regions and give a null far too narrow.
    """
    n_region, n_bin = len(select), len(test_cols)
    subjects = list(per_subject)
    rng = np.random.default_rng(seed)
    null_region = np.empty((n_perm, n_region))
    null_global = np.empty(n_perm)
    t_crit_cache = {}

    for p in range(n_perm):
        maps = np.empty((len(subjects), n_region, n_bin))
        for i, sid in enumerate(subjects):
            x, blocks = per_subject[sid]
            # `blocks` are indexed against ALL regions, because that is how the
            # subject's matrix was built. The restriction to the regions with data
            # happens HERE, inside the permutation, so the null is built on exactly
            # the map the observed statistic was computed from -- restricting
            # afterwards would leave empty rows contributing zeros to the null.
            # Compacted to the TESTED axis inside the loop as well, so the null
            # clusters form over exactly the adjacency the observed ones did.
            full = fullres_cells.coef_from_blocks(
                x[rng.permutation(len(x))], blocks,
                n_region_full * n_bin_full).reshape(n_region_full, n_bin_full)
            maps[i] = full[np.ix_(select, test_cols)]
        if transform is not None:
            maps = transform(maps)

        t, n_map = cp.onesample_t(maps)
        key = n_map.tobytes()
        if key not in t_crit_cache:
            t_crit_cache[key] = cp.critical_t(n_map, alpha)
        t_crit = t_crit_cache[key]
        valid_p = (valid & (n_map >= max(min_subjects, 2)) & np.isfinite(t)
                   & np.isfinite(t_crit))
        null_region[p], null_global[p] = cp.max_mass_per_region(
            t, valid_p, t_crit, min_extent, n_region)
        if (p + 1) % 250 == 0:
            logger.info('  shuffle null: %d/%d permutations', p + 1, n_perm)
    return null_region, null_global


# ============================================================================
# CLUSTER RECORDS
# ============================================================================

def cluster_rows(result, regions, axis, arm, test_cols, floor=None):
    """Cluster records as table rows, in ORIGINAL bin indices and real Hz.

    `test_cols` maps a position on the TESTED axis back to a position on the full
    native axis. Under `--notch-adjacency bridge` the two differ, and conflating
    them would mislabel every cluster above 58 Hz by the number of notched bins
    below it.

    `spans_removed_gap` is True when the cluster's original-index span contains a
    bin that was compacted out: the outline renders as two boxes, it is ONE
    cluster, and it rests on an adjacency that was asserted rather than measured.
    """
    rows = []
    freqs = axis['freq_hz'].to_numpy()
    for c in result['clusters']:
        lo, hi = c['bin_lo'], c['bin_hi']
        full_lo, full_hi = int(test_cols[lo]), int(test_cols[hi])
        cells = result['mean_map'][c['region_idx'], lo:hi + 1]
        mean_abs = float(np.nanmean(np.abs(cells)))
        rows.append({
            'arm': arm, 'contrast': 'pain_slope',
            'region': regions[c['region_idx']], 'region_idx': c['region_idx'],
            'bin_lo_idx': int(axis.index[full_lo]),
            'bin_hi_idx': int(axis.index[full_hi]),
            'freq_lo_hz': float(freqs[full_lo]), 'freq_hi_hz': float(freqs[full_hi]),
            'spans_removed_gap': bool(cp.spans_removed_gap(full_lo, full_hi,
                                                           test_cols)),
            'n_bins': c['n_bins'],
            'width_hz': float(freqs[full_hi] - freqs[full_lo]),
            'sign': c['sign'], 'mass': c['mass'], 'peak_t': c['peak_t'],
            'mean_signed_slope': float(np.nanmean(cells)),
            'mean_abs_slope': mean_abs,
            'peak_abs_slope': float(np.nanmax(np.abs(cells))),
            'floor_ratio': (mean_abs / floor) if floor else np.nan,
            'n_subjects_min': int(result['n_map'][c['region_idx'], lo:hi + 1].min()),
            'p_within_region': c['p_within_region'], 'p_global': c['p_global'],
            'region_p_bh': c['region_p_bh'],
            'sig_two_stage': c['sig_two_stage'], 'sig_global': c['sig_global'],
        })
    return rows


def add_shuffle_p(rows, result, null_region, null_global, alpha, q, n_region):
    """The shuffle null's p, under the SAME two-stage correction as the sign-flip.

    Lifted from `plot_pain_coef_heatmap` unchanged in substance: each region
    contributes its minimum cluster p, BH runs over the regions that were actually
    tested, and a cluster needs both its region to survive and its own p < alpha.
    Without this the two nulls are not comparable.
    """
    for row, c in zip(rows, result['clusters']):
        m = abs(c['mass'])
        row['p_shuffle_within_region'] = cp.permutation_p(
            m, null_region[:, c['region_idx']])
        row['p_shuffle_global'] = cp.permutation_p(m, null_global)

    region_p = np.ones(n_region)
    for row, c in zip(rows, result['clusters']):
        i = c['region_idx']
        region_p[i] = min(region_p[i], row['p_shuffle_within_region'])

    tested = result['region_tested']
    rejected = np.zeros(n_region, dtype=bool)
    adj = np.ones(n_region)
    if tested.any():
        rej_t, adj_t = cp.bh_fdr(region_p[tested], q)
        rejected[tested] = rej_t
        adj[tested] = adj_t
    for row, c in zip(rows, result['clusters']):
        i = c['region_idx']
        row['region_p_bh_shuffle'] = float(adj[i])
        row['sig_two_stage_shuffle'] = bool(rejected[i]
                                            and row['p_shuffle_within_region'] < alpha)
    return rows


def sig_mask(rows, result, n_region, n_bin, column='sig_two_stage'):
    """(n_region, n_bin_tested) bool of cells in a cluster significant by `column`.

    On the TESTED axis; `expand_to_full` puts it back on the native one, where a
    bridged cluster becomes two boxes with the notch grey between them.
    """
    mask = np.zeros((n_region, n_bin), dtype=bool)
    for row, c in zip(rows, result['clusters']):
        if row.get(column):
            mask[c['region_idx'], c['bin_lo']:c['bin_hi'] + 1] = True
    return mask


# ============================================================================
# FIGURE
# ============================================================================

def _outline(ax, mask, edges_x, color='black', lw=1.4):
    """Outline a mask on a pcolormesh in DATA coordinates, as cell-edge segments.

    The same argument as `common.draw_mask_outline`, which cannot be used here
    because it assumes imshow's unit cells while this axis is real log frequency:
    `contour()` interpolates between cell CENTRES and would put the boundary half
    a bin inside the cluster, which on a log axis is a visibly different frequency
    and a claim about which frequencies were tested that the test cannot support.

    Top and bottom edges are ALWAYS drawn, even between two vertically adjacent
    True cells. Rows are ROIs and the test's adjacency is along frequency only, so
    every cluster is exactly one row tall; suppressing horizontal edges would
    merge two regions' clusters into one staircase that reads as a single entity
    the test never formed.
    """
    from matplotlib.collections import LineCollection

    segs = []
    n_row, n_col = mask.shape
    for i, j in zip(*np.nonzero(mask)):
        x0, x1, y0, y1 = edges_x[j], edges_x[j + 1], i, i + 1
        segs.append([(x0, y0), (x1, y0)])
        segs.append([(x0, y1), (x1, y1)])
        if j == 0 or not mask[i, j - 1]:
            segs.append([(x0, y0), (x0, y1)])
        if j == n_col - 1 or not mask[i, j + 1]:
            segs.append([(x1, y0), (x1, y1)])
    if segs:
        ax.add_collection(LineCollection(segs, colors=color, linewidths=lw,
                                         zorder=5))


def expand_to_full(mat, test_cols, n_bin_full, fill=np.nan):
    """A tested-axis map back on the full native axis, gaps left as `fill`.

    Figures are drawn on real log frequency, so a compacted map cannot be plotted
    directly: it would silently slide every value above 58 Hz leftward by the
    number of notched bins below it and mislabel the whole high-frequency half.
    """
    out = np.full((mat.shape[0], n_bin_full), fill, dtype=float)
    out[:, test_cols] = mat
    return out


def _mesh(ax, mat, edges_x, regions, cmap, vmin, vmax):
    im = ax.pcolormesh(edges_x, np.arange(len(regions) + 1),
                       np.ma.masked_invalid(mat), cmap=cmap, vmin=vmin, vmax=vmax,
                       shading='flat')
    ax.set_xscale('log')
    ax.set_xlim(edges_x[0], edges_x[-1])
    ax.set_xticks(LOG_TICKS)
    ax.set_xticklabels([str(t) for t in LOG_TICKS], fontsize=7)
    ax.minorticks_off()
    ax.set_yticks(np.arange(len(regions)) + 0.5)
    ax.set_ylim(len(regions), 0)
    ax.set_xlabel('frequency (Hz)', fontsize=8)
    common.add_band_boundary_lines(ax)
    return im


def figure(results, regions, axis, out_path, alpha, q, min_hz, n_perm,
           shuffle_n_perm):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    edges_x = np.append(axis['bin_low_hz'].to_numpy(),
                        axis['bin_high_hz'].to_numpy()[-1])
    div = plt.get_cmap('RdBu_r').copy()
    div.set_bad('0.85')

    fig, axes = plt.subplots(len(results), 2,
                             figsize=(15.5, 0.40 * len(regions) * len(results) + 4.2),
                             squeeze=False)

    n_full = len(axis)
    for r, arm in enumerate(results):
        res, rows = results[arm]['result'], results[arm]['rows']
        test_cols = results[arm]['test_cols']
        n_tested = len(test_cols)
        sig = expand_to_full(sig_mask(rows, res, len(regions), n_tested),
                             test_cols, n_full, fill=0).astype(bool)
        sig_sh = expand_to_full(
            sig_mask(rows, res, len(regions), n_tested,
                     column='sig_two_stage_shuffle'),
            test_cols, n_full, fill=0).astype(bool)

        mean_full = expand_to_full(res['mean_map'], test_cols, n_full)
        t_full = expand_to_full(res['t_map'], test_cols, n_full)
        slope_cap = float(np.nanpercentile(np.abs(mean_full), 99))
        t_cap = float(np.nanpercentile(np.abs(t_full), 99))
        for col, (mat, cap, label) in enumerate((
                (mean_full, slope_cap,
                 'EFFECT SIZE: group mean per-subject slope\n'
                 'd log10 power per pain point'),
                (t_full, t_cap,
                 'TEST STATISTIC: one-sample t across subjects\n'
                 '(what the clusters are formed on)'))):
            ax = axes[r][col]
            im = _mesh(ax, mat, edges_x, regions, div, -cap, cap)
            _outline(ax, sig, edges_x, color='black', lw=1.5)
            _outline(ax, sig_sh, edges_x, color='#1a9850', lw=0.9)
            ax.set_title(f'{arm.upper()} arm -- {label}  (clipped at |{cap:.3g}|)',
                         fontsize=9.5)
            ax.set_yticklabels(regions if col == 0 else [], fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.03,
                         pad=0.02).ax.tick_params(labelsize=7)

    n_sig = sum(sum(x['sig_two_stage'] for x in results[a]['rows']) for a in results)
    fig.suptitle('Cluster-based permutation test on the native 0.5 Hz map\n'
                 f'{n_sig} significant cluster(s) by the sign-flip null, '
                 'two-stage correction', fontsize=13)
    fig.tight_layout(rect=(0, 0.115, 1, 0.945))
    fig.text(0.01, 0.005,
             f'{cp.BOUNDARY_CAVEAT}\n'
             f'BLACK outlines = significant under the SIGN-FLIP null '
             f'({n_perm:,} permutations); GREEN = significant under the '
             f'PREDICTOR-SHUFFLE null ({shuffle_n_perm:,}), which asks whether '
             'pairing a pain score to an epoch carries information and is the '
             'stronger claim. Both use the same two-stage correction: family-wise '
             f'across frequency within a region at alpha={alpha}, then BH over the '
             f'regions tested at q={q}. A cluster forms from cells past their own '
             f'per-cell critical t (df = n-1, coverage varies by region) and must '
             f'span >={min_hz:g} Hz. The 60/120/180/240 Hz notch bins are REMOVED and '
             'the axis CLOSED over them, so a cluster may bridge a harmonic: a '
             'cluster drawn as two boxes with a grey gap is ONE cluster resting on '
             'an adjacency that was asserted, not measured (spans_removed_gap in '
             'clusters.parquet). The RAW arm '
             'tests whether the slope is non-zero; the DETREND arm removes each '
             "subject x region map's mean over frequency and tests whether the "
             'spectral SHAPE is non-flat -- read the raw arm knowing a broadband '
             'low-frequency offset can fill a whole row with one cluster. '
             f'{TWO_STAGE_NOTE}\n{DISCLAIMER}',
             fontsize=7, va='bottom', ha='left', color='0.35', wrap=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--view-dir', default=None)
    ap.add_argument('--reference-run', default=str(reference_run.CONTPAIN_HEATMAP))
    ap.add_argument('--mask-label', default=None)
    ap.add_argument('--cohort', choices=['reference', 'eligible-discovery'],
                    default='reference')
    ap.add_argument('--allow-cohort-drift', action='store_true')
    ap.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    ap.add_argument('--n-perm', type=int, default=10000,
                    help='Sign-flip permutations (default 10,000, as the reference '
                         'run used).')
    ap.add_argument('--shuffle-n-perm', type=int, default=2000,
                    help='Predictor-shuffle permutations (default 2,000, as the '
                         'reference run used). 0 skips that null.')
    ap.add_argument('--alpha', type=float, default=0.05,
                    help='Cluster-FORMING threshold, as a two-sided per-cell t.')
    ap.add_argument('--fdr-q', type=float, default=0.05)
    ap.add_argument('--min-cluster-hz', type=float, default=4.0,
                    help='A run must span at least this many Hz. Given in Hz rather '
                         'than bins on purpose: 3 bins was ~a band on the old log '
                         'axis and is 1.5 Hz here.')
    ap.add_argument('--min-subjects', type=int, default=None,
                    help='Coverage floor for a cell to enter a cluster. Default: '
                         'inherited from the reference run (8).')
    ap.add_argument('--notch-half-width-hz', type=float, default=None)
    ap.add_argument('--notch-adjacency', choices=['bridge', 'break'],
                    default='bridge',
                    help="'bridge' (default) removes the line-noise bins and CLOSES "
                         'the axis over them, so one broadband effect stays one '
                         'cluster instead of being split into three by an '
                         'instrumentation hole. Every cluster whose span contains a '
                         "removed bin is flagged spans_removed_gap. 'break' keeps "
                         'them in the axis as invalid cells so runs terminate at '
                         'each harmonic (docs/cluster_permutation.md §5).')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--question', default=QUESTION)
    ap.add_argument('--view-scheme', default=VIEW_SCHEME)
    ap.add_argument('--run-name', default=RUN_NAME)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    ref = reference_run.load(args.reference_run)
    ref.describe()
    epoch_minutes = ref.view_params.get('epoch_minutes')

    view_dir = fullres_cells.resolve_view_dir(
        args.view_dir, mask_label=args.mask_label or ref.view_params.get('mask_label'),
        max_excluded_frac=ref.view_params.get('max_excluded_frac'),
        epoch_minutes=epoch_minutes)
    paths, _, diagnostics, subjects, roi_by_subject, no_roi = resolve_cohort(
        ref, view_dir, cohort=args.cohort)
    ref.assert_cohort_matches(subjects, allow_drift=args.allow_cohort_drift
                              or args.cohort != 'reference')

    roi_scheme = ref.view_params.get('roi_scheme', 'roi_v2')
    regions = view_tables.roi_regions_for({'roi_scheme': roi_scheme})
    min_subjects = int(args.min_subjects if args.min_subjects is not None
                       else ref.criteria.get('min_subjects', mm.MIN_SUBJECTS))

    # ---------------- the axis: FULL, with the notch marked invalid -------------
    axis = fullres_reader.freq_table(epoch_minutes).set_index('freq_bin_index')
    axis['freq_hz'] = 0.5 * (axis['bin_low_hz'] + axis['bin_high_hz'])
    df_hz = float(fullres_reader.manifest(epoch_minutes)['params']['df_hz'])
    notched = [int(b) for b in fullres_reader.notch_freqs(
        half_width_hz=args.notch_half_width_hz, epoch_minutes=epoch_minutes)]
    min_extent = max(2, int(round(args.min_cluster_hz / df_hz)))

    # `test_cols` are the positions on the native axis that enter the test, in
    # order. ONE mapping drives everything downstream -- the observed statistic,
    # both nulls, the cluster labels and the figure -- so the two adjacency modes
    # cannot disagree about which frequency a cluster sits at.
    positions = {int(b): i for i, b in enumerate(axis.index)}
    notch_pos = [positions[b] for b in notched]
    if args.notch_adjacency == 'bridge':
        test_cols = np.array([i for i in range(len(axis)) if i not in set(notch_pos)])
        notch_valid = True
    else:
        test_cols = np.arange(len(axis))
        notch_valid = False
    logger.info('axis: %d native bins at %.2f Hz; %d notch bins %s; tested axis '
                '%d bins; min extent %.1f Hz = %d bins (%.1f Hz of axis width)',
                len(axis), df_hz, len(notched),
                'REMOVED and the axis closed over them (clusters may BRIDGE a '
                'harmonic, and are flagged when they do)' if notch_valid
                else 'kept as INVALID cells (runs terminate at a harmonic)',
                len(test_cols), args.min_cluster_hz, min_extent,
                min_extent * df_hz)

    # ---------------- per-subject slope maps ------------------------------------
    t0 = time.time()
    freq_cols = fullres_reader.freq_columns(epoch_minutes)
    coef, subj_ids, per_subject, coverage = fullres_cells.subject_coef_matrix(
        paths, subjects, roi_by_subject, regions, freq_cols)
    logger.info('built %d subject slope map(s) in %.0fs (%d missingness pattern(s) '
                'median per subject)', len(subj_ids), time.time() - t0,
                int(coverage['n_missingness_patterns'].median()))

    present = [i for i, _ in enumerate(regions)
               if np.isfinite(coef[:, i, :]).any()]
    regions_present = [regions[i] for i in present]
    coef = coef[:, present, :]
    logger.info('%d region(s) with data: %s', len(regions_present), regions_present)

    coef = coef[:, :, test_cols]
    valid = np.ones((len(regions_present), len(test_cols)), dtype=bool)
    if not notch_valid:
        valid[:, notch_pos] = False

    # ---------------- the test, per arm ----------------------------------------
    results, all_rows = {}, []
    for arm in args.arms:
        logger.info('=' * 70)
        logger.info('ARM %s', arm.upper())
        x_arm = (cp.detrend_over_frequency(coef, valid) if arm == 'detrend'
                 else coef)
        transform = ((lambda m: cp.detrend_over_frequency(m, valid))
                     if arm == 'detrend' else None)

        t0 = time.time()
        result = cp.cluster_test(x_arm, valid=valid, alpha=args.alpha,
                                 min_extent=min_extent, n_perm=args.n_perm,
                                 q=args.fdr_q, seed=args.seed,
                                 min_subjects=min_subjects)
        logger.info('sign-flip null: %d permutations in %.0fs -> %d cluster(s)',
                    args.n_perm, time.time() - t0, len(result['clusters']))

        rows = cluster_rows(result, regions_present, axis, arm, test_cols)
        if args.shuffle_n_perm:
            t0 = time.time()
            null_region, null_global = shuffle_null(
                per_subject, len(regions), len(axis), present, test_cols, valid,
                args.alpha, min_extent, args.shuffle_n_perm, seed=args.seed,
                min_subjects=min_subjects, transform=transform)
            rows = add_shuffle_p(rows, result, null_region, null_global,
                                 args.alpha, args.fdr_q, len(regions_present))
            logger.info('shuffle null: %d permutations in %.0fs', args.shuffle_n_perm,
                        time.time() - t0)
        else:
            for row in rows:
                row.update({'p_shuffle_within_region': np.nan,
                            'p_shuffle_global': np.nan,
                            'region_p_bh_shuffle': np.nan,
                            'sig_two_stage_shuffle': False})

        n_sig = sum(r['sig_two_stage'] for r in rows)
        n_sh = sum(r['sig_two_stage_shuffle'] for r in rows)
        logger.info('%s: %d cluster(s), %d significant (sign-flip, two-stage), '
                    '%d significant (shuffle, same correction)',
                    arm, len(rows), n_sig, n_sh)
        for r in sorted(rows, key=lambda d: -abs(d['mass']))[:8]:
            logger.info('  %-18s %6.1f-%6.1f Hz  %+d  mass %9.1f  peak t %+5.2f  '
                        'mean slope %+.5f  p_wr %.4f  p_shuf %.4f%s',
                        r['region'], r['freq_lo_hz'], r['freq_hi_hz'], r['sign'],
                        r['mass'], r['peak_t'], r['mean_signed_slope'],
                        r['p_within_region'], r['p_shuffle_within_region'],
                        '  SIG' if r['sig_two_stage'] else '')
        n_gap = sum(r['spans_removed_gap'] for r in rows)
        if n_gap:
            logger.info('%s: %d cluster(s) BRIDGE a line-noise gap -- one cluster '
                        'drawn as two boxes, resting on an adjacency that was '
                        'asserted rather than measured', arm, n_gap)
        results[arm] = {'result': result, 'rows': rows, 'test_cols': test_cols}
        all_rows.extend(rows)

    # ---------------- outputs ---------------------------------------------------
    run_dir = config.analysis_run_dir(question=args.question,
                                      output_type=OUTPUT_TYPE,
                                      view_scheme=args.view_scheme,
                                      run_name=args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info('run dir: %s', run_dir)

    params = {'statistic': 'one-sample t of unpooled per-subject OLS slopes',
              'channel_aggregation': 'mean of logs within ROI',
              'alpha': args.alpha, 'fdr_q': args.fdr_q,
              'min_cluster_hz': args.min_cluster_hz, 'min_extent_bins': min_extent,
              'n_perm_signflip': args.n_perm,
              'n_perm_shuffle': args.shuffle_n_perm,
              'min_subjects': min_subjects, 'seed': args.seed,
              'notched_bins_removed': notched,
              'notch_adjacency': args.notch_adjacency,
              'n_bins_tested': int(len(test_cols)), 'arms': list(args.arms),
              'roi_scheme': roi_scheme, 'epoch_minutes': epoch_minutes}

    io.write_table(pd.DataFrame(all_rows), run_dir / 'clusters.parquet',
                   script=SCRIPT, params=params,
                   parents=[str(Path(args.reference_run) / 'provenance.json'),
                            str(view_dir)],
                   subjects=sorted(subjects),
                   extra={'interpretation_caveat': cp.BOUNDARY_CAVEAT,
                          'two_stage_caveat': TWO_STAGE_NOTE,
                          'gap_caveat':
                              'With notch_adjacency=bridge the line-noise bins were '
                              'REMOVED and the frequency axis closed over them, so a '
                              'cluster may assert adjacency across frequencies that '
                              'were never measured. spans_removed_gap flags every '
                              'such cluster; it is ONE cluster even though the '
                              'outline renders as two boxes.',
                          'status': DISCLAIMER})

    maps = []
    for arm in results:
        res = results[arm]['result']
        tested_index = [int(axis.index[i]) for i in test_cols]
        maps.append(pd.DataFrame({
            'arm': arm,
            'region': np.repeat(regions_present, len(test_cols)),
            'freq_bin_index': np.tile(tested_index, len(regions_present)),
            'freq_hz': np.tile(axis['freq_hz'].to_numpy()[test_cols],
                               len(regions_present)),
            'mean_slope': res['mean_map'].ravel(), 't': res['t_map'].ravel(),
            'n_subjects': res['n_map'].ravel(),
            'valid': res['valid_map'].ravel()}))
    io.write_table(pd.concat(maps, ignore_index=True), run_dir / 'group_map.parquet',
                   script=SCRIPT, params=params, subjects=sorted(subjects))
    io.write_table(coverage, run_dir / 'subject_coverage.parquet', script=SCRIPT,
                   params=params)

    fig_path = run_dir / 'fig_fullres_clusters.png'
    figure(results, regions_present, axis, fig_path, args.alpha, args.fdr_q,
           args.min_cluster_hz, args.n_perm, args.shuffle_n_perm)
    logger.info('wrote %s', fig_path)

    write_methods(run_dir, results, regions_present, axis, params, no_roi)
    io.write_run_provenance(
        run_dir, script=SCRIPT, params={**params, 'view_dir': str(view_dir),
                                        'view_params': ref.view_params,
                                        'criteria': ref.criteria},
        parents=[str(Path(args.reference_run) / 'provenance.json'), str(view_dir)],
        subjects=sorted(subjects),
        extra={'status': DISCLAIMER, 'interpretation_caveat': cp.BOUNDARY_CAVEAT,
               'two_stage_caveat': TWO_STAGE_NOTE, 'estimand': ESTIMAND_CAVEAT,
               'mask_content': CONFOUND_CAVEAT,
               'subjects_without_roi': sorted(no_roi)})
    io.log_analysis('cluster permutation on the native 0.5 Hz region x frequency '
                    'map, raw + detrended arms, sign-flip + predictor-shuffle '
                    'nulls (EXPLORATORY)', run_dir)
    print(run_dir)


def write_methods(run_dir, results, regions, axis, params, no_roi):
    n_notch = len(params.get('notched_bins_removed', []))
    lines = [f"""# Cluster permutation on the native 0.5 Hz region x frequency map

{DISCLAIMER}

Read `docs/cluster_permutation.md` for the test's design. This run differs from
the 50-log-bin version it inherits in four places, all recorded in
`provenance.json`.

## What is clustered

The bin-level statistic is a one-sample t ACROSS SUBJECTS of the unpooled
per-subject OLS slope of ROI log power on pain score, with df from that cell's own
n (coverage varies by region). NOT the mixed-model Wald z: a cluster null needs
the whole map per permutation, the grid cost 415 CPU-minutes of mixedlm fitting,
and a fixed effect cannot be sign-flipped because the exchangeable unit is the
epoch -> NRS pairing.

{TWO_STAGE_NOTE}

Channels are averaged within an ROI as a MEAN OF LOGS rather than
linear-then-log, so the ROI slope is an average of per-contact log slopes -- the
quantity the mixed model's fixed effect approximates. This differs from the
2026-08-08 two-stage reference run and is one reason magnitudes are not
identical to it.

## Clustering

Adjacency is along FREQUENCY ONLY, within a region; region rows are not
neighbours. Cluster statistic is mass (sum of t), positive and negative runs
formed separately. A run must span at least {params['min_cluster_hz']:g} Hz
({params['min_extent_bins']} native bins) -- specified in Hz because three bins
was ~a band on the log axis and is 1.5 Hz here. The extent filter is enforced
inside the permutation loop as well as on the observed map.

The {n_notch} line-noise bins are handled by `notch_adjacency=
{params['notch_adjacency']}`. Under `bridge` they are REMOVED and the axis closed
over them ({params['n_bins_tested']} bins tested), so a broadband effect stays one
cluster instead of being cut into three by an instrumentation hole -- at the cost
of asserting adjacency across 4.5 Hz per harmonic that was never measured. Every
cluster whose span contains a removed bin carries `spans_removed_gap = True`,
renders as two boxes on the figure, and is ONE cluster. Under `break` the bins stay
in the axis as invalid cells and runs terminate at each harmonic
(docs/cluster_permutation.md §5).

## The two nulls

`signflip` ({params['n_perm_signflip']:,} permutations): one +/-1 per subject per
permutation, applied across all regions at once. Tests symmetry about zero.

`shuffle` ({params['n_perm_shuffle']:,} permutations): one permutation of the
epoch -> pain-score pairing per subject per permutation, applied across all cells
at once, slopes refitted. Tests whether pairing a pain score to an epoch carries
information. This null is also the EFFECT-SIZE FLOOR that the `none` control bin
provides in the baseline-referenced design; a slope design has no such bin.

Both use the same two-stage correction: family-wise across frequency within a
region at alpha={params['alpha']}, then BH across the regions tested at
q={params['fdr_q']}. Regions with data but no cluster stay in the BH family.

## Arms

`raw` tests whether the pain slope is non-zero. `detrend` subtracts each
subject x region map's mean over valid bins first, which changes the hypothesis to
"the spectral shape is not flat"; it exists because the broadband low-frequency
effect otherwise absorbs into one enormous cluster per region that spans delta
through beta and cannot be attributed to a band.

## Results
"""]
    for arm, payload in results.items():
        rows = payload['rows']
        sig = [r for r in rows if r['sig_two_stage']]
        sh = [r for r in rows if r['sig_two_stage_shuffle']]
        lines.append(
            f"\n### {arm}\n\n{len(rows)} cluster(s) formed; {len(sig)} significant "
            f"under the sign-flip null and {len(sh)} under the shuffle null "
            f"(same correction).\n")
        if sig:
            lines.append('| region | Hz | sign | width | peak t | mean slope | '
                         'p (within region) | p (shuffle) |\n|---|---|---|---|---|'
                         '---|---|---|\n')
            for r in sorted(sig, key=lambda d: -abs(d['mass'])):
                lines.append(
                    '| {} | {:.1f}-{:.1f} | {} | {:.1f} Hz | {:+.2f} | {:+.5f} '
                    '| {:.4f} | {:.4f} |\n'.format(
                        r['region'], r['freq_lo_hz'], r['freq_hi_hz'],
                        '+' if r['sign'] > 0 else '-', r['width_hz'],
                        r['peak_t'], r['mean_signed_slope'],
                        r['p_within_region'], r['p_shuffle_within_region']))

    lines.append(f"""
## The reporting constraint

> {cp.BOUNDARY_CAVEAT}

At 0.5 Hz the temptation to read a cluster's edges is far stronger than on log
bins and the licence is no greater: the extent is where the statistic happened to
cross an arbitrary threshold in this sample, and it is not itself tested.

## Known limitations

- {ESTIMAND_CAVEAT}
- {CONFOUND_CAVEAT}
- Subjects contributing no ROI-labelled channel: {sorted(no_roi) or 'none'}.
- The test is equal-weighted across subjects, so a region's power depends on how
  many subjects have any contact there, not on how many contacts they have.
""")
    (run_dir / 'METHODS.md').write_text(''.join(lines))


if __name__ == '__main__':
    main()

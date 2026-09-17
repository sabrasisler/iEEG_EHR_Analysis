"""Mixed-effects models of BAND POWER on pain score, one fit per region x band.

    analysis/pain/bandpower/mixed_effects/<view_scheme>/<run>_<timestamp>/

A deliberate new level-2 question (`bandpower`), opened because this asks
something the `psd_physiology` grid cannot: not "which frequency" but "which
BAND", with one estimate per band that carries its own standard error. 20 regions
x 6 bands = 120 fits, against the grid's 9,723 cells.

THE REGION SET IS `roi_v2_ofc`, NOT `roi_v2`: mOFC and lOFC are fused into one
OFC (2026-09-16). roi_v2 measured 94 contacts in mOFC against 191 in lOFC and 504
in Insula, and the pain literature usually reports orbitofrontal cortex
undivided. Pass `--roi-scheme roi_v2` to keep them split; the scheme's full
CONTENTS go into provenance either way, so a run is never ambiguous about which
region set produced it.

WHY THIS EXISTS RATHER THAN AVERAGING THE GRID'S BETAS
------------------------------------------------------
Averaging the per-frequency betas inside a band gives a defensible POINT
ESTIMATE -- measured on 6 region x band cells, it agrees with a refit to under 1%
when the aggregation is a mean of logs -- but it cannot give an interval. The SE
of a mean of 463 estimates depends on their cross-bin covariance, and every cell
of the grid was fitted independently, so that covariance was never computed.
Averaging the per-cell SEs is not a substitute: measured against a refit it ran
from 5% to 67% too large. A band-level p-value therefore has to come from a
band-level fit.

The aggregation rule is also not free. `log10(mean(10**x))` (registry AXIS 5,
what a band POWER actually is) gave betas 12-31% LARGER in magnitude than the
average of the per-bin betas on those same 6 cells, and moved one from z=1.74 to
z=2.07. So the band value is built with `axes.aggregate_bands`, the one sanctioned
implementation, and never by averaging betas.

WHAT IS FITTED
--------------
One row = ONE CHANNEL x ONE 5-minute pre-report epoch. The outcome is that
channel's log10 band power in that epoch: the bins whose geometric centre falls
in the band, line-noise bins EXCLUDED, combined linear-then-log. A coefficient
reads as d log10(band power) per pain point.

    log10_power ~ NRS_within + NRS_submean
                  + (NRS_within || subject) + (1 | subject:channel)

Identical to `run_mixed_model_grid` -- same cohort, same ROI scheme, same
coverage floor, same REML fit. Only the outcome's frequency support changed. No
normalization and no baseline: contact amplitude differences are multiplicative,
so in log space they are additive and the channel random intercept absorbs them
exactly.

BAND SETS. The default is `paper_bands_6_hg200`: the target paper's six bands
with high_gamma extended to 70-200 Hz, because 170 Hz was their ceiling and not
this dataset's -- the native-resolution map shows Thalamus high-frequency
structure running past it. The published `paper_bands_6` stays selectable and
UNEDITED, since the decoding replication's edges must remain the paper's. Both
keep the paper's GAP AT 12-15 HZ, reproduced rather than closed. Bands that cross
a 60 Hz harmonic have the notched bins removed before aggregation; `band_caveat()`
states each set's own gaps and crossings, and it is computed rather than written
down so it cannot misdescribe a variant.

INFERENCE, IN TWO STAGES BECAUSE THEY COST DIFFERENT AMOUNTS
------------------------------------------------------------
    fit + collect   Wald z, BH over the 120-cell family. Seconds. Assumes
                    z ~ N(0,1) under the null, which needs the SE to be right,
                    which needs the random-effect structure approximately right
                    and enough subjects for the asymptotics.
    perm + collect  Within-subject shuffle of the epoch -> pain-score pairing,
                    refitting each cell. ~1.6 s a fit, so 120 cells x 1,000
                    shuffles is ~53 CPU-hours -- affordable here precisely
                    because there are 120 cells and not 9,723. Assumes only
                    exchangeability of that pairing.

NEITHER FIXES THE CONFOUND THEY SHARE: epochs within a subject are exchangeable
only if nothing else drifts with pain over a hospital stay. No time-of-day term,
no time-since-admission term, and the QC mask is signal quality only -- opioid
administrations and post-ictal periods are not excluded. A permutation test
replaces the reference distribution, not the design.

    python -m ieeg_ehr.analysis.run_bandpower_mixed --stage fit
    python -m ieeg_ehr.analysis.run_bandpower_mixed --stage collect --run-dir R

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
from ieeg_ehr.analysis.plot_mixed_model_subject_lines import epoch_level, subject_slopes
from ieeg_ehr.analysis.run_fullres_grid import (CONFOUND_CAVEAT, resolve_cohort)
from ieeg_ehr.views import axes, fullres_reader

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/run_bandpower_mixed.py'

QUESTION = 'bandpower'
OUTPUT_TYPE = 'mixed_effects'
VIEW_SCHEME = 'paperbands6-roiv2ofc'
RUN_NAME = 'paperbands6_mixedlm'

FDR_Q = 0.05

#: `paper_bands_6_hg200` is the DEFAULT rather than the published
#: `paper_bands_6`: 170 Hz was the target paper's ceiling, not this dataset's,
#: and the native-resolution map shows Thalamus high-frequency structure running
#: past it. The published set stays selectable, unedited, because the decoding
#: replication's edges must remain the paper's. The chosen set is in the run's
#: folder name, so a path never misstates which edges produced it.
BAND_SETS = {'paper_bands_6_hg200': config.PAPER_BANDS_6_HG200_HZ,
             'paper_bands_6': config.PAPER_BANDS_6_HZ,
             'canonical': config.CANONICAL_BANDS_HZ}

#: Level-4 folder per band set, so the path says which edges were used.
VIEW_SCHEMES = {'paper_bands_6_hg200': 'paperbands6hg200-roiv2ofc',
                'paper_bands_6': 'paperbands6-roiv2ofc',
                'canonical': 'canonicalbands-roiv2ofc'}

DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. '
              'Not confirmed out of sample.')

def band_caveat(band_set):
    """The band set's own gaps and harmonic crossings, in words.

    Computed rather than written down, because the two paper variants differ in
    exactly the property a reader needs (what the top of high_gamma is), and a
    fixed string would misdescribe one of them.
    """
    bands = BAND_SETS[band_set]
    top = max(hi for _, hi in bands.values())
    parts = []
    if band_set.startswith('paper_bands_6'):
        parts.append(
            'The band edges are the target paper\'s (Huang et al. 2025), which '
            'leaves a GAP AT 12-15 HZ -- alpha ends at 12 and beta starts at 15 '
            '-- reproduced rather than closed.')
    if band_set == 'paper_bands_6':
        parts.append(
            'Nothing covers above 170 Hz, so the pain-related high-frequency '
            'increase the native-resolution map shows in Thalamus above ~170 Hz '
            'has no band to appear in. `--band-set paper_bands_6_hg200` extends '
            'high_gamma to 200 Hz for that reason.')
    elif band_set == 'paper_bands_6_hg200':
        parts.append(
            'high_gamma is EXTENDED to 70-200 Hz, not the paper\'s 70-170: 170 Hz '
            'was their ceiling, not this dataset\'s, and the native-resolution map '
            'shows Thalamus structure running past it. The published set is '
            'unedited and still selectable as `--band-set paper_bands_6`. Nothing '
            f'covers {top:g}-250 Hz.')
    parts.append(
        'Bands that CROSS a 60 Hz harmonic have the notched bins removed before '
        'aggregation -- mandatory, not optional: without it gamma would absorb '
        'the 58-62 Hz residue and high_gamma the 118-122 and 178-182 Hz ones.')
    return ' '.join(parts)


WALD_CAVEAT = (
    'p and p_bh are PARAMETRIC Wald. They assume z ~ N(0,1) under the null, which '
    'requires the standard error to be correct and therefore the random-effect '
    'structure to be approximately right with enough subjects for the '
    'asymptotics. The Phase-1 pilot measured the null Wald z SD at ~1.03 on 19 '
    'cells -- near-calibrated, slightly anticonservative. `--stage perm` replaces '
    'them with a permutation p that assumes only exchangeability of the '
    'epoch -> pain-score pairing.')


# ============================================================================
# THE BAND OUTCOME
# ============================================================================

def band_table(band_set, notch_half_width_hz=None, epoch_minutes=None):
    """(bin_table with the notch REMOVED, band edge dict).

    The notch is removed from the table handed to `axes.aggregate_bands`, which
    is what excludes those frequencies from every band: that function does not
    drop flagged bins itself, it relies on them being absent or already NaN.
    Mandatory for paper_bands_6, whose gamma and high_gamma cross 60 and 120 Hz.
    """
    table = fullres_reader.freq_table(epoch_minutes).set_index('freq_bin_index')
    notched = [int(b) for b in fullres_reader.notch_freqs(
        half_width_hz=notch_half_width_hz, epoch_minutes=epoch_minutes)]
    kept = table.drop(index=[b for b in notched if b in table.index])
    bands = BAND_SETS[band_set]
    centres = np.sqrt(kept['bin_low_hz'].to_numpy() * kept['bin_high_hz'].to_numpy())
    for name, (lo, hi) in bands.items():
        n = int(((centres >= lo) & (centres < hi)).sum())
        logger.info('  %-12s %3g-%3g Hz : %3d native bins', name, lo, hi, n)
    return kept, bands, notched


def aggregate(values, kept_table, bands):
    """(n_rows, n_bins) log10 power -> (n_rows, n_bands) log10 BAND power.

    Straight through `axes.aggregate_bands` with `is_difference=False,
    domain='log'`, i.e. log10(mean(10**x)) -- linear-then-log, because raw log
    power does not average arithmetically (a mean of logs is a geometric mean).
    On this axis every bin is the same width, so uniform and width weighting are
    numerically identical and the default stands.
    """
    return axes.aggregate_bands(np.asarray(values, dtype=np.float64), kept_table,
                                bands=bands, is_difference=False, domain='log',
                                weighting='uniform')


# ============================================================================
# STAGE: fit
# ============================================================================

def fit_one_cell(df, meta):
    """(record, per-subject unpooled slopes) for one region x band cell."""
    ok, reason = mm.cell_is_fittable(df)
    if not ok:
        rec = mm.failed_record(meta['region'], meta['band_index'], meta['band_lo_hz'],
                               meta['band_hi_hz'], reason, df=df)
        rec.update({k: meta[k] for k in ('band', 'cell_index')})
        return rec, None, None

    t0 = time.time()
    try:
        res, warn = mm.fit_cell(df, mm.VC_FULL)
    except mm.CellFitError as exc:
        rec = mm.failed_record(meta['region'], meta['band_index'], meta['band_lo_hz'],
                               meta['band_hi_hz'], f'full: {exc}', df=df,
                               fit_seconds=time.time() - t0)
        rec.update({k: meta[k] for k in ('band', 'cell_index')})
        return rec, None, None

    # The reduced model IS fitted here, unlike the 9,723-cell grid. At 120 cells
    # the second fit is affordable, and the heterogeneity LRT is a question worth
    # answering per band: "subjects respond, but not in a consistent direction"
    # is a different claim from "no effect", and only the LRT separates them.
    try:
        res_red, warn_red = mm.fit_cell(df, mm.VC_REDUCED)
    except mm.CellFitError as exc:
        logger.warning('%s %s reduced model failed: %s', meta['region'], meta['band'],
                       exc)
        res_red, warn_red = None, [f'reduced failed: {exc}']

    rec = mm.cell_record(res, res_red, df, region=meta['region'],
                         freq_bin_index=meta['band_index'],
                         bin_low_hz=meta['band_lo_hz'],
                         bin_high_hz=meta['band_hi_hz'],
                         fit_seconds=time.time() - t0,
                         warnings_full=warn, warnings_reduced=warn_red)
    rec.update({k: meta[k] for k in ('band', 'cell_index')})
    rec['nrs_within_var'] = float(np.var(df['NRS_within'].to_numpy(), ddof=0))

    slopes = subject_slopes(epoch_level(df))
    slopes['region'] = meta['region']
    slopes['band'] = meta['band']

    # BLUPs as well as the unpooled slopes. They are NOT interchangeable -- partial
    # pooling drags every subject toward the group, which is exactly why the
    # caterpillar plot wants both: the gap between them IS the shrinkage, and a
    # cell whose spread only exists before pooling is telling you something.
    blups = pd.DataFrame(mm.blup_rows(res, df, region=meta['region'],
                                      freq_bin_index=meta['band_index']))
    if len(blups):
        blups['band'] = meta['band']
    logger.info('%-18s %-11s | n=%2d subj %4d chan %6d rows | beta %+.5f z %+5.2f '
                'p %.4g | LRT p %.4g | %.1fs',
                meta['region'], meta['band'], rec['n_subjects'], rec['n_channels'],
                rec['n_rows'], rec['beta_nrs_within'], rec['z'], rec['p'],
                rec['p_lrt_mixture'], rec['fit_seconds'])
    return rec, slopes, blups


def stage_fit(args):
    ref = reference_run.load(args.reference_run)
    ref.describe()
    epoch_minutes = ref.view_params.get('epoch_minutes')

    view_dir = fullres_cells.resolve_view_dir(
        args.view_dir, mask_label=args.mask_label or ref.view_params.get('mask_label'),
        max_excluded_frac=ref.view_params.get('max_excluded_frac'),
        epoch_minutes=epoch_minutes)
    logger.info('epoch-mean full-res view: %s', view_dir)

    paths, scores, diagnostics, subjects, roi_by_subject, no_roi = resolve_cohort(
        ref, view_dir, cohort=args.cohort, roi_scheme=args.roi_scheme)
    ref.assert_cohort_matches(subjects, allow_drift=args.allow_cohort_drift
                              or args.cohort != 'reference')

    roi_scheme = args.roi_scheme or ref.view_params.get('roi_scheme', 'roi_v2')
    regions = view_tables.roi_regions_for({'roi_scheme': roi_scheme})
    if args.exclude_regions:
        unknown = [r for r in args.exclude_regions if r not in regions]
        if unknown:
            raise SystemExit(
                f'--exclude-regions {unknown} are not regions of {roi_scheme!r}. '
                f'Known: {regions}. Refusing rather than silently excluding '
                'nothing, which would look like the exclusion worked.')
        regions = [r for r in regions if r not in set(args.exclude_regions)]
        logger.warning('EXCLUDED from this run entirely (not fitted, not in the BH '
                       'family, not plotted): %s -- %s', args.exclude_regions,
                       args.exclude_reason or 'no reason recorded')
    logger.info('ROI scheme %r -> %d region(s): %s', roi_scheme, len(regions), regions)
    logger.info('band set %r:', args.band_set)
    kept, bands, notched = band_table(args.band_set, args.notch_half_width_hz,
                                      epoch_minutes)
    band_names = list(bands)

    run_dir = (Path(args.run_dir) if args.run_dir else
               config.analysis_run_dir(question=args.question,
                                       output_type=OUTPUT_TYPE,
                                       view_scheme=args.view_scheme,
                                       run_name=args.run_name))
    (run_dir / 'cells').mkdir(parents=True, exist_ok=True)
    (run_dir / 'frames').mkdir(parents=True, exist_ok=True)

    todo = (regions if args.region_index is None
            else [regions[args.region_index]])
    if args.region_index is not None and not 0 <= args.region_index < len(regions):
        raise SystemExit(f'--region-index {args.region_index} outside '
                         f'0..{len(regions) - 1}')

    records, slope_parts, blup_parts = [], [], []
    for region in todo:
        ri = regions.index(region)
        t0 = time.time()
        index, values, stats = fullres_cells.load_region_matrix(
            paths, subjects, region, roi_by_subject, list(kept.index),
            epoch_minutes=epoch_minutes)
        if not len(index):
            logger.warning('%s: no rows after the ROI join, skipped', region)
            continue
        band_values, names = aggregate(values, kept, bands)
        logger.info('%s: %d rows x %d bins -> %d band(s) in %.1fs (%d subject files)',
                    region, len(index), values.shape[1], len(names),
                    time.time() - t0, stats['n_files'])

        for bi, band in enumerate(names):
            lo, hi = bands[band]
            meta = {'region': region, 'band': band, 'band_index': band_names.index(band),
                    'band_lo_hz': float(lo), 'band_hi_hz': float(hi),
                    'cell_index': ri * len(band_names) + band_names.index(band)}
            frame = pd.DataFrame({
                'subject_id': index['subject_id'].to_numpy(),
                'channel': index['channel'].to_numpy(),
                'epoch_id': index['epoch_id'].to_numpy(),
                'pain_score': index['pain_score'].to_numpy(),
                'value': band_values[:, bi]})
            df = mm.build_cell_frame(frame, region=region,
                                     freq_bin_index=meta['band_index'])
            rec, slopes, blups = fit_one_cell(df, meta)
            records.append(rec)
            if slopes is not None:
                slope_parts.append(slopes)
            if blups is not None and len(blups):
                blup_parts.append(blups)
            # The frame, so `--stage perm` refits without re-reading the view.
            if len(df):
                io.write_table(df, run_dir / 'frames' /
                               f"cell_{meta['cell_index']:03d}.parquet",
                               params={'region': region, 'band': band},
                               script=SCRIPT)

    tag = 'all' if args.region_index is None else f'{args.region_index:03d}'
    io.write_table(pd.DataFrame(records), run_dir / 'cells' / f'region_{tag}.parquet',
                   params={'band_set': args.band_set, 'n_cells': len(records)},
                   script=SCRIPT)
    io.write_table(pd.concat(slope_parts, ignore_index=True) if slope_parts
                   else pd.DataFrame(columns=['subject', 'slope', 'region', 'band']),
                   run_dir / 'cells' / f'slopes_{tag}.parquet',
                   params={'source': 'unpooled per-subject OLS, not BLUPs'},
                   script=SCRIPT)
    io.write_table(pd.concat(blup_parts, ignore_index=True) if blup_parts
                   else pd.DataFrame(columns=['subject', 'region', 'band']),
                   run_dir / 'cells' / f'blups_{tag}.parquet',
                   params={'source': 'model BLUPs -- SHRUNK toward the group, keep '
                                     'beside the unpooled slopes, never instead'},
                   script=SCRIPT)

    if args.region_index is None:
        io.write_table(inventory(scores, diagnostics), run_dir / 'inventory_subjects.parquet',
                       script=SCRIPT)
        io.write_run_provenance(
            run_dir, script=SCRIPT,
            params={'stage': 'fit', 'band_set': args.band_set,
                    'bands': {k: list(v) for k, v in bands.items()},
                    'view_dir': str(view_dir), 'view_params': ref.view_params,
                    'criteria': ref.criteria, 'roi_scheme': roi_scheme,
                    'roi_scheme_contents': __import__(
                        'ieeg_ehr.config.roi_schemes', fromlist=['x']
                    ).scheme_provenance(roi_scheme),
                    'cohort': args.cohort, 'epoch_minutes': epoch_minutes,
                    'notched_bins_excluded': notched,
                    'excluded_regions': list(args.exclude_regions),
                    'excluded_regions_reason': args.exclude_reason,
                    'aggregation': 'linear_then_log via axes.aggregate_bands',
                    'n_cells': len(records)},
            parents=[str(Path(args.reference_run) / 'provenance.json'), str(view_dir)],
            subjects=sorted(subjects),
            extra={'status': DISCLAIMER,
                   'band_caveat': band_caveat(args.band_set),
                   'inference_caveat': WALD_CAVEAT,
                   'mask_content': CONFOUND_CAVEAT,
                   'subjects_without_roi': sorted(no_roi)})
    print(run_dir)
    return run_dir


def inventory(scores, diagnostics):
    g = scores.groupby('subject_id')['pain_score']
    inv = pd.DataFrame({'n_reports': g.size(), 'nrs_mean': g.mean(),
                        'nrs_sd': g.std(ddof=1), 'nrs_min': g.min(),
                        'nrs_max': g.max(), 'n_distinct': g.nunique()}).reset_index()
    return inv.merge(diagnostics[['subject_id', 'included', 'excluded_because']],
                     on='subject_id', how='left')


# ============================================================================
# STAGE: perm  (one array task per cell)
# ============================================================================

def stage_perm(args):
    run_dir = Path(args.run_dir)
    frame_path = run_dir / 'frames' / f'cell_{args.cell_index:03d}.parquet'
    (run_dir / 'perm').mkdir(exist_ok=True)
    out_path = run_dir / 'perm' / f'perm_{args.cell_index:03d}.parquet'

    if not frame_path.exists():
        logger.warning('cell %d has no saved frame (not fittable); empty shard',
                       args.cell_index)
        io.write_table(pd.DataFrame(columns=['perm', 'beta', 'z', 'converged']),
                       out_path, params={'cell_index': args.cell_index},
                       script=SCRIPT)
        return

    df = io.read_table(frame_path, on_stale='ignore')
    res, _ = mm.fit_cell(df, mm.VC_FULL)
    logger.info('cell %d: observed beta %+.6f z %+.3f p %.4g over %d rows',
                args.cell_index, res.fe_params['NRS_within'],
                res.tvalues['NRS_within'], res.pvalues['NRS_within'], len(df))

    t0 = time.time()
    # Warm-started from the observed fit: the variance components barely move
    # under a permutation of the predictor, and the starting point is identical
    # for every shuffle so it cannot bias the null.
    out = mm.permutation_null(df, args.n_perm, seed=args.seed,
                              start_params=res.params_object, n_jobs=args.n_jobs)
    logger.info('%d shuffles in %.0fs (%.2fs each, %d jobs), %d failed',
                args.n_perm, time.time() - t0,
                (time.time() - t0) / max(args.n_perm, 1), args.n_jobs,
                int((~out['converged']).sum()))
    out.insert(0, 'cell_index', args.cell_index)
    io.write_table(out, out_path,
                   params={'n_perm': args.n_perm, 'seed': args.seed,
                           'cell_index': args.cell_index,
                           'exchangeability': 'epoch -> pain-score pairing, within '
                                              'subject, epoch relabelled as a whole'},
                   script=SCRIPT)


# ============================================================================
# STAGE: collect
# ============================================================================

def stage_collect(args):
    run_dir = Path(args.run_dir)
    parts = [io.read_table(p, on_stale='ignore')
             for p in sorted((run_dir / 'cells').glob('region_*.parquet'))]
    if not parts:
        raise SystemExit(f'no cell tables in {run_dir / "cells"}')
    cells = pd.concat(parts, ignore_index=True).sort_values('cell_index')
    cells = cells.reset_index(drop=True)

    usable = cells['p'].notna()
    cells['p_bh'] = np.nan
    cells['p_bh_reject'] = pd.NA
    if usable.any():
        _, adj = cp.bh_fdr(cells.loc[usable, 'p'].to_numpy(), q=args.fdr_q)
        cells.loc[usable, 'p_bh'] = adj
        cells.loc[usable, 'p_bh_reject'] = adj <= args.fdr_q

    # The heterogeneity LRT gets its own family. Two questions, two corrections;
    # pooling them would correct each for the other's tests.
    u2 = cells['p_lrt_mixture'].notna()
    cells['p_lrt_bh'] = np.nan
    cells['p_lrt_bh_reject'] = pd.NA
    if u2.any():
        _, adj = cp.bh_fdr(cells.loc[u2, 'p_lrt_mixture'].to_numpy(), q=args.fdr_q)
        cells.loc[u2, 'p_lrt_bh'] = adj
        cells.loc[u2, 'p_lrt_bh_reject'] = adj <= args.fdr_q

    # Permutation shards, if `--stage perm` has run.
    shards = sorted((run_dir / 'perm').glob('perm_*.parquet')) if (
        run_dir / 'perm').exists() else []
    if shards:
        nulls = pd.concat([io.read_table(p, on_stale='ignore') for p in shards],
                          ignore_index=True)
        rows = []
        for ci, grp in nulls.groupby('cell_index'):
            obs = cells.loc[cells['cell_index'] == ci, 'beta_nrs_within']
            obs = float(obs.iloc[0]) if len(obs) else np.nan
            p_perm, n_used = mm.permutation_p(obs, grp['beta'].to_numpy())
            rows.append({'cell_index': int(ci), 'p_perm': p_perm,
                         'n_perm_used': n_used})
        cells = cells.merge(pd.DataFrame(rows), on='cell_index', how='left')
        up = cells['p_perm'].notna()
        cells['p_perm_bh'] = np.nan
        cells['p_perm_bh_reject'] = pd.NA
        if up.any():
            _, adj = cp.bh_fdr(cells.loc[up, 'p_perm'].to_numpy(), q=args.fdr_q)
            cells.loc[up, 'p_perm_bh'] = adj
            cells.loc[up, 'p_perm_bh_reject'] = adj <= args.fdr_q
        logger.info('permutation p present for %d/%d cells', int(up.sum()), len(cells))
        io.write_table(nulls, run_dir / 'permutation_null.parquet',
                       params={'n_shards': len(shards)}, script=SCRIPT)

    blup_parts = [io.read_table(p, on_stale='ignore')
                  for p in sorted((run_dir / 'cells').glob('blups_*.parquet'))]
    blup_parts = [b for b in blup_parts if len(b)]
    if blup_parts:
        io.write_table(pd.concat(blup_parts, ignore_index=True),
                       run_dir / 'blups.parquet',
                       params={'source': 'model BLUPs, shrunk toward the group'},
                       script=SCRIPT)

    slopes = [io.read_table(p, on_stale='ignore')
              for p in sorted((run_dir / 'cells').glob('slopes_*.parquet'))]
    cons = pd.DataFrame()
    if slopes:
        s = pd.concat(slopes, ignore_index=True)
        merged = s.merge(cells[['region', 'band', 'beta_nrs_within']],
                         on=['region', 'band'], how='inner')
        ok = merged[merged['slope'].notna() & merged['beta_nrs_within'].notna()].copy()
        ok['agrees'] = np.sign(ok['slope']) == np.sign(ok['beta_nrs_within'])
        cons = (ok.groupby(['region', 'band'])
                .agg(frac_sign_consistent=('agrees', 'mean'),
                     n_with_slope=('agrees', 'size')).reset_index())
        cells = cells.merge(cons, on=['region', 'band'], how='left')
        io.write_table(s, run_dir / 'subject_slopes.parquet',
                       params={'source': 'unpooled per-subject OLS'}, script=SCRIPT)

    io.write_table(cells, run_dir / 'band_cells.parquet',
                   params={'fdr_q': args.fdr_q,
                           'families': 'NRS_within across all fitted cells; the '
                                       'heterogeneity LRT separately; the '
                                       'permutation p separately'},
                   script=SCRIPT,
                   extra={'status': DISCLAIMER, 'inference_caveat': WALD_CAVEAT,
                          'band_caveat': band_caveat(args.band_set)})

    report(cells, args.fdr_q)
    figures(run_dir, cells, args)
    write_methods(run_dir, cells, args)
    io.log_analysis(f'band-power mixed-effects models, {len(cells)} region x band '
                    'cells, BH-corrected (EXPLORATORY)', run_dir)
    print(run_dir)


def report(cells, q):
    conv = int(cells['converged'].fillna(False).sum())
    logger.info('=' * 74)
    logger.info('BAND-POWER MIXED MODELS: %d cells, %d converged', len(cells), conv)
    sig = cells[cells['p_bh_reject'] == True]                    # noqa: E712
    logger.info('  BH-significant (Wald, q=%.2f): %d', q, len(sig))
    for r in sig.sort_values('p').itertuples():
        extra = (f"  p_perm {r.p_perm:.4g}" if 'p_perm' in cells.columns
                 and np.isfinite(getattr(r, 'p_perm', np.nan)) else '')
        logger.info('    %-18s %-11s beta %+.5f  z %+5.2f  p_bh %.4g  '
                    'sign-consistent %.2f%s', r.region, r.band, r.beta_nrs_within,
                    r.z, r.p_bh, getattr(r, 'frac_sign_consistent', np.nan), extra)
    if 'p_perm' in cells.columns and cells['p_perm'].notna().any():
        d = cells.dropna(subset=['p', 'p_perm'])
        n_anti = int((d['p_perm'] > d['p']).sum())
        logger.info('  permutation vs Wald: p_perm LARGER (Wald anticonservative) '
                    'in %d/%d cells; median log10 ratio %+.3f', n_anti, len(d),
                    float(np.median(np.log10(d['p_perm'] / d['p']))))
    het = cells[cells['p_lrt_bh_reject'] == True]                # noqa: E712
    logger.info('  heterogeneity LRT BH-significant: %d (subjects differ in their '
                'slope)', len(het))
    logger.info('  %s', WALD_CAVEAT)
    logger.info('=' * 74)


def figures(run_dir, cells, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ieeg_ehr.features import common

    bands = list(BAND_SETS[args.band_set])
    regions = [r for r in view_tables.roi_regions_for({'roi_scheme': args.roi_scheme})
               if r in set(cells['region'])]
    beta = (cells.pivot_table(index='region', columns='band', values='beta_nrs_within')
            .reindex(index=regions, columns=bands))
    sig = (cells.assign(rej=cells['p_bh_reject'].fillna(False).astype(bool))
           .pivot_table(index='region', columns='band', values='rej')
           .reindex(index=regions, columns=bands).fillna(0).astype(bool))

    # --- 1. the map: 6 columns, so imshow over categorical cells is right here
    cap = float(np.nanmax(np.abs(beta.to_numpy(dtype=float))))
    cm = plt.get_cmap('RdBu_r').copy()
    cm.set_bad('0.85')
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(regions) + 3.0))
    im = ax.imshow(beta.to_numpy(dtype=float), aspect='auto', cmap=cm,
                   vmin=-cap, vmax=cap, interpolation='nearest')
    common.draw_mask_outline(ax, sig.to_numpy())
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([f'{b}\n{BAND_SETS[args.band_set][b][0]}-'
                        f'{BAND_SETS[args.band_set][b][1]} Hz' for b in bands],
                       fontsize=8)
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(regions, fontsize=8)
    ax.set_title(f'Band power vs pain, mixed-effects beta\n'
                 f'{int(sig.to_numpy().sum())} of {len(cells)} cells BH-significant '
                 f'at q={args.fdr_q}', fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03,
                 label='d log10(band power) per pain point')
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.text(0.01, 0.005,
             'Outlines mark BH-significant cells across all fitted cells at '
             f'q={args.fdr_q}. {WALD_CAVEAT} {band_caveat(args.band_set)}\n'
             f'{DISCLAIMER}',
             fontsize=6.5, va='bottom', ha='left', color='0.35', wrap=True)
    p1 = run_dir / 'fig_band_map.png'
    fig.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- 2. the forest plot: 120 estimates with intervals, which a heatmap hides
    fig, axs = plt.subplots(1, len(bands), figsize=(3.0 * len(bands), 0.30 * len(regions) + 3.0),
                            sharey=True, squeeze=False)
    y = np.arange(len(regions))
    xmax = float(np.nanmax(np.abs(np.concatenate(
        [(cells['beta_nrs_within'] + 1.96 * cells['se']).to_numpy(),
         (cells['beta_nrs_within'] - 1.96 * cells['se']).to_numpy()])))) * 1.05
    for j, band in enumerate(bands):
        ax = axs[0][j]
        d = cells[cells['band'] == band].set_index('region').reindex(regions)
        b = d['beta_nrs_within'].to_numpy(dtype=float)
        se = d['se'].to_numpy(dtype=float)
        rej = d['p_bh_reject'].fillna(False).to_numpy(dtype=bool)
        ax.errorbar(b[~rej], y[~rej], xerr=1.96 * se[~rej], fmt='o', ms=4, lw=1,
                    capsize=2, color='0.6')
        ax.errorbar(b[rej], y[rej], xerr=1.96 * se[rej], fmt='o', ms=5.5, lw=1.4,
                    capsize=2, color='#b03a2e')
        ax.axvline(0, color='0.4', lw=0.9, ls='--')
        ax.set_xlim(-xmax, xmax)
        ax.set_title(f'{band}\n{BAND_SETS[args.band_set][band][0]}-'
                     f'{BAND_SETS[args.band_set][band][1]} Hz', fontsize=9)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(regions, fontsize=8)
            ax.set_ylim(len(regions) - 0.5, -0.5)
    fig.suptitle('Band power vs pain: beta with 95% CI, per region\n'
                 'red = BH-significant', fontsize=12)
    fig.tight_layout(rect=(0, 0.09, 1, 0.92))
    fig.text(0.01, 0.005,
             'Intervals are Wald 95% CIs from the mixed model, so they inherit the '
             f'assumption in the caveat. {WALD_CAVEAT}\n{DISCLAIMER}',
             fontsize=6.5, va='bottom', ha='left', color='0.35', wrap=True)
    p2 = run_dir / 'fig_band_forest.png'
    fig.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('wrote %s and %s', p1.name, p2.name)


def write_methods(run_dir, cells, args):
    bands = BAND_SETS[args.band_set]
    sig = cells[cells['p_bh_reject'] == True]                    # noqa: E712
    lines = [f"""# Band-power mixed-effects models

{DISCLAIMER}

## What is fitted

One row = ONE CHANNEL x ONE 5-minute pre-report epoch. The outcome is that
channel's log10 BAND power for that epoch: the native 0.5 Hz bins whose geometric
centre falls in the band, line-noise bins excluded, combined LINEAR-THEN-LOG via
`views.axes.aggregate_bands` -- log10(mean(10**x)), because raw log power does not
average arithmetically. A coefficient is d log10(band power) per pain point.

    log10_power ~ NRS_within + NRS_submean
                  + (NRS_within || subject) + (1 | subject:channel)

`NRS_within` is the subject-mean-centred pain score and is the effect of
interest. `NRS_submean` is a nuisance term that keeps the between-subject
contrast out of the within-subject slope. The by-subject random slope lets
patients differ in how strongly power tracks their pain; the channel random
intercept absorbs each contact's own amplitude, which is why no normalization is
applied. REML, statsmodels.

Band set `{args.band_set}`: {', '.join(f'{k} {v[0]}-{v[1]} Hz' for k, v in bands.items())}.

{band_caveat(args.band_set)}

## Why a refit rather than averaging the grid's betas

Averaging the per-frequency betas within a band is a fine point estimate -- it
agreed with a refit to under 1% on 6 test cells -- but it has no usable standard
error: the SE of a mean of 463 estimates needs their cross-bin covariance, which
independent per-cell fits never produce, and the mean of the per-cell SEs ran 5%
to 67% away from the refit's. The aggregation rule also matters on its own:
linear-then-log gave betas 12-31% larger in magnitude than the averaged betas on
those cells, and moved one cell from z=1.74 to z=2.07.

## Significance

`p` is the parametric Wald z on `NRS_within`; `p_bh` is Benjamini-Hochberg over
all fitted cells at q={args.fdr_q}. {WALD_CAVEAT}

The heterogeneity LRT (`p_lrt_mixture`, referenced to the 50:50 mixture of a
point mass at 0 and chi2(1) because the variance sits at the parameter-space
boundary) gets its OWN BH family: "do subjects differ in their slope" is a
separate question from "is the mean slope non-zero", and correcting each for the
other's tests would be wrong.

`frac_sign_consistent` is the fraction of subjects whose UNPOOLED OLS slope shares
the sign of the group fixed effect -- not BLUPs, which partial pooling drags
toward the group.

## Known limitations

- {CONFOUND_CAVEAT}
- **No covariates**: no time of day, no time since admission, no temporal term.
  Epochs within a subject are treated as exchangeable, and BOTH the Wald and the
  permutation p depend on that. A permutation test replaces the reference
  distribution, not the design.
- Channels are kept as channels (not averaged into the ROI), so a region's
  estimate is precision-weighted toward subjects with more contacts there.

## Results: {len(sig)} of {len(cells)} cells BH-significant

"""]
    if len(sig):
        lines.append('| region | band | beta | SE | z | p_bh | sign consistency |\n'
                     '|---|---|---|---|---|---|---|\n')
        for r in sig.sort_values('p').itertuples():
            lines.append('| {} | {} | {:+.5f} | {:.5f} | {:+.2f} | {:.4g} | {:.2f} |\n'
                         .format(r.region, r.band, r.beta_nrs_within, r.se, r.z,
                                 r.p_bh, getattr(r, 'frac_sign_consistent', np.nan)))
    (run_dir / 'METHODS.md').write_text(''.join(lines))


# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=['fit', 'perm', 'collect'], default='fit')
    ap.add_argument('--run-dir', default=None)
    ap.add_argument('--region-index', type=int, default=None,
                    help='Fit one region only (array mode). Default: all of them, '
                         'which takes minutes at 120 cells.')
    ap.add_argument('--cell-index', type=int, default=None,
                    help='Which cell `--stage perm` shuffles.')
    ap.add_argument('--band-set', choices=list(BAND_SETS),
                    default='paper_bands_6_hg200')
    ap.add_argument('--exclude-regions', nargs='*', default=[],
                    help='Regions to leave out of the run ENTIRELY -- not fitted, '
                         'not in the BH family, not on the figures. Use for a '
                         'region whose data is under review, and say why: the '
                         'reason goes into provenance. Excluding a region because '
                         'it was not significant is a different and illegitimate '
                         'act, which is why this takes names rather than a '
                         'threshold.')
    ap.add_argument('--exclude-reason', default=None,
                    help='Recorded verbatim in provenance beside --exclude-regions.')
    ap.add_argument('--roi-scheme', default='roi_v2_ofc',
                    help="Region set. Default 'roi_v2_ofc' = roi_v2 with mOFC and "
                         'lOFC fused into one OFC (20 regions). Pass roi_v2 to keep '
                         'them split, or a path to a JSON scheme.')
    ap.add_argument('--view-dir', default=None)
    ap.add_argument('--reference-run', default=str(reference_run.CONTPAIN_HEATMAP))
    ap.add_argument('--mask-label', default=None)
    ap.add_argument('--cohort', choices=['reference', 'eligible-discovery'],
                    default='reference')
    ap.add_argument('--allow-cohort-drift', action='store_true')
    ap.add_argument('--notch-half-width-hz', type=float, default=None)
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--n-jobs', type=int,
                    default=int(__import__('os').environ.get('SLURM_CPUS_PER_TASK', 1)))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fdr-q', type=float, default=FDR_Q)
    ap.add_argument('--question', default=QUESTION)
    ap.add_argument('--view-scheme', default=None,
                    help='Level-4 folder. Default: derived from --band-set, so '
                         'the path cannot claim band edges the run did not use.')
    ap.add_argument('--run-name', default=RUN_NAME)
    args = ap.parse_args()

    if args.view_scheme is None:
        args.view_scheme = VIEW_SCHEMES.get(args.band_set, VIEW_SCHEME)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    if args.stage == 'fit':
        stage_fit(args)
    elif args.stage == 'perm':
        if not args.run_dir or args.cell_index is None:
            raise SystemExit('--stage perm needs --run-dir and --cell-index')
        stage_perm(args)
    else:
        if not args.run_dir:
            raise SystemExit('--stage collect needs --run-dir')
        stage_collect(args)


if __name__ == '__main__':
    main()

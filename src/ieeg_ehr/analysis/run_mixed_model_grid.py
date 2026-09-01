"""PHASE 2: per-cell mixed-effects models over the FULL region x frequency grid.

Phase 1 (`run_mixed_model_pilot`) fitted 19 hand-picked cells to answer the design
questions. This fits every cell that clears the coverage floor, on the same model,
the same cohort and the same inherited criteria:

    log10_power ~ NRS_within + NRS_submean
                  + (NRS_within || subject) + (1 | subject:channel)

WHAT PHASE 1 SETTLED, AND IS THEREFORE NOT RE-ASKED HERE

- A channel random SLOPE improved fit in 1 of 19 cells, so the intercept-only
  prior stands and `--with-channel-slope` is OFF by default. That is a third fit
  per cell avoided across ~900 cells.
- Fits are cheap (median 1.1 s, max 7.0 s) and 19/19 converged with no variance
  component at the boundary, so the grid is a single-pass array over regions
  rather than anything adaptive.
- Model objects are NOT saved. The pilot saved them because 19 objects are worth
  keeping; ~900 are not, and the per-cell table carries everything downstream
  needs.

WHAT IS STILL OPEN, AND WHY THE p COLUMNS ARE PROVISIONAL

The pilot's permutation stage -- whether the Wald z is calibrated for this design
-- had not finished when this grid was first run. `p` and `p_bh` here are
PARAMETRIC. If the permutation shows Wald is anticonservative, every p column in
this table must be replaced, and `p_provisional` in the run provenance says so.
Nothing else is affected: betas, variance components, LRT statistics, BLUPs and
coverage counts are what they are regardless of how the null is referenced.

STAGES, mirroring the pilot so the sbatch pattern is the same:

    prepare                 -> resolve cohort + manifest, create the run dir,
                               echo RUN_DIR on stdout and nothing else
    fit --region-index N    -> every bin of one region; one array task per region
    collect                 -> concatenate, apply BH, write the run tables

    python -m ieeg_ehr.analysis.run_mixed_model_grid --stage prepare
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import cluster_permutation, mixed_model as mm, pain_coef
from ieeg_ehr.analysis import reference_run, view_tables
from ieeg_ehr.analysis.run_mixed_model_pilot import (
    DISCLAIMER, coverage_map, fit_one_cell, inventory_subjects, load_cell_frames,
    load_epoch_scores, resolve_view_dir, roi_maps, view_subject_paths)
from ieeg_ehr.views import cache_reader

logger = logging.getLogger(__name__)

QUESTION = 'psd_physiology'
OUTPUT_TYPE = 'univariate_analysis'
VIEW_SCHEME = 'cont_pain_scratch'
RUN_NAME = 'fullgrid_mixedlm'

FDR_Q = 0.05


# ============================================================================
# STAGE: prepare
# ============================================================================

def resolve_cohort(ref, view_dir):
    """The reference's 51, re-derived and asserted -- same rule as the pilot."""
    paths = view_subject_paths(view_dir)
    scores = load_epoch_scores(paths)
    eligible, diagnostics = pain_coef.eligible_subjects(
        scores,
        min_epochs=ref.criteria.get('min_epochs', pain_coef.MIN_EPOCHS),
        min_range=ref.criteria.get('min_range', pain_coef.MIN_RANGE),
        min_non_modal=ref.criteria.get('min_non_modal', pain_coef.MIN_NON_MODAL))
    subjects = set(eligible)
    roi_scheme = ref.view_params.get('roi_scheme', 'roi_v2')
    roi_by_subject, no_roi = roi_maps(paths, subjects, roi_scheme)
    subjects -= set(no_roi)
    return paths, scores, diagnostics, subjects, roi_by_subject, no_roi


def build_grid_manifest(regions, bin_table, coverage, min_subjects):
    """Every region x bin, with its coverage and whether it clears the floor.

    Cells BELOW the floor stay in the manifest as rows rather than vanishing.
    "Which cells were not fitted, and why" is part of the result; a manifest that
    silently contains only the fittable cells cannot answer it, and the reference
    run's coverage threshold is exactly the kind of choice that has to remain
    visible downstream.
    """
    rows = []
    for region in regions:
        for b in bin_table.index:
            n = int(coverage.get((region, int(b)), 0))
            rows.append({'region': region, 'freq_bin_index': int(b),
                         'bin_low_hz': float(bin_table.loc[b, 'bin_low_hz']),
                         'bin_high_hz': float(bin_table.loc[b, 'bin_high_hz']),
                         'n_subjects_covered': n,
                         'above_coverage_floor': n >= min_subjects})
    man = pd.DataFrame(rows)
    man.insert(0, 'cell_index', range(len(man)))
    return man


def stage_prepare(args):
    ref = reference_run.load(args.reference_run)
    ref.describe()

    view_dir = resolve_view_dir(args.view_dir,
                                mask_label=args.mask_label
                                or ref.view_params.get('mask_label'),
                                roi_scheme=ref.view_params.get('roi_scheme', 'roi_v2'))
    logger.info('per-channel view: %s', view_dir)

    paths, scores, diagnostics, subjects, roi_by_subject, no_roi = resolve_cohort(
        ref, view_dir)
    ref.assert_cohort_matches(subjects, allow_drift=args.allow_cohort_drift)

    roi_scheme = ref.view_params.get('roi_scheme', 'roi_v2')
    regions = view_tables.roi_regions_for({'roi_scheme': roi_scheme})

    epoch_minutes = ref.view_params.get('epoch_minutes')
    bin_table = cache_reader.bin_edges(epoch_minutes).set_index('freq_bin_index')
    line_noise = list(ref.line_noise_bins_removed
                      or cache_reader.line_noise_bins(epoch_minutes))
    bin_table = bin_table.drop(index=[b for b in line_noise if b in bin_table.index])

    # Bins narrower than the 0.5 Hz FFT resolution are COPIES of a neighbour, not
    # measurements. Keeping them pads the correction family with duplicates and
    # makes a repeated number look like broadband low-frequency structure.
    unresolvable = []
    if not args.keep_unresolvable_bins:
        unresolvable = [int(b) for b in cache_reader.unresolvable_bins(epoch_minutes)
                        if b in bin_table.index]
        bin_table = bin_table.drop(index=unresolvable)
        logger.info('dropped %d unresolvable (duplicate) bin(s): %s',
                    len(unresolvable), unresolvable)

    min_subjects = int(args.min_subjects if args.min_subjects is not None
                       else ref.criteria.get('min_subjects', mm.MIN_SUBJECTS))
    if args.min_subjects is not None:
        logger.warning('coverage floor OVERRIDDEN to min_subjects=%d (the reference '
                       'run used %d). Cells and whole regions below it are excluded, '
                       'so this run is NOT cell-for-cell comparable with it.',
                       min_subjects,
                       int(ref.criteria.get('min_subjects', mm.MIN_SUBJECTS)))
    coverage = coverage_map(paths, subjects, roi_by_subject, bin_table.index)
    manifest = build_grid_manifest(regions, bin_table, coverage, min_subjects)

    n_fit = int(manifest['above_coverage_floor'].sum())
    logger.info('%d regions x %d bins = %d cells; %d clear the coverage floor '
                '(min_subjects=%d), %d do not and will be reported unfitted',
                len(regions), len(bin_table), len(manifest), n_fit, min_subjects,
                len(manifest) - n_fit)
    logger.info('regions below the floor entirely: %s',
                sorted(manifest.loc[~manifest['above_coverage_floor'], 'region']
                       .unique()) or 'none')

    run_dir = config.analysis_run_dir(question=args.question,
                                      output_type=OUTPUT_TYPE,
                                      view_scheme=args.view_scheme,
                                      run_name=args.run_name)
    (run_dir / 'grid').mkdir(parents=True, exist_ok=True)

    io.write_table(manifest, run_dir / 'grid_cell_manifest.parquet',
                   params={'min_subjects': min_subjects,
                           'line_noise_bins_removed': line_noise,
                           'unresolvable_bins_removed': unresolvable,
                           'roi_scheme': roi_scheme},
                   parents=[str(Path(args.reference_run) / 'provenance.json'),
                            str(view_dir)],
                   subjects=sorted(subjects),
                   script='ieeg_ehr/analysis/run_mixed_model_grid.py')
    io.write_table(inventory_subjects(scores, diagnostics),
                   run_dir / 'inventory_subjects.parquet',
                   script='ieeg_ehr/analysis/run_mixed_model_grid.py')
    io.write_table(pd.DataFrame({'region_index': range(len(regions)),
                                 'region': regions}),
                   run_dir / 'region_index.parquet',
                   script='ieeg_ehr/analysis/run_mixed_model_grid.py')

    io.write_run_provenance(
        run_dir, script='ieeg_ehr/analysis/run_mixed_model_grid.py',
        params={'stage': 'prepare', 'view_dir': str(view_dir),
                'view_params': ref.view_params, 'criteria': ref.criteria,
                'min_subjects': min_subjects, 'roi_scheme': roi_scheme,
                'unresolvable_bins_removed': unresolvable,
                'line_noise_bins_removed': line_noise,
                'with_channel_slope': bool(args.with_channel_slope),
                'n_regions': len(regions), 'n_bins': len(bin_table),
                'n_cells': len(manifest), 'n_cells_fittable': n_fit},
        parents=[str(Path(args.reference_run) / 'provenance.json'), str(view_dir)],
        subjects=sorted(subjects),
        extra={'status': 'EXPLORATORY Phase 2 full grid, NOT a finding',
               'subjects_without_roi': sorted(no_roi),
               'p_provisional': 'PARAMETRIC Wald p. The pilot permutation '
                                'calibration was not complete when this grid was '
                                'built; if Wald proves anticonservative every p '
                                'and p_bh column here must be replaced. Betas, '
                                'variance components, LRT statistics and BLUPs '
                                'are unaffected.',
               'mask_content': 'Signal quality ONLY. Opioid-administration windows '
                               'and post-ictal periods are NOT excluded; both are '
                               'first-order confounds for low-frequency power.'})

    # The ONLY thing on stdout. Every log line above went to stderr.
    print(run_dir)
    return run_dir


# ============================================================================
# STAGE: fit (one region per array task)
# ============================================================================

def stage_fit(args):
    ref = reference_run.load(args.reference_run)
    run_dir = Path(args.run_dir)
    manifest = io.read_table(run_dir / 'grid_cell_manifest.parquet', on_stale='warn')
    region_index = io.read_table(run_dir / 'region_index.parquet', on_stale='warn')

    match = region_index[region_index['region_index'] == args.region_index]
    if match.empty:
        raise SystemExit(f'--region-index {args.region_index} is not one of the '
                         f'{len(region_index)} regions in this run')
    region = str(match['region'].iloc[0])

    todo = manifest[(manifest['region'] == region)
                    & manifest['above_coverage_floor']].sort_values('freq_bin_index')
    skipped = manifest[(manifest['region'] == region)
                       & ~manifest['above_coverage_floor']]
    logger.info('region %d/%d: %s -- %d cells to fit, %d below the coverage floor',
                args.region_index, len(region_index), region, len(todo), len(skipped))

    if todo.empty:
        # Still write the file. A missing output is indistinguishable from a task
        # that died, and collect would have no way to tell the two apart. The
        # frame carries its columns so an empty region concatenates cleanly.
        io.write_table(pd.DataFrame(columns=['region', 'freq_bin_index']),
                       run_dir / 'grid' / f'region_{args.region_index:03d}.parquet',
                       params={'region': region, 'n_cells': 0},
                       script='ieeg_ehr/analysis/run_mixed_model_grid.py')
        logger.info('nothing fittable in %s; wrote an empty result', region)
        return

    view_dir = resolve_view_dir(args.view_dir,
                                mask_label=args.mask_label
                                or ref.view_params.get('mask_label'),
                                roi_scheme=ref.view_params.get('roi_scheme', 'roi_v2'))
    paths, _, _, subjects, roi_by_subject, _ = resolve_cohort(ref, view_dir)

    wanted = {(region, int(b)) for b in todo['freq_bin_index']}
    t0 = time.time()
    frames = load_cell_frames(paths, subjects, wanted, roi_by_subject)
    logger.info('loaded %d cell frame(s) for %s in %.1fs (%d rows total)',
                len(frames), region, time.time() - t0,
                sum(len(f) for f in frames.values()))

    records, blup_parts = [], []
    for row in todo.itertuples():
        meta = {'region': region, 'freq_bin_index': int(row.freq_bin_index),
                'bin_low_hz': float(row.bin_low_hz),
                'bin_high_hz': float(row.bin_high_hz),
                'cell_index': int(row.cell_index), 'group': 'grid'}
        df = frames.get((region, int(row.freq_bin_index)))
        if df is None or df.empty:
            # df=None, not an empty frame: failed_record reads df['subject'] to
            # count what was there, and an empty frame has no such column.
            records.append(mm.failed_record(region, int(row.freq_bin_index),
                                            float(row.bin_low_hz),
                                            float(row.bin_high_hz),
                                            'no rows after the ROI join', df=None))
            records[-1]['cell_index'] = int(row.cell_index)
            continue
        rec, blups, _ = fit_one_cell(df, meta,
                                     with_channel_slope=args.with_channel_slope)
        records.append(rec)
        if blups:
            blup_parts.append(pd.DataFrame(blups))

    cells = pd.DataFrame(records)
    io.write_table(cells, run_dir / 'grid' / f'region_{args.region_index:03d}.parquet',
                   params={'region': region, 'n_cells': len(cells),
                           'with_channel_slope': bool(args.with_channel_slope)},
                   parents=[str(run_dir / 'grid_cell_manifest.parquet')],
                   script='ieeg_ehr/analysis/run_mixed_model_grid.py')
    if blup_parts:
        io.write_table(pd.concat(blup_parts, ignore_index=True),
                       run_dir / 'grid' / f'blups_{args.region_index:03d}.parquet',
                       params={'region': region},
                       script='ieeg_ehr/analysis/run_mixed_model_grid.py')

    ok = int(cells['converged'].fillna(False).sum()) if 'converged' in cells else 0
    logger.info('%s done: %d/%d converged, %.0fs total fitting',
                region, ok, len(cells), cells.get('fit_seconds', pd.Series(dtype=float)).sum())


# ============================================================================
# STAGE: collect
# ============================================================================

def add_fdr(cells, p_col, out_prefix, q=FDR_Q):
    """BH across ALL fitted cells (primary) and within each region (secondary).

    Two corrections, both reported, neither silently preferred -- the spec asks
    for the global family as primary and the within-region family as a more
    liberal secondary. A cell with no usable p (unfittable, non-convergent) is
    held out of the family entirely rather than entered as p=1: it was never a
    test, and padding the family with non-tests would make the correction look
    harsher than the number of things actually examined.
    """
    cells = cells.copy()
    if p_col not in cells.columns:
        logger.warning('no %s column to correct; every cell failed to fit?', p_col)
        cells[p_col] = np.nan
    usable = cells[p_col].notna()
    cells[f'{out_prefix}_bh'] = np.nan
    cells[f'{out_prefix}_bh_reject'] = pd.NA
    if usable.any():
        _, adj = cluster_permutation.bh_fdr(cells.loc[usable, p_col].to_numpy(), q=q)
        cells.loc[usable, f'{out_prefix}_bh'] = adj
        cells.loc[usable, f'{out_prefix}_bh_reject'] = adj <= q

    cells[f'{out_prefix}_bh_within_region'] = np.nan
    cells[f'{out_prefix}_bh_within_region_reject'] = pd.NA
    for region, idx in cells[usable].groupby('region').groups.items():
        _, adj = cluster_permutation.bh_fdr(cells.loc[idx, p_col].to_numpy(), q=q)
        cells.loc[idx, f'{out_prefix}_bh_within_region'] = adj
        cells.loc[idx, f'{out_prefix}_bh_within_region_reject'] = adj <= q
    return cells


def stage_collect(args):
    run_dir = Path(args.run_dir)
    manifest = io.read_table(run_dir / 'grid_cell_manifest.parquet', on_stale='warn')

    parts, missing = [], []
    region_index = io.read_table(run_dir / 'region_index.parquet', on_stale='warn')
    for r in region_index.itertuples():
        p = run_dir / 'grid' / f'region_{int(r.region_index):03d}.parquet'
        if not p.exists():
            missing.append(r.region)
            continue
        df = io.read_table(p, on_stale='ignore')
        if len(df):
            parts.append(df)
    if missing:
        # Loud, and recorded -- a silently short grid is the failure mode that
        # looks exactly like a real result with fewer significant cells.
        logger.error('MISSING region outputs (array tasks that did not finish): %s',
                     missing)
    if not parts:
        raise SystemExit('no region outputs to collect')

    cells = pd.concat(parts, ignore_index=True)
    cells = add_fdr(cells, 'p', 'p')
    cells = add_fdr(cells, 'p_lrt_mixture', 'p_lrt')

    io.write_table(cells, run_dir / 'grid_cells.parquet',
                   params={'fdr_q': FDR_Q,
                           'families': 'global across fitted cells (primary); '
                                       'within region (secondary)',
                           'p_provisional': True},
                   parents=[str(run_dir / 'grid_cell_manifest.parquet')],
                   script='ieeg_ehr/analysis/run_mixed_model_grid.py')

    blup_parts = [io.read_table(p, on_stale='ignore')
                  for p in sorted((run_dir / 'grid').glob('blups_*.parquet'))]
    if blup_parts:
        io.write_table(pd.concat(blup_parts, ignore_index=True),
                       run_dir / 'grid_blups.parquet',
                       parents=[str(run_dir / 'grid_cells.parquet')],
                       script='ieeg_ehr/analysis/run_mixed_model_grid.py')

    n_fit = len(cells)
    conv = int(cells['converged'].fillna(False).sum())
    sig = int((cells['p_bh_reject'] == True).sum())  # noqa: E712
    sig_lrt = int((cells['p_lrt_bh_reject'] == True).sum())  # noqa: E712
    sing = int(cells['singular_flag'].fillna(False).sum())
    logger.info('=' * 70)
    logger.info('FULL GRID SUMMARY')
    logger.info('  manifest cells                 : %d (%d above the coverage floor)',
                len(manifest), int(manifest['above_coverage_floor'].sum()))
    logger.info('  fitted rows collected          : %d from %d region file(s)',
                n_fit, len(parts))
    logger.info('  converged                      : %d/%d', conv, n_fit)
    logger.info('  singular (a VC at boundary)    : %d', sing)
    logger.info('  fixed effect BH-significant    : %d  (global family, q=%.2f)',
                sig, FDR_Q)
    logger.info('  heterogeneity BH-significant   : %d  (global family, q=%.2f)',
                sig_lrt, FDR_Q)
    logger.info('  PARAMETRIC p -- provisional until the pilot permutation '
                'calibration is read.')
    logger.info('=' * 70)

    write_methods(run_dir, cells, manifest, missing)
    io.log_analysis('mixed-model Phase 2: full region x frequency grid, parametric p '
                    '(EXPLORATORY, nominations not findings)', run_dir)
    print(run_dir)


def write_methods(run_dir, cells, manifest, missing):
    import json
    try:
        prov = json.loads((Path(run_dir) / 'provenance.json').read_text())
        params = prov.get('params', {})
    except (OSError, ValueError):
        params = {}
    dropped = params.get('unresolvable_bins_removed', [])
    floor = params.get('min_subjects', '?')

    text = f"""# Phase 2: mass-univariate mixed-effects models, full grid

{DISCLAIMER}

## Model

    log10_power ~ NRS_within + NRS_submean
                  + (NRS_within || subject) + (1 | subject:channel)

One row = one channel x one 5-minute pre-report epoch. NRS is split into a
within-subject deviation and a subject mean so the between-subject contrast
cannot leak into the within-subject slope. Fitted with REML.

## Cells

{len(manifest)} region x frequency cells in the manifest
({manifest['region'].nunique()} regions x {manifest['freq_bin_index'].nunique()} bins);
{int(manifest['above_coverage_floor'].sum())} clear the coverage floor of
min_subjects={floor} and were fitted. Cells below the floor remain as manifest rows
and are not silently absent. {len(cells)} fitted rows were collected.

## Frequency axis

Line-noise bins are removed, as in the reference run. In addition, bins
{dropped} are dropped as UNRESOLVABLE: the axis is 50 log-spaced bins from 1 Hz
but the PSD comes from 2-second windows, so the real resolution is a flat 0.5 Hz.
Below ~4.7 Hz a log bin is narrower than that and contains no FFT frequency of its
own; the cache builder fills it from the nearest neighbour, making it an exact
duplicate. Retained bins are those whose nominal range contains the frequency they
report. Keeping the duplicates would pad the correction family with copies and
make one repeated number read as broadband low-frequency structure.

## Multiple comparisons

Benjamini-Hochberg at q={FDR_Q}, two families, both reported:
`p_bh` across every fitted cell (PRIMARY) and `p_bh_within_region` within each
region (SECONDARY, more liberal). The same treatment is applied separately to the
heterogeneity LRT as `p_lrt_bh` / `p_lrt_bh_within_region`. Cells with no usable
p-value are excluded from the family rather than entered as p=1.

## Heterogeneity

`p_lrt_mixture` is the likelihood-ratio test for dropping the by-subject random
slope, referenced to the 50:50 mixture of a point mass at 0 and chi2(1) because
the variance is at the boundary of the parameter space.

## Known limitations, carried forward from Phase 1

- **The p-values here are PARAMETRIC (Wald z) and provisional.** The pilot's
  permutation calibration decides whether that is legitimate. If Wald proves
  anticonservative, every p column must be replaced; betas, variance components,
  LRT statistics and BLUPs are unaffected.
- **The inherited QC mask is signal quality only.** Opioid-administration windows
  and post-ictal periods are NOT excluded -- no medication-state or
  seizure-proximity table exists in this project. Both are first-order confounds
  for low-frequency power.
- **No covariates**: no time of day, no time since admission, and no temporal
  term of any kind. Epochs within a subject are treated as exchangeable.
- **A channel random slope was not fitted** (Phase 1: improved fit in 1 of 19
  cells).
- sub-071 is in the cohort despite physically implausible MNI coordinates,
  carried through unchanged for comparability with the reference run.
"""
    if missing:
        text += f'\n## INCOMPLETE\n\nRegion outputs missing at collect time: {missing}\n'
    (run_dir / 'METHODS.md').write_text(text)


# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=['prepare', 'fit', 'collect'], required=True)
    ap.add_argument('--run-dir', default=None,
                    help='Required for --stage fit/collect; printed by prepare.')
    ap.add_argument('--region-index', type=int, default=None,
                    help='Which region this array task fits (--stage fit).')
    ap.add_argument('--view-dir', default=None)
    ap.add_argument('--reference-run', default=str(reference_run.CONTPAIN_HEATMAP))
    ap.add_argument('--mask-label', default=None)
    ap.add_argument('--allow-cohort-drift', action='store_true')
    ap.add_argument('--min-subjects', type=int, default=None,
                    help='Coverage floor. Default: inherited from the reference run '
                         '(8). Raising it drops whole regions and breaks cell-for-cell '
                         'comparability with the reference.')
    ap.add_argument('--keep-unresolvable-bins', action='store_true',
                    help='Keep log bins narrower than the FFT resolution. They are '
                         'exact duplicates of a neighbour; only for reproducing an '
                         'older run.')
    ap.add_argument('--with-channel-slope', action='store_true',
                    help='Also fit the channel random-slope model. Phase 1 found it '
                         'improved fit in 1/19 cells, so this is off by default.')
    ap.add_argument('--question', default=QUESTION)
    ap.add_argument('--view-scheme', default=VIEW_SCHEME)
    ap.add_argument('--run-name', default=RUN_NAME)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    if args.stage == 'prepare':
        stage_prepare(args)
    elif args.stage == 'fit':
        if not args.run_dir or args.region_index is None:
            raise SystemExit('--stage fit needs --run-dir and --region-index')
        stage_fit(args)
    else:
        if not args.run_dir:
            raise SystemExit('--stage collect needs --run-dir')
        stage_collect(args)


if __name__ == '__main__':
    main()

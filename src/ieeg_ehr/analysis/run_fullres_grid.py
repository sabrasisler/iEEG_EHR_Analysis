"""Mass-univariate mixed-effects models on the NATIVE 0.5 Hz frequency axis.

The same analysis as `run_mixed_model_grid` -- same model, same cohort, same ROI
scheme, same coverage floor -- moved off the 50 log-spaced bins and onto the
native FFT grid of `features/pain/psd_epochs_fullres/`:

    log10_power ~ NRS_within + NRS_submean
                  + (NRS_within || subject) + (1 | subject:channel)

21 regions x 463 bins = 9,723 cells, against the old 21 x 44 = 924.

WHAT IS DELIBERATELY DIFFERENT FROM THE 50-BIN GRID
---------------------------------------------------
- NO MULTIPLE-COMPARISON CORRECTION AND NO SIGNIFICANCE ANYWHERE. At 9,723
  cells a BH family is both enormous and almost entirely redundant -- adjacent
  0.5 Hz bins are not independent tests, they are a smooth function sampled 463
  times -- so a q-value here would answer a question nobody asked. The Wald `p`
  column still falls out of each fit and is written as data, but nothing
  corrects it, no figure outlines it, and no claim rests on it. This run exists
  to show whether the SHAPE of the old map survives a 10x finer axis.
- NO REDUCED MODEL, so no heterogeneity LRT. Dropping it halves the compute and
  costs only a p-value that would not be shown. `var_subj_slope` still comes out
  of the full fit, and the between-subject slope SD is the heterogeneity EFFECT
  SIZE the figures actually use.
- THE LINE-NOISE NOTCH IS A PARAMETER. `--notch-half-width-hz` (default 2.0 Hz,
  the structural floor in psd_params: a 2 s Hann main lobe is +/-0.5 Hz and mains
  drifts a few tenths) drops 9 bins per harmonic, 36 of 499 in total -- 58.0-62.0
  Hz and its three harmonics, 18 Hz of spectrum. The old axis dropped 6 of 50,
  where 60 Hz alone cost 13.2 Hz.
- UNPOOLED PER-SUBJECT SLOPES ARE COMPUTED IN THE FIT STAGE, not by a second
  array job over the same data (`compute_subject_slopes_grid`). They are an OLS
  line per subject -- microseconds -- and the region's rows are already resident,
  so the only thing a separate job bought was a second pass over Lustre.

STAGES, mirroring the 50-bin grid so the sbatch pattern is identical:

    prepare                 -> resolve cohort + view, create the run dir, echo
                               RUN_DIR on stdout and nothing else
    fit --region-index N     -> every bin of one region; one array task per region
    collect                 -> concatenate, derive sign consistency, write tables

EXPLORATORY. Discovery cohort only. Nominations, not findings.
"""

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import fullres_cells, mixed_model as mm
from ieeg_ehr.analysis import pain_coef, reference_run, view_tables
from ieeg_ehr.analysis.plot_mixed_model_subject_lines import epoch_level, subject_slopes
from ieeg_ehr.analysis.run_mixed_model_pilot import coverage_map, inventory_subjects, roi_maps

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/run_fullres_grid.py'

QUESTION = 'psd_physiology'
OUTPUT_TYPE = 'univariate_analysis'
#: Level 4. A SEPARATE scheme from the 50-bin grid's `cont_pain_scratch`: the
#: frequency axis is what changed, and two runs whose cells are not in
#: correspondence should not share a folder.
VIEW_SCHEME = 'cont_pain_fullres'
RUN_NAME = 'fullres_grid_mixedlm'

DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. No '
              'multiple-comparison correction and no significance testing: this '
              'run asks whether the pattern survives a 10x finer frequency axis, '
              'not whether any cell is real.')

CONFOUND_CAVEAT = (
    'The inherited QC mask is SIGNAL QUALITY ONLY (gross artifact, saturation, '
    'square wave, flatline, bipolar variance). Opioid-administration windows and '
    'post-ictal periods are NOT excluded, and both are first-order confounds for '
    'low-frequency power.')

ESTIMAND_CAVEAT = (
    'The epoch mean here is ARITHMETIC (log10 of the mean linear power, as the '
    'fullresmean view stores it); the 50-bin grid averaged log power directly, '
    'which is a GEOMETRIC mean. A channel-constant offset between the two is '
    'absorbed by the channel random intercept, so the pain slope should be '
    'nearly unaffected -- but the two maps are not estimates of literally the '
    'same quantity.')


# ============================================================================
# COHORT
# ============================================================================

def resolve_cohort(ref, view_dir, cohort='reference'):
    """(paths, scores, diagnostics, subjects, roi map, subjects without ROI).

    Same eligibility rule as the 50-bin grid, applied to the same epochs -- the
    fullres unit copies `epoch_defs` from psd_epochs, so a subject's report count
    and NRS spread are identical. It is re-derived and checked rather than copied
    precisely so that if it does NOT come out the same, we hear about it.

    TWO GATES, IN THIS ORDER, and neither is optional.

    1. THE SPLIT. The epoch-mean view was built over 81 subjects -- 30 of them
       NOT in the locked discovery cohort, because materializing a view
       legitimately runs over whatever is on disk while an ANALYSIS may not
       (CLAUDE.md, Cohorts). Those subjects are `unassigned`, i.e. still
       hold-out-eligible, and looking at one during exploration cannot be undone.
       So the candidate set is intersected with the discovery split HERE, before
       eligibility is even computed.
    2. THE REFERENCE COHORT. `cohort='reference'` (default) then restricts to the
       51 subjects of the run this is being compared against, which is what makes
       the comparison a change of frequency axis rather than a change of axis AND
       cohort. `cohort='eligible-discovery'` keeps every eligible discovery
       subject instead -- more power, but every comparison against the old map
       then spans two cohorts and has to say so.
    """
    all_paths = fullres_cells.subject_paths(view_dir)
    discovery = set(config.discovery_subjects())
    paths, outside = [], []
    for p in all_paths:
        subject, _ = fullres_cells.subject_session_of(p)
        if subject in discovery:
            paths.append(p)
        else:
            outside.append(f'sub-{subject}')
    if outside:
        logger.warning('SPLIT GATE: %d subject-session(s) in this view are NOT in '
                       'the discovery cohort and are excluded before anything is '
                       'computed: %s', len(outside), sorted(set(outside)))

    scores = fullres_cells.load_epoch_scores(paths)
    eligible, diagnostics = pain_coef.eligible_subjects(
        scores,
        min_epochs=ref.criteria.get('min_epochs', pain_coef.MIN_EPOCHS),
        min_range=ref.criteria.get('min_range', pain_coef.MIN_RANGE),
        min_non_modal=ref.criteria.get('min_non_modal', pain_coef.MIN_NON_MODAL))
    subjects = set(eligible)

    if cohort == 'reference':
        extra = sorted(subjects - set(ref.subjects))
        subjects &= set(ref.subjects)
        if extra:
            logger.info('%d eligible discovery subject(s) held out of this run '
                        'because they are not in the reference cohort: %s',
                        len(extra), extra)
    else:
        logger.warning("cohort='%s': keeping every eligible discovery subject. Any "
                       'comparison with the reference map now spans two cohorts.',
                       cohort)

    roi_scheme = ref.view_params.get('roi_scheme', 'roi_v2')
    roi_by_subject, no_roi = roi_maps(paths, subjects, roi_scheme)
    subjects -= set(no_roi)
    return paths, scores, diagnostics, subjects, roi_by_subject, no_roi


# ============================================================================
# STAGE: prepare
# ============================================================================

def build_grid_manifest(regions, bin_table, coverage, min_subjects):
    """Every region x bin, with its coverage and whether it clears the floor.

    Cells BELOW the floor stay as ROWS rather than vanishing: "which cells were
    not fitted, and why" is part of the result, and a manifest holding only the
    fittable cells cannot answer it.
    """
    rows = []
    for region in regions:
        for b in bin_table.index:
            lo = float(bin_table.loc[b, 'bin_low_hz'])
            hi = float(bin_table.loc[b, 'bin_high_hz'])
            n = int(coverage.get((region, int(b)), 0))
            rows.append({'region': region, 'freq_bin_index': int(b),
                         'freq_hz': 0.5 * (lo + hi),
                         'bin_low_hz': lo, 'bin_high_hz': hi,
                         'n_subjects_covered': n,
                         'above_coverage_floor': n >= min_subjects})
    man = pd.DataFrame(rows)
    man.insert(0, 'cell_index', range(len(man)))
    return man


def stage_prepare(args):
    ref = reference_run.load(args.reference_run)
    ref.describe()

    view_dir = fullres_cells.resolve_view_dir(
        args.view_dir,
        mask_label=args.mask_label or ref.view_params.get('mask_label'),
        max_excluded_frac=ref.view_params.get('max_excluded_frac'),
        epoch_minutes=ref.view_params.get('epoch_minutes'))
    logger.info('epoch-mean full-res view: %s', view_dir)

    paths, scores, diagnostics, subjects, roi_by_subject, no_roi = resolve_cohort(
        ref, view_dir, cohort=args.cohort)
    ref.assert_cohort_matches(subjects, allow_drift=args.allow_cohort_drift
                              or args.cohort != 'reference')

    roi_scheme = ref.view_params.get('roi_scheme', 'roi_v2')
    regions = view_tables.roi_regions_for({'roi_scheme': roi_scheme})
    epoch_minutes = ref.view_params.get('epoch_minutes')

    bin_table, notched = fullres_cells.analysis_freq_table(
        args.notch_half_width_hz, epoch_minutes)

    # What happened to every contact the view carries. `absent_from_meta` is the
    # only half that is a problem -- a pair channel_meta has never heard of --
    # and it is reported separately from the non-ROI contacts (white matter, CSF,
    # Exclude), which every subject has and the 50-bin analysis dropped too.
    audit = fullres_cells.channel_audit(paths, subjects, roi_by_subject)
    absent = {k: v['absent_from_meta'] for k, v in audit.items()
              if v['absent_from_meta']}
    if absent:
        logger.error('%d subject(s) carry bipolar pairs the channel_meta table has '
                     'never heard of. These enter NO region: %s', len(absent),
                     {k: len(v) for k, v in sorted(absent.items())})
    logger.info('contacts: %d in the view, %d with an ROI, %d labelled non-ROI '
                '(white matter / CSF / Exclude / Other -- expected, and the same '
                'contacts the 50-bin analysis dropped)',
                sum(v['n_in_view'] for v in audit.values()),
                sum(v['n_with_roi'] for v in audit.values()),
                sum(v['n_non_roi'] for v in audit.values()))

    min_subjects = int(args.min_subjects if args.min_subjects is not None
                       else ref.criteria.get('min_subjects', mm.MIN_SUBJECTS))
    coverage = coverage_map(paths, subjects, roi_by_subject, bin_table.index)
    manifest = build_grid_manifest(regions, bin_table, coverage, min_subjects)

    n_fit = int(manifest['above_coverage_floor'].sum())
    logger.info('%d regions x %d bins = %d cells; %d clear the coverage floor '
                '(min_subjects=%d), %d do not and are reported unfitted',
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
    (run_dir / 'subject_slopes').mkdir(parents=True, exist_ok=True)

    params = {'cohort': args.cohort,
              'min_subjects': min_subjects,
              'notch_half_width_hz': (args.notch_half_width_hz
                                      if args.notch_half_width_hz is not None
                                      else config.PSD_NOTCH_HALF_WIDTH_HZ),
              'line_noise_freqs_hz': list(config.PSD_LINE_NOISE_FREQS_HZ),
              'notched_bins_removed': notched,
              'roi_scheme': roi_scheme,
              'epoch_minutes': epoch_minutes}

    io.write_table(manifest, run_dir / 'grid_cell_manifest.parquet',
                   params=params,
                   parents=[str(Path(args.reference_run) / 'provenance.json'),
                            str(view_dir)],
                   subjects=sorted(subjects), script=SCRIPT)
    io.write_table(inventory_subjects(scores, diagnostics),
                   run_dir / 'inventory_subjects.parquet', script=SCRIPT)
    io.write_table(pd.DataFrame({'region_index': range(len(regions)),
                                 'region': regions}),
                   run_dir / 'region_index.parquet', script=SCRIPT)

    io.write_run_provenance(
        run_dir, script=SCRIPT,
        params={'stage': 'prepare', 'view_dir': str(view_dir),
                'view_params': ref.view_params, 'criteria': ref.criteria,
                'n_regions': len(regions), 'n_bins': len(bin_table),
                'n_cells': len(manifest), 'n_cells_fittable': n_fit,
                'with_reduced_model': False, 'fdr': None, **params},
        parents=[str(Path(args.reference_run) / 'provenance.json'), str(view_dir)],
        subjects=sorted(subjects),
        extra={'status': DISCLAIMER,
               'subjects_without_roi': sorted(no_roi),
               'channel_audit': {k: v for k, v in sorted(audit.items())},
               'inference': 'NONE. No BH, no permutation, no outlines. The Wald p '
                            'column is written as data and is not corrected; at '
                            '0.5 Hz spacing adjacent cells are a smooth function '
                            'sampled 463 times, not 463 independent tests.',
               'estimand': ESTIMAND_CAVEAT,
               'mask_content': CONFOUND_CAVEAT})

    # The ONLY thing on stdout. Every log line above went to stderr.
    print(run_dir)
    return run_dir


# ============================================================================
# STAGE: fit (one region per array task)
# ============================================================================

def fit_one_cell(df, meta):
    """(record, per-subject slope frame) for one cell. FULL model only.

    No reduced model, so `p_lrt_mixture` comes out NaN by construction rather
    than by omission -- `mixed_model.cell_record` is passed None and records the
    absence, which is why this returns the same schema as the 50-bin grid.
    """
    region, b = meta['region'], meta['freq_bin_index']
    lo, hi = meta['bin_low_hz'], meta['bin_high_hz']

    ok, reason = mm.cell_is_fittable(df)
    if not ok:
        rec = mm.failed_record(region, b, lo, hi, reason, df=df)
        rec.update({'cell_index': meta['cell_index'], 'freq_hz': meta['freq_hz']})
        return rec, None

    t0 = time.time()
    try:
        res, warn = mm.fit_cell(df, mm.VC_FULL)
    except mm.CellFitError as exc:
        rec = mm.failed_record(region, b, lo, hi, f'full: {exc}', df=df,
                               fit_seconds=time.time() - t0)
        rec.update({'cell_index': meta['cell_index'], 'freq_hz': meta['freq_hz']})
        return rec, None

    rec = mm.cell_record(res, None, df, region=region, freq_bin_index=b,
                         bin_low_hz=lo, bin_high_hz=hi,
                         fit_seconds=time.time() - t0, warnings_full=warn)
    rec['cell_index'] = meta['cell_index']
    rec['freq_hz'] = meta['freq_hz']
    rec['nrs_within_var'] = float(np.var(df['NRS_within'].to_numpy(), ddof=0))

    # UNPOOLED per-subject slopes, for sign consistency. Not the BLUPs: partial
    # pooling drags every subject toward the group, so BLUP-based consistency
    # sits at 0.8-1.0 almost everywhere and separates nothing.
    slopes = subject_slopes(epoch_level(df))
    slopes['region'] = region
    slopes['freq_bin_index'] = int(b)
    slopes['freq_hz'] = meta['freq_hz']
    return rec, slopes


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
    below = int(((manifest['region'] == region)
                 & ~manifest['above_coverage_floor']).sum())
    logger.info('region %d/%d: %s -- %d cells to fit, %d below the coverage floor',
                args.region_index, len(region_index), region, len(todo), below)

    cells_path = run_dir / 'grid' / f'region_{args.region_index:03d}.parquet'
    slopes_path = run_dir / 'subject_slopes' / f'region_{args.region_index:03d}.parquet'

    if todo.empty:
        # Still write both files. A missing output is indistinguishable from a
        # task that died, and collect would have no way to tell them apart.
        for path in (cells_path, slopes_path):
            io.write_table(pd.DataFrame(columns=['region', 'freq_bin_index']), path,
                           params={'region': region, 'n_cells': 0}, script=SCRIPT)
        logger.info('nothing fittable in %s; wrote empty results', region)
        return

    view_dir = fullres_cells.resolve_view_dir(
        args.view_dir,
        mask_label=args.mask_label or ref.view_params.get('mask_label'),
        max_excluded_frac=ref.view_params.get('max_excluded_frac'),
        epoch_minutes=ref.view_params.get('epoch_minutes'))
    paths, _, _, subjects, roi_by_subject, _ = resolve_cohort(ref, view_dir,
                                                              cohort=args.cohort)

    bins = [int(b) for b in todo['freq_bin_index']]
    t0 = time.time()
    index, values, stats = fullres_cells.load_region_matrix(
        paths, subjects, region, roi_by_subject, bins,
        epoch_minutes=ref.view_params.get('epoch_minutes'))
    logger.info('%s: loaded %d rows x %d bins from %d subject file(s) in %.1fs '
                '(%.0f MB, %d non-finite values)', region, len(index), len(bins),
                stats['n_files'], time.time() - t0, values.nbytes / 1e6,
                stats['n_nonfinite'])
    if not len(index):
        raise SystemExit(f'{region}: no rows after the ROI join, yet the manifest '
                         'says its cells clear the coverage floor. The channel map '
                         'and the view disagree; do not fit past this.')

    metas = [{'region': region, 'freq_bin_index': int(r.freq_bin_index),
              'freq_hz': float(r.freq_hz), 'bin_low_hz': float(r.bin_low_hz),
              'bin_high_hz': float(r.bin_high_hz),
              'cell_index': int(r.cell_index)} for r in todo.itertuples()]

    t0 = time.time()
    results = _fit_all(index, values, metas, n_jobs=args.n_jobs)
    records = [rec for rec, _ in results]
    slope_parts = [s for _, s in results if s is not None]

    cells = pd.DataFrame(records)
    io.write_table(cells, cells_path,
                   params={'region': region, 'n_cells': len(cells),
                           'with_reduced_model': False},
                   parents=[str(run_dir / 'grid_cell_manifest.parquet')],
                   script=SCRIPT)
    io.write_table(pd.concat(slope_parts, ignore_index=True) if slope_parts
                   else pd.DataFrame(columns=['subject', 'slope', 'se', 'p',
                                              'region', 'freq_bin_index']),
                   slopes_path,
                   params={'region': region,
                           'source': 'unpooled per-subject OLS, not BLUPs'},
                   parents=[str(run_dir / 'grid_cell_manifest.parquet')],
                   script=SCRIPT)

    ok = int(cells['converged'].fillna(False).sum()) if 'converged' in cells else 0
    logger.info('%s done: %d/%d converged, %.0fs wall over %d job(s) '
                '(%.0fs of fitting)', region, ok, len(cells), time.time() - t0,
                args.n_jobs, cells.get('fit_seconds', pd.Series(dtype=float)).sum())


def _fit_all(index, values, metas, n_jobs=1):
    """Fit every cell of one region, optionally across processes.

    The per-cell FRAME is built in the parent and handed to the worker, rather
    than handing every worker the whole region matrix: the matrix is tens of MB
    and would be pickled once per task, while a frame is ~2 MB. Passing a
    GENERATOR keeps joblib's dispatch lazy, so at most `pre_dispatch` frames are
    alive at once instead of all 463.

    BLAS is pinned to one thread per worker. These are small dense problems and
    letting each process grab every core costs more in contention than it wins --
    `mixed_model.permutation_null` makes the same call for the same reason.
    """
    def one(df, meta):
        return fit_one_cell(df, meta)

    def frames():
        for j, meta in enumerate(metas):
            yield fullres_cells.cell_frame(
                index, values, j, region=meta['region'],
                freq_bin_index=meta['freq_bin_index']), meta

    if n_jobs and n_jobs > 1:
        from joblib import Parallel, delayed
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            return Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(one)(df, meta) for df, meta in frames())
    return [one(df, meta) for df, meta in frames()]


# ============================================================================
# STAGE: collect
# ============================================================================

def sign_consistency(cells, slopes):
    """Per cell: the fraction of subjects whose UNPOOLED slope shares the group's
    sign, and how many subjects had a slope at all.

    A subject with fewer than two distinct pain scores has no line, and must not
    enter the denominator as though they had disagreed -- so `n_with_slope` is
    reported beside the fraction rather than assumed equal to the cohort.
    """
    merged = slopes.merge(cells[['region', 'freq_bin_index', 'beta_nrs_within']],
                          on=['region', 'freq_bin_index'], how='inner')
    ok = merged[merged['slope'].notna() & merged['beta_nrs_within'].notna()].copy()
    if ok.empty:
        return pd.DataFrame(columns=['region', 'freq_bin_index',
                                     'frac_sign_consistent', 'n_with_slope'])
    ok['agrees'] = np.sign(ok['slope']) == np.sign(ok['beta_nrs_within'])
    return (ok.groupby(['region', 'freq_bin_index'])
            .agg(frac_sign_consistent=('agrees', 'mean'),
                 n_with_slope=('agrees', 'size'))
            .reset_index())


def _collect_parts(run_dir, subdir, region_index):
    parts, missing = [], []
    for r in region_index.itertuples():
        p = Path(run_dir) / subdir / f'region_{int(r.region_index):03d}.parquet'
        if not p.exists():
            missing.append(r.region)
            continue
        df = io.read_table(p, on_stale='ignore')
        if len(df):
            parts.append(df)
    return parts, missing


def stage_collect(args):
    run_dir = Path(args.run_dir)
    manifest = io.read_table(run_dir / 'grid_cell_manifest.parquet', on_stale='warn')
    region_index = io.read_table(run_dir / 'region_index.parquet', on_stale='warn')

    parts, missing = _collect_parts(run_dir, 'grid', region_index)
    if missing:
        # Loud and recorded -- a silently short grid looks exactly like a real
        # result with a few quiet regions.
        logger.error('MISSING region outputs (array tasks that did not finish): %s',
                     missing)
    if not parts:
        raise SystemExit('no region outputs to collect')
    cells = pd.concat(parts, ignore_index=True)

    slope_parts, slope_missing = _collect_parts(run_dir, 'subject_slopes',
                                                region_index)
    slopes = (pd.concat(slope_parts, ignore_index=True) if slope_parts
              else pd.DataFrame())
    if slope_missing:
        logger.error('MISSING per-subject slope outputs: %s', slope_missing)

    io.write_table(cells, run_dir / 'grid_cells.parquet',
                   params={'fdr': None, 'with_reduced_model': False,
                           'inference': 'none -- betas and effect sizes only'},
                   parents=[str(run_dir / 'grid_cell_manifest.parquet')],
                   script=SCRIPT)

    cons = pd.DataFrame()
    if len(slopes):
        io.write_table(slopes, run_dir / 'subject_slopes.parquet',
                       params={'source': 'unpooled per-subject OLS'},
                       parents=[str(run_dir / 'grid_cells.parquet')],
                       subjects=sorted(slopes['subject'].dropna().unique()),
                       script=SCRIPT)
        cons = sign_consistency(cells, slopes)
        io.write_table(cons, run_dir / 'sign_consistency.parquet',
                       params={'definition': 'fraction of subjects whose UNPOOLED '
                                             'slope shares the sign of the group '
                                             'fixed effect'},
                       parents=[str(run_dir / 'subject_slopes.parquet')],
                       script=SCRIPT)

    report(cells, cons, manifest, missing)
    write_methods(run_dir, cells, manifest, missing)
    io.log_analysis('mass-univariate mixed models on the NATIVE 0.5 Hz PSD axis '
                    '(no correction, no significance; EXPLORATORY)', run_dir)
    print(run_dir)


def report(cells, cons, manifest, missing):
    conv = int(cells['converged'].fillna(False).sum())
    sing = int(cells['singular_flag'].fillna(False).sum())
    logger.info('=' * 72)
    logger.info('FULL-RESOLUTION GRID SUMMARY')
    logger.info('  manifest cells            : %d (%d above the coverage floor)',
                len(manifest), int(manifest['above_coverage_floor'].sum()))
    logger.info('  fitted rows collected     : %d', len(cells))
    logger.info('  converged                 : %d/%d', conv, len(cells))
    logger.info('  singular (a VC at boundary): %d', sing)
    if 'beta_nrs_within' in cells:
        b = cells['beta_nrs_within'].dropna()
        logger.info('  |beta| median %.5f  p95 %.5f  max %.5f',
                    b.abs().median(), b.abs().quantile(0.95), b.abs().max())
        logger.info('  sign split                : %d negative / %d positive',
                    int((b < 0).sum()), int((b > 0).sum()))
    if len(cons):
        logger.info('  sign consistency          : median %.3f, range %.3f-%.3f',
                    cons['frac_sign_consistent'].median(),
                    cons['frac_sign_consistent'].min(),
                    cons['frac_sign_consistent'].max())
    logger.info('  fit time                  : median %.2fs, total %.0f min',
                cells['fit_seconds'].median(), cells['fit_seconds'].sum() / 60)
    if missing:
        logger.error('  INCOMPLETE -- missing regions: %s', missing)
    logger.info('  NO p-VALUE IS CORRECTED OR SHOWN. %s', DISCLAIMER)
    logger.info('=' * 72)


def write_methods(run_dir, cells, manifest, missing):
    import json
    try:
        prov = json.loads((Path(run_dir) / 'provenance.json').read_text())
        params = prov.get('params', {})
    except (OSError, ValueError):
        params = {}
    notched = params.get('notched_bins_removed', [])
    half = params.get('notch_half_width_hz', '?')
    floor = params.get('min_subjects', '?')

    text = f"""# Mass-univariate mixed models on the native 0.5 Hz PSD axis

{DISCLAIMER}

## Model

    log10_power ~ NRS_within + NRS_submean
                  + (NRS_within || subject) + (1 | subject:channel)

One row = one channel x one 5-minute pre-report epoch. NRS is split into a
within-subject deviation and a subject mean so the between-subject contrast
cannot leak into the within-subject slope. Fitted with REML. Identical to the
50-log-bin grid; only the frequency axis changed.

## Cells

{len(manifest)} region x frequency cells in the manifest
({manifest['region'].nunique()} regions x {manifest['freq_bin_index'].nunique()} bins);
{int(manifest['above_coverage_floor'].sum())} clear the coverage floor of
min_subjects={floor} and were fitted. {len(cells)} fitted rows were collected.
The 50-bin grid this replaces had 924 cells.

## Frequency axis

The NATIVE FFT grid from the unit's manifest: 0.5 Hz spacing, 1-250 Hz
inclusive, 499 bins, identical at 500/1000/2000 Hz sampling because df =
1/PSD_WINDOW_SEC. No log binning, no band aggregation. {len(notched)} bins are
removed as line noise, at +/-{half} Hz around each 60 Hz harmonic. The selection
is inclusive at both ends, so that is 9 bins per harmonic (58.0-62.0 Hz and its
three harmonics, 4.5 Hz of the axis each, 18 Hz in total) -- against the old
axis's 6 of 50 bins, where 60 Hz alone cost 13.2 Hz. The notch width is a view
parameter (`--notch-half-width-hz`), not a stored flag.

None of the old axis's *unresolvable* bins exist here: that problem was log bins
narrower than 0.5 Hz being filled with a copy of a neighbour, and every bin on
this axis is one real FFT frequency.

## Inference: none

No Benjamini-Hochberg, no permutation, no cluster test, no outlines on any
figure. At 0.5 Hz spacing the 463 bins of a region are a smooth function sampled
463 times, not 463 independent tests, so a correction family over them would
answer a question that was not asked. The Wald `p` column falls out of each fit
and is written as data; it is uncorrected and nothing here rests on it.

No reduced model was fitted, so `lrt_stat` and `p_lrt_mixture` are NaN by
construction. Heterogeneity is reported as an EFFECT SIZE instead:
`var_subj_slope` from the full fit, shown as the between-subject slope SD
divided by the residual SD.

## Sign consistency

`sign_consistency.parquet` is the fraction of subjects whose UNPOOLED per-subject
OLS slope shares the sign of the group fixed effect, computed in the fit stage
from the same rows. Not BLUPs: partial pooling drags every subject toward the
group, so BLUP-based consistency sits at 0.8-1.0 almost everywhere and separates
nothing. `n_with_slope` is the denominator -- a subject with one distinct pain
score has no line and is not counted as disagreeing.

## Known limitations

- **{ESTIMAND_CAVEAT}**
- {CONFOUND_CAVEAT}
- **No covariates**: no time of day, no time since admission, no temporal term of
  any kind. Epochs within a subject are treated as exchangeable.
- A channel random slope was not fitted (Phase 1: improved fit in 1 of 19 cells).
- sub-071 is in the cohort despite physically implausible MNI coordinates,
  carried through unchanged for comparability with the reference run.
- ~649M non-finite values across 53 subject-sessions of the per-window cache are
  uncharacterized upstream. Non-finite epoch means are dropped per cell by
  `mixed_model.build_cell_frame`, and each region's count is in its fit log.
"""
    if missing:
        text += f'\n## INCOMPLETE\n\nRegion outputs missing at collect: {missing}\n'
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
    ap.add_argument('--view-dir', default=None,
                    help='Epoch-mean full-res view directory. Default: resolved the '
                         'same way its builder resolves it, so the config hash is '
                         'never spelled twice.')
    ap.add_argument('--reference-run', default=str(reference_run.CONTPAIN_HEATMAP))
    ap.add_argument('--mask-label', default=None,
                    help="Override the reference run's mask label. Almost never right.")
    ap.add_argument('--cohort', choices=['reference', 'eligible-discovery'],
                    default='reference',
                    help="'reference' (default) uses the reference run's cohort, so "
                         'the only thing that changed vs the old map is the '
                         'frequency axis. \'eligible-discovery\' uses every eligible '
                         'DISCOVERY subject in the view instead -- more power, but '
                         'the comparison then spans two cohorts. Non-discovery '
                         'subjects are excluded under either choice.')
    ap.add_argument('--allow-cohort-drift', action='store_true')
    ap.add_argument('--min-subjects', type=int, default=None,
                    help='Coverage floor. Default: inherited from the reference run.')
    ap.add_argument('--notch-half-width-hz', type=float, default=None,
                    help=f'Half-width of the line-noise notch around each of '
                         f'{config.PSD_LINE_NOISE_FREQS_HZ} Hz. Default: '
                         f'{config.PSD_NOTCH_HALF_WIDTH_HZ} Hz, which takes 9 bins '
                         'per harmonic and 36 of 499 in total (the comparison is '
                         'inclusive at both ends) -- the structural floor: '
                         'a 2 s Hann main lobe is +/-0.5 Hz and mains drifts a few '
                         'tenths.')
    ap.add_argument('--n-jobs', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK', 1)),
                    help='Processes fitting cells within one region task.')
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

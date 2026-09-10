"""Medication-stratified mixed-effects models over the region x frequency grid.

The full grid (`run_mixed_model_grid`) fits one pain slope per cell across every
epoch. Its own METHODS.md names analgesia as the largest untested confound: the
inherited QC mask is signal quality only, and a drug given shortly before an
assessment plausibly moves low-frequency power directly. This splits each epoch
by whether an analgesic was given in the 2 hours before its pain score and fits
the same model in each stratum.

THREE FITS PER CELL, and the third is the one that licenses a comparison:

    medicated     log10_power ~ NRS_within + NRS_submean            (stratum only)
    unmedicated   log10_power ~ NRS_within + NRS_submean            (stratum only)
    interaction   log10_power ~ NRS_within * med_state + NRS_submean  (all rows)

Two stratified maps cannot support a claim that they differ -- "significant in
one, not the other" is not a test of a difference. The interaction gives that
test directly, on one cohort, with medication state varying WITHIN subject (all
51 subjects contribute epochs to both strata).

CENTRING IS PER STRATUM, and it matters. `add_nrs_components` centres over the
rows it is given, so re-running it after the split makes `NRS_within` a
deviation from that subject's mean pain WITHIN that stratum. Reusing the
whole-data centring instead would leave the stratum means in the predictor and
the within/between split would no longer be orthogonal in the fitted sample.

CONFOUNDING BY INDICATION -- read this before reading any output. Patients are
medicated BECAUSE they are in pain. On this cohort the medicated epochs average
NRS 4.15 and the unmedicated 2.01 (opioids: 4.67 vs 2.22). The strata are not
exchangeable, the unmedicated stratum's slope is estimated over a compressed pain
range, and a difference between the two maps confounds pharmacology with pain
severity. This is reported, not adjusted for, by decision.

    python -m ieeg_ehr.analysis.run_mixed_model_med_strata --stage prepare \
        --drug-set analgesics
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import (cluster_permutation, med_state, mixed_model as mm,
                               pain_coef, reference_run, view_tables)
from ieeg_ehr.analysis.plot_mixed_model_subject_lines import (epoch_level,
                                                              subject_slopes)
from ieeg_ehr.analysis.run_mixed_model_grid import (OUTPUT_TYPE, QUESTION, add_fdr,
                                                    build_grid_manifest,
                                                    resolve_cohort)
from ieeg_ehr.analysis.run_mixed_model_pilot import (coverage_map, fit_one_cell,
                                                     load_cell_frames,
                                                     load_epoch_scores, roi_maps,
                                                     resolve_view_dir,
                                                     view_subject_paths)
from ieeg_ehr.views import cache_reader

logger = logging.getLogger(__name__)

RUN_NAME = 'medstrata'
FDR_Q = 0.05

#: Subdirectory per fit. Named `grid_cells.parquet` inside each so every existing
#: plotting script works on them unchanged.
STRATA = ('medpos', 'medneg', 'interaction', 'matched')


def stratum_dir(run_dir, name):
    return Path(run_dir) / name


def run_provenance(run_dir):
    """The run's provenance.json as a dict, or {} if unreadable."""
    import json
    try:
        return json.loads((Path(run_dir) / 'provenance.json').read_text())
    except (OSError, ValueError):
        return {}


# ============================================================================
# STAGE: prepare
# ============================================================================

def stratum_eligibility(scores, state, ref):
    """(eligible per stratum, intersection). Eligibility RE-DERIVED per stratum.

    A subject who reported ten scores of which two fall in a stratum has almost
    no predictor spread there, and the reference criteria exist precisely to
    exclude that. Applying them to the pooled data and then splitting would carry
    a subject into a stratum their data cannot support.

    The INTERSECTION is the run's cohort: both maps then describe the same
    patients, which is the only way a cell-for-cell comparison means anything.
    """
    keyed = state.assign(subject_id='sub-' + state['subject'])
    per_stratum = {}
    for name, want in (('medpos', True), ('medneg', False)):
        keep = keyed.loc[keyed['med_state'] == want, ['subject_id', 'epoch_id']]
        sub = scores.merge(keep, on=['subject_id', 'epoch_id'], how='inner')
        eligible, _ = pain_coef.eligible_subjects(
            sub,
            min_epochs=ref.criteria.get('min_epochs', pain_coef.MIN_EPOCHS),
            min_range=ref.criteria.get('min_range', pain_coef.MIN_RANGE),
            min_non_modal=ref.criteria.get('min_non_modal', pain_coef.MIN_NON_MODAL))
        per_stratum[name] = sorted(eligible)
        logger.info('%s: %d epochs, %d subjects eligible', name, len(sub),
                    len(eligible))
    both = sorted(set(per_stratum['medpos']) & set(per_stratum['medneg']))
    logger.info('intersection cohort: %d subjects', len(both))
    return per_stratum, both


def stage_prepare(args):
    ref = reference_run.load(args.reference_run)
    ref.describe()

    view_dir = resolve_view_dir(args.view_dir,
                                mask_label=ref.view_params.get('mask_label'),
                                roi_scheme=ref.view_params.get('roi_scheme', 'roi_v2'))
    logger.info('per-channel view: %s', view_dir)

    paths, _, _, subjects, roi_by_subject, no_roi = resolve_cohort(ref, view_dir)
    ref.assert_cohort_matches(subjects, allow_drift=args.allow_cohort_drift)

    subclasses = med_state.DRUG_SETS[args.drug_set]
    admin = med_state.load_admin_table(subclasses=subclasses)
    defs = med_state.load_epoch_defs(subjects=subjects)
    state = med_state.epoch_med_state(defs, admin, hours=args.hours)

    missing = sorted(set(defs['subject']) - set(admin['subject']))
    if missing:
        # Not fatal -- 'no analgesic charted' is a real observation -- but it must
        # be visible, because a session absent from the MAR export would look
        # exactly like a patient who was never medicated.
        logger.warning('%d cohort subject(s) have NO %s administrations at all: %s',
                       len(missing), args.drug_set, missing)

    summary = med_state.stratum_summary(state)
    logger.info('\n%s', summary.to_string(index=False))

    scores = load_epoch_scores(paths)
    per_stratum, cohort_ids = stratum_eligibility(scores, state, ref)
    if args.per_stratum_cohort:
        logger.warning('--per-stratum-cohort: each stratum keeps its own eligible '
                       'set, so the two maps describe DIFFERENT patients and are '
                       'not comparable cell for cell.')
    cohort = set(cohort_ids) if not args.per_stratum_cohort else subjects
    if not cohort:
        raise SystemExit('no subjects eligible in both strata')

    roi_scheme = ref.view_params.get('roi_scheme', 'roi_v2')
    regions = view_tables.roi_regions_for({'roi_scheme': roi_scheme})

    epoch_minutes = ref.view_params.get('epoch_minutes')
    bin_table = cache_reader.bin_edges(epoch_minutes).set_index('freq_bin_index')
    line_noise = list(ref.line_noise_bins_removed
                      or cache_reader.line_noise_bins(epoch_minutes))
    bin_table = bin_table.drop(index=[b for b in line_noise if b in bin_table.index])
    unresolvable = [int(b) for b in cache_reader.unresolvable_bins(epoch_minutes)
                    if b in bin_table.index]
    bin_table = bin_table.drop(index=unresolvable)
    logger.info('dropped %d unresolvable (duplicate) bin(s): %s',
                len(unresolvable), unresolvable)

    coverage = coverage_map(paths, cohort, roi_by_subject, bin_table.index)
    manifest = build_grid_manifest(regions, bin_table, coverage, args.min_subjects)
    logger.info('%d regions x %d bins = %d cells; %d clear min_subjects=%d',
                len(regions), len(bin_table), len(manifest),
                int(manifest['above_coverage_floor'].sum()), args.min_subjects)

    run_dir = config.analysis_run_dir(
        question=args.question, output_type=OUTPUT_TYPE,
        view_scheme=f'med_strata_{args.drug_set}', run_name=args.run_name)
    for name in STRATA:
        (stratum_dir(run_dir, name)).mkdir(parents=True, exist_ok=True)
    (run_dir / 'cells').mkdir(parents=True, exist_ok=True)

    params = {'drug_set': args.drug_set, 'subclasses': list(subclasses),
              'window_hours': args.hours, 'min_subjects': args.min_subjects,
              'unresolvable_bins_removed': unresolvable,
              'line_noise_bins_removed': line_noise, 'roi_scheme': roi_scheme,
              'per_stratum_cohort': bool(args.per_stratum_cohort)}

    io.write_table(state, run_dir / 'epoch_med_state.parquet', params=params,
                   parents=[str(med_state.ADMIN_TABLE)],
                   script='ieeg_ehr/analysis/run_mixed_model_med_strata.py')
    io.write_table(summary, run_dir / 'stratum_summary.csv', params=params,
                   script='ieeg_ehr/analysis/run_mixed_model_med_strata.py')
    io.write_table(manifest, run_dir / 'grid_cell_manifest.parquet', params=params,
                   parents=[str(Path(args.reference_run) / 'provenance.json'),
                            str(view_dir)],
                   subjects=sorted(cohort),
                   script='ieeg_ehr/analysis/run_mixed_model_med_strata.py')
    io.write_table(pd.DataFrame({'region_index': range(len(regions)),
                                 'region': regions}),
                   run_dir / 'region_index.parquet',
                   script='ieeg_ehr/analysis/run_mixed_model_med_strata.py')

    io.write_run_provenance(
        run_dir, script='ieeg_ehr/analysis/run_mixed_model_med_strata.py',
        params={**params, 'stage': 'prepare', 'view_dir': str(view_dir),
                'view_params': ref.view_params, 'criteria': ref.criteria,
                'eligible_medpos': per_stratum['medpos'],
                'eligible_medneg': per_stratum['medneg'],
                'n_cells': len(manifest)},
        parents=[str(Path(args.reference_run) / 'provenance.json'), str(view_dir),
                 str(med_state.ADMIN_TABLE)],
        subjects=sorted(cohort),
        extra={'status': 'EXPLORATORY medication-stratified grid, NOT a finding',
               'subjects_without_roi': sorted(no_roi),
               'subjects_without_administrations': missing,
               'confounding_by_indication':
                   'Patients are medicated BECAUSE they are in pain. Medicated '
                   f'epochs average NRS {summary.set_index("stratum").loc["medicated", "nrs_mean"]:.2f} '
                   f'and unmedicated {summary.set_index("stratum").loc["unmedicated", "nrs_mean"]:.2f}. '
                   'The strata are NOT exchangeable; the unmedicated slope is '
                   'estimated over a compressed pain range; a difference between '
                   'the maps confounds pharmacology with pain severity. Not '
                   'adjusted for, by decision.',
               'p_provisional': 'Parametric Wald p; the pilot permutation put the '
                                'null z SD at ~1.03.'})

    print(run_dir)
    return run_dir


# ============================================================================
# STAGE: fit
# ============================================================================

def fit_interaction(df, meta):
    """The interaction fit. Separate from `fit_one_cell` because the formula and
    the extracted terms both differ; everything else is deliberately identical."""
    t0 = time.time()
    try:
        res, warn_full = mm.fit_cell(df, mm.VC_FULL,
                                     formula=mm.FORMULA_MED_INTERACTION)
    except mm.CellFitError as exc:
        rec = mm.failed_record(meta['region'], meta['freq_bin_index'],
                               meta['bin_low_hz'], meta['bin_high_hz'],
                               f'interaction: {exc}', df=df,
                               fit_seconds=time.time() - t0)
        return rec, []
    t_full = time.time() - t0

    try:
        res_red, warn_red = mm.fit_cell(df, mm.VC_REDUCED,
                                        formula=mm.FORMULA_MED_INTERACTION)
    except mm.CellFitError as exc:
        res_red, warn_red = None, [f'reduced failed: {exc}']

    rec = mm.cell_record(res, res_red, df, region=meta['region'],
                         freq_bin_index=meta['freq_bin_index'],
                         bin_low_hz=meta['bin_low_hz'],
                         bin_high_hz=meta['bin_high_hz'], fit_seconds=t_full,
                         warnings_full=warn_full, warnings_reduced=warn_red,
                         extra_terms=[(mm.MED_INTERACTION_TERM, 'med_ix'),
                                      ('med_state', 'med_main')])
    rec['cell_index'] = meta['cell_index']
    rec['group'] = 'interaction'
    blups = mm.blup_rows(res, df, region=meta['region'],
                         freq_bin_index=meta['freq_bin_index'])
    logger.info('%-18s bin %2d | INTERACTION beta %+.5f p %.4g | base %+.5f',
                meta['region'], meta['freq_bin_index'], rec['med_ix_beta'],
                rec['med_ix_p'], rec['beta_nrs_within'])
    return rec, blups


def matched_subject_diffs(df):
    """Per subject: mean over NRS levels of (medicated power - unmedicated power).

    A NONPARAMETRIC companion to the matched model, and the honest one. For each
    (subject, NRS level) that has epochs in BOTH strata it takes the difference of
    the two means, then averages those differences within subject. Levels present
    in only one stratum contribute nothing, so nothing is ever compared across
    pain levels and the shape of the pain->power curve is irrelevant by
    construction.

    Averaging the level differences UNWEIGHTED is deliberate: weighting by epoch
    count would let NRS=0, which is a third of all unmedicated epochs, dominate a
    quantity that is supposed to describe the whole scale.
    """
    e = (df.groupby(['subject', 'epoch_id'], as_index=False)
         .agg(y=('log10_power', 'mean'), NRS=('NRS', 'first'),
              med=('med_state', 'first')))
    rows = []
    for subject, g in e.groupby('subject'):
        diffs = []
        for nrs, gg in g.groupby('NRS'):
            pos, neg = gg[gg['med'] == 1.0], gg[gg['med'] == 0.0]
            if len(pos) and len(neg):
                diffs.append(pos['y'].mean() - neg['y'].mean())
        if diffs:
            rows.append({'subject': subject, 'diff': float(np.mean(diffs)),
                         'n_levels': len(diffs)})
    return pd.DataFrame(rows)


def fit_matched(df, meta):
    """The matched-NRS fit: is power different at the SAME reported pain?"""
    t0 = time.time()
    n_sub = df['subject'].nunique()
    base = {'region': meta['region'], 'freq_bin_index': meta['freq_bin_index'],
            'freq_bin_low': meta['bin_low_hz'], 'freq_bin_high': meta['bin_high_hz'],
            'cell_index': meta['cell_index'], 'n_subjects': n_sub,
            'n_channels': df['channel_uid'].nunique(), 'n_rows': len(df)}

    diffs = matched_subject_diffs(df)
    base['nonparam_diff'] = float(diffs['diff'].mean()) if len(diffs) else np.nan
    base['n_subjects_matched'] = int(len(diffs))
    base['frac_sign_consistent'] = (
        float((np.sign(diffs['diff']) == np.sign(diffs['diff'].mean())).mean())
        if len(diffs) else np.nan)

    # Both strata must be present, or `med_state` is constant and unidentified.
    if df['med_state'].nunique() < 2 or n_sub < mm.MIN_SUBJECTS:
        return {**base, 'converged': False,
                'error': 'med_state constant or too few subjects'}
    try:
        res, warn = mm.fit_cell(df, mm.VC_MATCHED, formula=mm.FORMULA_MATCHED)
    except mm.CellFitError as exc:
        return {**base, 'converged': False, 'error': f'matched: {exc}'[:200]}
    try:
        res_red, _ = mm.fit_cell(df, mm.VC_MATCHED_REDUCED,
                                 formula=mm.FORMULA_MATCHED)
        stat, p_lrt = mm.lrt(res, res_red)
    except (mm.CellFitError, Exception):                     # noqa: BLE001
        stat, p_lrt = np.nan, np.nan

    vc = mm.vcomp_by_name(res)
    return {**base,
            'beta_med': float(res.fe_params['med_state']),
            'se': float(res.bse['med_state']),
            'z': float(res.tvalues['med_state']),
            'p': float(res.pvalues['med_state']),
            'var_subj_int': float(vc.get('subj_int', np.nan)),
            'var_subj_med': float(vc.get('subj_med', np.nan)),
            'var_channel': float(vc.get('channel', np.nan)),
            'var_resid': float(res.scale),
            'lrt_stat': float(stat), 'p_lrt_mixture': float(p_lrt),
            'converged': bool(res.converged),
            'n_warnings': len(warn), 'error': '',
            'fit_seconds': time.time() - t0}


def unpooled_slopes(df, meta, stratum):
    """Per-subject OLS slopes for this cell, computed while the frame is in hand.

    `compute_subject_slopes_grid` does the same thing for the unstratified grid,
    but as a second pass over the view. Three strata times two drug sets would be
    six more full Lustre reads for numbers that are free right here.
    """
    s = subject_slopes(epoch_level(df))
    s['region'] = meta['region']
    s['freq_bin_index'] = meta['freq_bin_index']
    s['stratum'] = stratum
    return s


def stage_fit(args):
    ref = reference_run.load(args.reference_run)
    run_dir = Path(args.run_dir)
    manifest = io.read_table(run_dir / 'grid_cell_manifest.parquet', on_stale='warn')
    region_index = io.read_table(run_dir / 'region_index.parquet', on_stale='warn')
    state = io.read_table(run_dir / 'epoch_med_state.parquet', on_stale='warn')

    match = region_index[region_index['region_index'] == args.region_index]
    if match.empty:
        raise SystemExit(f'--region-index {args.region_index} is not one of the '
                         f'{len(region_index)} regions')
    region = str(match['region'].iloc[0])

    todo = manifest[(manifest['region'] == region)
                    & manifest['above_coverage_floor']].sort_values('freq_bin_index')
    logger.info('region %d: %s -- %d cells', args.region_index, region, len(todo))

    # The run's cohort, from its own provenance -- never re-derived here, or a
    # fit task could silently disagree with what prepare recorded.
    cohort = set(run_provenance(run_dir).get('subjects') or [])
    if not cohort:
        raise SystemExit(f'{run_dir}/provenance.json records no subjects')

    view_dir = resolve_view_dir(args.view_dir,
                                mask_label=ref.view_params.get('mask_label'),
                                roi_scheme=ref.view_params.get('roi_scheme', 'roi_v2'))
    paths = view_subject_paths(view_dir)
    roi_by_subject, _ = roi_maps(paths, cohort,
                                 ref.view_params.get('roi_scheme', 'roi_v2'))

    if todo.empty:
        for name in STRATA:
            io.write_table(pd.DataFrame(columns=['region', 'freq_bin_index']),
                           run_dir / 'cells' / f'{name}_{args.region_index:03d}.parquet',
                           params={'region': region, 'n_cells': 0},
                           script='ieeg_ehr/analysis/run_mixed_model_med_strata.py')
        logger.info('nothing fittable in %s', region)
        return

    wanted = {(region, int(b)) for b in todo['freq_bin_index']}
    t0 = time.time()
    frames = load_cell_frames(paths, cohort, wanted, roi_by_subject)
    logger.info('loaded %d cell frame(s) in %.1fs (%d rows)', len(frames),
                time.time() - t0, sum(len(f) for f in frames.values()))

    # (subject_id, epoch_id) -> med_state. The view's `subject` is the prefixed
    # form; med_state carries the bare one.
    lookup = state.assign(subject='sub-' + state['subject'])[
        ['subject', 'epoch_id', 'med_state']]

    records = {name: [] for name in STRATA}
    blups = {name: [] for name in STRATA}
    slopes = []

    for row in todo.itertuples():
        b = int(row.freq_bin_index)
        meta = {'region': region, 'freq_bin_index': b,
                'bin_low_hz': float(row.bin_low_hz),
                'bin_high_hz': float(row.bin_high_hz),
                'cell_index': int(row.cell_index), 'group': 'medstrata'}
        df = frames.get((region, b))
        if df is None or df.empty:
            for name in STRATA:
                rec = mm.failed_record(region, b, meta['bin_low_hz'],
                                       meta['bin_high_hz'],
                                       'no rows after the ROI join', df=None)
                rec['cell_index'] = meta['cell_index']
                records[name].append(rec)
            continue

        df = df.merge(lookup, on=['subject', 'epoch_id'], how='inner')
        # Numeric, not bool: patsy would treat a boolean as categorical and name
        # the term `med_state[T.True]`, which MED_INTERACTION_TERM would miss.
        df['med_state'] = df['med_state'].astype(float)

        for name, want in (() if args.matched_only
                           else (('medpos', 1.0), ('medneg', 0.0))):
            sub = df[df['med_state'] == want]
            # Re-centre WITHIN the stratum -- see the module docstring.
            sub = mm.add_nrs_components(sub)
            rec, bl, _ = fit_one_cell(sub, meta, with_channel_slope=False)
            rec['stratum'] = name
            records[name].append(rec)
            if bl:
                blups[name].append(pd.DataFrame(bl))
            if len(sub):
                slopes.append(unpooled_slopes(sub, meta, name))

        if args.matched_only:
            records['matched'].append(fit_matched(df, meta))
            continue

        rec, bl = fit_interaction(df, meta)
        rec['stratum'] = 'interaction'
        records['interaction'].append(rec)
        if bl:
            blups['interaction'].append(pd.DataFrame(bl))
        records['matched'].append(fit_matched(df, meta))

    for name in STRATA:
        # Skip strata this task did not compute -- with --matched-only that is
        # everything but `matched`, and an empty file would look to `collect`
        # like a region that produced no fittable cells.
        if not records[name]:
            continue
        io.write_table(pd.DataFrame(records[name]),
                       run_dir / 'cells' / f'{name}_{args.region_index:03d}.parquet',
                       params={'region': region, 'stratum': name},
                       parents=[str(run_dir / 'grid_cell_manifest.parquet')],
                       script='ieeg_ehr/analysis/run_mixed_model_med_strata.py')
        if blups[name]:
            io.write_table(pd.concat(blups[name], ignore_index=True),
                           run_dir / 'cells' / f'blups_{name}_{args.region_index:03d}.parquet',
                           params={'region': region, 'stratum': name},
                           script='ieeg_ehr/analysis/run_mixed_model_med_strata.py')
    if slopes:
        io.write_table(pd.concat(slopes, ignore_index=True),
                       run_dir / 'cells' / f'slopes_{args.region_index:03d}.parquet',
                       params={'region': region,
                               'source': 'unpooled per-subject OLS'},
                       script='ieeg_ehr/analysis/run_mixed_model_med_strata.py')
    logger.info('%s done', region)


# ============================================================================
# STAGE: collect
# ============================================================================

def sign_consistency_from(slopes, cells):
    """Fraction of subjects whose unpooled slope matches the group sign."""
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


def stage_collect(args):
    run_dir = Path(args.run_dir)
    region_index = io.read_table(run_dir / 'region_index.parquet', on_stale='warn')

    slope_parts = []
    for r in region_index.itertuples():
        p = run_dir / 'cells' / f'slopes_{int(r.region_index):03d}.parquet'
        if p.exists():
            slope_parts.append(io.read_table(p, on_stale='ignore'))
    all_slopes = (pd.concat(slope_parts, ignore_index=True) if slope_parts
                  else pd.DataFrame())

    for name in STRATA:
        parts, missing = [], []
        for r in region_index.itertuples():
            p = run_dir / 'cells' / f'{name}_{int(r.region_index):03d}.parquet'
            if not p.exists():
                missing.append(r.region)
                continue
            df = io.read_table(p, on_stale='ignore')
            if len(df):
                parts.append(df)
        if missing:
            logger.error('[%s] MISSING region outputs: %s', name, missing)
        if not parts:
            logger.error('[%s] nothing to collect', name)
            continue

        cells = pd.concat(parts, ignore_index=True)
        # FDR is applied WITHIN each stratum: each map is its own family of tests.
        cells = add_fdr(cells, 'p', 'p')
        cells = add_fdr(cells, 'p_lrt_mixture', 'p_lrt')
        if name == 'interaction':
            # BOTH medication terms get their own family. The main effect is the
            # power LEVEL difference at average pain; the interaction is the
            # change in the pain slope. They answer different questions and the
            # main effect turns out to be the larger of the two, so leaving it
            # uncorrected made the smaller effect the only one on the record.
            cells = add_fdr(cells, 'med_ix_p', 'med_ix')
            cells = add_fdr(cells, 'med_main_p', 'med_main')

        out = stratum_dir(run_dir, name)
        out.mkdir(parents=True, exist_ok=True)   # `matched` may postdate prepare
        io.write_table(cells, out / 'grid_cells.parquet',
                       params={'stratum': name, 'fdr_q': FDR_Q,
                               'families': 'global within this stratum (primary); '
                                           'within region (secondary)'},
                       parents=[str(run_dir / 'grid_cell_manifest.parquet')],
                       script='ieeg_ehr/analysis/run_mixed_model_med_strata.py')

        bl = [io.read_table(p, on_stale='ignore') for p in
              sorted((run_dir / 'cells').glob(f'blups_{name}_*.parquet'))]
        if bl:
            io.write_table(pd.concat(bl, ignore_index=True),
                           out / 'grid_blups.parquet', params={'stratum': name},
                           script='ieeg_ehr/analysis/run_mixed_model_med_strata.py')

        # `matched` carries its own frac_sign_consistent, computed from matched
        # per-subject differences rather than from slopes, so the slope-based
        # version would be meaningless there.
        if len(all_slopes) and name not in ('interaction', 'matched'):
            cons = sign_consistency_from(
                all_slopes[all_slopes['stratum'] == name], cells)
            io.write_table(cons, out / 'sign_consistency.parquet',
                           params={'stratum': name,
                                   'definition': 'fraction of subjects whose '
                                                 'UNPOOLED slope shares the sign '
                                                 'of the group fixed effect'},
                           script='ieeg_ehr/analysis/run_mixed_model_med_strata.py')

        sig = int((cells['p_bh_reject'] == True).sum())          # noqa: E712
        extra = ''
        if name == 'interaction':
            extra = (f" | interaction BH-significant: "
                     f"{int((cells['med_ix_bh_reject'] == True).sum())}")  # noqa: E712
        logger.info('[%-11s] %d cells, %d converged, %d BH-significant%s',
                    name, len(cells), int(cells['converged'].fillna(False).sum()),
                    sig, extra)

    write_methods(run_dir)
    io.log_analysis('medication-stratified mixed-effects grid: medicated vs '
                    'unmedicated plus interaction (EXPLORATORY)', run_dir)
    print(run_dir)


def write_methods(run_dir):
    prov = run_provenance(run_dir)
    params = prov.get('params', {})
    extra = prov.get('extra', {})
    summary = pd.read_csv(Path(run_dir) / 'stratum_summary.csv')

    (Path(run_dir) / 'METHODS.md').write_text(f"""# Medication-stratified mixed-effects grid

EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS.

## Stratification

Each pain assessment is labelled by whether any **{params.get('drug_set')}**
administration ({', '.join(params.get('subclasses', []))}) falls in the
**{params.get('window_hours')} hours** before it, as the half-open interval
`(pain_time - hours, pain_time]`. A dose stamped in the same minute counts as
prior: charting is minute-resolution and the nursing sequence is assess ->
administer -> chart.

```
{summary.to_string(index=False)}
```

## Models

    medicated     log10_power ~ NRS_within + NRS_submean
    unmedicated   log10_power ~ NRS_within + NRS_submean
    interaction   log10_power ~ NRS_within * med_state + NRS_submean

all with `(NRS_within || subject) + (1 | subject:channel)`. In the interaction
model `med_state` is 0/1, so `NRS_within` is the slope in the unmedicated stratum
and `NRS_within:med_state` is the difference between strata.

NRS is re-centred **within each stratum**, so `NRS_within` is a deviation from
that subject's mean pain in that stratum.

## Cohort

{'Per-stratum eligible sets (NOT comparable cell for cell).' if params.get('per_stratum_cohort') else 'The INTERSECTION of the two strata eligible sets, so both maps describe the same patients.'}
Eligibility is re-derived per stratum under the inherited criteria.
medicated: {len(params.get('eligible_medpos', []))} eligible;
unmedicated: {len(params.get('eligible_medneg', []))} eligible.

## Multiple comparisons

Benjamini-Hochberg at q={FDR_Q}, applied **within each stratum** -- each map is
its own family. Global (`p_bh`) and within-region (`p_bh_within_region`) are both
reported, as is the interaction term's own correction (`med_ix_bh`).

## Limitations

- **CONFOUNDING BY INDICATION.** {extra.get('confounding_by_indication', '')}
- No dose, route weighting, or opioid equivalence: the split is binary.
- Scheduled vs PRN cannot be separated from the MAR export.
- Administration time is when the dose was charted, at minute resolution.
- A dose preceding an assessment does not make it the cause of anything the
  assessment records; an assessment is often charted BECAUSE a PRN dose was
  requested. The arrow can point either way.
- Inherited unchanged from the parent grid: epochs within a subject are treated
  as exchangeable (no temporal term), post-ictal windows are not excluded,
  electrode impedance drift is untested, and p-values are parametric Wald.
""")


# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=['prepare', 'fit', 'collect'], required=True)
    ap.add_argument('--run-dir', default=None)
    ap.add_argument('--region-index', type=int, default=None)
    ap.add_argument('--drug-set', choices=sorted(med_state.DRUG_SETS),
                    default='analgesics')
    ap.add_argument('--hours', type=float, default=med_state.DEFAULT_WINDOW_HOURS)
    ap.add_argument('--view-dir', default=None)
    ap.add_argument('--reference-run', default=str(reference_run.CONTPAIN_HEATMAP))
    ap.add_argument('--allow-cohort-drift', action='store_true')
    ap.add_argument('--min-subjects', type=int, default=10)
    ap.add_argument('--matched-only', action='store_true',
                    help='Fit ONLY the matched-NRS model, reusing an existing '
                         "run's cohort, manifest and med state -- so the matched "
                         'analysis lands on exactly the same cells as the others.')
    ap.add_argument('--per-stratum-cohort', action='store_true',
                    help='Let each stratum keep its own eligible set. The two maps '
                         'then describe different patients and are NOT comparable '
                         'cell for cell.')
    ap.add_argument('--question', default=QUESTION)
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

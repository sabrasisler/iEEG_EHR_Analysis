"""Paired change-score models over the region x frequency grid.

Rows are (consecutive assessment pair x channel); the outcome is the CHANGE in
log power across the pair. See `analysis/change_score.py` for why this design is
preferred to the 2-hour lookback, and for the two confounds it must carry
(regression to the mean, and gap being entangled with baseline pain).

    d_log10_power ~ d_pain * med_between + pain_1 + gap_h + med_between:gap_h
                    + (d_pain || subject) + (1 | subject:channel)

Stages mirror the other grid drivers so the sbatch pattern is identical:

    prepare                 -> pair index, summary, manifest; echoes RUN_DIR
    fit --region-index N    -> one array task per region
    collect                 -> concatenate, BH, METHODS.md

    python -m ieeg_ehr.analysis.run_change_score_grid --stage prepare
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import (change_score, med_state, mixed_model as mm,
                               reference_run, view_tables)
from ieeg_ehr.analysis.run_mixed_model_grid import (OUTPUT_TYPE, QUESTION, add_fdr,
                                                    build_grid_manifest,
                                                    resolve_cohort)
from ieeg_ehr.analysis.run_mixed_model_pilot import (coverage_map, load_cell_frames,
                                                     resolve_view_dir, roi_maps,
                                                     view_subject_paths)
from ieeg_ehr.views import cache_reader

logger = logging.getLogger(__name__)

RUN_NAME = 'changescore'
VIEW_SCHEME = 'med_change_score'
FDR_Q = 0.05


def run_provenance(run_dir):
    import json
    try:
        return json.loads((Path(run_dir) / 'provenance.json').read_text())
    except (OSError, ValueError):
        return {}


# ============================================================================

def stage_prepare(args):
    ref = reference_run.load(args.reference_run)
    ref.describe()

    view_dir = resolve_view_dir(args.view_dir,
                                mask_label=ref.view_params.get('mask_label'),
                                roi_scheme=ref.view_params.get('roi_scheme', 'roi_v2'))
    paths, _, _, subjects, roi_by_subject, no_roi = resolve_cohort(ref, view_dir)
    ref.assert_cohort_matches(subjects, allow_drift=args.allow_cohort_drift)

    subclasses = med_state.DRUG_SETS[args.drug_set]
    admin = med_state.load_admin_table(subclasses=subclasses)

    # A subset arm's `med_between` is blind to the classes it excludes, so
    # without this its control group contains them. Opt-in and per-run rather
    # than a property of the drug set, because the correct answer is asymmetric
    # (see change_score.build_pairs) and an asymmetry belongs in the recorded
    # params where it can be read off, not buried in a lookup table.
    exclude_admin, excluded_subclasses = None, []
    if args.exclude_coexposure:
        excluded_subclasses = [s for s in med_state.DRUG_SETS['analgesics']
                               if s not in subclasses]
        if not excluded_subclasses:
            raise SystemExit(
                f'--exclude-coexposure is meaningless for --drug-set '
                f'{args.drug_set}: it already covers every analgesic subclass, '
                f'so there is no other class to exclude.')
        logger.info('CO-EXPOSURE EXCLUSION ON: a pair is dropped if any of %s '
                    'was administered between its two assessments',
                    excluded_subclasses)
        exclude_admin = med_state.load_admin_table(subclasses=excluded_subclasses)

    defs = med_state.load_epoch_defs(subjects=subjects)
    pairs = change_score.build_pairs(defs, admin, args.min_gap_min,
                                     args.max_gap_min,
                                     exclude_admin=exclude_admin)
    summary = change_score.pair_summary(pairs)
    logger.info('\n%s', summary.to_string(index=False))

    # A subject needs pairs in BOTH exposure states or they inform only the
    # nuisance terms; recorded rather than silently dropped.
    per = pairs.groupby('subject')['med_between'].nunique()
    both = sorted(per[per > 1].index)
    logger.info('%d/%d subjects have pairs both with AND without a dose between',
                len(both), pairs['subject'].nunique())

    cohort = {f'sub-{s}' for s in pairs['subject'].unique()} & set(subjects)

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

    coverage = coverage_map(paths, cohort, roi_by_subject, bin_table.index)
    manifest = build_grid_manifest(regions, bin_table, coverage, args.min_subjects)
    logger.info('%d regions x %d bins = %d cells; %d clear min_subjects=%d',
                len(regions), len(bin_table), len(manifest),
                int(manifest['above_coverage_floor'].sum()), args.min_subjects)

    run_dir = config.analysis_run_dir(
        question=args.question, output_type=OUTPUT_TYPE,
        view_scheme=args.view_scheme,
        run_name=(f'{args.run_name}_{args.drug_set}_min{int(args.min_gap_min)}'
                  + ('_noco' if args.exclude_coexposure else '')))
    (run_dir / 'cells').mkdir(parents=True, exist_ok=True)

    params = {'drug_set': args.drug_set, 'subclasses': list(subclasses),
              'min_gap_min': args.min_gap_min, 'max_gap_min': args.max_gap_min,
              'min_subjects': args.min_subjects,
              'exclude_coexposure': bool(args.exclude_coexposure),
              'excluded_subclasses': excluded_subclasses,
              'unresolvable_bins_removed': unresolvable,
              'line_noise_bins_removed': line_noise, 'roi_scheme': roi_scheme,
              'formula': mm.FORMULA_CHANGE,
              # Recorded because it CHANGED on 2026-09-10: the channel component
              # was dropped as unidentifiable after differencing, so a run's
              # random-effects spec can no longer be inferred from the formula.
              'vc': sorted(mm.VC_CHANGE)}

    io.write_table(pairs, run_dir / 'pair_index.parquet', params=params,
                   parents=[str(med_state.ADMIN_TABLE)], subjects=sorted(cohort),
                   script='ieeg_ehr/analysis/run_change_score_grid.py')
    io.write_table(summary, run_dir / 'pair_summary.csv', params=params,
                   script='ieeg_ehr/analysis/run_change_score_grid.py')
    io.write_table(manifest, run_dir / 'grid_cell_manifest.parquet', params=params,
                   subjects=sorted(cohort),
                   script='ieeg_ehr/analysis/run_change_score_grid.py')
    io.write_table(pd.DataFrame({'region_index': range(len(regions)),
                                 'region': regions}),
                   run_dir / 'region_index.parquet',
                   script='ieeg_ehr/analysis/run_change_score_grid.py')

    io.write_run_provenance(
        run_dir, script='ieeg_ehr/analysis/run_change_score_grid.py',
        params={**params, 'stage': 'prepare', 'view_dir': str(view_dir),
                'view_params': ref.view_params, 'criteria': ref.criteria,
                'n_pairs': len(pairs), 'n_cells': len(manifest),
                'subjects_both_exposures': both},
        parents=[str(Path(args.reference_run) / 'provenance.json'), str(view_dir),
                 str(med_state.ADMIN_TABLE)],
        subjects=sorted(cohort),
        extra={'status': 'EXPLORATORY paired change-score grid, NOT a finding',
               'subjects_without_roi': sorted(no_roi),
               'coexposure_note': (
                   '## Co-exposure\n\n'
                   f'Pairs with a dose of {excluded_subclasses} between their two '
                   'assessments are DROPPED, from both the exposed and the control '
                   'group. Without this the arm is not a partition: measured '
                   '2026-09-10, 22.3% of `non_opioid_analgesics` control pairs had '
                   'an opioid administered between, so the weakest analgesic was '
                   'being contrasted against a control group containing the '
                   'strongest one. Dropping from the exposed group too removes the '
                   '74 doubly-exposed pairs, which would otherwise let an opioid '
                   'effect be reported as a non-opioid one. `h_since_prior` is '
                   'measured over the UNION of both class sets for the same reason.'
                   if args.exclude_coexposure else
                   '## Co-exposure\n\n'
                   'NOT excluded in this run. `med_between` is computed only from '
                   f'{list(subclasses)}, so for a SUBSET drug set the control group '
                   'still contains doses of the other analgesic classes. Harmless '
                   'for `--drug-set analgesics`, which covers every class; for a '
                   'subset arm see --exclude-coexposure.'),
               'regression_to_the_mean':
                   'Mean change in pain runs from +2.12 at baseline 0 to -3.00 at '
                   'baseline 10 on this cohort, and medication is given BECAUSE '
                   'pain is high. `pain_1` is in the model for this reason; a '
                   'change-score model without it reports regression to the mean '
                   'as a drug effect.',
               'gap_baseline_confound':
                   'Short-gap pairs start around NRS 4.2-4.6 and long-gap pairs '
                   'around 2.2-2.9 -- reassessment is faster when pain is high. A '
                   'difference between gap bands is therefore NOT cleanly a '
                   'timescale effect; the med_between x gap_h term inside one '
                   'model is the better instrument.',
               'min_gap_rationale':
                   f'Epochs are 5 min pre-score, so pairs closer than that overlap '
                   f'and their difference is compressed toward zero. Minimum is '
                   f'{args.min_gap_min} min, applied GLOBALLY -- a route-dependent '
                   f'minimum is undefined for unmedicated pairs and would make the '
                   f'gap distribution differ between the compared groups.'})
    print(run_dir)
    return run_dir


# ============================================================================

def fit_change_cell(df, meta):
    """One cell of the change-score grid."""
    t0 = time.time()
    base = {'region': meta['region'], 'freq_bin_index': meta['freq_bin_index'],
            'freq_bin_low': meta['bin_low_hz'], 'freq_bin_high': meta['bin_high_hz'],
            'cell_index': meta['cell_index'],
            'n_subjects': df['subject'].nunique(),
            'n_channels': df['channel_uid'].nunique(),
            'n_pairs': df['pair_id'].nunique(), 'n_rows': len(df),
            'n_pairs_dosed': int(df.loc[df['med_between'] == 1.0,
                                        'pair_id'].nunique())}

    if (df['med_between'].nunique() < 2 or df['d_pain'].std(ddof=0) == 0
            or base['n_subjects'] < mm.MIN_SUBJECTS):
        return {**base, 'converged': False,
                'error': 'no exposure contrast, no d_pain variance, or too few '
                         'subjects'}
    try:
        res, warn = mm.fit_cell(df, mm.VC_CHANGE, formula=mm.FORMULA_CHANGE)
    except mm.CellFitError as exc:
        return {**base, 'converged': False, 'error': f'change: {exc}'[:200]}
    try:
        res_red, _ = mm.fit_cell(df, mm.VC_CHANGE_REDUCED,
                                 formula=mm.FORMULA_CHANGE)
        stat, p_lrt = mm.lrt(res, res_red)
    except Exception:                                        # noqa: BLE001
        stat, p_lrt = np.nan, np.nan

    rec = dict(base)
    for term, prefix in mm.CHANGE_TERMS:
        rec.update(mm.term_stats(res, term, prefix))
    vc = mm.vcomp_by_name(res)
    rec.update({'var_subj_int': float(vc.get('subj_int', np.nan)),
                'var_subj_dpain': float(vc.get('subj_dpain', np.nan)),
                'var_channel': float(vc.get('channel', np.nan)),
                'var_resid': float(res.scale),
                'lrt_stat': float(stat), 'p_lrt_mixture': float(p_lrt),
                'converged': bool(res.converged), 'n_warnings': len(warn),
                'error': '', 'fit_seconds': time.time() - t0})
    logger.info('%-18s bin %2d | dpain %+.5f p %.3g | med %+.5f p %.3g | '
                'ix %+.5f p %.3g | %d pairs (%d dosed)',
                meta['region'], meta['freq_bin_index'], rec['dpain_beta'],
                rec['dpain_p'], rec['med_beta'], rec['med_p'],
                rec['med_ix_beta'], rec['med_ix_p'], rec['n_pairs'],
                rec['n_pairs_dosed'])
    return rec


def stage_fit(args):
    ref = reference_run.load(args.reference_run)
    run_dir = Path(args.run_dir)
    manifest = io.read_table(run_dir / 'grid_cell_manifest.parquet', on_stale='warn')
    region_index = io.read_table(run_dir / 'region_index.parquet', on_stale='warn')
    pairs = io.read_table(run_dir / 'pair_index.parquet', on_stale='warn')

    match = region_index[region_index['region_index'] == args.region_index]
    if match.empty:
        raise SystemExit(f'--region-index {args.region_index} out of range')
    region = str(match['region'].iloc[0])
    todo = manifest[(manifest['region'] == region)
                    & manifest['above_coverage_floor']].sort_values('freq_bin_index')
    logger.info('region %d: %s -- %d cells, %d pairs',
                args.region_index, region, len(todo), len(pairs))

    cohort = set(run_provenance(run_dir).get('subjects') or [])
    view_dir = resolve_view_dir(args.view_dir,
                                mask_label=ref.view_params.get('mask_label'),
                                roi_scheme=ref.view_params.get('roi_scheme', 'roi_v2'))
    view_paths = view_subject_paths(view_dir)
    roi_by_subject, _ = roi_maps(view_paths, cohort,
                                 ref.view_params.get('roi_scheme', 'roi_v2'))

    if todo.empty:
        io.write_table(pd.DataFrame(columns=['region', 'freq_bin_index']),
                       run_dir / 'cells' / f'region_{args.region_index:03d}.parquet',
                       params={'region': region, 'n_cells': 0},
                       script='ieeg_ehr/analysis/run_change_score_grid.py')
        return

    wanted = {(region, int(b)) for b in todo['freq_bin_index']}
    t0 = time.time()
    frames = load_cell_frames(view_paths, cohort, wanted, roi_by_subject)
    logger.info('loaded %d frame(s) in %.1fs', len(frames), time.time() - t0)

    records = []
    for row in todo.itertuples():
        b = int(row.freq_bin_index)
        meta = {'region': region, 'freq_bin_index': b,
                'bin_low_hz': float(row.bin_low_hz),
                'bin_high_hz': float(row.bin_high_hz),
                'cell_index': int(row.cell_index)}
        cell = frames.get((region, b))
        if cell is None or cell.empty:
            records.append({**meta, 'converged': False,
                            'error': 'no rows after the ROI join'})
            continue
        records.append(fit_change_cell(change_score.build_change_frame(cell, pairs),
                                       meta))

    io.write_table(pd.DataFrame(records),
                   run_dir / 'cells' / f'region_{args.region_index:03d}.parquet',
                   params={'region': region},
                   parents=[str(run_dir / 'grid_cell_manifest.parquet')],
                   script='ieeg_ehr/analysis/run_change_score_grid.py')
    logger.info('%s done', region)


# ============================================================================

def stage_collect(args):
    run_dir = Path(args.run_dir)
    region_index = io.read_table(run_dir / 'region_index.parquet', on_stale='warn')
    parts, missing = [], []
    for r in region_index.itertuples():
        p = run_dir / 'cells' / f'region_{int(r.region_index):03d}.parquet'
        if not p.exists():
            missing.append(r.region)
            continue
        df = io.read_table(p, on_stale='ignore')
        if len(df):
            parts.append(df)
    if missing:
        logger.error('MISSING region outputs: %s', missing)
    if not parts:
        raise SystemExit('nothing to collect')

    cells = pd.concat(parts, ignore_index=True)
    # Each coefficient is its own family of tests.
    for col, prefix in (('dpain_p', 'dpain'), ('med_p', 'med'),
                        ('med_ix_p', 'med_ix'), ('med_gap_p', 'med_gap')):
        if col in cells.columns:
            cells = add_fdr(cells, col, prefix)
    if 'p_lrt_mixture' in cells.columns:
        cells = add_fdr(cells, 'p_lrt_mixture', 'p_lrt')

    io.write_table(cells, run_dir / 'grid_cells.parquet',
                   params={'fdr_q': FDR_Q, 'formula': mm.FORMULA_CHANGE},
                   parents=[str(run_dir / 'grid_cell_manifest.parquet')],
                   script='ieeg_ehr/analysis/run_change_score_grid.py')

    n = len(cells)
    conv = int(cells['converged'].fillna(False).sum())
    logger.info('=' * 70)
    logger.info('CHANGE-SCORE GRID: %d cells, %d converged', n, conv)
    for prefix, label in (('dpain', 'd_pain (power tracks pain change)'),
                          ('med', 'med_between (dose shifts power)'),
                          ('med_ix', 'd_pain x med (dose alters coupling)'),
                          ('med_gap', 'med x gap (TIME STRUCTURE)')):
        col = f'{prefix}_bh_reject'
        if col in cells.columns:
            logger.info('  %-42s BH-significant: %d', label,
                        int((cells[col] == True).sum()))       # noqa: E712
    logger.info('=' * 70)

    write_methods(run_dir, cells)
    io.log_analysis('paired change-score grid: power change vs pain change with '
                    'dose between as a factor (EXPLORATORY)', run_dir)
    print(run_dir)


def write_methods(run_dir, cells):
    prov = run_provenance(run_dir)
    p = prov.get('params', {})
    extra = prov.get('extra', {})
    summary = pd.read_csv(Path(run_dir) / 'pair_summary.csv')
    (Path(run_dir) / 'METHODS.md').write_text(f"""# Paired change-score grid

EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS.

## Design

Consecutive pain assessments {p.get('min_gap_min')}-{p.get('max_gap_min')} minutes
apart. One row per (pair x channel); the outcome is the CHANGE in log10 power
across the pair. A dose of **{p.get('drug_set')}** administered BETWEEN the two
assessments is the exposure.

```
{p.get('formula')}
```
with random effects `{' + '.join(p.get('vc') or ['?'])}` (i.e.
`(d_pain || subject)`). {p.get('n_pairs')} pairs, {len(cells)} cells.

**There is no `(1 | subject:channel)` term, deliberately.** A channel's own
baseline power cancels in `y2 - y1`, so the differenced outcome retains almost no
channel-to-channel variance for it to estimate -- 2e-04 against a 3e-02 residual,
where the LEVEL grids put it at 1.8e-01 against 3.4e-02. Asking for a variance
the design set to ~zero flattens the likelihood and the optimizer never settles,
which made the WHOLE fit report `converged=False` even though the fixed effects
were fine: 3/10 cells converged with the term, 10/10 without it, at 1/24th the
fit time, with `med_between` agreeing to 4e-04. Dropping the random SLOPE instead
was strictly worse (0/10), so `subj_dpain` is load-bearing and stays.

```
{summary.to_string(index=False)}
```

{extra.get('coexposure_note', '')}

## Why differences

The exposure is better defined than a lookback -- a dose either did or did not
fall between two measurements, with no ambiguity about rebound. Differencing also
removes each channel's level and any drift slow enough to be common to both
epochs, which is the only handle this project has on electrode impedance drift.
And it never fits a slope across the raw 0-10 pain scale, which is non-monotone
and badly unevenly sampled here.

## Minimum gap

{extra.get('min_gap_rationale', '')}

## Limitations

- **{extra.get('regression_to_the_mean', '')}**
- **Gap is confounded with baseline.** {extra.get('gap_baseline_confound', '')}
- Overlapping pairs share an assessment, so consecutive pairs are correlated;
  the subject random effects absorb some of this but not all.
- A large share of pairs have `d_pain == 0`; they inform the `med_between` main
  effect but contribute nothing to the `d_pain` slope.
- Still observational. Nothing randomised who received a dose.
- Inherited: post-ictal windows are not excluded, and p-values are parametric.
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
    ap.add_argument('--min-gap-min', type=float,
                    default=change_score.DEFAULT_MIN_GAP_MIN)
    ap.add_argument('--max-gap-min', type=float,
                    default=change_score.DEFAULT_MAX_GAP_MIN)
    ap.add_argument('--min-subjects', type=int, default=10)
    ap.add_argument('--exclude-coexposure', action='store_true',
                    help='Drop any pair with a dose from a NON-tested analgesic '
                         'subclass between its two assessments. Required for a '
                         'subset drug set to mean anything: 22.3%% of the '
                         'non_opioid_analgesics control pairs had an opioid '
                         'between. Meaningless for --drug-set analgesics.')
    ap.add_argument('--view-dir', default=None)
    ap.add_argument('--reference-run', default=str(reference_run.CONTPAIN_HEATMAP))
    ap.add_argument('--allow-cohort-drift', action='store_true')
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

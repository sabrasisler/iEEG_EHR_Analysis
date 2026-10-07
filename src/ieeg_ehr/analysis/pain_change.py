"""pain_change: does the change in z-scored power track the change in pain?

    python -m ieeg_ehr.analysis.pain_change --roi-scheme roi_v3_ins \\
        --mask-label std10_rv-gross-std3_satmargin15_sw_logz4

Input is the `fullresz` z-score table (`views/build_pain_epoch_fullres_zscore.py`):
per subject-session, channel and native 0.5 Hz frequency, the epoch mean of
per-window z-scores against all of that session's pain-epoch windows.

Pairs are CONSECUTIVE assessments within a session, built by
`change_score.build_pairs` with its gap window (30-240 min by default). Per pair
x channel x frequency

    d_z = zmean(e2) - zmean(e1)

then the mean d_z over the native frequencies inside each band of
`--band-set` (default `paper_bands_6_hg200`, the six bands of
`run_bandpower_mixed.BAND_SETS`), arithmetic because d_z is a difference
(`axes.aggregate_bands`), line-noise bins removed first. `--band-set fullres`
keeps every native frequency as its own cell.

One mixed model per ROI x band cell, rows = pair x channel:

    d_z ~ d_pain + pain_1_within + gap_h + (1 | subject) + (0 + d_pain | subject)

The random effects are `mixed_model.VC_CHANGE`: uncorrelated subject intercept
and subject `d_pain` slope, no channel component (a channel's level cancels in
the difference, see mixed_model.py). `pain_1_within` is `pain_1` centred within
subject over the cell's rows, the same centring `mixed_model.add_nrs_components`
applies to NRS. The reported term is `d_pain`: change in z per point of NRS
change. The LRT on the random `d_pain` slope uses `VC_CHANGE_REDUCED`.

BH FDR across all fitted cells and within each region (`add_fdr`). Discovery
cohort only. Output is NOMINATIONS, not findings.

Each run lands in `analysis/pain/pain_change/bandpower/<run_name>_<timestamp>/`
(`fullres/` for `--freq fullres`): `cells.csv`, `pair_index.csv`, `METHODS.md`,
`provenance.json`, and every figure in `figures/`.
"""

import argparse
import inspect
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import (change_score, fullres_cells, med_state,
                               mixed_model as mm, view_tables)
from ieeg_ehr.analysis.run_bandpower_mixed import BAND_SETS
from ieeg_ehr.analysis.run_mixed_model_grid import FDR_Q, add_fdr
from ieeg_ehr.analysis.run_mixed_model_pilot import roi_maps
from ieeg_ehr.views import axes, fullres_reader, view_config
from ieeg_ehr.views import build_pain_epoch_fullres_zscore as zview

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/pain_change.py'
#: Level-3 folder per frequency axis. Figures go in each run's FIGURES_SUBDIR.
OUTPUT_TYPE = {'bands': 'bandpower', 'fullres': 'fullres'}
FIGURES_SUBDIR = 'figures'
RUN_NAME = 'painchange'

FORMULA = 'd_z ~ d_pain + pain_1_within + gap_h'
TERMS = (('d_pain', 'dpain'), ('pain_1_within', 'baseline'), ('gap_h', 'gap'))


def discovery_paths(zdir):
    """z-score tables for discovery subjects only; the rest are logged and dropped."""
    discovery = set(config.discovery_subjects())
    paths, outside = [], []
    for p in sorted(zdir.glob('zmean_sub-*_ses-*.parquet')):
        subject, _ = fullres_cells.subject_session_of(p)
        (paths if subject in discovery else outside).append(p)
    if outside:
        logger.warning('SPLIT GATE: %d subject-session(s) outside discovery '
                       'excluded: %s', len(outside), [p.name for p in outside])
    if not paths:
        raise SystemExit(f'no discovery zmean_sub-*.parquet in {zdir}')
    return paths


def change_matrix(table, pairs, freq_cols):
    """{channel: (n_pairs, n_freq) d_z}, NaN where the channel is absent from either epoch."""
    table = table.set_index(['channel', 'epoch_id'])[freq_cols]
    e1, e2 = pairs['e1'].to_numpy(), pairs['e2'].to_numpy()
    out = {}
    for channel, sub in table.groupby(level='channel', sort=True):
        sub = sub.droplevel('channel')
        out[channel] = (sub.reindex(e2).to_numpy(dtype=config.CACHE_ACCUMULATE_DTYPE)
                        - sub.reindex(e1).to_numpy(dtype=config.CACHE_ACCUMULATE_DTYPE))
    return out


def session_rows(path, pairs, region_of, freq_table, bands):
    """(index frame, (n_rows, n_cols) d_z) for one subject-session, ROI-mapped channels only."""
    subject, session = fullres_cells.subject_session_of(path)
    all_cols = fullres_reader.freq_columns()
    freq_cols = [all_cols[i] for i in freq_table.index]
    table = io.read_table(path, columns=['epoch_id', 'channel'] + freq_cols,
                          on_stale='warn')
    bin_table = freq_table.reset_index()
    index, values = [], []
    for channel, dz in change_matrix(table, pairs, freq_cols).items():
        region = region_of.get(channel)
        if region is None:
            continue
        if bands is not None:
            dz, _ = axes.aggregate_bands(dz, bin_table, bands=bands, is_difference=True)
        index.append(pd.DataFrame({
            'subject': f'sub-{subject}',
            'channel_uid': f'sub-{subject}|ses-{session}|{channel}',
            'pair_id': pairs['pair_id'].to_numpy(), 'region': region}))
        values.append(dz)
    if not index:
        return None, None
    return pd.concat(index, ignore_index=True), np.vstack(values)


def cell_frame(index, values, j, pairs):
    """One cell's model frame: rows with a finite d_z, covariates joined by pair."""
    df = index.assign(d_z=values[:, j])
    df = df[np.isfinite(df['d_z'])]
    df = df.merge(pairs[['pair_id', 'd_pain', 'pain_1', 'gap_h']], on='pair_id')
    df['pain_1_within'] = df['pain_1'] - df.groupby('subject')['pain_1'].transform('mean')
    return df.reset_index(drop=True)


def fit_cell_record(df, meta, min_subjects):
    """One row of results for one ROI x frequency cell."""
    t0 = time.time()
    base = {**meta, 'n_subjects': df['subject'].nunique(),
            'n_channels': df['channel_uid'].nunique(),
            'n_pairs': df['pair_id'].nunique(), 'n_rows': len(df)}
    if (len(df) < mm.MIN_ROWS or base['n_subjects'] < min_subjects
            or df['d_pain'].std(ddof=0) == 0):
        return {**base, 'converged': False,
                'error': f'below floor: {base["n_rows"]} rows, '
                         f'{base["n_subjects"]} subjects (min {min_subjects})'}
    try:
        res, warn = mm.fit_cell(df, mm.VC_CHANGE, formula=FORMULA)
    except mm.CellFitError as exc:
        return {**base, 'converged': False, 'error': str(exc)[:200]}
    try:
        res_red, _ = mm.fit_cell(df, mm.VC_CHANGE_REDUCED, formula=FORMULA)
        stat, p_lrt = mm.lrt(res, res_red)
    except mm.CellFitError:
        stat, p_lrt = np.nan, np.nan

    rec = dict(base)
    for term, prefix in TERMS:
        rec.update(mm.term_stats(res, term, prefix))
    vc = mm.vcomp_by_name(res)
    rec.update({'var_subj_int': float(vc.get('subj_int', np.nan)),
                'var_subj_dpain': float(vc.get('subj_dpain', np.nan)),
                'var_resid': float(res.scale),
                'lrt_stat': float(stat), 'p_lrt_mixture': float(p_lrt),
                'converged': bool(res.converged), 'n_warnings': len(warn),
                'error': '', 'fit_seconds': round(time.time() - t0, 1)})
    return rec


def random_effects_text(vc):
    """'(1 | subject) + (0 + d_pain | subject)' from a vc_formula dict, uncorrelated by construction."""
    return ' + '.join(f'({v} | subject)' for v in vc.values())


def write_methods(run_dir, cells, params, n_pairs, cohort):
    """METHODS.md built from the constants the fit used, so the text cannot drift from the code."""
    fit = inspect.signature(mm.fit_cell).parameters
    bands = params['bands']
    freq_text = (', '.join(f'{b} {lo}-{hi} Hz' for b, (lo, hi) in bands.items())
                 if bands else 'every native 0.5 Hz frequency as its own cell')
    n_conv = int(cells['converged'].sum())
    text = f"""# pain_change mixed-effects grid

EXPLORATORY, discovery cohort. NOMINATIONS, NOT FINDINGS.

## Model

One linear mixed model per ROI x frequency cell. Rows are pair x channel.

```
{FORMULA} + {random_effects_text(mm.VC_CHANGE)}
```

- Random effects: `{mm.VC_CHANGE}` as statsmodels `vc_formula` components grouped by
  subject, with `re_formula='0'`. The intercept and `d_pain` slope are independent
  variance components, so they are uncorrelated (lme4 `(d_pain || subject)`).
  No channel component: a channel's level cancels in the pair difference.
- Estimation: statsmodels `MixedLM`, REML={fit['reml'].default}, maxiter={fit['maxiter'].default},
  default optimizer. A cell that raises or returns non-finite fixed effects is
  recorded with `converged=False` and its error.
- Reported term: `d_pain`, the change in z per point of NRS change. Also reported:
  `pain_1_within`, `gap_h`, the variance components, and a likelihood-ratio test of
  the random `d_pain` slope against `{random_effects_text(mm.VC_CHANGE_REDUCED)}`
  on the 50:50 chi2(0)/chi2(1) mixture (`p_lrt_mixture`).
- Multiple comparisons: Benjamini-Hochberg at q={FDR_Q} on `dpain_p` across all
  fitted cells (`dpain_bh`) and within each region (`dpain_bh_within_region`).
- Inclusion floors per cell: at least {mm.MIN_ROWS} rows, at least
  {params['min_subjects']} subjects, and non-zero `d_pain` variance.

## Variables

- `d_z`: epoch-mean z of the later assessment minus the earlier one. Each 2 s
  window is z-scored per channel x frequency against the mean and SD over every
  masked window of every pain epoch in that session (AXIS 2 `all_pain_epochs`),
  then averaged within the 5-min epoch. Tables: `{params['zscore_dir']}`.
- Frequency cells (`{params['band_set']}`): {freq_text}. Bands average d_z arithmetically over the native
  frequencies inside them after removing {len(params['notch_bins_removed'])} line-noise bins.
- `d_pain`: NRS of the later assessment minus the earlier one.
- `pain_1_within`: NRS of the earlier assessment, centred on the subject's mean
  over the cell's rows.
- `gap_h`: hours between the two assessments.

## Data

Consecutive assessment pairs within a session, {params['min_gap_min']:g}-{params['max_gap_min']:g} min apart
(`change_score.build_pairs`). ROI scheme `{params['roi_scheme']}`. {len(cohort)} subjects,
{n_pairs} pairs, {len(cells)} cells, {n_conv} converged.

## Limitations

- Consecutive pairs share an assessment, so neighbouring rows are correlated
  beyond what the subject random effects absorb.
- Pairs with `d_pain == 0` inform the intercept and covariates only.
- No medication term: a dose between the two assessments is not modelled.
- p-values are parametric (Wald z), not permutation-based.
"""
    (run_dir / 'METHODS.md').write_text(text)


def main(argv=None):
    from joblib import Parallel, delayed

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--min-gap-min', type=float, default=change_score.DEFAULT_MIN_GAP_MIN)
    ap.add_argument('--max-gap-min', type=float, default=change_score.DEFAULT_MAX_GAP_MIN)
    ap.add_argument('--min-subjects', type=int, default=mm.MIN_SUBJECTS)
    ap.add_argument('--notch-half-width-hz', type=float, default=None)
    ap.add_argument('--n-jobs', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))
    ap.add_argument('--run-name', default=RUN_NAME)
    view_config.add_view_arguments(ap)
    ap.add_argument('--band-set', choices=list(BAND_SETS) + ['fullres'],
                    default='paper_bands_6_hg200')
    ap.set_defaults(baseline='all_pain_epochs', freq='fullres', region='none')
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    vc = view_config.from_args(args)
    zdir = zview.zscore_dir(vc)
    paths = discovery_paths(zdir)
    subjects = {f'sub-{fullres_cells.subject_session_of(p)[0]}' for p in paths}

    split_report = {}
    roi_by_subject, no_roi = roi_maps(paths, subjects, args.roi_scheme,
                                      report=split_report)
    regions = view_tables.roi_regions_for({'roi_scheme': args.roi_scheme})

    admin = med_state.load_admin_table(subclasses=med_state.DRUG_SETS['analgesics'])
    defs = med_state.load_epoch_defs(subjects=sorted(subjects))
    pairs = change_score.build_pairs(defs, admin, args.min_gap_min, args.max_gap_min)

    freq_table, notch = fullres_cells.analysis_freq_table(args.notch_half_width_hz)
    bands = BAND_SETS.get(args.band_set)
    if bands is None:
        all_cols = fullres_reader.freq_columns()
        cols = [all_cols[i] for i in freq_table.index]
        col_hz = {c: (float(freq_table.at[i, 'bin_low_hz']),
                      float(freq_table.at[i, 'bin_high_hz']))
                  for c, i in zip(cols, freq_table.index)}
    else:
        cols, col_hz = list(bands), dict(bands)

    index_parts, value_parts = [], []
    for path in paths:
        subject, session = fullres_cells.subject_session_of(path)
        sid = f'sub-{subject}'
        sp = pairs[(pairs['subject'] == subject) & (pairs['session'] == session)]
        if sid not in roi_by_subject or sp.empty:
            logger.warning('%s ses-%s: %s, skipped', sid, session,
                           'no pairs' if sp.empty else 'no ROI-labelled channel')
            continue
        idx, vals = session_rows(path, sp.reset_index(drop=True),
                                 roi_by_subject[sid], freq_table, bands)
        if idx is not None:
            index_parts.append(idx)
            value_parts.append(vals)
    index = pd.concat(index_parts, ignore_index=True)
    values = np.vstack(value_parts)
    logger.info('%d pair x channel rows, %d subjects, %d regions x %d %s cells',
                len(index), index['subject'].nunique(), len(regions), len(cols),
                args.band_set)

    def cells_to_fit():
        # A generator, so joblib builds each cell frame only when a worker is free.
        for region in regions:
            in_region = (index['region'] == region).to_numpy()
            if not in_region.any():
                continue
            sub_index, sub_values = index[in_region], values[in_region]
            for j, col in enumerate(cols):
                meta = {'region': region, 'freq': col, 'freq_low_hz': col_hz[col][0],
                        'freq_high_hz': col_hz[col][1]}
                yield delayed(fit_cell_record)(
                    cell_frame(sub_index, sub_values, j, pairs), meta, args.min_subjects)

    logger.info('fitting up to %d cells on %d worker(s)', len(regions) * len(cols),
                args.n_jobs)
    records = Parallel(n_jobs=args.n_jobs, verbose=5)(cells_to_fit())
    cells = add_fdr(pd.DataFrame(records), 'dpain_p', 'dpain')

    run_dir = config.analysis_run_dir(
        question=config.PAIN_CHANGE_QUESTION,
        output_type=OUTPUT_TYPE['fullres' if bands is None else 'bands'], run_name=args.run_name)
    cohort = sorted(index['subject'].unique())
    params = {'formula': FORMULA, 'vc': mm.VC_CHANGE, 'band_set': args.band_set,
              'bands': bands, 'roi_scheme': args.roi_scheme,
              'min_gap_min': args.min_gap_min, 'max_gap_min': args.max_gap_min,
              'min_subjects': args.min_subjects, 'notch_bins_removed': notch,
              'zscore_dir': str(zdir), 'view_config': vc.provenance()}
    io.write_table(pairs[pairs['subject_id'].isin(cohort)], run_dir / 'pair_index.csv',
                   params=params, parents=[str(med_state.ADMIN_TABLE)],
                   subjects=cohort, script=SCRIPT)
    io.write_table(cells, run_dir / 'cells.csv', params=params,
                   parents=[str(zdir)], subjects=cohort, script=SCRIPT)
    write_methods(run_dir, cells, params, int(pairs['subject_id'].isin(cohort).sum()), cohort)
    io.write_run_provenance(
        run_dir, script=SCRIPT, params=params,
        parents=[str(zdir), str(med_state.ADMIN_TABLE)], subjects=cohort,
        extra={'status': 'EXPLORATORY mixed-model grid, NOT a finding',
               'n_cells': len(cells), 'n_converged': int(cells['converged'].sum()),
               'subjects_without_roi': sorted(no_roi), **split_report})
    io.log_analysis(f'pain_change mixed model, {args.band_set}, {args.roi_scheme}, '
                    f'{len(cohort)} subjects, {len(cells)} cells', run_dir)
    logger.info('%d/%d cells converged; %d pass BH q<%.2f on d_pain',
                int(cells['converged'].sum()), len(cells),
                int(cells['dpain_bh_reject'].eq(True).sum()), 0.05)
    print(run_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())

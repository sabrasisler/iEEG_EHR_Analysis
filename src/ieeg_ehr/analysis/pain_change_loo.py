"""Leave-one-subject-out refits of a pain_change grid, for out-of-sample prediction weights.

    python -m ieeg_ehr.analysis.pain_change_loo --reference-run <pain_change run> \\
        --roi-scheme roi_v3_ins --mask-level bipolar \\
        --mask-label std10_rv-gross-std3_satmargin15_sw_logz4

For each held-out subject, every ROI x band cell of the reference run is refitted
on the same pair x channel rows minus that subject's: same model, random
effects, floors, gap window, first-hours exclusion and pinned insula cut (`pain_change.load_rows`,
`pain_change.cell_jobs`). The likelihood-ratio test is skipped, since scoring
needs only the fixed effects and the residual variance.
Each fold uses the reference run's engine: statsmodels (`pain_change.cell_jobs`) or
lme4 (`pain_change.lme4_records`, one `pain_change_lmer.R` call per fold).

The training subjects are the reference run's `subjects[]`, so a fold differs from
the reference fit by exactly one subject. The folds are the training subjects that
are still discovery, the ones the prediction scores.

GATE. A `none` fold refits every subject first and must reproduce the reference
`cells.csv` (d_pain beta and var_resid to REPRODUCE_TOL, converged exactly) before
any fold runs.

Output: beside the reference run, `<reference run token>_loo_<timestamp>/`
`cells_loo.csv`, one row per fold x cell with `heldout_subject` (`none` for the
gate fold), plus METHODS.md and provenance.json. Export the folds as weights with
`pain_change_prediction weights --source-run <this run>`.

EXPLORATORY. Discovery cohort only. Nominations, not findings.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import fullres_cells, pain_change
from ieeg_ehr.views import view_config
from ieeg_ehr.views import build_pain_epoch_fullres_zscore as zview

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/pain_change_loo.py'
NO_HOLDOUT = 'none'
REPRODUCE_TOL = 1e-8
CELL_KEY = ['region', 'freq']


def reproduce_reference(records, reference_cells):
    """Max |diff| of beta and var_resid against the reference fit; raises past tolerance."""
    mine = pd.DataFrame(records).set_index(CELL_KEY)
    ref = reference_cells.set_index(CELL_KEY).loc[mine.index]
    if not (mine['converged'].astype(bool) == ref['converged'].astype(bool)).all():
        raise AssertionError('the full refit converges in different cells than the reference')
    ok = ref['converged'].astype(bool)
    diffs = {c: float(np.nanmax(np.abs(mine.loc[ok, c] - ref.loc[ok, c])))
             for c in ('dpain_beta', 'var_resid')}
    logger.info('full refit vs reference: %s over %d converged cells', diffs, int(ok.sum()))
    if max(diffs.values()) > REPRODUCE_TOL:
        raise AssertionError(f'the full refit does not reproduce the reference: {diffs}')
    return diffs


def write_methods(run_dir, reference_run, model, engine, folds, n_train, diffs, cells):
    loo = cells[cells['heldout_subject'] != NO_HOLDOUT]
    per_fold = loo.groupby('heldout_subject')['converged'].sum()
    text = f"""# pain_change leave-one-subject-out refits

EXPLORATORY, discovery cohort. NOMINATIONS, NOT FINDINGS.

The model of `{Path(reference_run).name}`, refitted once per held-out subject
with its engine, `{engine}`:

```
{pain_change.MODELS[model].formula}
```

- Training subjects: the reference run's {n_train}. {len(folds)} folds, one per
  training subject that is still discovery.
- Each fold drops every pair x channel row of its subject and refits every cell
  with the reference floors. No likelihood-ratio test.
- Gate: the `{NO_HOLDOUT}` fold refits all {n_train} subjects and reproduces the
  reference `cells.csv`: max |diff| d_pain beta {diffs['dpain_beta']:.2g},
  var_resid {diffs['var_resid']:.2g}.
- Converged cells per fold: median {int(per_fold.median())}, range
  {int(per_fold.min())}-{int(per_fold.max())}.
"""
    (run_dir / 'METHODS.md').write_text(text)


def main(argv=None):
    from joblib import Parallel

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference-run', required=True)
    ap.add_argument('--n-jobs', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))
    view_config.add_view_arguments(ap)
    ap.set_defaults(baseline='all_pain_epochs', freq='fullres', region='none')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    reference_run = Path(args.reference_run)
    prov = json.loads((reference_run / 'provenance.json').read_text())
    params = prov['params']
    vc = view_config.from_args(args)
    if vc.provenance() != params['view_config']:
        raise ValueError('view arguments differ from the reference run:\n'
                         f'  here      {vc.provenance()}\n'
                         f'  reference {params["view_config"]}')
    train = set(prov['subjects'])
    outside = sorted(s for s in train
                     if s.replace('sub-', '') not in config.locked_discovery_subjects())
    if outside:
        raise ValueError(f'reference subjects outside the locked discovery set: {outside}')
    folds = sorted(s for s in train if s.replace('sub-', '') in config.discovery_subjects())

    zdir = zview.zscore_dir(vc)
    paths = [p for p in sorted(zdir.glob('zmean_sub-*_ses-*.parquet'))
             if f'sub-{fullres_cells.subject_session_of(p)[0]}' in train]
    rows = pain_change.load_rows(paths, params['roi_scheme'], params['band_set'],
                                 params['min_gap_min'], params['max_gap_min'],
                                 insula_threshold=prov['insula_split']['thresholds'],
                                 exclude_first_h=params['exclude_first_h'])
    if rows.notch != params['notch_bins_removed']:
        raise ValueError('the notch differs from the reference run')
    if set(rows.index['subject']) != train:
        raise ValueError(f'rows cover {sorted(set(rows.index["subject"]) ^ train)} '
                         'differently from the reference run')

    min_subjects, model, engine = params['min_subjects'], params['model'], params['engine']

    def fit(holdout=None):
        if engine == 'lme4':
            return pain_change.lme4_records(rows, min_subjects, model, args.n_jobs, holdout)
        return Parallel(n_jobs=args.n_jobs, verbose=2)(
            pain_change.cell_jobs(rows, min_subjects, model, holdout, with_lrt=False))

    full = fit()
    diffs = reproduce_reference(full, io.read_table(reference_run / 'cells.csv',
                                                    on_stale='warn'))
    logger.info('refitting %d folds x up to %d cells on %d worker(s)', len(folds),
                len(rows.regions) * len(rows.cols), args.n_jobs)
    loo = []
    for i, s in enumerate(folds, 1):
        logger.info('fold %d/%d: %s held out', i, len(folds), s)
        loo.extend(fit(s))
    cells = pd.concat([pd.DataFrame(full).assign(heldout_subject=NO_HOLDOUT),
                       pd.DataFrame(loo)], ignore_index=True)

    run_dir = config.analysis_run_dir(question=config.PAIN_CHANGE_QUESTION,
                                      output_type=reference_run.parent.parent.name,
                                      view_scheme=reference_run.parent.name,
                                      run_name=f"{params['run_name']}_loo")
    run_params = {'reference_run': str(reference_run), 'model': model, 'engine': engine,
                  'formula': pain_change.MODELS[model].formula,
                  'vc': pain_change.mm.VC_CHANGE, 'band_set': params['band_set'],
                  'roi_scheme': params['roi_scheme'], 'min_subjects': min_subjects,
                  'min_gap_min': params['min_gap_min'], 'max_gap_min': params['max_gap_min'],
                  'insula_threshold': prov['insula_split']['thresholds'],
                  'exclude_first_h': params['exclude_first_h'],
                  'with_lrt': False, 'folds': folds}
    subjects = sorted(train)
    io.write_table(cells, run_dir / 'cells_loo.csv', params=run_params,
                   parents=[str(zdir), str(reference_run / 'cells.csv')],
                   subjects=subjects, script=SCRIPT)
    write_methods(run_dir, reference_run, model, engine, folds, len(train), diffs, cells)
    io.write_run_provenance(
        run_dir, script=SCRIPT, params=run_params,
        parents=[str(zdir), str(reference_run)], subjects=subjects,
        extra={'status': 'EXPLORATORY leave-one-subject-out refits, NOT a finding',
               'reproduce_reference': diffs, 'n_folds': len(folds)})
    io.log_analysis(f'pain_change LOO refits, {len(folds)} folds of '
                    f'{reference_run.name}', run_dir)
    print(run_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())

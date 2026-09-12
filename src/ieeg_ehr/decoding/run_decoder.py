"""Decode one subject-session. ONE unit per invocation = one Slurm array task.

    python -m ieeg_ehr.decoding.run_decoder --subject 183 --view-dir <dir> \
        --run-timestamp 20260908-150000

WHY THE TIMESTAMP IS AN ARGUMENT AND NOT COMPUTED HERE
------------------------------------------------------
45 array tasks starting at slightly different seconds would each build a
DIFFERENT run directory, and the run would shatter into 45 sibling folders that
nothing could aggregate. The submitting sbatch computes it once and passes the
same value to every task, so all of them write into one run.

Each task writes only its own files under `units/`; `aggregate.py` combines them
afterwards. Nothing is appended to a shared file, so there is no concurrent-write
race between array tasks.
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from ieeg_ehr import config, io
from ieeg_ehr.decoding import arms as arms_mod
from ieeg_ehr.decoding import cascade, cv, eligible, features

logger = logging.getLogger(__name__)

DEFAULT_VIEW_SCHEME = 'chan-paper6-raw'

#: The level-5 view_scheme FOLDER, derived from the feature source rather than
#: passed by hand. Two sources must never share a run directory -- they hold
#: different channel sets and different preprocessing -- and relying on a CLI
#: flag to keep them apart failed immediately: the first Laplacian run wrote into
#: `chan-paper6-raw/`, the bipolar view's folder, because --view-scheme simply
#: defaulted. Deriving it means the mistake is not available.
SCHEME_BY_SOURCE = {
    'psd_view': 'chan-paper6-raw',
    'laplacian_bandpass_rms': 'perchannel-laplacian-bandrms',
}


def unit_dir(run_timestamp, arm, view_scheme=DEFAULT_VIEW_SCHEME, run_name='per_subject'):
    """The shared run directory for one arm, plus its per-unit subfolder."""
    run = config.decoding_run_dir(output_type=arm, run_name=run_name,
                                  view_scheme=view_scheme, timestamp=run_timestamp)
    return run, run / 'units'


def fit_index_model(arm, X, y, alpha, l1_ratio, seed=0):
    """Refit on ALL of this unit's epochs with the chosen penalty.

    The index model is not cross-validated -- it is the model you would deploy,
    and it is what the paper's timescale analysis re-applies to later windows.
    Because it uses every epoch, the CV SCHEME is irrelevant to it: there is one
    index model per (unit, arm), not one per (unit, arm, cv_scheme).
    """
    if alpha is None or not np.isfinite(alpha):
        raise arms_mod.NotFittableError(
            'no penalty was selected across bootstraps, so there is nothing to '
            'pin the index model at')
    estimator = arms_mod.pinned_estimator(arm, alpha, l1_ratio or 0.5, len(y),
                                          random_state=seed)
    pipe = arms_mod.make_pipeline(estimator)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=ConvergenceWarning)
        pipe.fit(X, y)
    return pipe


def run_unit(subject, session, view_dir, run_timestamp, arms=arms_mod.ARMS,
             schemes=cv.CV_SCHEMES, n_bootstraps=cv.N_BOOTSTRAPS,
             view_scheme=DEFAULT_VIEW_SCHEME, run_name='per_subject',
             save_models=True, min_epochs=30):
    fm = features.build_matrix(subject, session, view_dir, min_epochs=min_epochs)
    X, y_raw = fm.X, fm.y
    written = []

    # The folder follows the FEATURE SOURCE unless the caller overrode it.
    if view_scheme in (None, DEFAULT_VIEW_SCHEME):
        source = fm.report.get('source', 'psd_view')
        view_scheme = SCHEME_BY_SOURCE.get(source, view_scheme or DEFAULT_VIEW_SCHEME)
        logger.info('view_scheme resolved to %r from source %r', view_scheme, source)

    for arm in arms:
        y = cv.make_labels(y_raw, arm)
        if arm == 'classification':
            # The paper's inclusion criteria 2 and 3, applied PER ARM: a subject
            # that fails them is still fine for regression and ordinal. Skipping
            # rather than failing, because "this subject cannot support a median
            # split" is a result.
            reason = eligible.classification_ok(
                float(np.median(y_raw)), float(y_raw.max() - y_raw.min()))
            if reason:
                logger.warning('sub-%s ses-%s: skipping the classification arm '
                               '(%s)', subject, session, reason)
                continue
            balance = float(np.mean(y))
            if min(balance, 1 - balance) == 0:
                logger.warning('sub-%s ses-%s: median split leaves an empty '
                               'class; skipping the classification arm',
                               subject, session)
                continue

        run_dir, units = unit_dir(run_timestamp, arm, view_scheme, run_name)
        units.mkdir(parents=True, exist_ok=True)

        boot_rows, summary_row = [], {'unit': fm.unit, 'arm': arm}
        summary_row.update({k: v for k, v in fm.report.items()
                            if not isinstance(v, (list, dict))})
        if arm == 'classification':
            summary_row['class_balance_high'] = balance

        coef_rows, pred_frames = [], []
        for scheme in schemes:
            # Per-feature coefficient SUMMARIES for the real and permuted runs.
            # Both are needed: the paper's "significant feature" is one whose
            # coefficient distribution differs from the permuted models, which
            # cannot be asked of the real run alone.
            store_obs = cv.new_coefficient_store(X.shape[1])
            store_null = cv.new_coefficient_store(X.shape[1])
            preds = []
            try:
                obs = cv.run_bootstraps(arm, X, y, scheme, n_bootstraps,
                                        shuffle_labels=False, base_seed=0,
                                        collect=store_obs, predictions=preds)
                null = cv.run_bootstraps(arm, X, y, scheme, n_bootstraps,
                                         shuffle_labels=True, base_seed=10_000,
                                         collect=store_null, predictions=preds)
            except arms_mod.NotFittableError as exc:
                logger.warning('sub-%s %s/%s: %s', subject, arm, scheme, exc)
                continue
            boot_rows.extend(obs)
            boot_rows.extend(null)
            pred_frames.extend(preds)

            for shuffled, store in ((False, store_obs), (True, store_null)):
                stats_ = cv.finalize_coefficients(store)
                coef_rows.append(pd.DataFrame({
                    'unit': fm.unit, 'arm': arm, 'cv_scheme': scheme,
                    'shuffled': shuffled, 'feature': fm.feature_names,
                    'channel': [f.split('|')[0] for f in fm.feature_names],
                    'band': [f.split('|')[1] for f in fm.feature_names],
                    'coef_mean': stats_['coef_mean'], 'coef_sd': stats_['coef_sd'],
                    'selection_frequency': stats_['selection_frequency'],
                    'n_fits': stats_['n_fits']}))

            metrics = [k for k in obs[0]
                       if k not in ('bootstrap', 'cv_scheme', 'arm', 'shuffled')]
            for metric in metrics:
                for key, value in cv.summarize(obs, null, metric).items():
                    summary_row[f'{scheme}__{key}'] = value

        if not boot_rows:
            logger.warning('sub-%s ses-%s %s: nothing fitted, unit skipped',
                           subject, session, arm)
            continue

        stem = f'{fm.unit}'
        io.write_table(pd.DataFrame([summary_row]), units / f'{stem}_metrics.csv',
                       params=fm.report, subjects=[f'sub-{subject}'])
        io.write_table(pd.DataFrame(boot_rows), units / f'{stem}_bootstraps.csv',
                       params={'arm': arm, 'n_bootstraps': n_bootstraps},
                       subjects=[f'sub-{subject}'])

        # Feature STABILITY across bootstraps, real and permuted. This is what
        # makes "significant feature" mean what the paper means rather than
        # "nonzero in one index fit".
        if coef_rows:
            io.write_table(pd.concat(coef_rows, ignore_index=True),
                           units / f'{stem}_coef_stability.csv',
                           params={'arm': arm, 'n_bootstraps': n_bootstraps},
                           subjects=[f'sub-{subject}'])
        # Out-of-fold predictions, so an ROC can be built without re-running.
        if pred_frames:
            io.write_table(pd.concat(pred_frames, ignore_index=True),
                           units / f'{stem}_predictions.csv',
                           params={'arm': arm, 'n_bootstraps': n_bootstraps},
                           subjects=[f'sub-{subject}'])

        # Coefficients: one index-model fit at the modal penalty, plus the
        # SELECTION FREQUENCY that is the more trustworthy statistic under
        # collinearity (see arms.py).
        primary = [r for r in boot_rows if not r['shuffled']]
        alpha = float(np.nanmedian([r['alpha_median'] for r in primary]))
        l1_ratio = float(np.nanmedian([r['l1_ratio_median'] for r in primary]))
        try:
            index_model = fit_index_model(arm, X, y, alpha, l1_ratio)
            coefs = arms_mod.coefficients(index_model)
            coef_table = pd.DataFrame({
                'unit': fm.unit, 'arm': arm, 'feature': fm.feature_names,
                'channel': [f.split('|')[0] for f in fm.feature_names],
                'band': [f.split('|')[1] for f in fm.feature_names],
                'coefficient': coefs,
                'selected': (np.abs(coefs) > 0).astype(int),
            })
            io.write_table(coef_table, units / f'{stem}_coefficients.csv',
                           params={'arm': arm, 'alpha': alpha, 'l1_ratio': l1_ratio},
                           subjects=[f'sub-{subject}'])
            if save_models:
                io.save_model(index_model, units / f'{stem}_index_model.joblib',
                              params={'arm': arm, 'alpha': alpha,
                                      'l1_ratio': l1_ratio, 'n_features': X.shape[1]},
                              subjects=[f'sub-{subject}'],
                              script='ieeg_ehr/decoding/run_decoder.py')
        except (arms_mod.NotFittableError, ValueError) as exc:
            logger.warning('sub-%s %s: index model failed (%s); metrics still '
                           'written', subject, arm, exc)

        written.append(run_dir)
        logger.info('sub-%s ses-%s %s -> %s', subject, session, arm, units)

    return fm, written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subject', required=True)
    ap.add_argument('--session', default='01')
    ap.add_argument('--view-dir', required=True,
                    help='materialized chan-paper6-raw view directory')
    ap.add_argument('--run-timestamp', required=True,
                    help='shared across every array task of one run')
    ap.add_argument('--arms', nargs='+', default=list(arms_mod.ARMS),
                    choices=list(arms_mod.ARMS))
    ap.add_argument('--cv-schemes', nargs='+', default=list(cv.CV_SCHEMES),
                    choices=list(cv.CV_SCHEMES))
    ap.add_argument('--n-bootstraps', type=int, default=cv.N_BOOTSTRAPS)
    ap.add_argument('--run-name', default='per_subject')
    ap.add_argument('--view-scheme', default=DEFAULT_VIEW_SCHEME)
    ap.add_argument('--min-epochs', type=int, default=30)
    ap.add_argument('--no-save-models', action='store_true')
    ap.add_argument('--log-level', default='INFO')
    args = ap.parse_args(argv)

    logging.basicConfig(level=args.log_level,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    try:
        fm, run_dirs = run_unit(
            args.subject, args.session, Path(args.view_dir), args.run_timestamp,
            arms=args.arms, schemes=args.cv_schemes, n_bootstraps=args.n_bootstraps,
            view_scheme=args.view_scheme, run_name=args.run_name,
            save_models=not args.no_save_models, min_epochs=args.min_epochs)
    except (cascade.NoUsableDataError, FileNotFoundError,
            features.ViewMismatchError) as exc:
        # A unit that cannot be decoded is a RESULT, not a crash -- but only when
        # the reason is the data. A ViewMismatchError is a wiring error and must
        # be loud, so it exits non-zero.
        loud = isinstance(exc, features.ViewMismatchError)
        logger.error('sub-%s ses-%s not decodable: %s', args.subject,
                     args.session, exc)
        return 1 if loud else 0

    for run_dir in dict.fromkeys(run_dirs):
        io.log_analysis('per-subject pain decoder (EXPLORATORY)', run_dir)
    logger.info('done: %r', fm)
    return 0


if __name__ == '__main__':
    sys.exit(main())

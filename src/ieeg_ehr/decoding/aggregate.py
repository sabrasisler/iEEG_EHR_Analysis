"""Combine per-unit decoder outputs into one run's tables.

    python -m ieeg_ehr.decoding.aggregate --run-timestamp 20260908-150000

Run AFTER the array drains. Each array task wrote only its own files under
`units/`, so this is a pure gather -- no task ever appended to a shared file and
there is nothing to have raced.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not pool coefficients across subjects, and it does not make regional
claims. Coefficient magnitudes are not comparable between units: under
collinearity the elastic net picks one of a correlated group, so a channel's
coefficient depends on what ELSE that subject has coverage in. The group-level
statistic here is therefore "how many units decode better than their own null",
never "how big is the amygdala coefficient". Regional interpretation is a
separate analysis with a better-suited estimator (per-subject effect + sign
consistency), and is out of scope for v1.

Group summaries are also reported WITHOUT any pooled p-value that ignores
per-subject structure (CLAUDE.md), and FDR is applied within the discovery set
only.
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from ieeg_ehr import io
from ieeg_ehr.decoding import arms as arms_mod
from ieeg_ehr.decoding import cv
from ieeg_ehr.decoding.run_decoder import DEFAULT_VIEW_SCHEME, unit_dir

logger = logging.getLogger(__name__)


def _read_all(units_dir, suffix):
    paths = sorted(units_dir.glob(f'*_{suffix}.csv'))
    if not paths:
        return pd.DataFrame(), []
    frames = [io.read_table(p, on_stale='ignore') for p in paths]
    return pd.concat(frames, ignore_index=True), paths


def benjamini_hochberg(p):
    """BH-FDR q-values. Within the discovery set only, by construction --
    nothing from the hold-out reaches this table."""
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    vals = p[ok]
    if vals.size == 0:
        return q
    order = np.argsort(vals)
    ranked = vals[order]
    n = vals.size
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]     # enforce monotonicity
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    q[ok] = out
    return q


def summarize_group(metrics, arm):
    """Per-arm group summary: the distribution of per-unit effects, and how many
    beat their own null. Never a pooled p-value."""
    primary = cv.primary_metric(arm)
    rows = []
    for scheme in cv.CV_SCHEMES:
        col = f'{scheme}__{primary}'
        pcol = f'{scheme}__{primary}_p_perm'
        if col not in metrics:
            continue
        vals = metrics[col].to_numpy(dtype=float)
        finite = np.isfinite(vals)
        row = {
            'arm': arm, 'cv_scheme': scheme, 'metric': primary,
            'n_units': int(finite.sum()),
            'mean': float(np.nanmean(vals)), 'sd': float(np.nanstd(vals)),
            'median': float(np.nanmedian(vals)),
            'min': float(np.nanmin(vals)) if finite.any() else np.nan,
            'max': float(np.nanmax(vals)) if finite.any() else np.nan,
        }
        if pcol in metrics:
            q = benjamini_hochberg(metrics[pcol].to_numpy(dtype=float))
            row['n_sig_uncorrected'] = int(np.nansum(
                metrics[pcol].to_numpy(dtype=float) < 0.05))
            row['n_sig_fdr'] = int(np.nansum(q < 0.05))
        rows.append(row)
    return rows


def aggregate(run_timestamp, arms=arms_mod.ARMS, view_scheme=DEFAULT_VIEW_SCHEME,
              run_name='per_subject'):
    group_rows, written = [], []
    for arm in arms:
        run_dir, units = unit_dir(run_timestamp, arm, view_scheme, run_name)
        if not units.exists():
            logger.warning('%s: no units/ directory, arm skipped', arm)
            continue

        metrics, metric_paths = _read_all(units, 'metrics')
        if metrics.empty:
            logger.warning('%s: no per-unit metrics found', arm)
            continue
        boots, _ = _read_all(units, 'bootstraps')
        coefs, _ = _read_all(units, 'coefficients')

        # Add FDR within this arm, so the q-value is visible beside its p.
        primary = cv.primary_metric(arm)
        for scheme in cv.CV_SCHEMES:
            pcol = f'{scheme}__{primary}_p_perm'
            if pcol in metrics:
                metrics[f'{scheme}__{primary}_q_fdr'] = benjamini_hochberg(
                    metrics[pcol].to_numpy(dtype=float))

        subjects = sorted(metrics['unit'].str.split('_').str[0].unique())
        params = {'arm': arm, 'view_scheme': view_scheme,
                  'n_units': int(metrics['unit'].nunique()),
                  'cv_schemes': list(cv.CV_SCHEMES)}

        io.write_table(metrics, run_dir / 'metrics.csv', params=params,
                       parents=[io.parent_ref(p) for p in metric_paths[:1]],
                       subjects=subjects)
        if not boots.empty:
            io.write_table(boots, run_dir / 'per_bootstrap.csv', params=params,
                           subjects=subjects)
        if not coefs.empty:
            io.write_table(coefs, run_dir / 'coefficients.csv', params=params,
                           subjects=subjects)

        io.write_run_provenance(run_dir, script='ieeg_ehr/decoding/aggregate.py',
                                params=params, subjects=subjects)
        group_rows.extend(summarize_group(metrics, arm))
        written.append(run_dir)
        logger.info('%s: %d units -> %s', arm, metrics['unit'].nunique(), run_dir)

    if group_rows:
        # The cross-arm summary lands beside the FIRST arm's run directory rather
        # than in a new folder: it describes this run, and a folder-per-summary is
        # exactly the level-1/2 proliferation CLAUDE.md forbids.
        summary_dir = written[0]
        io.write_table(pd.DataFrame(group_rows), summary_dir / 'group_summary.csv',
                       params={'arms': list(arms), 'run_timestamp': run_timestamp})
        logger.info('group summary -> %s', summary_dir / 'group_summary.csv')
        print(pd.DataFrame(group_rows).to_string(index=False))
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-timestamp', required=True)
    ap.add_argument('--arms', nargs='+', default=list(arms_mod.ARMS),
                    choices=list(arms_mod.ARMS))
    ap.add_argument('--view-scheme', default=DEFAULT_VIEW_SCHEME)
    ap.add_argument('--run-name', default='per_subject')
    ap.add_argument('--log-level', default='INFO')
    args = ap.parse_args(argv)

    logging.basicConfig(level=args.log_level,
                        format='%(asctime)s %(levelname)s %(message)s')
    written = aggregate(args.run_timestamp, arms=args.arms,
                        view_scheme=args.view_scheme, run_name=args.run_name)
    return 0 if written else 1


if __name__ == '__main__':
    sys.exit(main())

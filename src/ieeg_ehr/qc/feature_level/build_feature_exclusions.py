#!/usr/bin/env python3
"""
Feature-level QC, threshold stage (Step A): apply K and B to the stored
power-outlier metrics and emit per-window exclusions. Cheap, touches no NWB --
so a threshold sweep re-runs only this, never detect_power_outlier.py.

The rule (config/feature_qc_params.py owns the defaults):

    window (channel, run, window_idx) is excluded
      <=>  frac(z > K) > B
      <=>  z_b<B*100> > K            <-- the stored order statistic

That second equivalence is the whole reason this step is cheap: the metric stage
stored the order statistic of the per-bin z at each B in FEATURE_BIN_FRAC_GRID, so
thresholding is a single column comparison. See tests/test_feature_qc.py for the
proof that the two forms agree exactly (an interpolated quantile would NOT).

WHAT THIS DOES *NOT* DO
-----------------------
This is the WINDOW level only -- the K and B stages of the cascade. The epoch
levels (X: fraction of an epoch's windows flagged; Y: fraction of a channel's
epochs flagged -> drop channel; Z: fraction of surviving channels flagged -> drop
epoch) are NOT here, because they are defined over epochs and epochs only exist
relative to an epoch definition. They live in the view layer, applied to the epoch
cache at load time.

SPARSE OUTPUT, AND WHY THAT IS SAFE
-----------------------------------
The input metric table is sparse (only rows whose largest stored order statistic
cleared FEATURE_METRIC_STORE_FLOOR), so this output is sparse too: it contains
ONLY excluded windows. A window absent from the file is not excluded.

That is safe for exactly one reason, which this script ENFORCES rather than
assumes: K > FEATURE_METRIC_STORE_FLOOR. Any window omitted from the metric table
had every order statistic <= the floor < K, so it cannot be excluded at this K.
Thresholding below the floor would silently miss windows, so it is a hard error
here, not a warning.

Denominators (how many windows a channel HAS) are deliberately not duplicated
into every row -- they live in metrics/summary/, recorded as a parent in
params.json, and each output file's own sidecar carries this subject/session's
counts in `extra` so a rate is readable without opening the metrics tree.

Usage:
  python -m ieeg_ehr.qc.feature_level.build_feature_exclusions
  python -m ieeg_ehr.qc.feature_level.build_feature_exclusions --z-thresh 4 --bin-frac 0.10
  python -m ieeg_ehr.qc.feature_level.build_feature_exclusions --subjects 071,085
"""

import argparse
import json
import logging
import os
import re

import pandas as pd

from ieeg_ehr import config, io

logger = logging.getLogger(__name__)

ARTIFACT_TYPE = 'power_outlier'

_TAG_RE = re.compile(r'^sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)$')


def _parse_tag(path):
    m = _TAG_RE.match(path.stem)
    if not m:
        raise ValueError(f'Unexpected metrics filename: {path.name}')
    return m.group('subject'), m.group('session')


def stat_column(bin_frac):
    """The stored order-statistic column implementing this B, e.g. 0.20 -> 'z_b20'.

    Raises if B is not one of the stored grid values: the metric stage stored a
    fixed set of order statistics, and no column in the table answers a B outside
    it. Silently falling back to the nearest grid value would threshold on a
    different rule than the label claims.
    """
    if bin_frac not in tuple(config.FEATURE_BIN_FRAC_GRID):
        raise SystemExit(
            f'--bin-frac {bin_frac:g} is not in FEATURE_BIN_FRAC_GRID '
            f'{tuple(config.FEATURE_BIN_FRAC_GRID)}. The metric stage stores one order '
            f'statistic per grid value, so this B has no column. Either pick a grid value '
            f'or add it to the grid and re-run detect_power_outlier.py.')
    return f'z_b{bin_frac * 100:g}'


def build_one(metric_path, z_thresh, bin_frac, out_path, mask_label):
    col = stat_column(bin_frac)
    subject, session = _parse_tag(metric_path)

    metrics = io.read_table(metric_path,
                            columns=['run_id', 'channel', 'window_idx',
                                     'window_start_time', col, 'z_max',
                                     'n_bins_nonfinite', 'rv_excluded'])
    if config.FEATURE_Z_SIDE == 'both':
        # The metric stage already stored |z| order statistics when side='both',
        # so the comparison is identical; recorded here so the label and the
        # params.json agree about which convention produced the numbers.
        pass

    excluded = metrics[metrics[col] > z_thresh].copy()
    excluded = excluded.rename(columns={col: 'metric_value'})
    excluded['excluded'] = True
    excluded = (excluded[['run_id', 'channel', 'window_idx', 'window_start_time',
                          'metric_value', 'z_max', 'n_bins_nonfinite',
                          'rv_excluded', 'excluded']]
                .sort_values(['run_id', 'channel', 'window_idx'])
                .reset_index(drop=True))

    # Denominators from the metric stage's own summary table, so the rate this
    # logs and records is against every window that EXISTS, not just the ones
    # that survived the storage floor.
    summary_path = config.feature_metrics_path('summary', subject, session, mask_label)
    n_windows_total = n_rv = None
    if summary_path.exists():
        s = io.read_table(summary_path, columns=['n_windows', 'n_rv_excluded'])
        n_windows_total = int(s['n_windows'].sum())
        n_rv = int(s['n_rv_excluded'].sum())

    n_excl = len(excluded)
    n_incremental = int((~excluded['rv_excluded']).sum()) if n_excl else 0
    params = {
        'artifact_type': ARTIFACT_TYPE,
        'z_thresh': z_thresh, 'bin_frac': bin_frac,
        'z_side': config.FEATURE_Z_SIDE,
        'stat_column': col,
        'raw_voltage_mask_label': mask_label,
        'store_floor': config.FEATURE_METRIC_STORE_FLOOR,
        'level': 'window',
    }
    io.write_table(excluded, out_path, params=params,
                   parents=[str(metric_path)] + ([str(summary_path)] if summary_path.exists() else []),
                   subjects=[f'sub-{subject}'],
                   extra={'counts': {
                       'n_excluded': n_excl,
                       'n_excluded_not_rv_excluded': n_incremental,
                       'n_windows_total': n_windows_total,
                       'n_rv_excluded': n_rv,
                       'n_channels': int(excluded['channel'].nunique()) if n_excl else 0,
                   },
                       'note': 'SPARSE: excluded windows only. A window absent from this '
                               'file is not excluded. Safe because z_thresh > store_floor '
                               '(enforced at build time). Denominators: n_windows_total '
                               'above, or metrics/summary/.'})

    rate = (100.0 * n_excl / n_windows_total) if n_windows_total else float('nan')
    inc_rate = (100.0 * n_incremental / n_windows_total) if n_windows_total else float('nan')
    logger.info('  sub-%s ses-%s: %d excluded channel-windows (%.3f%% of %s), '
                'incremental over raw-voltage: %d (%.3f%%) -> %s',
                subject, session, n_excl, rate, n_windows_total, n_incremental,
                inc_rate, out_path.name)
    return {'subject': subject, 'session': session, 'n_excluded': n_excl,
            'n_excluded_not_rv_excluded': n_incremental,
            'n_windows_total': n_windows_total}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--z-thresh', type=float, default=None,
                    help=f'K (default: config {config.FEATURE_Z_THRESH})')
    ap.add_argument('--bin-frac', type=float, default=None,
                    help=f'B, must be in FEATURE_BIN_FRAC_GRID '
                         f'(default: config {config.FEATURE_BIN_FRAC})')
    ap.add_argument('--label', default=None,
                    help='Output folder label (default: auto from thresholds, e.g. z5_binfrac20)')
    ap.add_argument('--mask-label', default=None,
                    help='Which raw-voltage-scoped metrics to read (default: config '
                         f'{config.CANONICAL_MASK_LABEL}). Pass "none" for the unmasked-baseline metrics.')
    ap.add_argument('--subjects', default=None,
                    help='Comma-separated subject IDs to restrict to (default: all present)')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    z_thresh = config.FEATURE_Z_THRESH if args.z_thresh is None else args.z_thresh
    bin_frac = config.FEATURE_BIN_FRAC if args.bin_frac is None else args.bin_frac
    mask_label = config.CANONICAL_MASK_LABEL if args.mask_label is None else args.mask_label
    if mask_label == 'none':
        mask_label = None

    # The sparse-input guard. Not a warning: below the floor the metric table has
    # already dropped rows this K would have excluded, so the output would
    # understate exclusions with no way to tell from the file.
    if z_thresh <= config.FEATURE_METRIC_STORE_FLOOR:
        raise SystemExit(
            f'--z-thresh {z_thresh:g} is at or below FEATURE_METRIC_STORE_FLOOR '
            f'{config.FEATURE_METRIC_STORE_FLOOR:g}. The per-window metric table only '
            f'stores rows above that floor, so thresholding here would silently miss '
            f'excluded windows. Lower the floor and re-run detect_power_outlier.py, or '
            f'pick a higher K.')

    label = args.label or config.feature_exclusion_label(z_thresh, bin_frac)
    stat_column(bin_frac)          # validate B before doing any work

    metrics_dir = config.feature_metrics_dir('per_window', mask_label)
    paths = sorted(metrics_dir.glob('sub-*_ses-*.parquet'))
    if args.subjects:
        wanted = {s.strip().replace('sub-', '') for s in args.subjects.split(',')}
        paths = [p for p in paths if _parse_tag(p)[0] in wanted]
    if not paths:
        raise SystemExit(f'No per-window metric tables in {metrics_dir} '
                         '(run detect_power_outlier.py first).')

    out_dir = config.exclusion_dir(config.FEATURE_LEVEL_ROOT, ARTIFACT_TYPE, label)
    out_dir.mkdir(parents=True, exist_ok=True)

    io.warn_if_dirty()
    logger.info('=== build_feature_exclusions: %s (label=%s) K=%g B=%g col=%s mask=%s ===',
                ARTIFACT_TYPE, label, z_thresh, bin_frac, stat_column(bin_frac), mask_label)

    rows = []
    for metric_path in paths:
        subject, session = _parse_tag(metric_path)
        out_path = config.feature_exclusion_path(subject, session, ARTIFACT_TYPE, label)
        try:
            rows.append(build_one(metric_path, z_thresh, bin_frac, out_path, mask_label))
        except Exception:
            logger.exception('  sub-%s ses-%s: failed, skipping', subject, session)

    total_excl = sum(r['n_excluded'] for r in rows)
    total_inc = sum(r['n_excluded_not_rv_excluded'] for r in rows)
    total_win = sum(r['n_windows_total'] or 0 for r in rows)
    params_out = {
        'artifact_type': ARTIFACT_TYPE, 'label': label,
        'level': 'window',
        'thresholds': {'z_thresh': z_thresh, 'bin_frac': bin_frac,
                       'z_side': config.FEATURE_Z_SIDE,
                       'stat_column': stat_column(bin_frac)},
        'raw_voltage_mask_label': mask_label,
        'store_floor': config.FEATURE_METRIC_STORE_FLOOR,
        'metrics_per_window_dir': str(metrics_dir),
        'metrics_summary_dir': str(config.feature_metrics_dir('summary', mask_label)),
        'n_subject_sessions': len(rows),
        'totals': {'n_excluded': total_excl,
                   'n_excluded_not_rv_excluded': total_inc,
                   'n_windows_total': total_win,
                   'pct_excluded': (100.0 * total_excl / total_win) if total_win else None,
                   'pct_excluded_incremental': (100.0 * total_inc / total_win) if total_win else None},
        'run_timestamp': config.run_timestamp(),
        'git': config.git_provenance(),
        'sparse': True,
        'note': 'Window level (K, B) only. The epoch cascade (X, Y, Z) is applied in '
                'the view layer against the epoch cache.',
    }
    # Atomic write: concurrent invocations for different subjects share this file.
    tmp = out_dir / f'params.json.{os.getpid()}.tmp'
    tmp.write_text(json.dumps(params_out, indent=2, default=str))
    os.replace(tmp, out_dir / 'params.json')

    logger.info('%d subject/sessions: %d excluded channel-windows of %d (%.3f%%), '
                'incremental over raw-voltage %d (%.3f%%) -> %s',
                len(rows), total_excl, total_win,
                (100.0 * total_excl / total_win) if total_win else float('nan'),
                total_inc,
                (100.0 * total_inc / total_win) if total_win else float('nan'), out_dir)


if __name__ == '__main__':
    main()

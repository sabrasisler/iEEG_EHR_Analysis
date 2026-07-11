#!/usr/bin/env python3
"""
Mask-aware z-score exclusion on the bipolar variance metric (preprocessing/
run_pipeline_bipolar.py's output). QC ONLY -- reads exclusively the bipolar
variance-metric CSVs (qc/bipolar/metrics/per_window/); never opens the PSD/NWB
output. No QC currently runs on the FFT/PSD output, and this script should
never be extended to do so -- FFT is a separate downstream product.

Analogous to build_exclusions.py's gross_artifact path (same z=(var-mean)/std,
one-sided-high threshold), but with one deliberate difference: the session
baseline here is MASK-AWARE. It takes an existing qc/raw_voltage mask
(monopolar, 60s bins) and excludes any bipolar 2s window whose enclosing 60s
bin is already flagged excluded for its anode OR cathode monopolar channel
from the baseline accumulation -- so a known raw-voltage artifact doesn't
inflate this detector's own idea of "normal" variance. (raw_voltage's
gross_artifact is intentionally NOT mask-aware, for reasons documented in
detect_gross_artifact.py; this bipolar detector's requirement is different and
new, not a bug relative to that precedent.)

The row is still written to the output table either way -- only the BASELINE
computation ignores masked-out windows, not the reported per-window rows.

Usage:
  python -m qc_scripts.build_bipolar_exclusions \
      --level-root <qc/bipolar root> \
      --raw-voltage-mask <path to qc/raw_voltage/masks/<label>/> \
      --label std5 --std-thresh 5.0
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from qc_scripts import config
from qc_scripts.detect_gross_artifact import new_accumulator, finalize_baseline

BIN_SEC = 60.0
METRIC_USECOLS = ['subject_id', 'session_id', 'run_id', 'channel', 'anode_channel',
                   'cathode_channel', 'window_start_time', 'window_end_time', 'metric_value']
MASK_USECOLS = ['session_id', 'run_id', 'channel', 'bin_start', 'excluded']


def label_for(p):
    return f"std{p['std_thresh']:g}"


def default_params():
    return {'std_thresh': config.BIPOLAR_VARIANCE_STD_THRESH}


def _load_mask_lookup(raw_voltage_mask_dir, subject_id):
    """(session_id, run_id, channel, bin_start) -> excluded bool, for one subject."""
    mask_path = Path(raw_voltage_mask_dir) / f'{subject_id}.csv'
    if not mask_path.exists():
        return {}
    df = pd.read_csv(mask_path, usecols=MASK_USECOLS)
    return {
        (row.session_id, row.run_id, row.channel, row.bin_start): bool(row.excluded)
        for row in df.itertuples(index=False)
    }


def _is_masked(mask_lookup, session_id, run_id, channel, bin_start):
    return mask_lookup.get((session_id, run_id, channel, bin_start), False)


def build_one_subject(metric_csv, mask_lookup, std_thresh):
    df = pd.read_csv(metric_csv, usecols=METRIC_USECOLS)
    if df.empty:
        return None

    df['_bin'] = (df['window_start_time'] // BIN_SEC) * BIN_SEC
    df['_masked'] = [
        _is_masked(mask_lookup, s, r, a, b) or _is_masked(mask_lookup, s, r, c, b)
        for s, r, a, c, b in zip(df['session_id'], df['run_id'], df['anode_channel'],
                                  df['cathode_channel'], df['_bin'])
    ]

    # Per-(session, channel) baseline over the NON-masked-out subset.
    accs = {}
    for (session_id, channel), grp in df[~df['_masked']].groupby(['session_id', 'channel']):
        acc = new_accumulator()
        acc['n'] = len(grp)
        acc['sum'] = float(grp['metric_value'].sum())
        acc['sumsq'] = float(np.square(grp['metric_value'].to_numpy(dtype=np.float64)).sum())
        accs[(session_id, channel)] = finalize_baseline(acc)

    def _classify(row):
        mean, std = accs.get((row['session_id'], row['channel']), (np.nan, np.nan))
        if np.isnan(std) or std == 0:
            return True   # no usable baseline -> flag the whole channel, same convention as gross_artifact
        z = (row['metric_value'] - mean) / std
        return bool(z > std_thresh)

    df['_excl_2s'] = df.apply(_classify, axis=1)

    grouped = df.groupby(['session_id', 'run_id', 'channel', 'anode_channel', 'cathode_channel', '_bin'])['_excl_2s'].any()
    subject_id = metric_csv.name.split('_')[0]
    rows = [
        {'subject_id': subject_id, 'session_id': s, 'run_id': r, 'channel': c,
         'anode_channel': a, 'cathode_channel': cat, 'bin_start': b, 'bin_end': b + BIN_SEC, 'excluded': e}
        for (s, r, c, a, cat, b), e in grouped.items()
    ]
    return pd.DataFrame(rows).sort_values(['run_id', 'channel', 'bin_start']).reset_index(drop=True)


def run(level_root, raw_voltage_mask_dir, label, std_thresh, subjects=None):
    metrics_dir = config.metrics_per_window_dir(level_root)
    out_dir = Path(level_root) / 'exclusions' / 'bipolar_variance' / label
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_csvs = sorted(metrics_dir.glob('sub-*_bipolar_variance.csv'))
    if subjects:
        wanted = {f'sub-{s.replace("sub-", "")}' for s in subjects}
        metric_csvs = [p for p in metric_csvs if p.name.split('_')[0] in wanted]
    if not metric_csvs:
        print(f"  no bipolar_variance metric CSVs in {metrics_dir}, nothing to do.", flush=True)
        return

    for metric_csv in metric_csvs:
        subject_id = metric_csv.name.split('_')[0]
        mask_lookup = _load_mask_lookup(raw_voltage_mask_dir, subject_id)
        if not mask_lookup:
            print(f"  NOTE: no raw_voltage mask rows found for {subject_id} at {raw_voltage_mask_dir} "
                  f"-- baseline computed with NOTHING masked out.", flush=True)
        df = build_one_subject(metric_csv, mask_lookup, std_thresh)
        if df is None:
            print(f"  {subject_id}: no rows, skipping.", flush=True)
            continue
        out_path = out_dir / f'{subject_id}.csv'
        df.to_csv(out_path, index=False)
        print(f"  {subject_id}: {len(df)} bins, {int(df['excluded'].sum())} excluded -> {out_path}",
              flush=True)

    prov = config.warn_if_dirty()
    params_out = {
        'std_thresh': std_thresh,
        'label': label,
        'bin_sec': BIN_SEC,
        'raw_voltage_mask_used': str(raw_voltage_mask_dir),
        'metrics_dir': str(metrics_dir),
        'run_timestamp': config.run_timestamp(),
        'git': prov,
    }
    tmp = out_dir / f'params.json.{os.getpid()}.tmp'
    with open(tmp, 'w') as f:
        json.dump(params_out, f, indent=2, default=str)
    os.replace(tmp, out_dir / 'params.json')
    print(f"  wrote {out_dir / 'params.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--level-root', default=str(config.BIPOLAR_LEVEL_ROOT),
                     help=f'qc/bipolar level root (default: {config.BIPOLAR_LEVEL_ROOT})')
    ap.add_argument('--raw-voltage-mask', required=True,
                     help='Path to a qc/raw_voltage/masks/<label>/ directory')
    ap.add_argument('--label', default=None,
                     help='Folder label (default: auto from --std-thresh, e.g. std5)')
    ap.add_argument('--std-thresh', type=float, default=None)
    ap.add_argument('--subjects', default=None,
                     help='Comma-separated subject IDs to restrict to (default: all present)')
    args = ap.parse_args()

    params = default_params()
    if args.std_thresh is not None:
        params['std_thresh'] = args.std_thresh
    label = args.label or label_for(params)
    subjects = [s.strip() for s in args.subjects.split(',')] if args.subjects else None

    print(f"=== build_bipolar_exclusions: label={label} params={params} "
          f"raw_voltage_mask={args.raw_voltage_mask} ===", flush=True)
    run(args.level_root, args.raw_voltage_mask, label, params['std_thresh'], subjects=subjects)


if __name__ == '__main__':
    main()

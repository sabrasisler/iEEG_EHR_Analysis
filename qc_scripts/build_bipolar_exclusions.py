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

BIN_SEC = 60.0
METRIC_USECOLS = ['subject_id', 'session_id', 'run_id', 'channel', 'anode_channel',
                   'cathode_channel', 'window_start_time', 'window_end_time', 'metric_value']
MASK_USECOLS = ['run_id', 'channel', 'bin_start', 'excluded']


def label_for(p):
    return f"std{p['std_thresh']:g}"


def default_params():
    return {'std_thresh': config.BIPOLAR_VARIANCE_STD_THRESH}


def _load_mask_lookup(raw_voltage_mask_dir, subject_id):
    """Concatenated (session_id, run_id, channel, bin_start, excluded) DataFrame
    across all of one subject's session mask files -- used as a merge key, not
    a Python dict, so masking a subject's metric rows is a vectorized join
    rather than a per-row lookup.

    Mask files are now one-per-session (`sub-XXX_ses-YY.csv`, session dropped as a
    column -- see CONTEXT.md's 2026-07-14 filename migration), so glob across all
    of that subject's sessions and recover session_id from each filename."""
    mask_paths = sorted(Path(raw_voltage_mask_dir).glob(f'{subject_id}_ses-*.csv'))
    frames = []
    for mask_path in mask_paths:
        session_id = mask_path.stem.split('_ses-', 1)[1].split('_')[0]
        session_id = f'ses-{session_id}'
        df = pd.read_csv(mask_path, usecols=MASK_USECOLS)
        df.insert(0, 'session_id', session_id)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=['session_id', 'run_id', 'channel', 'bin_start', 'excluded'])
    return pd.concat(frames, ignore_index=True)


def _mask_flags(df, mask_df, channel_col):
    """Vectorized version of the old per-row `_is_masked` lookup: merge
    mask_df's `excluded` column onto df keyed on (session_id, run_id,
    channel_col, _bin), returning a boolean numpy array (unmatched -> False)."""
    merged = df[['session_id', 'run_id', channel_col, '_bin']].merge(
        mask_df.rename(columns={'channel': channel_col, 'bin_start': '_bin'}),
        on=['session_id', 'run_id', channel_col, '_bin'], how='left')
    return merged['excluded'].fillna(False).to_numpy(dtype=bool)


def build_one_subject(metric_csv, mask_df, std_thresh):
    df = pd.read_csv(metric_csv, usecols=METRIC_USECOLS)
    if df.empty:
        return None

    df['_bin'] = (df['window_start_time'] // BIN_SEC) * BIN_SEC
    anode_masked = _mask_flags(df, mask_df, 'anode_channel')
    cathode_masked = _mask_flags(df, mask_df, 'cathode_channel')
    df['_masked'] = anode_masked | cathode_masked

    # Per-(session, channel) baseline over the NON-masked-out subset -- same
    # math as new_accumulator/finalize_baseline (detect_gross_artifact.py),
    # vectorized via groupby().agg() instead of a Python accumulator loop.
    stats = (df.loc[~df['_masked']].groupby(['session_id', 'channel'])['metric_value']
             .agg(n='size', s='sum', ss=lambda x: float(np.square(x.to_numpy(dtype=np.float64)).sum())))
    baseline_mean = stats['s'] / stats['n']
    baseline_var = stats['ss'] / stats['n'] - baseline_mean ** 2
    baseline_std = np.sqrt(baseline_var.clip(lower=0.0))

    # Map each row's (session, channel) baseline mean/std in one vectorized reindex
    # instead of a per-row df.apply(...) dict lookup.
    idx = pd.MultiIndex.from_arrays([df['session_id'], df['channel']])
    row_mean = baseline_mean.reindex(idx).to_numpy()
    row_std = baseline_std.reindex(idx).to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (df['metric_value'].to_numpy() - row_mean) / row_std
    degenerate = ~np.isfinite(row_std) | (row_std == 0)   # no usable baseline -> flag the whole channel,
                                                            # same convention as gross_artifact
    df['_excl_2s'] = (z > std_thresh) | degenerate

    grouped = df.groupby(['session_id', 'run_id', 'channel', 'anode_channel', 'cathode_channel', '_bin'])['_excl_2s'].any()
    subject_id = metric_csv.name.split('_')[0]
    out = grouped.reset_index(name='excluded')
    out.insert(0, 'subject_id', subject_id)
    out = out.rename(columns={'_bin': 'bin_start'})
    out['bin_end'] = out['bin_start'] + BIN_SEC
    out = out[['subject_id', 'session_id', 'run_id', 'channel', 'anode_channel',
               'cathode_channel', 'bin_start', 'bin_end', 'excluded']]
    return out.sort_values(['run_id', 'channel', 'bin_start']).reset_index(drop=True)


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
        mask_df = _load_mask_lookup(raw_voltage_mask_dir, subject_id)
        if mask_df.empty:
            print(f"  NOTE: no raw_voltage mask rows found for {subject_id} at {raw_voltage_mask_dir} "
                  f"-- baseline computed with NOTHING masked out.", flush=True)
        df = build_one_subject(metric_csv, mask_df, std_thresh)
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

#!/usr/bin/env python3
"""
Dilate saturation/flatline exclusions by ±pad_sec (default 30s) using
ABSOLUTE time (session_start_time + window_start_time), not run-relative bin
position -- so padding correctly bridges a run boundary when runs are
recorded back-to-back (observed for at least sub-093: consecutive runs start
within ~5s of the previous run's end), but does not bridge a genuine gap
between separate recording sessions. Requires a run_start_times.csv from
build_run_start_times.py.

Pure post-hoc recomputation on the existing per-window CSVs -- no raw NWB
data is touched. gross_artifact is untouched (symlinked) per explicit scope:
only saturation/flatline get padded.

Grouping is by (subject, session, channel) -- pooling across every run in a
session before sorting by absolute time and dilating -- not per-run, since
bridging across run boundaries within a session is the whole point.

Dilation itself: for each already-excluded window, mark every window whose
absolute time falls within ±pad_sec of it. Implemented via searchsorted on
each channel's time-sorted array, one lookup per originally-excluded window
-- cheap even when a channel has thousands of excluded windows (each lookup
is O(log n)), and does no work at all for windows that start clean.

Usage:
  python -m ieeg_ehr.qc.pad_exclusions \
      --src-dir /path/to/qc_variance --dst-dir /path/to/qc_variance_padded30 \
      --run-start-times /path/to/run_start_times.csv --pad-sec 30
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def pad_channel_group(times_sec, excluded, pad_sec):
    """
    times_sec: sorted 1D array of absolute times in seconds (float) for one
    (subject, session, channel)'s windows, pooled across all its runs.
    excluded: bool array, same order/length as times_sec.
    Returns a new bool array with every window within pad_sec of an already-
    excluded window also set to True.
    """
    padded = excluded.copy()
    excluded_idx = np.flatnonzero(excluded)
    for idx in excluded_idx:
        lo = np.searchsorted(times_sec, times_sec[idx] - pad_sec, side='left')
        hi = np.searchsorted(times_sec, times_sec[idx] + pad_sec, side='right')
        padded[lo:hi] = True
    return padded


def pad_subject_artifact_file(src_path, dst_path, run_start_lookup, pad_sec):
    df = pd.read_csv(src_path)

    key_df = df[['subject_id', 'session_id', 'run_id']].drop_duplicates()
    key_df['session_start_time'] = key_df.apply(
        lambda r: run_start_lookup.get((r['subject_id'], r['session_id'], r['run_id'])), axis=1)
    missing = key_df['session_start_time'].isna()
    if missing.any():
        missing_runs = key_df.loc[missing, 'run_id'].tolist()
        raise ValueError(f"{src_path.name}: no run_start_times entry for run(s) {missing_runs}")

    df = df.merge(key_df, on=['subject_id', 'session_id', 'run_id'], how='left')
    df['abs_time_sec'] = (
        pd.to_datetime(df['session_start_time'], utc=True).astype('int64') / 1e9
        + df['window_start_time']
    )

    padded_excluded = np.empty(len(df), dtype=bool)
    for (subject, session, channel), group in df.groupby(
            ['subject_id', 'session_id', 'channel'], sort=False):
        order = np.argsort(group['abs_time_sec'].values)
        times_sorted = group['abs_time_sec'].values[order]
        excluded_sorted = group['excluded'].values[order]
        padded_sorted = pad_channel_group(times_sorted, excluded_sorted, pad_sec)
        # scatter back to original row order
        padded_excluded[group.index.values[order]] = padded_sorted

    df['excluded'] = padded_excluded
    df = df.drop(columns=['session_start_time', 'abs_time_sec'])
    df.to_csv(dst_path, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src-dir', required=True)
    ap.add_argument('--dst-dir', required=True)
    ap.add_argument('--run-start-times', required=True)
    ap.add_argument('--pad-sec', type=float, default=30.0)
    ap.add_argument('--skip-existing', action='store_true',
                     help='Skip a file if its destination already exists (for resuming a '
                          'partial/timed-out run) -- does NOT validate completeness, so remove '
                          'any file left mid-write by a killed job before using this')
    args = ap.parse_args()

    run_starts = pd.read_csv(args.run_start_times)
    run_start_lookup = {
        (row.subject_id, row.session_id, row.run_id): row.session_start_time
        for row in run_starts.itertuples()
    }

    src_per_window = Path(args.src_dir) / 'per_window'
    dst_per_window = Path(args.dst_dir) / 'per_window'
    dst_per_window.mkdir(parents=True, exist_ok=True)

    for src_path in sorted(src_per_window.glob('sub-*.csv')):
        dst_path = dst_per_window / src_path.name

        if args.skip_existing and (dst_path.exists() or dst_path.is_symlink()):
            print(f"Skipping {src_path.name} (already exists at {dst_path})", flush=True)
            continue

        if src_path.name.endswith('_gross_artifact.csv'):
            if dst_path.exists() or dst_path.is_symlink():
                dst_path.unlink()
            dst_path.symlink_to(src_path.resolve())
            print(f"Symlinked {src_path.name} -> {dst_path} (unpadded, out of scope)", flush=True)
        else:
            pad_subject_artifact_file(src_path, dst_path, run_start_lookup, args.pad_sec)
            print(f"Padded (+/-{args.pad_sec}s) {src_path.name} -> {dst_path}", flush=True)


if __name__ == '__main__':
    main()

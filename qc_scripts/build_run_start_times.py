#!/usr/bin/env python3
"""
One-time (cacheable) metadata pass: for every run of the given subjects,
record session_start_time from the NWB header. This never touches the raw
voltage array (series.data stays lazy/on-disk) -- just the NWBFile object's
top-level metadata -- so it's cheap (file-open overhead, not data volume)
compared to a real pipeline run.

This lookup is what lets pad_exclusions.py convert each window's run-relative
window_start_time into an absolute timestamp, so ±30s padding can correctly
bridge across a run boundary when runs are recorded back-to-back (as
observed for at least sub-093) without incorrectly bridging a real gap
between genuinely separate recording sessions.

Output is independent of any particular pipeline/threshold version (it's raw
NWB-file metadata, not a QC result), so it defaults to a stable location
shared across qc_session_rail / qc_variance / etc. rather than living inside
any one of those output folders.

Usage:
  python -m qc_scripts.build_run_start_times --subjects 093,217
  python -m qc_scripts.build_run_start_times --subjects 093,217 --out /path/to/run_start_times.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from pynwb import NWBHDF5IO

from qc_scripts import config, io_utils


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--subjects', required=True,
                     help='Comma-separated subject IDs, e.g. 093,217')
    ap.add_argument('--out', default=None,
                     help='Output CSV path (default: <qc-output-root-parent>/run_start_times.csv)')
    ap.add_argument('--append', action='store_true',
                     help='Append to an existing CSV (skipping subjects already present) '
                          'instead of overwriting')
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else config.OUTPUT_DIR.parent / 'run_start_times.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    subjects = [s.strip().replace('sub-', '') for s in args.subjects.split(',')]

    existing = None
    if args.append and out_path.exists():
        existing = pd.read_csv(out_path)
        already_done = set(existing['subject_id'].str.replace('sub-', '', regex=False))
        subjects = [s for s in subjects if s not in already_done]
        print(f"--append: skipping {len(already_done & set(subjects))} already-present subjects")

    rows = []
    for subject in subjects:
        session_runs = io_utils.get_session_runs(subject)
        for session, run, nwb_path in session_runs:
            try:
                with NWBHDF5IO(nwb_path, 'r') as io:
                    nwb = io.read()
                    start = nwb.session_start_time
            except Exception as e:
                print(f"  WARNING: failed to read metadata for sub-{subject} ses-{session} "
                      f"run-{run} ({e!r}); skipping.")
                continue
            rows.append({
                'subject_id': f'sub-{subject}',
                'session_id': f'ses-{session}',
                'run_id': f'run-{run}',
                'session_start_time': start.isoformat(),
            })
        print(f"sub-{subject}: {sum(1 for r in rows if r['subject_id'] == f'sub-{subject}')} run(s)")

    new_df = pd.DataFrame(rows)
    if existing is not None:
        new_df = pd.concat([existing, new_df], ignore_index=True)
    new_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(new_df)} rows)")


if __name__ == '__main__':
    main()

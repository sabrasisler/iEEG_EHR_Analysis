#!/usr/bin/env python3
"""
Builds a small per-(subject, session, run, channel) summary of the resolved
saturation rail (config.RAIL_VALUES_CSV) from the existing saturation
per-window CSVs. rail_value/rail_source are already recorded per window in
those files, so this just deduplicates down to one row per channel/run —
streamed in chunks so the (potentially 900MB+) per-window files are never
fully loaded into memory.

Since the rail is now resolved session-wide (see detect_saturation.py:
resolve_session_rails), rail_value should by construction be IDENTICAL across
every run of a given channel within a session — the per-channel-across-runs
check below is a sanity check that this actually holds, not a discovery tool
the way it was when rails were inferred independently per run. Distinct rail
values ACROSS channels within a session are expected whenever
SAT_AGREEMENT_THRESHOLD isn't met session-wide (some channels fall back to
their own 'session_individual'/'fallback'/'none' result) — not itself a bug.

Usage:
  python -m qc_scripts.build_rail_summary
  python -m qc_scripts.build_rail_summary --output-dir /path/to/alt/root
"""

import argparse

import pandas as pd

from qc_scripts import config


def build(chunksize=500_000):
    rows = []
    for path in sorted(config.PER_WINDOW_DIR.glob('sub-*_saturation.csv')):
        usecols = ['subject_id', 'session_id', 'run_id', 'channel', 'rail_value', 'rail_source']
        seen = set()
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
            dedup = chunk.drop_duplicates(subset=['subject_id', 'session_id', 'run_id', 'channel'])
            for _, row in dedup.iterrows():
                key = (row['subject_id'], row['session_id'], row['run_id'], row['channel'])
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
        print(f"  {path.name}: {len(seen)} channel/run combos")

    summary = pd.DataFrame(rows).rename(columns={'rail_value': 'rail_value_v'})
    config.save_table(summary, config.RAIL_VALUES_CSV)
    print(f"Wrote {config.RAIL_VALUES_CSV} ({len(summary)} rows)")
    return summary


def check_consistency(summary):
    print("\n--- Sanity check: rail values per (subject, session, channel), across runs ---")
    print("(should ALWAYS be 1 now — rail is resolved once per session, not per run)")
    per_channel = summary.groupby(['subject_id', 'session_id', 'channel'])['rail_value_v'].nunique()
    n_inconsistent_ch = (per_channel > 1).sum()
    print(f"{n_inconsistent_ch}/{len(per_channel)} subject-session-channels have >1 distinct rail value "
          f"across runs" + (" -- unexpected, investigate!" if n_inconsistent_ch else " -- as expected."))

    print("\n--- Informational: rail values per (subject, session), across all channels ---")
    print("(>1 is expected whenever SAT_AGREEMENT_THRESHOLD wasn't met and some channels")
    print(" fell back to their own per-channel result)")
    per_session = summary.groupby(['subject_id', 'session_id'])['rail_value_v'].nunique()
    print(per_session.to_string())

    print("\n--- rail_source breakdown ---")
    print(summary['rail_source'].value_counts().to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', default=None,
                     help=f'Alternate output root (default: {config.OUTPUT_DIR})')
    args = ap.parse_args()
    if args.output_dir:
        config.set_output_dir(args.output_dir)

    summary = build()
    check_consistency(summary)


if __name__ == '__main__':
    main()

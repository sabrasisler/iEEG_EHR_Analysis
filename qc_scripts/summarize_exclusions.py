#!/usr/bin/env python3
"""
Builds population-level exclusion-rate summaries from the per-window tables
written by run_pipeline.py.

Usage:
  python -m qc_scripts.summarize_exclusions
  python -m qc_scripts.summarize_exclusions --output-dir /path/to/alt/root
"""

import argparse
import glob

import pandas as pd

from qc_scripts import config


def load_per_window(artifact_type, usecols=None):
    """
    usecols: optionally restrict which columns get read (per-window CSVs can
    be very large — e.g. hundreds of MB for a single subject with many runs —
    so callers that only need a couple of columns should say so rather than
    loading everything into memory).
    """
    pattern = str(config.PER_WINDOW_DIR / f'sub-*_{artifact_type}.csv')
    paths = sorted(glob.glob(pattern))
    if not paths:
        return None
    return pd.concat((pd.read_csv(p, usecols=usecols) for p in paths), ignore_index=True)


def summarize(df):
    """One row per (subject_id, channel, artifact_type) -> pct_windows_excluded."""
    grouped = df.groupby(['subject_id', 'channel', 'artifact_type'])['excluded']
    summary = grouped.mean().reset_index()
    summary = summary.rename(columns={'excluded': 'pct_windows_excluded'})
    summary['pct_windows_excluded'] *= 100.0
    return summary


def top_n_and_stats(summary, n=10):
    stats = {
        'mean': summary['pct_windows_excluded'].mean(),
        'median': summary['pct_windows_excluded'].median(),
        'std': summary['pct_windows_excluded'].std(),
        'min': summary['pct_windows_excluded'].min(),
        'max': summary['pct_windows_excluded'].max(),
    }
    top = summary.sort_values('pct_windows_excluded', ascending=False).head(n)
    return stats, top


def flag_for_review(summary, std_thresh):
    mean = summary['pct_windows_excluded'].mean()
    std = summary['pct_windows_excluded'].std()
    cutoff = mean + std_thresh * std
    return summary[summary['pct_windows_excluded'] > cutoff].copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', default=None,
                     help=f'Alternate output root (default: {config.OUTPUT_DIR})')
    args = ap.parse_args()
    if args.output_dir:
        config.set_output_dir(args.output_dir)

    config.ensure_output_dirs()
    all_flagged = []

    for artifact_type in config.ARTIFACT_TYPES:
        df = load_per_window(artifact_type,
                              usecols=['subject_id', 'channel', 'artifact_type', 'excluded'])
        if df is None:
            print(f"No per-window data found for '{artifact_type}', skipping.")
            continue

        summary = summarize(df)
        out_path = config.SUMMARY_DIR / f'exclusion_rates_{artifact_type}.csv'
        config.save_table(summary, out_path)
        print(f"Wrote {out_path} ({len(summary)} rows)")

        stats, top = top_n_and_stats(summary)
        print(f"\n[{artifact_type}] pct_windows_excluded stats: {stats}")
        print(f"[{artifact_type}] top-10 highest:\n{top.to_string(index=False)}\n")

        flagged = flag_for_review(summary, config.FLAG_REVIEW_STD_THRESH)
        if not flagged.empty:
            all_flagged.append(flagged)

    if all_flagged:
        flagged_df = pd.concat(all_flagged, ignore_index=True)
        out_path = config.SUMMARY_DIR / 'flagged_for_review.csv'
        config.save_table(flagged_df, out_path)
        print(f"Wrote {out_path} ({len(flagged_df)} flagged subject/channel/artifact rows)")
    else:
        print("Nothing flagged for review.")


if __name__ == '__main__':
    main()

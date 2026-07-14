#!/usr/bin/env python3
"""
Cheap precompute step for a Slurm array of example plots: picks concrete
(subject, session, run, channel) targets -- some from flagged_for_review.csv,
some fully random -- and writes them to a CSV. No NWB files are touched here
(only streamed per-window CSV scanning, same primitives as
plot_flagged_runs.py's --targets/--random-any), so this step is fast and can
run as a single small job. The actual raw-trace plotting (the slow part, one
NWB read per row) then fans out across an array job, one row per task.

Usage:
  python -m qc_scripts.build_plot_targets --output-dir /path/to/qc_variance_padded30 \
      --n-flagged 20 --n-random 20 --out /path/to/plot_targets.csv --seed 7
"""

import argparse

import pandas as pd

from qc_scripts import config
from qc_scripts.plot_flagged_runs import top_runs_for_channel, find_random_any_examples


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', default=None,
                     help=f'Root to read per-window CSVs / flagged_for_review.csv from '
                          f'(default: {config.OUTPUT_DIR})')
    ap.add_argument('--n-flagged', type=int, default=20,
                     help='Number of rows to sample from flagged_for_review.csv, spread across '
                          'artifact types')
    ap.add_argument('--n-random', type=int, default=20,
                     help='Number of fully random (no exclusion requirement) channel/run combos')
    ap.add_argument('--session', default='01')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--out', required=True, help='Output targets CSV path')
    args = ap.parse_args()

    if args.output_dir:
        config.set_output_dir(args.output_dir)

    available_subjects = sorted({p.stem.split('_')[0].replace('sub-', '')
                                  for p in config.PER_WINDOW_DIR.glob('sub-*_ses-*.csv')})
    print(f"Subjects with per-window data available: {available_subjects}")

    rows = []

    if args.n_flagged:
        review_path = config.SUMMARY_DIR / 'flagged_for_review.csv'
        review = pd.read_csv(review_path)
        per_type = max(1, args.n_flagged // review['artifact_type'].nunique())
        sample = review.groupby('artifact_type', group_keys=False).apply(
            lambda g: g.sample(min(per_type, len(g)), random_state=args.seed), include_groups=False)
        sample = review.loc[sample.index]
        print(f"Sampled {len(sample)} flagged rows across {review['artifact_type'].nunique()} artifact types")

        for _, r in sample.iterrows():
            subject = str(r['subject_id']).replace('sub-', '')
            channel = r['channel']
            artifact_type = r['artifact_type']
            top = top_runs_for_channel(subject, channel, artifact_type, n=1)
            if not top:
                print(f"  No exclusions found for sub-{subject}/{channel}/{artifact_type}, skipping")
                continue
            run, _ = top[0]
            rows.append({'subject': f'sub-{subject}', 'session': f'ses-{args.session}',
                         'run': run, 'channel': channel})

    if args.n_random:
        examples = find_random_any_examples(available_subjects, args.n_random, seed=args.seed)
        for subject, channel, run in examples:
            rows.append({'subject': f'sub-{subject}', 'session': f'ses-{args.session}',
                         'run': run, 'channel': channel})

    targets = pd.DataFrame(rows)
    targets.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(targets)} targets)")


if __name__ == '__main__':
    main()

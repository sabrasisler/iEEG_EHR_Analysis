#!/usr/bin/env python3
"""
Plot raw voltage for specific subject/channel combos, shading the exact
windows already flagged by run_pipeline.py (read from the per-window CSVs —
no re-classification, no full-file loads). Two things this does efficiently:

1. Finding which RUN to plot: per-window CSVs can be huge (one subject's
   saturation.csv can be 900MB+), so we never load one in full. We stream it
   in chunks with only the columns we need (usecols) and filter to the
   requested channel(s) as we go, picking whichever run has the most
   excluded windows for that channel/artifact type.
2. Loading the raw trace: once a target run is known, io_utils.load_channels_subset
   reads only the requested channel's column out of the NWB file, not the
   full channels x samples array.

Usage:
  python -m qc_scripts.plot_flagged_runs --targets 093:LOF10,093:RINS7
  python -m qc_scripts.plot_flagged_runs --find-flatline-examples 2
"""

import argparse
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qc_scripts import config, io_utils

STATUS_COLORS = {
    'saturation': 'crimson',
    'flatline': 'dimgray',
    'gross_artifact': 'orange',
}

OUT_DIR = config.PLOTS_DIR / 'flagged_examples'


def _per_window_path(subject, artifact_type):
    return config.PER_WINDOW_DIR / f'sub-{subject}_{artifact_type}.csv'


def best_run_for_channel(subject, channel, artifact_type, chunksize=500_000):
    """
    Stream a per-window CSV in chunks, filtering to `channel`, without ever
    loading the full (potentially 900MB+) file into memory. Returns
    (run_id, windows_df) for the run with the most excluded windows for this
    channel/artifact type, or (None, None) if the channel has no exclusions
    at all for this artifact type.
    """
    path = _per_window_path(subject, artifact_type)
    if not path.exists():
        return None, None

    usecols = ['channel', 'run_id', 'window_start_time', 'window_end_time', 'excluded']
    matches = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        sub = chunk[chunk['channel'] == channel]
        if len(sub):
            matches.append(sub)
    if not matches:
        return None, None

    df = pd.concat(matches, ignore_index=True)
    counts = df[df['excluded']].groupby('run_id').size()
    if counts.empty:
        return None, None
    best_run = counts.idxmax()
    return best_run, df[df['run_id'] == best_run]


def find_flatline_examples(subjects, n_examples=2, chunksize=500_000):
    """
    Scan flatline per-window CSVs across `subjects` (streamed, channel-count
    only — never materializes a full file) to find (subject, channel, run_id)
    combos with the most flatlined windows, without needing to know which
    channel in advance.
    """
    candidates = []
    for subject in subjects:
        path = _per_window_path(subject, 'flatline')
        if not path.exists():
            continue
        usecols = ['channel', 'run_id', 'excluded']
        counts = {}
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
            flagged = chunk[chunk['excluded']]
            for (ch, run), n in flagged.groupby(['channel', 'run_id']).size().items():
                counts[(ch, run)] = counts.get((ch, run), 0) + n
        for (ch, run), n in counts.items():
            candidates.append((n, subject, ch, run))

    candidates.sort(reverse=True)
    return [(subject, ch, run) for _, subject, ch, run in candidates[:n_examples]]


def find_random_examples(subjects, artifact_types, n_examples, seed=None,
                          max_chunks_per_file=5, chunksize=500_000):
    """
    Sample N random (subject, channel, run, artifact_type) combos that have
    at least one excluded window, without exhaustively scanning every file:
    subjects are visited in random order and each file is only read up to
    `max_chunks_per_file` chunks (typically plenty to find candidates, since
    exclusion rates are rarely vanishingly rare across an entire subject).
    """
    rng = random.Random(seed)
    subjects_shuffled = list(subjects)
    rng.shuffle(subjects_shuffled)

    candidates = []
    for subject in subjects_shuffled:
        artifact_types_shuffled = list(artifact_types)
        rng.shuffle(artifact_types_shuffled)
        for artifact_type in artifact_types_shuffled:
            path = _per_window_path(subject, artifact_type)
            if not path.exists():
                continue
            usecols = ['channel', 'run_id', 'excluded']
            for chunk_i, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize)):
                flagged = chunk[chunk['excluded']]
                if len(flagged):
                    sample_n = min(5, len(flagged))
                    for _, row in flagged.sample(sample_n, random_state=rng.randrange(10**6)).iterrows():
                        candidates.append((subject, row['channel'], row['run_id'], artifact_type))
                if chunk_i + 1 >= max_chunks_per_file:
                    break
        if len(candidates) >= n_examples * 5:
            break  # enough of a pool to sample from without scanning every subject

    rng.shuffle(candidates)
    seen, unique = set(), []
    for c in candidates:
        key = c[:3]  # dedupe by (subject, channel, run) — a run may be flagged by multiple artifact types
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
        if len(unique) >= n_examples:
            break
    return unique


def get_windows(subject, channel, run, artifact_type, chunksize=500_000):
    """Same streaming filter as best_run_for_channel, but for a known run."""
    path = _per_window_path(subject, artifact_type)
    if not path.exists():
        return None
    usecols = ['channel', 'run_id', 'window_start_time', 'window_end_time', 'excluded']
    matches = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        sub = chunk[(chunk['channel'] == channel) & (chunk['run_id'] == run)]
        if len(sub):
            matches.append(sub)
    return pd.concat(matches, ignore_index=True) if matches else None


def plot_channel_run(subject, channel, run, session='01', highlight_types=None):
    """
    Load just this one channel's raw trace for this run, and shade the exact
    windows already flagged for each artifact type in highlight_types
    (default: all three).
    """
    highlight_types = highlight_types or list(STATUS_COLORS.keys())

    nwb_path = io_utils.get_session_runs(subject, session=session)
    nwb_path = next((p for s, r, p in nwb_path if r == run.replace('run-', '')), None)
    if nwb_path is None:
        print(f"  Could not find NWB path for sub-{subject} run-{run}")
        return

    data_v, found_channels, sfreq = io_utils.load_channels_subset(nwb_path, [channel])
    if channel not in found_channels:
        print(f"  Channel {channel} not found in sub-{subject} {run}")
        return
    trace_uv = data_v[:, found_channels.index(channel)] * 1e6
    t = np.arange(len(trace_uv)) / sfreq

    fig, ax = plt.subplots(figsize=(14, 3))
    counts = {}
    for artifact_type in highlight_types:
        windows = get_windows(subject, channel, run, artifact_type)
        if windows is None:
            counts[artifact_type] = 0
            continue
        flagged = windows[windows['excluded']]
        counts[artifact_type] = len(flagged)
        for _, w in flagged.iterrows():
            ax.axvspan(w['window_start_time'], w['window_end_time'],
                       color=STATUS_COLORS[artifact_type], alpha=0.3, linewidth=0, zorder=1)

    ax.plot(t, trace_uv, linewidth=0.5, color='black', zorder=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('µV')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, alpha=0.3, label=f'{a} ({counts.get(a, 0)})')
                       for a, c in STATUS_COLORS.items() if a in highlight_types]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    ax.set_title(f"sub-{subject} ses-{session} {run}  channel={channel}", fontsize=10)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"sub-{subject}_{run}_{channel}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}  (counts: {counts})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--targets', default=None,
                     help='Comma-separated subject:channel pairs, e.g. 093:LOF10,093:RINS7')
    ap.add_argument('--artifact-type', default=None,
                     help='Restrict --targets run-selection to one artifact type '
                          '(default: try saturation, then gross_artifact, then flatline)')
    ap.add_argument('--find-flatline-examples', type=int, default=0,
                     help='Also scan available subjects for N channels/runs with flatlined periods')
    ap.add_argument('--random', type=int, default=0,
                     help='Plot N random subject/channel/run combos with at least one exclusion, '
                          'sampled across all artifact types')
    ap.add_argument('--seed', type=int, default=None,
                     help='Seed for --random selection (default: unseeded/non-reproducible)')
    ap.add_argument('--session', default='01')
    args = ap.parse_args()

    available_subjects = sorted({p.stem.split('_')[0].replace('sub-', '')
                                  for p in config.PER_WINDOW_DIR.glob('sub-*.csv')})
    print(f"Subjects with per-window data available: {available_subjects}")

    if args.targets:
        for pair in args.targets.split(','):
            subject, channel = pair.split(':')
            print(f"sub-{subject} / {channel}:")

            preferred_order = [args.artifact_type] if args.artifact_type else \
                ['saturation', 'gross_artifact', 'flatline']
            run, windows = None, None
            for artifact_type in preferred_order:
                if artifact_type is None:
                    continue
                run, windows = best_run_for_channel(subject, channel, artifact_type)
                if run is not None:
                    print(f"  Picked run-{run} (most exclusions for '{artifact_type}')")
                    break
            if run is None:
                print(f"  No exclusions found for {channel} in any artifact type, skipping.")
                continue

            plot_channel_run(subject, channel, run, session=args.session)

    if args.find_flatline_examples:
        print(f"\nScanning for {args.find_flatline_examples} flatline example(s)...")
        examples = find_flatline_examples(available_subjects, n_examples=args.find_flatline_examples)
        for subject, channel, run in examples:
            print(f"sub-{subject} / {channel} / {run}:")
            plot_channel_run(subject, channel, run, session=args.session)

    if args.random:
        print(f"\nSampling {args.random} random flagged example(s) (seed={args.seed})...")
        examples = find_random_examples(available_subjects, config.ARTIFACT_TYPES, args.random,
                                         seed=args.seed)
        for subject, channel, run, artifact_type in examples:
            print(f"sub-{subject} / {channel} / {run}  (found via '{artifact_type}'):")
            plot_channel_run(subject, channel, run, session=args.session)


if __name__ == '__main__':
    main()

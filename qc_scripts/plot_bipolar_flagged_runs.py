#!/usr/bin/env python3
"""
Plot a bipolar-referenced trace for a specific subject/channel, shading the
60s bins already flagged excluded by build_bipolar_exclusions.py — modeled on
plot_flagged_runs.py's shape (find a run -> load trace -> shade -> save PNG),
with two differences suited to bipolar:

1. Trace source: the bipolar-referenced signal is never persisted by the
   production pipeline (preprocessing/bipolar_reref.py's docstring — it's
   transient, `del`eted right after use). This script instead reads the
   scratch cache written by preprocessing/save_bipolar_sample_traces.py
   (config.bipolar_trace_cache_dir()) — no NWB read here at all, which is the
   whole point of caching: try a new --std-thresh or raw-voltage mask without
   re-reading raw NWB / re-referencing.
2. Shading source: read straight from
   qc/bipolar/exclusions/bipolar_variance/<label>/sub-XXX.csv — that IS
   already the final per-channel-per-60s-bin table (no separate mask-rollup
   step needed, unlike raw_voltage's masks/<label>/).

Supports multiple --labels at once (e.g. std3,std4,std5) so you can visually
compare how a threshold change shifts the shading on the SAME trace panel.

Usage:
  python -m qc_scripts.plot_bipolar_flagged_runs --targets 085:LAMY1-LAMY2 --labels std4
  python -m qc_scripts.plot_bipolar_flagged_runs --targets 085:LAMY1-LAMY2 --labels std3,std4,std5
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from qc_scripts import config

# Same "deep" palette family as plot_flagged_runs.py's STATUS_COLORS, extended
# to however many labels get compared -- cycles if more than 4 are requested.
LABEL_COLORS = ['#4c72b0', '#dd8452', '#c44e52', '#55a868', '#8172b2', '#937860']


def _cache_dir(subject):
    return config.bipolar_trace_cache_dir() / f'sub-{subject}'


def cached_runs(subject):
    """[(session, run, npz_path, sidecar_dict), ...] for every run of this
    subject found in the trace cache."""
    out = []
    sub_dir = _cache_dir(subject)
    if not sub_dir.exists():
        return out
    for sidecar_path in sorted(sub_dir.glob('ses-*/*.json')):
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        npz_path = sidecar_path.with_suffix('.npz')
        if npz_path.exists():
            out.append((sidecar['session'], sidecar['run'], npz_path, sidecar))
    return out


def _exclusion_csv(label, subject):
    return config.BIPOLAR_LEVEL_ROOT / 'exclusions' / 'bipolar_variance' / label / f'sub-{subject}.csv'


def load_exclusion_windows(label, subject, channel, run):
    """Rows from the label's exclusion CSV for this channel/run, or None if
    the file/channel/run isn't present."""
    path = _exclusion_csv(label, subject)
    if not path.exists():
        print(f"  WARNING: no exclusion CSV at {path} for label '{label}'")
        return None
    df = pd.read_csv(path)
    sub = df[(df['channel'] == channel) & (df['run_id'] == run)]
    return sub if len(sub) else None


def all_cached_subjects():
    """Every subject with at least one cached run, from config.bipolar_trace_cache_dir()."""
    root = config.bipolar_trace_cache_dir()
    if not root.exists():
        return []
    return sorted(p.name.replace('sub-', '') for p in root.glob('sub-*') if p.is_dir())


def random_examples(n, seed=None):
    """N random (subject, channel, run) combos drawn from the trace cache, with
    NO requirement of any exclusion -- unlike pick_run's most-excluded search,
    this is for seeing typical/clean bipolar behavior too, not just artifacts
    (mirrors plot_flagged_runs.py's find_random_any_examples)."""
    rng = random.Random(seed)
    candidates = []
    for subject in all_cached_subjects():
        for session, run, npz_path, sidecar in cached_runs(subject):
            for channel in sidecar['channel']:
                candidates.append((subject, channel, run))
    rng.shuffle(candidates)
    return candidates[:n]


def pick_run(subject, channel, labels):
    """Among this subject's cached runs, pick whichever has the most excluded
    bins for `channel` under the FIRST label (used only to pick a run to
    plot -- all requested labels are then shaded on that same run)."""
    runs = cached_runs(subject)
    if not runs:
        return None
    best, best_n = None, -1
    for session, run, npz_path, sidecar in runs:
        windows = load_exclusion_windows(labels[0], subject, channel, run)
        n = int(windows['excluded'].sum()) if windows is not None else 0
        if n > best_n:
            best, best_n = (session, run, npz_path, sidecar), n
    return best


def plot_channel(subject, channel, labels, run=None, output_dir=None):
    if run is not None:
        match = next((r for r in cached_runs(subject) if r[1] == run), None)
        if match is None:
            print(f"  run-{run} not found in cache for sub-{subject}, skipping.")
            return
        session, run, npz_path, sidecar = match
    else:
        picked = pick_run(subject, channel, labels)
        if picked is None:
            print(f"  No cached runs found for sub-{subject}, skipping "
                  f"(run preprocessing/save_bipolar_sample_traces.py first).")
            return
        session, run, npz_path, sidecar = picked

    if channel not in sidecar['channel']:
        print(f"  Channel {channel} not in cached sub-{subject} {run} "
              f"({len(sidecar['channel'])} pairs available), skipping.")
        return
    col = sidecar['channel'].index(channel)
    bipolar_v = np.load(npz_path)['bipolar_v'][:, col]
    sfreq = sidecar['sfreq']
    t = np.arange(len(bipolar_v)) / sfreq
    trace_uv = bipolar_v * 1e6

    fig, ax = plt.subplots(figsize=(14, 3))
    counts = {}
    for i, label in enumerate(labels):
        color = LABEL_COLORS[i % len(LABEL_COLORS)]
        windows = load_exclusion_windows(label, subject, channel, run)
        flagged = windows[windows['excluded']] if windows is not None else None
        counts[label] = len(flagged) if flagged is not None else 0
        if flagged is not None:
            for _, w in flagged.iterrows():
                ax.axvspan(w['bin_start'], w['bin_end'], color=color, alpha=0.25,
                           linewidth=0, zorder=1 + i)

    ax.plot(t, trace_uv, linewidth=0.5, color='black', zorder=10)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('µV (bipolar)')
    legend_elements = [Patch(facecolor=LABEL_COLORS[i % len(LABEL_COLORS)], alpha=0.25,
                              label=f'{label} ({counts[label]} bins)')
                       for i, label in enumerate(labels)]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    ax.set_title(f"sub-{subject} {session} {run}  channel={channel}  (bipolar)", fontsize=10)
    fig.tight_layout()

    out_dir = Path(output_dir) if output_dir else (config.BIPOLAR_LEVEL_ROOT / 'plots' / 'bipolar_flagged_runs')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sub-{subject}_{session}_{run}_{channel}_{'-'.join(labels)}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}  (counts: {counts})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--targets', default=None,
                     help='Comma-separated subject:channel pairs, e.g. 085:LAMY1-LAMY2,085:RINS7-RINS8')
    ap.add_argument('--random', type=int, default=0,
                     help='Plot N random subject/channel/run combos from the trace cache, with NO '
                          'requirement that anything be flagged -- to see typical/clean behavior, '
                          'not just the most-artifact-heavy examples')
    ap.add_argument('--seed', type=int, default=None,
                     help='Seed for --random selection (default: unseeded/non-reproducible)')
    ap.add_argument('--labels', default='std4',
                     help='Comma-separated exclusion labels to overlay, e.g. std3,std4,std5 '
                          '(default: std4). The run to plot is picked using the FIRST label\'s '
                          'exclusion counts; all labels are shaded on that same run.')
    ap.add_argument('--run', default=None,
                     help='Force a specific run-XXXX instead of auto-picking the most-excluded '
                          'cached run for --targets')
    ap.add_argument('--output-dir', default=None,
                     help='Where to write PNGs (default: qc/bipolar/plots/bipolar_flagged_runs)')
    args = ap.parse_args()

    labels = [l.strip() for l in args.labels.split(',')]

    if args.targets:
        for pair in args.targets.split(','):
            subject, channel = pair.split(':')
            print(f"sub-{subject} / {channel}  (labels={labels}):")
            plot_channel(subject, channel, labels, run=args.run, output_dir=args.output_dir)

    if args.random:
        print(f"\nSampling {args.random} random cached example(s) (seed={args.seed})...")
        for subject, channel, run in random_examples(args.random, seed=args.seed):
            print(f"sub-{subject} / {channel} / {run}  (labels={labels}):")
            plot_channel(subject, channel, labels, run=run, output_dir=args.output_dir)


if __name__ == '__main__':
    main()

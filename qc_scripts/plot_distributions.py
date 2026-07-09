#!/usr/bin/env python3
"""
Distributional sanity-check plots of each detector's continuous metric,
computed with a STREAMING histogram (two cheap passes over the 2s metric CSVs in
metrics/per_window/) so it never concatenates the whole cohort into memory —
that all-subjects concat is what OOM'd the previous version at 64/150GB.

Pass 1 finds the value range; pass 2 accumulates counts into fixed bins. Log-y
throughout; log-x for the variance-based metrics (flatline, gross_artifact),
whose values span many orders of magnitude. For flatline/square_wave the config
default threshold is drawn (those thresholds act directly on the plotted metric);
saturation's percent and gross's z-threshold act on transformed quantities so no
line is drawn there.

Usage:
  python -m qc_scripts.plot_distributions --level-root /path/to/qc/raw_voltage [--output-dir DIR]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qc_scripts import config

LOG_X = {'saturation': False, 'flatline': True, 'square_wave': False, 'gross_artifact': True}
THRESHOLD_LINE = {   # only where the threshold acts directly on the plotted metric
    'flatline': config.FLATLINE_VAR_THRESH,
    'square_wave': config.SQUARE_FRAC_THRESH,
}
CHUNK = 500_000
NBINS = 100


def _metric_csvs(level_root, artifact_type):
    return sorted(config.metrics_per_window_dir(level_root).glob(f'sub-*_{artifact_type}.csv'))


def _range(csvs, log_x):
    lo, hi = np.inf, -np.inf
    for csv in csvs:
        for chunk in pd.read_csv(csv, usecols=['metric_value'], chunksize=CHUNK):
            v = chunk['metric_value'].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if log_x:
                v = v[v > 0]
            if v.size:
                lo = min(lo, v.min()); hi = max(hi, v.max())
    return lo, hi


def plot_type(level_root, artifact_type, out_dir):
    csvs = _metric_csvs(level_root, artifact_type)
    if not csvs:
        print(f"No metric CSVs for '{artifact_type}', skipping.")
        return
    log_x = LOG_X[artifact_type]
    lo, hi = _range(csvs, log_x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        print(f"[{artifact_type}] no plottable range ({lo}..{hi}), skipping.")
        return
    bins = (np.logspace(np.log10(lo), np.log10(hi), NBINS) if log_x
            else np.linspace(lo, hi, NBINS))
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    for csv in csvs:
        for chunk in pd.read_csv(csv, usecols=['metric_value'], chunksize=CHUNK):
            v = chunk['metric_value'].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if log_x:
                v = v[v > 0]
            counts += np.histogram(v, bins=bins)[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(bins[:-1], counts, width=np.diff(bins), align='edge', color='steelblue', alpha=0.8)
    ax.set_yscale('log')
    if log_x:
        ax.set_xscale('log')
    thr = THRESHOLD_LINE.get(artifact_type)
    if thr is not None:
        ax.axvline(thr, color='crimson', linestyle='--', linewidth=1.5, label=f'threshold = {thr:.3g}')
        ax.legend()
    ax.set_xlabel(f'{artifact_type} metric_value')
    ax.set_ylabel('Count (pooled subjects/channels/windows), log scale')
    ax.set_title(f'Metric distribution: {artifact_type}')
    fig.tight_layout()
    out_path = Path(out_dir) / f'metric_distribution_{artifact_type}.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--level-root', default=str(config.DEFAULT_LEVEL_ROOT))
    ap.add_argument('--output-dir', default=None,
                     help='Where to write plots (default: <level-root>/metrics/plots)')
    args = ap.parse_args()
    out_dir = args.output_dir or (config.metrics_root(args.level_root) / 'plots')
    for artifact_type in config.ARTIFACT_TYPES:
        plot_type(args.level_root, artifact_type, out_dir)


if __name__ == '__main__':
    main()

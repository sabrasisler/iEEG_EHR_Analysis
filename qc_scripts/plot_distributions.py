#!/usr/bin/env python3
"""
Distributional sanity-check plots: pooled metric_value histograms with the
threshold drawn as a vertical line, one plot per artifact type.

Metric values are z-scored per-subject before pooling (except saturation's
fraction-saturated metric and flatline's variance, both of which are compared
against a single fixed absolute threshold rather than a per-subject baseline
— z-scoring them would put the threshold line at a meaningless position on
the x-axis). All plots use a log-scale y-axis and, for flatline, a log-scale
x-axis too, since the near-zero/clean-window mode otherwise swamps the tail
these plots exist to show.

Usage:
  python -m qc_scripts.plot_distributions
  python -m qc_scripts.plot_distributions --output-dir /path/to/alt/root
"""

import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from qc_scripts import config
from qc_scripts.summarize_exclusions import load_per_window

THRESHOLDS = {
    'saturation': None,        # per-channel inferred rail, no single global threshold to draw
    'flatline': config.FLATLINE_VAR_THRESH,
    'gross_artifact': config.GROSS_STD_THRESH,
}

ZSCORE = {
    'saturation': False,
    'flatline': False,      # fixed absolute threshold, not per-subject-relative
    'gross_artifact': True,
}

LOG_X = {
    'saturation': False,
    'flatline': True,       # variance spans many orders of magnitude
    'gross_artifact': False,
}


def zscore_per_subject(df):
    """
    Z-score df['metric_value'] within each subject_id group. Uses
    groupby().transform() (not apply()) so the result is guaranteed to be a
    flat Series aligned to df's original index.
    """
    grouped = df.groupby('subject_id')['metric_value']
    mean = grouped.transform('mean')
    std = grouped.transform('std').replace(0, np.nan)
    return ((df['metric_value'] - mean) / std).fillna(0.0)


def plot_artifact_type(artifact_type, df):
    df = df[['subject_id', 'metric_value']].copy()
    df['metric_value'] = df['metric_value'].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['metric_value'])

    if ZSCORE[artifact_type]:
        values = zscore_per_subject(df)
        xlabel = f'{artifact_type} metric (z-scored per subject)'
    else:
        values = df['metric_value']
        xlabel = {
            'saturation': f'{artifact_type} metric (fraction of samples saturated)',
            'flatline': f'{artifact_type} metric (variance, V^2, log scale)',
        }.get(artifact_type, f'{artifact_type} metric')
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    log_x = LOG_X[artifact_type]
    if log_x:
        values = values[values > 0]  # log-scale can't show non-positive values
        bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 100)
    else:
        bins = 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins, color='steelblue', alpha=0.8)
    ax.set_yscale('log')
    if log_x:
        ax.set_xscale('log')

    threshold = THRESHOLDS[artifact_type]
    if threshold is not None:
        ax.axvline(threshold, color='crimson', linestyle='--', linewidth=1.5,
                    label=f'threshold = {threshold:.3g}')
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count (pooled across subjects/channels/windows), log scale')
    ax.set_title(f'Metric distribution: {artifact_type}')
    fig.tight_layout()

    out_path = config.PLOTS_DIR / f'metric_distribution_{artifact_type}.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_exclusion_rate_hist(artifact_type, df):
    from qc_scripts.summarize_exclusions import summarize
    summary = summarize(df[['subject_id', 'channel', 'artifact_type', 'excluded']])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(summary['pct_windows_excluded'], bins=50, color='darkorange', alpha=0.8)
    ax.set_yscale('log')
    ax.set_xlabel('% windows excluded (per subject/channel)')
    ax.set_ylabel('Count, log scale')
    ax.set_title(f'Exclusion rate distribution: {artifact_type}')
    fig.tight_layout()

    out_path = config.PLOTS_DIR / f'exclusion_rate_hist_{artifact_type}.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', default=None,
                     help=f'Alternate output root (default: {config.OUTPUT_DIR})')
    args = ap.parse_args()
    if args.output_dir:
        config.set_output_dir(args.output_dir)

    config.ensure_output_dirs()
    needed_cols = ['subject_id', 'channel', 'artifact_type', 'excluded', 'metric_value']
    for artifact_type in config.ARTIFACT_TYPES:
        df = load_per_window(artifact_type, usecols=needed_cols)
        if df is None:
            print(f"No per-window data found for '{artifact_type}', skipping.")
            continue
        plot_artifact_type(artifact_type, df)
        plot_exclusion_rate_hist(artifact_type, df)
        del df


if __name__ == '__main__':
    main()

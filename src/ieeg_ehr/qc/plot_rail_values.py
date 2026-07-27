#!/usr/bin/env python3
"""
Visualize inferred saturation rail values (config.RAIL_VALUES_CSV) across
subjects/sessions/runs/channels. The summary CSV is small (one row per
channel/run, not per window), so this loads it directly rather than
streaming.

Usage:
  python -m ieeg_ehr.qc.plot_rail_values
  python -m ieeg_ehr.qc.plot_rail_values --output-dir /path/to/alt/root
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ieeg_ehr import config

SOURCE_COLORS = {
    'session_agreement': 'steelblue',
    'session_individual': 'mediumpurple',
    'fallback': 'darkorange',
    'none': 'lightgray',
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', default=None,
                     help=f'Alternate output root (default: {config.OUTPUT_DIR})')
    args = ap.parse_args()
    if args.output_dir:
        config.set_output_dir(args.output_dir)

    df = pd.read_csv(config.RAIL_VALUES_CSV)
    df['rail_value_uv'] = df['rail_value_v'] * 1e6
    df['rail_source'] = df['rail_source'].fillna('none')

    subjects = sorted(df['subject_id'].unique())
    x_pos = {s: i for i, s in enumerate(subjects)}
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(max(10, len(subjects) * 0.6), 6))

    for source, color in SOURCE_COLORS.items():
        sub = df[df['rail_source'] == source]
        if sub.empty:
            continue
        jitter = rng.uniform(-0.3, 0.3, size=len(sub))
        xs = sub['subject_id'].map(x_pos).to_numpy() + jitter
        ax.scatter(xs, sub['rail_value_uv'], s=8, alpha=0.25, color=color, label=source,
                   linewidths=0)

    ax.set_yscale('log')
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects, rotation=90, fontsize=7)
    ax.set_xlabel('Subject')
    ax.set_ylabel('Inferred rail value (µV), log scale')
    ax.set_title('Saturation rail value per channel/run, by subject\n'
                  '(each point = one channel in one run; jittered horizontally within subject)')
    ax.legend(title='rail_source', markerscale=2, loc='upper right')
    fig.tight_layout()

    out_path = config.PLOTS_DIR / 'rail_values_by_subject.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    # Companion: per-subject spread summary (min/max/n_unique), sorted by inconsistency.
    spread = df.groupby('subject_id')['rail_value_uv'].agg(['nunique', 'min', 'max', 'std'])
    spread = spread.sort_values('nunique', ascending=False)
    print("\nPer-subject rail-value spread (most inconsistent first):")
    print(spread.to_string())


if __name__ == '__main__':
    main()

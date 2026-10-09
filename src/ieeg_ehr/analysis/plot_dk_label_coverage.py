#!/usr/bin/env python3
"""
How many subjects and how many electrodes carry each Desikan-Killiany label.

    analysis/electrode_localization/dk_coverage/label_counts/<run>_<timestamp>/

Two horizontal bar panels on one shared label axis: subjects per label (left)
and bipolar pairs per label (right). EVERY label string in the cohort's
channel_meta appears, including white matter, ventricles, `Unknown`,
`undefined` and blank labels, because the point is to see the raw atlas
coverage before any ROI scheme decides what to keep.

WHAT IS COUNTED
---------------
One electrode = one BIPOLAR PAIR, labeled by its ANODE (`dk_anode`), which is
how the analysis assigns regions. channel_meta repeats each pair once per run,
so pairs are de-duplicated on (subject, session, channel) before counting.

HEMISPHERES ARE MERGED. Cortical DK labels in this dataset mostly carry no
hemisphere at all (`insula`); subcortical and white-matter labels carry
`Left-`/`Right-`, and a few cortical ones carry `ctx-lh-`/`ctx-rh-`. Stripping
those four prefixes folds `ctx-lh-insula` into `insula` and `Left-Amygdala`
into `Amygdala`. No other spelling is merged: `Unknown` and `undefined` stay
two labels, as they are two strings in the source.

THE COHORT is the reference run's `subjects[]` (reference_run.CONTPAIN_HEATMAP
by default), so the counts describe the subjects the models saw.

Run on Slurm, never the login node:
    sbatch sbatch/dk_label_coverage.sbatch
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import reference_run
from ieeg_ehr.med_analysis import style

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_dk_label_coverage.py'
EVENT = 'electrode_localization'
QUESTION = 'dk_coverage'
OUTPUT_TYPE = 'label_counts'

HEMISPHERE_PREFIXES = ('ctx-lh-', 'ctx-rh-', 'Left-', 'Right-')
BLANK_LABEL = '(blank)'


def collapse_hemisphere(label):
    label = str(label).strip().strip("'\"")
    for prefix in HEMISPHERE_PREFIXES:
        if label.startswith(prefix):
            return label[len(prefix):]
    return label or BLANK_LABEL


def load_pairs(subjects, epoch_minutes=None):
    """One row per (subject, session, bipolar pair), every session on disk."""
    frames, paths = [], []
    for sid in subjects:
        bare = sid.replace('sub-', '')
        meta_dir = config.pain_epoch_channel_meta_path(bare, '01', epoch_minutes).parent
        hits = sorted(meta_dir.glob(f'sub-{bare}_ses-*_channels.parquet'))
        if not hits:
            raise SystemExit(f'{sid}: no channel_meta under {meta_dir}')
        for path in hits:
            meta = io.read_table(path, on_stale='warn')
            meta = meta.drop_duplicates('channel')[['channel', 'dk_anode']].copy()
            meta['subject_id'] = sid
            meta['session'] = path.name.split('_ses-')[1][:2]
            frames.append(meta)
            paths.append(path)
    pairs = pd.concat(frames, ignore_index=True)
    pairs['dk_label'] = pairs['dk_anode'].fillna('').map(collapse_hemisphere)
    return pairs, paths


def label_counts(pairs):
    """Subjects and pairs per label, most subjects first, ties by pairs."""
    counts = (pairs.groupby('dk_label')
              .agg(n_subjects=('subject_id', 'nunique'),
                   n_electrodes=('channel', 'size'))
              .reset_index())
    return counts.sort_values(['n_subjects', 'n_electrodes', 'dk_label'],
                              ascending=[False, False, True], ignore_index=True)


def plot_counts(counts, n_subjects, n_pairs, out_path):
    y = np.arange(len(counts))
    fig, axes = plt.subplots(1, 2, sharey=True,
                             figsize=(10, 0.19 * len(counts) + 1.6))
    panels = [('n_subjects', f'Subjects (of {n_subjects})'),
              ('n_electrodes', f'Bipolar pairs (of {n_pairs})')]
    for ax, (column, xlabel) in zip(axes, panels):
        values = counts[column].to_numpy()
        ax.barh(y, values, height=0.72, color=style.BAR_COLOR, zorder=2)
        for yi, v in zip(y, values):
            ax.text(v, yi, f' {v}', va='center', ha='left',
                    fontsize=style.TICK_SIZE - 2, color=style.TEXT_MUTED)
        ax.set_xlim(0, values.max() * 1.12)
        style.style_axes(ax, grid_axis='x')
        style.label_axes(ax, xlabel=xlabel)
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(counts['dk_label'], fontsize=style.TICK_SIZE - 1,
                            color=style.TEXT_PRIMARY)
    axes[0].set_ylim(len(counts) - 0.5, -0.5)
    axes[0].tick_params(axis='y', length=0)
    fig.suptitle(f'Desikan-Killiany anode labels, hemispheres merged: '
                 f'{len(counts)} labels, {n_subjects} subjects, {n_pairs} bipolar pairs',
                 fontsize=style.TITLE_SIZE, color=style.TEXT_PRIMARY,
                 x=0.01, ha='left')
    fig.tight_layout()
    style.save(fig, out_path, footnote='one electrode = one bipolar pair, by anode label')
    logger.info('Wrote %s', out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference-run', default=str(reference_run.CONTPAIN_HEATMAP))
    ap.add_argument('--epoch-minutes', type=float, default=None)
    ap.add_argument('--run-name', default='contpain_ref')
    args = ap.parse_args()

    io.warn_if_dirty()

    ref = reference_run.load(args.reference_run)
    subjects = sorted(ref.subjects)
    logger.info('cohort from %s: %d subject(s)', ref.run_dir.name, len(subjects))

    pairs, meta_paths = load_pairs(subjects, args.epoch_minutes)
    counts = label_counts(pairs)
    n_subjects, n_pairs = pairs['subject_id'].nunique(), len(pairs)
    logger.info('%d label(s) over %d pair(s), %d subject(s):\n%s',
                len(counts), n_pairs, n_subjects, counts.to_string())

    run_dir = config.analysis_run_dir(question=QUESTION, output_type=OUTPUT_TYPE,
                                      run_name=args.run_name, event=EVENT)
    parents = ([io.parent_ref(ref.run_dir / 'provenance.json', digest=False)]
               + [io.parent_ref(p, digest=False) for p in meta_paths])
    params = {'reference_run': str(ref.run_dir),
              'epoch_minutes': args.epoch_minutes,
              'label_column': 'dk_anode',
              'hemisphere_prefixes_stripped': list(HEMISPHERE_PREFIXES)}

    io.write_table(counts, run_dir / 'dk_label_counts.csv', script=SCRIPT,
                   params=params, parents=parents, subjects=subjects)
    io.write_run_provenance(run_dir, script=SCRIPT, params={**vars(args), **params},
                            parents=parents, subjects=subjects,
                            extra={'n_labels': int(len(counts)),
                                   'n_electrodes': int(n_pairs),
                                   'n_subjects': int(n_subjects)})
    plot_counts(counts, n_subjects, n_pairs, run_dir / 'dk_label_counts.png')

    io.log_analysis(f'DK label coverage (anode, hemispheres merged), {len(counts)} '
                    f'labels, {n_pairs} pairs, n={n_subjects}', run_dir)
    logger.info('figure + table + provenance -> %s', run_dir)


if __name__ == '__main__':
    main()

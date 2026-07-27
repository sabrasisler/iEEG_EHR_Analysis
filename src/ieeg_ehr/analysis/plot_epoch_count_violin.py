"""
Violin plot of per-subject pain-event counts, one violin per pain bin, with
each subject plotted as its own colored point (same color across all
violins) -- a sense-check for how much epoch count varies subject-to-subject
and bin-to-bin, independent of any region grouping (counts here are raw pain
events per subject/bin, before any channel/region mapping).

Supports two pain-bin schemes (--pain-bin-scheme), recomputed at plot time
from the cache's raw `pain_score` column -- no need to re-run
build_pain_epoch_power.py to switch:
- absolute (default): fixed cutpoints shared across all subjects
  (config.PAIN_BIN_EDGES) -- none/low/medium/high.
- subject_relative: 'none' still means score == 0, but 'low'/'high' splits
  at that SAME subject's own mean pain score among their non-zero events --
  no 'medium' bin (see common.assign_relative_pain_bins).

Run on a dev/interactive Slurm shell (never the login node):
    module load python/3.12
    source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate
    python -m ieeg_ehr.analysis.plot_epoch_count_violin
"""

import argparse
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from ieeg_ehr import config
from ieeg_ehr.features import common

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def epoch_counts_by_subject_bin(df, bin_order):
    """(subject, pain_bin) -> number of distinct pain events -- raw counts,
    not filtered to region-mapped channels (a subject with zero ROI-mapped
    channels should still show its true event count here)."""
    counts = (df.drop_duplicates(['subject', 'pain_bin', 'pain_event_id'])
                .groupby(['subject', 'pain_bin']).size().rename('n_epochs').reset_index())
    subjects = sorted(counts['subject'].unique())
    full_index = pd.MultiIndex.from_product([subjects, bin_order], names=['subject', 'pain_bin'])
    return counts.set_index(['subject', 'pain_bin']).reindex(full_index, fill_value=0)['n_epochs'].reset_index()


def plot_violin(counts, bin_order, out_path):
    subjects = sorted(counts['subject'].unique())
    subject_color = common.subject_color_map(subjects)

    fig, ax = plt.subplots(figsize=(8, 6))
    common.draw_violin_with_subject_dots(
        ax, counts.rename(columns={'n_epochs': 'value'}), subject_color, value_col='value', pain_bins=bin_order,
    )

    ax.set_xlabel('Pain bin')
    ax.set_ylabel('Number of epochs (pain events)')
    ax.set_title(f'Epoch count per subject/pain bin (n={len(subjects)} subjects)')
    handles = [plt.Line2D([0], [0], marker='o', linestyle='', markeredgecolor='black',
                          markeredgewidth=0.3, color=c, label=s) for s, c in subject_color.items()]
    ax.legend(handles=handles, title='Subject', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, ncol=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('Wrote %s', out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subjects', nargs='+', default=None,
                         help='Subject IDs without sub- prefix (default: all present in cache dir).')
    parser.add_argument('--run-name', default=None,
                         help='Name for this run\'s output subdirectory under '
                              f'{config.PLOTS_ROOT} (default: auto-generated from timestamp + subject count).')
    parser.add_argument('--pain-bin-scheme', choices=['absolute', 'subject_relative'], default='absolute',
                         help='"absolute" = fixed cutpoints shared across subjects (none/low/medium/high). '
                              '"subject_relative" = none is score==0, low/high split at that subject\'s own '
                              'mean non-zero pain score (no medium). (default: %(default)s)')
    args = parser.parse_args()

    df, cache_paths = common.load_cache(args.subjects)
    if args.pain_bin_scheme == 'subject_relative':
        df['pain_bin'] = common.assign_relative_pain_bins(df)

    bin_order = config.pain_bin_order(args.pain_bin_scheme)
    counts = epoch_counts_by_subject_bin(df, bin_order)
    logger.info('Epoch counts by subject/bin (%s scheme):\n%s', args.pain_bin_scheme,
                counts.pivot(index='subject', columns='pain_bin', values='n_epochs'))

    n_subjects = counts['subject'].nunique()
    run_dir = common.make_run_dir(args.run_name, n_subjects, category=f'epoch_count_violin/{args.pain_bin_scheme}')
    common.write_run_provenance(
        run_dir, 'ieeg_ehr/analysis/plot_epoch_count_violin.py', args, cache_paths,
        subjects=df['subject'].unique().tolist(),
        extra={'pain_bin_scheme': args.pain_bin_scheme},
    )

    plot_violin(counts, bin_order, run_dir / 'epoch_count_violin.png')

    logger.info('Run outputs + provenance.json written to %s', run_dir)


if __name__ == '__main__':
    main()

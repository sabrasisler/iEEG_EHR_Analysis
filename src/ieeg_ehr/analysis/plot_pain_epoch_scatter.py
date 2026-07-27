"""
Power-spectrum view of regional log-power vs. frequency, one subplot per ROI
region: x = frequency bin, y = mean log-power (averaged across channels
within region, per the usual within-subject weighting). Per pain_bin
(none/low/medium/high): a solid mean line with a shaded +/-1 SEM (standard
error of the mean) error band around it -- how precisely that mean is
pinned down -- instead of individual scatter dots (too much overlap to read
high-vs-low separation with 15+ subjects' worth of points on one axis).

Two figures per run:
- Group: mean/SEM computed across SUBJECTS at each freq_bin (one value per
  subject going into the band) -- shows how tightly the group mean is
  pinned down across subjects.
- Per-subject (by_subject/): mean/SEM computed across that ONE subject's own
  individual EPOCHS -- shows how tightly that subject's own mean is pinned
  down across their epochs.

This plots raw regional power (not delta-from-baseline like
plot_pain_heatmaps.py) specifically to let the 1/f-like spectral shape --
and how it shifts with pain bin -- be inspected directly.

Run on a dev/interactive Slurm shell (never the login node):
    module load python/3.12
    source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate
    python -m ieeg_ehr.analysis.plot_pain_epoch_scatter
"""

import argparse
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ieeg_ehr import config
from ieeg_ehr.features import common

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

PAIN_BIN_STYLE = {
    'none': {'color': 'tab:green', 'marker': 'o'},
    'low': {'color': 'tab:blue', 'marker': 's'},
    'medium': {'color': 'tab:orange', 'marker': '^'},
    'high': {'color': 'tab:red', 'marker': 'D'},
}


def mean_sem_table(samples):
    """(pain_bin, region, freq_bin_index) -> (mean, sem) of mean_log_power
    across whatever rows `samples` has per group -- one row per subject for
    the group figure, one row per epoch for a per-subject figure. ddof=1
    (sample SEM); NaN sem when a group has only 1 sample (undefined)."""
    grouped = samples.groupby(['pain_bin', 'region', 'freq_bin_index'])['mean_log_power']
    stats = grouped.agg(mean='mean', std='std', n='count').reset_index()
    stats['sem'] = stats['std'] / np.sqrt(stats['n'])
    return stats


def plot_region_scatter(samples, bin_labels, title, out_path, sample_label, n_cols=5):
    """samples: table with (pain_bin, region, freq_bin_index, mean_log_power)
    columns -- one row per underlying sample (a subject's region mean, for
    the group figure; a single epoch, for a per-subject figure). For each
    region/pain_bin, plots the across-sample mean line with a shaded +/-1
    SEM band computed from those same samples."""
    regions = [r for r in config.ROI_REGIONS if r in samples['region'].unique()]
    if not regions:
        logger.warning('No ROI regions present in data for %s, skipping', title)
        return

    freq_bins = bin_labels.index.tolist()
    x_hz = bin_labels['bin_low_hz'].to_numpy()
    pain_bins = [b for b in config.PAIN_BIN_ORDER if b in samples['pain_bin'].unique()]
    stats = mean_sem_table(samples)

    n_rows = int(np.ceil(len(regions) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.0 * n_rows),
                              sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.ravel()

    for ax, region in zip(axes_flat, regions):
        region_stats = stats[stats['region'] == region]
        for pain_bin in pain_bins:
            style = PAIN_BIN_STYLE[pain_bin]
            line = region_stats[region_stats['pain_bin'] == pain_bin].set_index('freq_bin_index')
            line = line.reindex(freq_bins)
            ax.plot(x_hz, line['mean'], color=style['color'], linewidth=2, label=pain_bin)
            ax.fill_between(x_hz, line['mean'] - line['sem'], line['mean'] + line['sem'],
                            color=style['color'], alpha=0.2, linewidth=0)

        ax.set_xscale('log')
        ax.set_title(region, fontsize=9)
        common.add_band_boundary_lines(ax)

    for ax in axes_flat[len(regions):]:
        ax.axis('off')

    for row in range(n_rows):
        axes[row, 0].set_ylabel('Mean log10(V^2/Hz)')
    for col in range(n_cols):
        axes[-1, col].set_xlabel('Freq bin low edge (Hz)')

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(pain_bins),
               bbox_to_anchor=(0.5, 1.02),
               title=f'Pain bin (line=mean across {sample_label}, band=+/-1 SEM)')
    fig.suptitle(title, y=1.06)

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
    parser.add_argument('--n-cols', type=int, default=5, help='Subplot grid columns (one subplot per region).')
    args = parser.parse_args()

    df, cache_paths = common.load_cache(args.subjects)
    df = common.add_region(df)
    bin_labels = common.bin_label_table(df)

    region_mean = common.subject_region_table(df)
    epoch_table = common.subject_region_epoch_table(df)

    run_dir = common.make_run_dir(args.run_name, region_mean['subject'].nunique(), category='epoch_scatter')
    common.write_run_provenance(
        run_dir, 'ieeg_ehr/analysis/plot_pain_epoch_scatter.py', args, cache_paths,
        subjects=df['subject'].unique().tolist(),
    )

    plot_region_scatter(
        region_mean, bin_labels,
        f"Group (n={region_mean['subject'].nunique()} subjects)",
        run_dir / 'group_epoch_scatter.png', sample_label='subjects', n_cols=args.n_cols,
    )

    for subject, sub_epochs in epoch_table.groupby('subject'):
        plot_region_scatter(
            sub_epochs, bin_labels, f'sub-{subject}',
            run_dir / 'by_subject' / f'sub-{subject}_epoch_scatter.png',
            sample_label='epochs', n_cols=args.n_cols,
        )

    logger.info('Run outputs + provenance.json written to %s', run_dir)


if __name__ == '__main__':
    main()

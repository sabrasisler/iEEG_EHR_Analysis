"""Do the medication estimates agree across specifications?

Four models now estimate a medication effect on the same cells, and they disagree
wildly about significance while agreeing closely about magnitude:

    interaction main effect   444/722 BH-significant   |beta| median 0.0137
    matched-NRS               24/722                   |beta| median 0.0150

Same quantity, correlation 0.80, four-fold difference in how many cells clear
correction. The reason is the pain adjustment: the interaction model adjusts for
pain LINEARLY, which is efficient but assumes a shape we measured and found to be
wrong; the matched model adjusts nonparametrically, which is assumption-free but
discards most comparisons. Neither is simply right.

That disagreement is the most important thing to look at, and no map shows it,
because a map shows one estimate at a time. This figure puts them against each
other so it is visible WHERE they diverge -- if divergence concentrates in
particular regions or frequencies, that localises the problem; if it is uniform
scatter, it is precision rather than bias.

    python -m ieeg_ehr.analysis.plot_med_specification_agreement --run-dir <run>
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import io

logger = logging.getLogger(__name__)

DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. '
              'Parametric Wald p. Not confirmed out of sample.')


def load_estimates(run_dir):
    """One row per cell with every medication estimate available for it."""
    run_dir = Path(run_dir)
    ix = io.read_table(run_dir / 'interaction' / 'grid_cells.parquet',
                       on_stale='warn')
    mt = io.read_table(run_dir / 'matched' / 'grid_cells.parquet', on_stale='warn')
    k = ['region', 'freq_bin_index', 'freq_bin_low']
    cols_ix = k + ['med_main_beta', 'med_main_se', 'med_main_bh_reject',
                   'med_ix_beta', 'med_ix_bh_reject', 'n_subjects']
    cols_mt = ['region', 'freq_bin_index', 'beta_med', 'se', 'p_bh_reject',
               'frac_sign_consistent', 'n_subjects_matched']
    df = ix[[c for c in cols_ix if c in ix.columns]].merge(
        mt[[c for c in cols_mt if c in mt.columns]].rename(
            columns={'se': 'matched_se', 'p_bh_reject': 'matched_bh_reject'}),
        on=['region', 'freq_bin_index'], how='inner')
    return df.dropna(subset=['med_main_beta', 'beta_med'])


def build(df, out_path, title_suffix=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))

    # -- 1. the two estimates against each other --------------------------
    ax = axes[0]
    lim = float(np.nanmax(np.abs(np.concatenate(
        [df['med_main_beta'].to_numpy(), df['beta_med'].to_numpy()])))) * 1.05
    both = (df['med_main_bh_reject'] == True) & (df['matched_bh_reject'] == True)  # noqa: E712
    only_main = (df['med_main_bh_reject'] == True) & ~both                          # noqa: E712
    neither = ~(df['med_main_bh_reject'] == True) & ~(df['matched_bh_reject'] == True)  # noqa: E712
    for sel, c, lab in ((neither, '0.75', 'neither significant'),
                        (only_main, '#4a7ba7', 'only the linear-adjusted model'),
                        (both, '#c1442f', 'both significant')):
        ax.scatter(df.loc[sel, 'med_main_beta'], df.loc[sel, 'beta_med'],
                   s=10, c=c, alpha=0.75, linewidths=0, label=f'{lab} ({int(sel.sum())})')
    ax.plot([-lim, lim], [-lim, lim], ls='--', color='0.4', lw=1, zorder=0,
            label='identity')
    ax.axhline(0, color='0.85', lw=0.8, zorder=0)
    ax.axvline(0, color='0.85', lw=0.8, zorder=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    r = float(np.corrcoef(df['med_main_beta'], df['beta_med'])[0, 1])
    ax.set_title(f'the same effect, two pain adjustments\nr = {r:.3f}', fontsize=10)
    ax.set_xlabel('interaction model main effect\n(pain adjusted LINEARLY)', fontsize=9)
    ax.set_ylabel('matched-NRS model\n(pain adjusted as a FACTOR)', fontsize=9)
    ax.legend(fontsize=7, loc='upper left')

    # -- 2. precision, which is where they actually differ ----------------
    ax = axes[1]
    ax.scatter(df['med_main_se'], df['matched_se'], s=10, c='0.45', alpha=0.7,
               linewidths=0)
    hi = float(np.nanmax(np.concatenate(
        [df['med_main_se'].to_numpy(), df['matched_se'].to_numpy()]))) * 1.05
    ax.plot([0, hi], [0, hi], ls='--', color='0.4', lw=1)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect('equal')
    ratio = float(np.nanmedian(df['matched_se'] / df['med_main_se']))
    ax.set_title('standard errors\nmatched SE is %.1fx larger (median)' % ratio,
                 fontsize=10)
    ax.set_xlabel('SE, linear adjustment', fontsize=9)
    ax.set_ylabel('SE, factor adjustment', fontsize=9)

    # -- 3. where the disagreement lives ----------------------------------
    ax = axes[2]
    df = df.copy()
    df['disagree'] = (df['med_main_beta'] - df['beta_med']).abs()
    by_region = (df.groupby('region')['disagree'].median()
                 .sort_values(ascending=False))
    ax.barh(range(len(by_region)), by_region.to_numpy(), color='#4a7ba7')
    ax.set_yticks(range(len(by_region)))
    ax.set_yticklabels(by_region.index, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('median |difference| between the two estimates', fontsize=9)
    ax.set_title('where the specifications disagree', fontsize=10)
    ax.tick_params(labelsize=7)

    fig.suptitle('Medication effect: does it survive changing how pain is '
                 'adjusted for?' + title_suffix, fontsize=13)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    fig.text(0.01, 0.01,
             'Both panels estimate the SAME quantity -- the power difference '
             'between medicated and unmedicated epochs at matched pain -- and '
             'differ only in how pain is adjusted for. Points near the identity '
             'line mean the estimate does not depend on that choice. The '
             'significance difference is a PRECISION difference, not a '
             'disagreement about the effect: the factor adjustment conditions on '
             'an 11-level pain variable and keeps only within-level contrasts, so '
             'it is assumption-free and imprecise, while the linear adjustment is '
             'efficient and rests on a shape the data do not support. Blue points '
             'are cells where only the efficient-but-assuming model calls '
             'significance -- those are the ones to distrust.\n' + DISCLAIMER,
             fontsize=7, va='bottom', ha='left', color='0.35', wrap=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return {'r': r, 'se_ratio': ratio, 'n_cells': int(len(df))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True,
                    help='A med-strata run containing interaction/ and matched/.')
    ap.add_argument('--out-name', default='fig_specification_agreement.png')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    run_dir = Path(args.run_dir)
    df = load_estimates(run_dir)
    stats = build(df, run_dir / args.out_name)
    logger.info('%d cells | r=%.3f | matched SE %.1fx larger',
                stats['n_cells'], stats['r'], stats['se_ratio'])
    logger.info('wrote %s', run_dir / args.out_name)
    io.log_analysis('medication effect: agreement between linear and factor pain '
                    'adjustment (EXPLORATORY)', run_dir)


if __name__ == '__main__':
    main()

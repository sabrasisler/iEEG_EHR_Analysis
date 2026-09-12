"""Nomination 1: does the broadband-positive medication effect survive differencing?

THE CLAIM IS ABOUT A DISTRIBUTION, NOT A CELL, so the figure is a distribution
and not a heatmap. The 2-hour lookback returned a medication main effect that was
positive across nearly every region and frequency -- the signature of a
MULTIPLICATIVE GAIN on the signal (impedance, reference, amplitude), which is not
band-limited and therefore not plausibly physiology. The paired change-score
design exists to test exactly that, because differencing cancels any drift slow
enough to be common to both epochs of a pair.

If the level model's betas sit almost entirely above zero and the change model's
straddle it, that is the whole argument for the design, and it belongs on one
axis where a reader can see both at once rather than as two numbers in prose.

The level-model comparator is `medw_beta` from the `decomposed/` fit -- the
WITHIN-subject medication effect. That is the right analogue: `med_between` is
also a within-subject contrast (this subject's dosed pairs against their own
undosed ones), so pairing it with the between-subject term would compare two
different estimands and flatter the design unfairly.

    python -m ieeg_ehr.analysis.plot_change_score_med_distribution
"""

import argparse
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis.plot_change_score_diagnostics import (COLOR_BAD, COLOR_OK,
                                                             INK, INK_MUTED,
                                                             arm_label, load_arms)

logger = logging.getLogger(__name__)

LEVEL_RUNS = {
    'analgesics': ('med_strata_analgesics', 'medstrata_20260904-162541'),
    'opioids': ('med_strata_opioids', 'medstrata_20260904-162613'),
}

DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. '
              'Distributions are over CELLS, which are correlated; this figure '
              'is a shape comparison, not a test.')


def load_level(base_univariate, key):
    """The level model's WITHIN-subject medication beta, converged cells only."""
    if key not in LEVEL_RUNS:
        return None
    scheme, run = LEVEL_RUNS[key]
    path = Path(base_univariate) / scheme / run / 'decomposed' / 'grid_cells.parquet'
    if not path.exists():
        logger.warning('no level comparator at %s', path)
        return None
    d = pd.read_parquet(path)
    if 'converged' in d.columns:
        d = d[d['converged'].fillna(False)]
    return d['medw_beta'].dropna().to_numpy() if 'medw_beta' in d.columns else None


def fig_distribution(arms, base_univariate, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(arms)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 4.8), squeeze=False)

    for i, (label, _rd, cells) in enumerate(arms):
        ax = axes[0][i]
        ok = cells[cells['converged'].fillna(False)]
        change = ok['med_beta'].dropna().to_numpy()
        key = label.split(' ')[0]
        level = load_level(base_univariate,
                           'analgesics' if key == 'analgesics'
                           else ('opioids' if key == 'opioids' else None))

        lo, hi = np.percentile(change, [0.5, 99.5])
        if level is not None and len(level):
            lo = min(lo, np.percentile(level, 0.5))
            hi = max(hi, np.percentile(level, 99.5))
        span = max(abs(lo), abs(hi))
        edges = np.linspace(-span, span, 61)

        if level is not None and len(level):
            ax.hist(level, bins=edges, density=True, color=COLOR_BAD, alpha=0.55,
                    label=f'LEVEL model (lookback), {100 * (level > 0).mean():.0f}% positive')
        ax.hist(change, bins=edges, density=True, color=COLOR_OK, alpha=0.75,
                label=f'CHANGE score, {100 * (change > 0).mean():.0f}% positive')

        ax.axvline(0, color=INK, lw=1.4)
        ax.set_title(f'{label}\nn={len(change)} converged cells', fontsize=10,
                     color=INK)
        ax.set_xlabel('medication effect  (d log10 power)', fontsize=9, color=INK)
        if i == 0:
            ax.set_ylabel('density', fontsize=9, color=INK)
        ax.tick_params(labelsize=8, colors=INK_MUTED)
        ax.grid(axis='y', color='#EEEEEE', lw=0.6)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color('#CCCCCC')
        ax.legend(fontsize=8, frameon=False, loc='upper left')

    fig.suptitle('Does the broadband-positive medication effect survive '
                 'differencing?\nA level model massed above zero beside a change '
                 'model straddling it is the argument for the paired design.',
                 fontsize=12, color=INK)
    fig.text(0.5, 0.005, DISCLAIMER, ha='center', fontsize=7, color=INK_MUTED)
    fig.tight_layout(rect=[0, 0.03, 1, 0.88])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def summary_table(arms, base_univariate):
    rows = []
    for label, _rd, cells in arms:
        ok = cells[cells['converged'].fillna(False)]
        change = ok['med_beta'].dropna().to_numpy()
        key = label.split(' ')[0]
        level = load_level(base_univariate, key if key in LEVEL_RUNS else None)
        rec = {'arm': label, 'n_cells_change': len(change),
               'change_median': float(np.median(change)),
               'change_frac_positive': float((change > 0).mean()),
               'change_iqr': float(np.subtract(*np.percentile(change, [75, 25])))}
        if level is not None and len(level):
            rec.update(n_cells_level=len(level),
                       level_median=float(np.median(level)),
                       level_frac_positive=float((level > 0).mean()))
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', default=None)
    ap.add_argument('--run-name', default='changescore_med_distribution')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    base = args.base or str(config.analysis_run_dir(
        question='psd_physiology', output_type='univariate_analysis',
        view_scheme='med_change_score', run_name='').parent)
    base_univariate = str(Path(base).parent)

    arms = load_arms(base)
    if not arms:
        raise SystemExit(f'no grid_cells.parquet under {base}')

    out_dir = config.analysis_run_dir(
        question='psd_physiology', output_type='univariate_analysis',
        view_scheme='med_change_score', run_name=args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = summary_table(arms, base_univariate)
    logger.info('\n%s', table.to_string(index=False))
    io.write_table(table, out_dir / 'med_distribution_summary.csv',
                   params={'level_runs': {k: list(v) for k, v in LEVEL_RUNS.items()},
                           'level_column': 'medw_beta (within-subject)'},
                   parents=[str(a[1] / 'grid_cells.parquet') for a in arms],
                   script='ieeg_ehr/analysis/plot_change_score_med_distribution.py')

    fig_distribution(arms, base_univariate, out_dir / 'fig_med_distribution.png')
    logger.info('wrote %s', out_dir / 'fig_med_distribution.png')

    io.write_run_provenance(
        out_dir, script='ieeg_ehr/analysis/plot_change_score_med_distribution.py',
        params={'arms': [a[0] for a in arms]},
        parents=[str(a[1] / 'provenance.json') for a in arms],
        extra={'status': 'EXPLORATORY, NOT a finding',
               'caveat': 'Cells are CORRELATED -- adjacent log-spaced bins come '
                         'from one FFT and regions share subjects and channels. '
                         'This figure compares the SHAPE of two distributions; '
                         'it is not a test, and the cell count is not a sample '
                         'size.'})
    io.log_analysis('change-score vs level medication effect distributions '
                    '(EXPLORATORY)', out_dir)
    print(out_dir)


if __name__ == '__main__':
    main()

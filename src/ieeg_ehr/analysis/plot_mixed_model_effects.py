"""Fixed effect vs random effects, per pilot cell: a caterpillar plot.

`fig_pilot_subject_lines` shows each subject's RAW fit. This one shows what the
MODEL believes about each subject, which is a different object and the one the
terms "fixed" and "random effect" actually refer to.

    model's slope for subject i  =  beta        +   b_i
                                    ^fixed          ^random deviation (a BLUP)

Per cell, subjects are sorted by that model slope and drawn twice: an open marker
at their own unpooled OLS slope with its 95% CI, and a filled marker at the
model's estimate. The line joining them is SHRINKAGE -- how far partial pooling
moved that subject toward the group. Subjects with few reports or noisy data move
furthest, which is the whole point of fitting a mixed model instead of averaging
per-subject slopes.

THREE DIFFERENT SIGNIFICANCE QUESTIONS, which this figure keeps visually apart:

1. "Is there a group effect?" -- the black vertical line and its band. This is the
   fixed-effect Wald test, the number the pilot is really about.
2. "Do subjects differ from one another?" -- NOT visible as any one subject's
   interval; it is the LRT on var(subj_slope), a single test per cell, reported in
   the panel annotation. The visual proxy is the SPREAD of filled markers.
3. "Does THIS subject respond?" -- whether an open marker's CI clears zero. This
   is the weakest of the three and is annotated as a count, not celebrated: it is
   ~30-50 uncorrected tests per cell on 10-90 epochs each, so at alpha=0.05 a
   handful will clear zero in a cell with no effect at all. Read it as texture,
   never as a per-patient finding.

A BLUP is a PREDICTION, not a fitted parameter, so the filled markers deliberately
carry no interval. Putting a confidence interval on one invites reading it as a
per-subject test, which is question 3 -- and question 3 is better answered by the
subject's own unpooled fit, already drawn beside it.

    python -m ieeg_ehr.analysis.plot_mixed_model_effects --run-dir <run>
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import io
from ieeg_ehr.analysis.plot_mixed_model_subject_lines import (
    DISCLAIMER, NEG_COLOR, POS_COLOR, epoch_level, subject_slopes)

logger = logging.getLogger(__name__)

RAW_COLOR = '0.45'


def cell_effects(run_dir, cell, blups):
    """Join a cell's unpooled per-subject fits to the model's BLUP estimates."""
    frame_path = Path(run_dir) / 'frames' / f"cell_{int(cell['cell_index']):03d}.parquet"
    if not frame_path.exists():
        return None
    raw = subject_slopes(epoch_level(io.read_table(frame_path, on_stale='ignore')))
    mine = blups[(blups['region'] == cell['region'])
                 & (blups['freq_bin'] == int(cell['freq_bin_index']))]
    merged = raw.merge(mine[['subject', 'blup_slope', 'subject_slope', 'n_reports',
                             'n_channels']],
                       on='subject', how='inner')
    return merged.sort_values('subject_slope').reset_index(drop=True)


def draw_panel(ax, eff, cell):
    beta = float(cell['beta_nrs_within'])
    se = float(cell['se'])
    y = np.arange(len(eff))

    # The fixed effect and its interval, behind everything else.
    ax.axvspan(beta - 1.96 * se, beta + 1.96 * se, color='0.25', alpha=0.15, lw=0,
               zorder=0)
    ax.axvline(beta, color='black', lw=1.8, zorder=1)
    ax.axvline(0, color='0.75', lw=0.8, ls='--', zorder=0)

    # Each subject's own evidence: unpooled slope, 95% CI.
    has_raw = eff['slope'].notna() & eff['se'].notna()
    ax.hlines(y[has_raw], (eff['slope'] - 1.96 * eff['se'])[has_raw],
              (eff['slope'] + 1.96 * eff['se'])[has_raw],
              color=RAW_COLOR, lw=0.6, alpha=0.55, zorder=2)
    ax.scatter(eff['slope'][has_raw], y[has_raw], s=9, facecolors='none',
               edgecolors=RAW_COLOR, linewidths=0.6, zorder=3)

    # Shrinkage, then the model's estimate on top.
    ax.hlines(y[has_raw], eff['slope'][has_raw], eff['subject_slope'][has_raw],
              color='0.7', lw=0.5, alpha=0.6, zorder=2)
    colors = [POS_COLOR if s > 0 else NEG_COLOR for s in eff['subject_slope']]
    ax.scatter(eff['subject_slope'], y, s=13, c=colors, zorder=4, linewidths=0)

    # Robust x-limits: one subject with 3 epochs can have a CI wider than the
    # entire scientific range of interest and would flatten the panel.
    span = np.nanpercentile(
        np.abs(np.concatenate([
            (eff['slope'] + 1.96 * eff['se']).to_numpy(dtype='float64'),
            (eff['slope'] - 1.96 * eff['se']).to_numpy(dtype='float64'),
            eff['subject_slope'].to_numpy(dtype='float64')])), 92)
    span = float(span) if np.isfinite(span) and span > 0 else 0.05
    ax.set_xlim(-span, span)
    ax.set_ylim(-1, len(eff))
    ax.set_yticks([])

    n_sig = int((eff['p'] < 0.05).sum())
    n_raw = int(has_raw.sum())
    ax.set_title(f"{cell['region']}  {cell['freq_bin_low']:.1f}-"
                 f"{cell['freq_bin_high']:.1f} Hz\n{cell['group']}", fontsize=8.5)
    ax.text(0.03, 0.97,
            f"group beta {beta:+.4f} (p {float(cell['p']):.2g})\n"
            f"heterogeneity LRT p {float(cell['p_lrt_mixture']):.1g}\n"
            f"{n_sig}/{n_raw} subjects' own CI excludes 0",
            transform=ax.transAxes, fontsize=6.2, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.26', fc='white', ec='0.8', alpha=0.88))
    ax.tick_params(labelsize=7)
    return {'region': cell['region'], 'freq_bin_index': int(cell['freq_bin_index']),
            'n_subjects': n_raw, 'n_subject_sig': n_sig,
            'frac_subject_sig': n_sig / n_raw if n_raw else np.nan,
            'median_abs_shrinkage': float(
                np.nanmedian(np.abs(eff['slope'] - eff['subject_slope'])))}


def build(run_dir, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    run_dir = Path(run_dir)
    cells = io.read_table(run_dir / 'pilot_cells.parquet', on_stale='warn')
    cells = cells.sort_values('cell_index').reset_index(drop=True)
    blups = io.read_table(run_dir / 'pilot_blups.parquet', on_stale='warn')

    ncol = 5
    nrow = int(np.ceil(len(cells) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.6 * nrow),
                             squeeze=False)

    summaries = []
    for i, cell in cells.iterrows():
        ax = axes[i // ncol][i % ncol]
        eff = cell_effects(run_dir, cell, blups)
        if eff is None or eff.empty:
            ax.set_visible(False)
            continue
        summaries.append(draw_panel(ax, eff, cell))
        logger.info('cell %2d %-14s bin %2d | %d subjects', int(cell['cell_index']),
                    cell['region'], int(cell['freq_bin_index']), len(eff))

    for j in range(len(cells), nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)

    fig.supxlabel('pain slope  (d log10 power per pain point)', fontsize=10, y=0.055)
    fig.supylabel('subjects, sorted by the model\'s estimate for them', fontsize=10)
    fig.suptitle('Fixed effect vs random effects in each pilot cell\n'
                 'open marker = subject\'s own unpooled fit (95% CI)   filled = the '
                 'model\'s estimate for them   black line = group fixed effect',
                 fontsize=12)

    handles = [
        Line2D([], [], marker='o', ls='none', mfc='none', mec=RAW_COLOR,
               label='subject\'s own OLS slope, 95% CI'),
        Line2D([], [], marker='o', ls='none', color=POS_COLOR,
               label='model estimate (beta + BLUP), positive'),
        Line2D([], [], marker='o', ls='none', color=NEG_COLOR,
               label='model estimate (beta + BLUP), negative'),
        Line2D([], [], color='black', lw=1.8, label='group fixed effect (+/- 1.96 SE)')]
    fig.legend(handles=handles, loc='lower right', fontsize=8.5, ncol=4,
               bbox_to_anchor=(0.995, 0.082))

    fig.tight_layout(rect=(0.02, 0.095, 1, 0.952))
    fig.text(0.01, 0.004,
             'The grey segment joining a subject\'s two markers is SHRINKAGE: partial '
             'pooling pulls a subject toward the group in proportion to how little '
             'evidence they carry, so subjects with few reports or noisy data move '
             'furthest. Spread of the FILLED markers is the heterogeneity the LRT '
             'tests. The "own CI excludes 0" count is ~30-50 UNCORRECTED tests per '
             'cell -- at alpha=0.05 a few clear zero even under a true null, so it is '
             'texture, not a per-patient result. BLUPs are predictions rather than '
             'fitted parameters and deliberately carry no interval here.\n' + DISCLAIMER,
             fontsize=6.5, va='bottom', ha='left', color='0.35', wrap=True)

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return pd.DataFrame(summaries)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--out-name', default='fig_pilot_caterpillar.png')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    run_dir = Path(args.run_dir)
    out_path = run_dir / args.out_name
    summary = build(run_dir, out_path)

    logger.info('\n%s', summary.to_string(index=False))
    logger.info('wrote %s', out_path)
    io.log_analysis('mixed-model pilot: fixed vs random effects caterpillar per cell '
                    '(EXPLORATORY pilot figure, not a finding)', run_dir)


if __name__ == '__main__':
    main()

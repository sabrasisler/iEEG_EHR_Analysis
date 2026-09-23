"""Basic figure for a band-power diagnosis-interaction run: one row per cell.

    left   the pain slope in each arm (MDD- and MDD+), from the ONE interaction fit
    right  the difference (the interaction beta) with its Wald 95% CI;
           BH-significant cells in ink, the rest muted, p_bh at the right edge

Reads `band_cells.parquet` from a `run_bandpower_mixed --dx-model interaction`
run and writes `fig_dx_cells.png` into that same run directory. Rows are ordered
by the interaction p, top = smallest.

The arm slopes carry NO interval: `mm.simple_slopes` records point estimates
only. The right panel is the only one that tests a difference, which is also
why the left panel is not marked for significance -- "significant in one arm,
not the other" is not evidence that the arms differ.

    python -m ieeg_ehr.analysis.plot_dx_cells --run-dir <run>

EXPLORATORY -- discovery cohort, nominations not findings.
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from ieeg_ehr import io

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_dx_cells.py'

#: Categorical slots 1-2 of the reference palette, validated (CVD dE 24.7).
C_CTRL = '#2a78d6'
C_CASE = '#eb6834'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, AXIS = '#e1e0d9', '#c3c2b7'


def draw(cells, out, label, fdr_q):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    d = cells.sort_values('dx_ix_p', ascending=True).reset_index(drop=True)
    n = len(d)
    y = np.arange(n)
    rows = [f'{r.region} · {r.band}' for r in d.itertuples()]
    sig = d['dx_ix_p_bh_reject'].fillna(False).astype(bool).to_numpy()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 0.42 * n + 1.9),
                                 sharey=True, gridspec_kw={'wspace': 0.08})
    for ax in (a1, a2):
        ax.axvline(0, color=AXIS, lw=1, zorder=1)
        ax.grid(axis='x', color=GRID, lw=0.6, zorder=0)
        ax.tick_params(colors=INK2, labelsize=8.5, length=0)
        for s in ax.spines.values():
            s.set_visible(False)

    # --- left: the two arm slopes, joined so each cell reads as one pair
    for i, r in enumerate(d.itertuples()):
        a1.plot([r.slope_ref, r.slope_mod], [i, i], color=AXIS, lw=2, zorder=2)
    a1.scatter(d['slope_ref'], y, s=48, color=C_CTRL, edgecolor='white', lw=1.5,
               zorder=3, label=f'{label}−')
    a1.scatter(d['slope_mod'], y, s=48, color=C_CASE, edgecolor='white', lw=1.5,
               zorder=3, label=f'{label}+')
    a1.set_yticks(y)
    a1.set_yticklabels([f'{t}   ({int(c)}/{int(k)})' for t, c, k in
                        zip(rows, d['n_subjects_case'], d['n_subjects_control'])],
                       color=INK)
    a1.set_ylim(n - 0.5, -0.5)
    a1.set_xlabel('pain slope  (Δ log10 band power per pain point)',
                  color=INK2, fontsize=9)
    a1.set_title('Pain slope in each arm', loc='left', color=INK, fontsize=10.5)
    a1.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=2,
              frameon=False, fontsize=8.5, labelcolor=INK2, borderaxespad=0.2,
              handletextpad=0.3)

    # --- right: the interaction, which is the test
    ci = 1.96 * d['dx_ix_se'].to_numpy()
    b = d['dx_ix_beta'].to_numpy()
    for i in range(n):
        col = INK if sig[i] else MUTED
        a2.plot([b[i] - ci[i], b[i] + ci[i]], [i, i], color=col, lw=2,
                solid_capstyle='round', zorder=2)
        a2.scatter(b[i], i, s=48, color=col, edgecolor='white', lw=1.5, zorder=3)
    lim = float(np.nanmax(np.abs(np.r_[b - ci, b + ci]))) * 1.08
    a2.set_xlim(-lim, lim)
    a2.set_xlabel(f'{label}+ minus {label}− slope  (β ± 95% CI)',
                  color=INK2, fontsize=9)
    a2.set_title('Difference (interaction)', loc='left', color=INK, fontsize=10.5)
    # p_bh as a right-hand column, outside the plot, so it never sits on a mark.
    for i, p in enumerate(d['dx_ix_p_bh']):
        a2.text(1.02, i, f'{p:.3f}', transform=a2.get_yaxis_transform(),
                va='center', ha='left', fontsize=8.5,
                color=INK if sig[i] else MUTED,
                fontweight='bold' if sig[i] else 'normal')
    a2.text(1.02, -0.9, 'p_bh', transform=a2.get_yaxis_transform(),
            va='center', ha='left', fontsize=8.5, color=INK2)

    fig.suptitle(f'Does {label} change the pain slope?  '
                 f'{int(sig.sum())} of {n} cells BH-significant (q={fdr_q:g}, '
                 f'BH over these {n})', x=0.02, ha='left', color=INK, fontsize=11.5)
    fig.text(0.02, -0.02,
             f'Rows: region · band  ({label}+/{label}− subjects). Arm slopes are '
             'point estimates from the single interaction fit (no interval stored). '
             'Dark = BH-significant interaction.\nCells were selected as the '
             'largest pain effects of an earlier run on the same subjects, so p '
             'is conditional on that selection. EXPLORATORY -- nominations, not '
             'findings.', fontsize=7, color=MUTED, ha='left', va='top')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--label', default='MDD')
    ap.add_argument('--fdr-q', type=float, default=0.05)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    run_dir = Path(args.run_dir)
    cells = io.read_table(run_dir / 'band_cells.parquet', on_stale='refuse')
    cells = cells[cells['model'].str.startswith('dx_')
                  & cells['dx_ix_beta'].notna()]
    if not len(cells):
        raise SystemExit(f'no fitted dx-interaction cells in {run_dir}')
    out = run_dir / 'fig_dx_cells.png'
    draw(cells, out, args.label, args.fdr_q)
    logger.info('wrote %s', out)
    io.log_analysis(f'dx-interaction cell figure ({len(cells)} cells)', run_dir)


if __name__ == '__main__':
    main()

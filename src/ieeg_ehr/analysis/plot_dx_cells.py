"""Basic figure for a band-power diagnosis-interaction run: one row per cell.

    left   the pain slope in each arm (MDD- and MDD+), from the ONE interaction fit
    right  the difference (the interaction beta) with its Wald 95% CI;
           BH-significant cells in ink, the rest muted, p_bh at the right edge

Reads `band_cells.parquet` from a `run_bandpower_mixed --dx-model interaction`
run and writes `fig_dx_cells.png` into that same run directory. Rows are ordered
by the interaction p, top = smallest.

This figure draws the arm slopes as points only. `mm.simple_slopes` does
record each arm's SE (`se_ref`, `se_mod`, covariance-correct), and `--poster`
draws those intervals. The right panel is the only one that tests a difference, which is also
why the left panel is not marked for significance -- "significant in one arm,
not the other" is not evidence that the arms differ.

    python -m ieeg_ehr.analysis.plot_dx_cells --run-dir <run>

EXPLORATORY -- discovery cohort, nominations not findings.
"""

import argparse
import json
import logging
from datetime import datetime
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


#: The POSTER cut (`--poster`), in the house style of the domain-model poster
#: figures (`plot_domain_lmer_med.POSTER`): one row per cell, both arms' 95%
#: Wald CIs on that row, grey unless the INTERACTION is BH-significant, which
#: is inked, thickened and starred. No p values, n's or footnote -- the
#: full-record figure beside the run has them.
#: 6 in tall like the domain posters, and as wide as one of their 2 x 2
#: columns (7 in / 2), so it can stand beside a single domain panel.
POSTER_SIZE = (3.5, 6.0)
POSTER_DPI = 300
POSTER_TITLE = 'Pain–power slopes by\n{label} diagnosis across regions'


def draw_poster(cells, out, label):
    """The arm slopes with their intervals, one row per cell.

    `se_ref` / `se_mod` are `mm.simple_slopes`' SEs; the moderated one uses the
    covariance of the two terms, so each interval is that arm's own. The star
    still comes ONLY from the interaction test: two intervals that do or do
    not overlap are not a test of the difference.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    from matplotlib.transforms import blended_transform_factory
    from ieeg_ehr.analysis.plot_domain_lmer_med import POSTER_BAND_SYMBOL
    from ieeg_ehr.analysis.plot_domain_lmer import BAND_ORDER
    from ieeg_ehr.med_analysis import style
    from ieeg_ehr.med_analysis.plot_poster_epoch_meds import GROUPED_SIZES

    # Band order (low to high frequency), then the interaction p within a band.
    rank = {b: i for i, b in enumerate(BAND_ORDER)}
    d = (cells.assign(_band=cells['band'].map(rank))
         .sort_values(['_band', 'dx_ix_p']).reset_index(drop=True))
    n = len(d)
    sig = d['dx_ix_p_bh_reject'].fillna(False).astype(bool).to_numpy()
    OFF = 0.18
    GREY = '0.35'

    saved = {k: getattr(style, k) for k in GROUPED_SIZES}
    for k, v in GROUPED_SIZES.items():
        setattr(style, k, v)
    try:
        W, H = POSTER_SIZE
        fig, ax = plt.subplots(figsize=(W, H))
        fig.subplots_adjust(left=0.30, right=0.955, top=1 - 0.62 / H,
                            bottom=0.74 / H)
        star = blended_transform_factory(ax.transAxes, ax.transData)
        lo = np.r_[d['slope_ref'] - 1.96 * d['se_ref'],
                   d['slope_mod'] - 1.96 * d['se_mod']]
        hi = np.r_[d['slope_ref'] + 1.96 * d['se_ref'],
                   d['slope_mod'] + 1.96 * d['se_mod']]
        lim = float(np.nanmax(np.abs(np.r_[lo, hi]))) * 1.12
        for i, r in enumerate(d.itertuples()):
            col = INK if sig[i] else GREY
            lw = 2.0 if sig[i] else 1.0
            ax.plot([r.slope_ref, r.slope_mod], [i - OFF, i + OFF], '-',
                    color=col, lw=lw, alpha=0.9 if sig[i] else 0.6, zorder=2)
            for b, se, yy, fmt, fc, ms in (
                    (r.slope_ref, r.se_ref, i - OFF, 'o', 'white', 4.5),
                    (r.slope_mod, r.se_mod, i + OFF, 'D', col, 4.0)):
                ax.errorbar(b, yy, xerr=1.96 * se, fmt=fmt, ms=ms, color=col,
                            markerfacecolor=fc, lw=lw, capsize=2.0,
                            markeredgewidth=1.4 if sig[i] else 1.0, zorder=3)
            if sig[i]:
                ax.text(0.985, i, '*', transform=star, ha='right',
                        va='center', fontsize=15, color=INK, zorder=4)
        style.style_axes(ax, grid_axis=None, tick_color=style.TEXT_PRIMARY)
        ax.axvline(0, color=style.ZERO_LINE_COLOR, lw=0.8, ls='--', zorder=1)
        ax.set_xlim(-lim, lim)
        ax.xaxis.set_major_locator(MaxNLocator(3, symmetric=True))
        ax.tick_params(axis='x', labelsize=style.TICK_SIZE - 1)
        ax.tick_params(axis='y', length=0)
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels(
            [f'{r.region} {POSTER_BAND_SYMBOL.get(r.band, r.band)}'
             for r in d.itertuples()], fontsize=style.TICK_SIZE + 0.5)
        for t, s in zip(ax.get_yticklabels(), sig):
            t.set_fontweight('bold' if s else 'normal')
            t.set_color(INK if s else style.TEXT_PRIMARY)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_xlabel('Pain slope\n(Δ log$_{10}$ power per NRS point)',
                      fontsize=style.LABEL_SIZE - 1, color=style.TEXT_PRIMARY)

        pos = ax.get_position()
        # Title and legend are centred on the FIGURE: at 3.5 in the axes sit
        # right of centre (row labels), and either would run off the edge.
        fig.text(0.5, 1 - 0.08 / H,
                 POSTER_TITLE.format(label=label), fontsize=12, linespacing=1.15,
                 fontweight='bold', color=style.TEXT_PRIMARY, ha='center',
                 va='top')
        # Inside the axes, top right: with the rows in band order the delta rows
        # come first, and every delta interval ends at or left of zero.
        ax.legend(handles=[
            Line2D([], [], marker='o', color=GREY, markerfacecolor='white',
                   ls='none', ms=5, label=f'{label}−'),
            Line2D([], [], marker='D', color=GREY, markerfacecolor=GREY,
                   ls='none', ms=4.5, label=f'{label}+'),
            Line2D([], [], marker='$*$', color=INK, ls='none', ms=8,
                   label='slope difference\nBH q < .05')],
            loc='upper right', bbox_to_anchor=(1.05, 1.03), ncol=1,
            fontsize=style.LEGEND_SIZE - 1, frameon=False,
            handletextpad=0.25, borderaxespad=0.0,
            labelspacing=0.5)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=POSTER_DPI, facecolor='white')
        plt.close(fig)
    finally:
        for k, v in saved.items():
            setattr(style, k, v)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--label', default='MDD')
    ap.add_argument('--fdr-q', type=float, default=0.05)
    ap.add_argument('--poster', action='store_true',
                    help='draw the 3.5 x 6 in POSTER cut (both arms with CIs, '
                         'one row per cell) into <run-dir>/poster/<ts>/')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    run_dir = Path(args.run_dir)
    cells = io.read_table(run_dir / 'band_cells.parquet', on_stale='refuse')
    cells = cells[cells['model'].str.startswith('dx_')
                  & cells['dx_ix_beta'].notna()]
    if not len(cells):
        raise SystemExit(f'no fitted dx-interaction cells in {run_dir}')
    if args.poster:
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        poster_dir = run_dir / 'poster' / f'dx_cells_{stamp}'
        out = poster_dir / 'fig_dx_cells.png'
        draw_poster(cells, out, args.label)
        parent = json.loads((run_dir / 'provenance.json').read_text())
        io.write_run_provenance(
            poster_dir, script=SCRIPT,
            params={'label': args.label, 'fdr_q': args.fdr_q,
                    'size_in': POSTER_SIZE, 'source_run': str(run_dir)},
            parents=[str(run_dir / 'band_cells.parquet')],
            subjects=parent.get('subjects'),
            extra={'status': 'EXPLORATORY -- nominations, not findings.'})
        logger.info('wrote %s', out)
        io.log_analysis(f'dx-interaction cell figure ({len(cells)} cells) '
                        '-- poster', poster_dir)
        return
    out = run_dir / 'fig_dx_cells.png'
    draw(cells, out, args.label, args.fdr_q)
    logger.info('wrote %s', out)
    io.log_analysis(f'dx-interaction cell figure ({len(cells)} cells)', run_dir)


if __name__ == '__main__':
    main()

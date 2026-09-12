"""Convergence, coverage and p-value diagnostics for the paired change-score grids.

THESE ARE RESULTS, NOT HOUSEKEEPING. They gate how the medication panels may be
described. The non-opioid arm has the fewest pairs (1262), the worst convergence
(75.1%) and by far the most BH-significant cells (133/599 = 22%, against 11% and
13% for the other two). That could be real -- but SELECTIVE non-convergence
produces exactly this pattern, because every cell that fails to converge leaves
the BH denominator as well as the numerator. So the question "which cells failed,
and do they differ systematically from the ones that passed" has to be answered
before 22% means anything.

`fig_convergence` -- per arm, three region x frequency panels: which cells
converged, how many subjects each cell had, and how many of its pairs were dosed.
Read down a column: if the non-converged cells cluster where n_pairs_dosed is
thinnest, the extra BH hits in that arm are a survivorship artefact.

`fig_pvalues` -- p-value histogram per term per arm, against the uniform density
expected under a global null. A histogram that is FLAT WITH A SPIKE AT ZERO is
what a real effect plus valid parametric p-values looks like, and it supports the
BH counts. A BOWED or sloped histogram says the p-values are misspecified and the
BH counts are not interpretable at face value, whichever direction the bow goes.

    python -m ieeg_ehr.analysis.plot_change_score_diagnostics
"""

import argparse
import glob
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import view_tables
from ieeg_ehr.analysis.plot_mixed_model_grid import order_regions, pivot

logger = logging.getLogger(__name__)

#: Binary STATUS, not magnitude, so a two-colour categorical pair rather than a
#: ramp. Validated with the dataviz palette checker: chroma floor pass, CVD
#: separation dE 28.2 (protan) / 30.8 (tritan), normal-vision dE 36.3, contrast
#: >= 3:1 on a light surface. The repo's older #4C78A8 FAILS the chroma floor
#: (0.089 -- it reads grey), so it is deliberately not reused here.
COLOR_OK = '#1F77D0'
COLOR_BAD = '#F58518'

#: Counts are MAGNITUDE. One hue, light to dark -- never the diverging RdBu_r the
#: beta maps use, which would invent a meaningful midpoint where there is none.
CMAP_COUNT = 'Blues'

#: Text wears text tokens, never a series colour.
INK = '#222222'
INK_MUTED = '#666666'

TERMS = (('dpain_p', 'd_pain\n(power tracks pain change)'),
         ('med_p', 'med_between\n(dose shifts power)'),
         ('med_ix_p', 'd_pain x med\n(dose alters coupling)'),
         ('med_gap_p', 'med x gap_h\n(time structure)'))

DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. '
              'Parametric Wald p. Not confirmed out of sample.')


def arm_label(run_dir):
    """Short arm name from the run directory, plus its co-exposure status."""
    name = Path(run_dir).name
    for key, short in (('non_opioid_analgesics', 'non-opioid'),
                       ('opioids', 'opioids'),
                       ('analgesics', 'analgesics')):
        if key in name:
            noco = '_noco' in name
            return f'{short}' + (' (co-exp excluded)' if noco else '')
    return name


def load_arms(base):
    """[(label, run_dir, cells)] ordered analgesics -> opioids -> non-opioid."""
    arms = []
    for rd in sorted(glob.glob(f'{base}/changescore_*/')):
        p = Path(rd) / 'grid_cells.parquet'
        if not p.exists():
            continue
        cells = io.read_table(p, on_stale='warn')
        arms.append((arm_label(rd), Path(rd), cells))
    order = {'analgesics': 0, 'opioids': 1}
    arms.sort(key=lambda a: order.get(a[0].split(' ')[0], 2))
    return arms


def _grid_axes(ax, regions, bin_labels, bins, show_x):
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(regions, fontsize=7, color=INK)
    if show_x:
        lo = bin_labels['bin_low_hz'].reindex(bins)
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels([f'{v:.0f}' if v >= 1 else f'{v:.1f}' for v in lo],
                           fontsize=5.5, rotation=90, color=INK_MUTED)
        ax.set_xlabel('frequency bin, low edge (Hz)', fontsize=8, color=INK)
    else:
        ax.set_xticks([])
    for s in ax.spines.values():
        s.set_color('#CCCCCC')


def fig_convergence(arms, regions, bins, bin_labels, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    n = len(arms)
    fig, axes = plt.subplots(n, 3, figsize=(21, 3.4 * n + 1.6), squeeze=False)
    binary = ListedColormap([COLOR_BAD, COLOR_OK])

    for r, (label, _rd, cells) in enumerate(arms):
        conv = cells.copy()
        conv['ok'] = conv['converged'].fillna(False).astype(float)
        rate = conv['ok'].mean()

        m = pivot(conv, 'ok', regions, bins).astype(float)
        ax = axes[r][0]
        ax.imshow(m.to_numpy(), aspect='auto', cmap=binary, vmin=0, vmax=1,
                  interpolation='nearest')
        ax.set_title(f'{label} — CONVERGED  ({int(conv["ok"].sum())}/{len(conv)}'
                     f' = {rate:.1%})', fontsize=9, color=INK)
        _grid_axes(ax, regions, bin_labels, bins, show_x=(r == n - 1))
        if r == 0:
            # Legend, because identity must never be colour-alone.
            ax.legend(handles=[Patch(facecolor=COLOR_OK, label='converged'),
                               Patch(facecolor=COLOR_BAD, label='did NOT converge')],
                      loc='upper left', bbox_to_anchor=(0, 1.32), ncol=2,
                      fontsize=7, frameon=False)

        for c, (col, name) in enumerate(
                (('n_subjects', 'n subjects per cell'),
                 ('n_pairs_dosed', 'n pairs WITH a dose between')), start=1):
            ax = axes[r][c]
            if col not in cells.columns:
                ax.set_visible(False)
                continue
            g = pivot(cells, col, regions, bins).astype(float)
            im = ax.imshow(g.to_numpy(), aspect='auto', cmap=CMAP_COUNT,
                           interpolation='nearest')
            ax.set_title(f'{label} — {name}', fontsize=9, color=INK)
            _grid_axes(ax, regions, bin_labels, bins, show_x=(r == n - 1))
            cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.01)
            cb.ax.tick_params(labelsize=6, colors=INK_MUTED)

    fig.suptitle('Change-score grids: convergence and coverage\n'
                 'Read DOWN a column — if non-converged cells sit where dosed '
                 'pairs are thinnest, that arm\'s BH count is a survivorship '
                 'artefact, not a result.',
                 fontsize=12, color=INK)
    fig.text(0.5, 0.005, DISCLAIMER, ha='center', fontsize=7, color=INK_MUTED)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_pvalues(arms, out_path, nbins=20):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(arms)
    fig, axes = plt.subplots(n, len(TERMS), figsize=(4.1 * len(TERMS), 2.9 * n),
                             squeeze=False)
    for r, (label, _rd, cells) in enumerate(arms):
        ok = cells[cells['converged'].fillna(False)]
        for c, (col, title) in enumerate(TERMS):
            ax = axes[r][c]
            if col not in ok.columns:
                ax.set_visible(False)
                continue
            p = ok[col].dropna().to_numpy()
            ax.hist(p, bins=nbins, range=(0, 1), color=COLOR_OK,
                    edgecolor='white', linewidth=0.6)
            # The uniform density under a global null -- the shape to compare to.
            ax.axhline(len(p) / nbins, color=INK_MUTED, lw=1.4, ls='--',
                       label='uniform (global null)')
            frac_lo = float((p < 0.05).mean()) if len(p) else np.nan
            ax.set_title(f'{title}\n{label}  ·  n={len(p)}  ·  '
                         f'{frac_lo:.0%} below .05', fontsize=8, color=INK)
            ax.tick_params(labelsize=7, colors=INK_MUTED)
            ax.grid(axis='y', color='#EEEEEE', lw=0.6)
            ax.set_axisbelow(True)
            for s in ax.spines.values():
                s.set_color('#CCCCCC')
            if r == n - 1:
                ax.set_xlabel('p', fontsize=8, color=INK)
            if c == 0:
                ax.set_ylabel('cells', fontsize=8, color=INK)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, frameon=False)

    fig.suptitle('p-value distributions, converged cells only\n'
                 'FLAT WITH A SPIKE AT ZERO supports the BH counts. '
                 'A bowed or sloped histogram means the p-values are '
                 'misspecified and BH is not interpretable at face value.',
                 fontsize=12, color=INK)
    fig.text(0.5, 0.002, DISCLAIMER, ha='center', fontsize=7, color=INK_MUTED)
    fig.tight_layout(rect=[0, 0.015, 1, 0.90])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def convergence_table(arms):
    """The numbers behind the figures, so a reader never has to eyeball a rate."""
    rows = []
    for label, rd, cells in arms:
        ok = cells['converged'].fillna(False)
        g = cells[ok]
        rec = {'arm': label, 'run': Path(rd).name,
               'n_cells': len(cells), 'n_converged': int(ok.sum()),
               'conv_rate': round(float(ok.mean()), 4)}
        for col, prefix in (('dpain', 'dpain'), ('med', 'med'),
                            ('med_ix', 'med_ix'), ('med_gap', 'med_gap')):
            rc = f'{prefix}_bh_reject'
            if rc in g.columns:
                rec[f'BH_{col}'] = int((g[rc] == True).sum())     # noqa: E712
                rec[f'BHfrac_{col}'] = (round(float((g[rc] == True).mean()), 4)
                                        if len(g) else np.nan)    # noqa: E712
        # Does non-convergence track coverage? If it does, the BH counts in the
        # worst-converging arm are survivorship, not signal.
        for col in ('n_subjects', 'n_pairs_dosed', 'n_pairs'):
            if col in cells.columns:
                rec[f'{col}_conv'] = round(float(cells.loc[ok, col].median()), 1)
                rec[f'{col}_notconv'] = (
                    round(float(cells.loc[~ok, col].median()), 1)
                    if (~ok).any() else np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', default=None,
                    help='level-4 med_change_score directory holding the arms')
    ap.add_argument('--roi-scheme', default='roi_v2')
    ap.add_argument('--run-name', default='changescore_diagnostics')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    base = args.base or str(config.analysis_run_dir(
        question='psd_physiology', output_type='univariate_analysis',
        view_scheme='med_change_score', run_name='').parent)
    arms = load_arms(base)
    if not arms:
        raise SystemExit(f'no grid_cells.parquet under {base}')
    logger.info('arms: %s', [a[0] for a in arms])

    ref = arms[0][2]
    regions = order_regions(ref, view_tables.roi_regions_for(
        {'roi_scheme': args.roi_scheme}))
    bins = sorted(ref['freq_bin_index'].unique())
    bin_labels = (ref.drop_duplicates('freq_bin_index')
                  .set_index('freq_bin_index')[['freq_bin_low', 'freq_bin_high']]
                  .rename(columns={'freq_bin_low': 'bin_low_hz',
                                   'freq_bin_high': 'bin_high_hz'})
                  .sort_index())

    out_dir = config.analysis_run_dir(
        question='psd_physiology', output_type='univariate_analysis',
        view_scheme='med_change_score', run_name=args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    table = convergence_table(arms)
    logger.info('\n%s', table.to_string(index=False))
    io.write_table(table, out_dir / 'convergence_summary.csv',
                   params={'arms': [a[0] for a in arms]},
                   parents=[str(a[1] / 'grid_cells.parquet') for a in arms],
                   script='ieeg_ehr/analysis/plot_change_score_diagnostics.py')

    fig_convergence(arms, regions, bins, bin_labels,
                    out_dir / 'fig_convergence_coverage.png')
    logger.info('wrote %s', out_dir / 'fig_convergence_coverage.png')
    fig_pvalues(arms, out_dir / 'fig_pvalue_histograms.png')
    logger.info('wrote %s', out_dir / 'fig_pvalue_histograms.png')

    io.write_run_provenance(
        out_dir, script='ieeg_ehr/analysis/plot_change_score_diagnostics.py',
        params={'arms': [a[0] for a in arms], 'roi_scheme': args.roi_scheme},
        parents=[str(a[1] / 'provenance.json') for a in arms],
        extra={'status': 'EXPLORATORY diagnostics for the change-score arms',
               'why': 'Convergence is a RESULT here: a cell that fails to '
                      'converge leaves the BH denominator as well as the '
                      'numerator, so selective non-convergence inflates the '
                      'BH FRACTION in exactly the arm that converged worst.'})
    io.log_analysis('change-score diagnostics: convergence/coverage grids and '
                    'p-value histograms per arm (EXPLORATORY)', out_dir)
    print(out_dir)


if __name__ == '__main__':
    main()

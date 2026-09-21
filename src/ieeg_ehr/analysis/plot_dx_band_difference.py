"""One band, one question: is the pain slope different in MDD, per circuit?

This is the narrow figure. `fig_dx_domain.png` and `fig_dx_circuit.png` show all
six bands and lead with the omnibus, which tests "do circuits differ FROM EACH
OTHER in how MDD changes pain encoding". That is a different question from the
one asked here and it is deliberately absent: this script takes ONE band, asks
per circuit whether the MDD and non-MDD pain slopes differ, and reports nothing
else.

WHY THE MULTIPLE-COMPARISON FAMILY IS THE WHOLE STORY, and why this script
makes you name it. The domain pipeline BH-corrects `pain_x_dx` over its full
grid -- 5 circuits x 6 bands = 30 cells. Restricting attention to one band
makes the family 5 tests instead of 30, and on this cohort that is the
difference between a result and a nomination:

    Cognitive / beta, MDD - non-MDD = +0.0108, uncorrected p = 0.0069
        BH over 30 cells (all bands)      p_bh = 0.205   NOT significant
        BH over 5 circuits within beta    p_bh = 0.034   significant at q=0.05

BOTH ARE ARITHMETICALLY CORRECT. Which one is honest depends entirely on
something not in the data: whether the band was chosen BEFORE looking. If beta
was pre-specified, 5 tests is the right family. If beta was chosen because it
had the largest effect, the 30-cell family is the right one and the 5-test
number is a post-hoc reduction -- the standard way a nomination is dressed up
as a finding. So `--band` has NO DEFAULT. You have to type it, and what you
typed is written into the output and the caption.

The uncorrected 95% CI is drawn because the effect size and its precision are
what travel, but a CI EXCLUDING ZERO IS NOT THE TEST -- it is the uncorrected
p < 0.05 boundary, and it is why `fig_dx_domain.png` can show an interval clear
of zero on a cell its own BH family does not reject. The two are labelled
separately here rather than left to be conflated.

Usage:

    python -m ieeg_ehr.analysis.plot_dx_band_difference \\
        --run-dir $DERIV/analysis/pain/mdd/domain_model/.../domain_mdd_ever_.../ \\
        --band beta
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import io
from ieeg_ehr.analysis import cluster_permutation as cp
from ieeg_ehr.analysis.run_domain_model import (DISCLAIMER, DX_LABEL,
                                                NON_DX_LABEL, DX_DISPLAY,
                                                NON_DX_DISPLAY, norm_strata)
from ieeg_ehr.analysis.run_bandpower_mixed import DX_CAVEAT

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_dx_band_difference.py'


def load(run_dir, band):
    """(both strata, the difference, cells row) for one band of a domain run."""
    run_dir = Path(run_dir)
    slopes = norm_strata(io.read_table(run_dir / 'domain_slopes.parquet',
                                       on_stale='warn'))
    cells = io.read_table(run_dir / 'domain_bands.parquet', on_stale='warn')
    if band not in set(slopes['band']):
        raise SystemExit(f'band {band!r} not in this run; it has '
                         f'{sorted(set(slopes["band"]))}')
    row = cells[cells['band'] == band]

    # REFUSE on an ill-conditioned fit rather than plotting its intervals. A
    # non-positive-definite Hessian means the SEs came out of a non-invertible
    # curvature estimate, so the CI this figure is entirely about is not an
    # interval estimate at all.
    warn = str(row['warnings'].iloc[0]) if len(row) else ''
    if 'not positive definite' in warn:
        raise SystemExit(
            f'band {band!r} in this run has a NON-POSITIVE-DEFINITE Hessian, so '
            'its standard errors are not interval estimates and this figure '
            'would be meaningless. Refit (see TASKS.md: warm-start the '
            'three-way fits) or pick another band.')

    both = slopes[(slopes['term'] == 'pain_slope_by_stratum')
                  & (slopes['band'] == band)]
    diff = slopes[(slopes['term'] == 'pain_x_dx') & (slopes['band'] == band)].copy()
    if both.empty or diff.empty:
        raise SystemExit('this run has no diagnosis strata -- was it fitted '
                         'with --dx-model interaction?')
    return both, diff, row


def correct(diff, q):
    """BH within THIS band's circuits, kept beside the grid-wide correction.

    Both columns survive into the output table on purpose. `p_bh_grid` is what
    the pipeline computed over all 30 cells; `p_bh_band` is the narrower family
    this figure is scoped to. Reporting only one of them is how the choice of
    family stops being visible.
    """
    out = diff.copy()
    out = out.rename(columns={'p_bh': 'p_bh_grid'})
    _, adj = cp.bh_fdr(out['p'].to_numpy(), q=q)
    out['p_bh_band'] = adj
    out['reject_band'] = adj <= q
    out['reject_grid'] = out['p_bh_grid'] <= q
    out['ci_excludes_zero'] = (out['ci_lo'] > 0) | (out['ci_hi'] < 0)
    return out


def figure(both, diff, out_path, band, args, n_dx, n_non):
    """Left: both strata's slopes. Right: the difference, which is the answer."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    doms = list(diff['domain'])
    y = np.arange(len(doms))
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 0.62 * len(doms) + 4.0),
                            squeeze=False, sharey=True)

    # --- left: the two strata, no per-arm significance marks
    ax = axs[0][0]
    for stratum, disp, off, colour in (
            (NON_DX_LABEL, NON_DX_DISPLAY, -0.15, '#4a7fb5'),
            (DX_LABEL, DX_DISPLAY, +0.15, '#b03a2e')):
        d = (both[both['stratum'] == stratum].set_index('domain').reindex(doms))
        n = n_dx if stratum == DX_LABEL else n_non
        ax.errorbar(d['beta'].to_numpy(dtype=float), y + off,
                    xerr=1.96 * d['se'].to_numpy(dtype=float), fmt='o', ms=6,
                    lw=1.5, capsize=3, color=colour,
                    label=f'{disp} (n={n})')
    ax.axvline(0, color='0.4', lw=1.0, ls='--')
    ax.set_yticks(y)
    ax.set_yticklabels(doms, fontsize=10)
    ax.set_ylim(len(doms) - 0.5, -0.5)
    ax.set_xlabel(f'{band} pain slope  (d log10 power / pain point)', fontsize=9)
    ax.set_title(f'{band}: pain slope in each group\n'
                 '95% CI, no significance marks (see caption)', fontsize=10.5)
    ax.legend(fontsize=9, frameon=False, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)

    # --- right: the difference. THIS is the test.
    ax = axs[0][1]
    d = diff.set_index('domain').reindex(doms)
    b = d['beta'].to_numpy(dtype=float)
    se = d['se'].to_numpy(dtype=float)
    rej = d['reject_band'].fillna(False).to_numpy(dtype=bool)
    ax.errorbar(b[~rej], y[~rej], xerr=1.96 * se[~rej], fmt='o', ms=6, lw=1.5,
                capsize=3, color='0.55')
    ax.errorbar(b[rej], y[rej], xerr=1.96 * se[rej], fmt='o', ms=8.5, lw=2.0,
                capsize=3, color='#7d3c98')
    ax.axvline(0, color='0.4', lw=1.0, ls='--')
    for i, dom in enumerate(doms):
        r = d.loc[dom]
        star = ' *' if bool(r['reject_band']) else ''
        ax.annotate(f"{r['beta']:+.4f}   p={r['p']:.3g}   "
                    f"p_bh={r['p_bh_band']:.3g}{star}",
                    xy=(1.02, i), xycoords=('axes fraction', 'data'),
                    va='center', fontsize=8.5,
                    color='#7d3c98' if star else '0.35')
    ax.set_xlabel(f'{DX_DISPLAY} - {NON_DX_DISPLAY} difference in the {band} '
                  'pain slope', fontsize=9)
    ax.set_title(f'{band}: the DIFFERENCE  ({DX_DISPLAY} - {NON_DX_DISPLAY})\n'
                 f'purple = BH-significant among the {len(doms)} circuits '
                 f'in {band}, q={args.fdr_q}', fontsize=10.5)
    ax.spines[['top', 'right']].set_visible(False)
    for s in ('left',):
        ax.spines[s].set_visible(False)

    n_rej_band = int(diff['reject_band'].sum())
    n_rej_grid = int(diff['reject_grid'].fillna(False).sum())
    fig.suptitle(
        f'Does the {band}-band pain slope differ in MDD?  '
        f'{n_dx} {DX_DISPLAY} vs {n_non} {NON_DX_DISPLAY}\n'
        f'{n_rej_band} of {len(doms)} circuits significant correcting within '
        f'{band}; {n_rej_grid} correcting over all 30 circuit x band cells',
        fontsize=12.5)
    fig.tight_layout(rect=(0, 0.20, 0.97, 0.90))
    fig.text(0.01, 0.005,
             'Both groups come from ONE three-way fit (pain x circuit x MDD) as '
             'linear combinations with SEs from the fitted covariance; neither '
             'group was refitted alone. NO OMNIBUS HERE, deliberately: this '
             'figure asks whether each circuit\'s slope differs between groups, '
             'not whether circuits differ from each other. '
             'THE LEFT PANEL CARRIES NO SIGNIFICANCE MARKS -- the '
             f'{DX_DISPLAY} group is smaller and has wider intervals everywhere '
             'from power alone, so "one group excludes zero and the other does '
             'not" is the difference-of-significance fallacy. AND AN INTERVAL '
             'CLEAR OF ZERO IS NOT THE TEST: that is the uncorrected p < 0.05 '
             'boundary, which is why a cell can show a CI off zero and still '
             f'not be rejected. The family here is the {len(doms)} circuits '
             f'WITHIN {band} ({len(doms)} tests); the pipeline\'s own '
             '`p_bh_grid` corrects over all 30 cells and is kept in the output '
             f'table. If {band} was NOT chosen before looking at the data, the '
             '30-cell family is the honest one.\n'
             f'{DX_CAVEAT}\n{DISCLAIMER}',
             fontsize=6.6, va='bottom', ha='left', color='0.35', wrap=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True,
                    help='A domain-model run fitted with --dx-model interaction.')
    ap.add_argument('--band', required=True,
                    help='NO DEFAULT, deliberately: the band decides the '
                         'multiple-comparison family, so it has to be typed and '
                         'it is recorded in the output.')
    ap.add_argument('--fdr-q', type=float, default=0.05)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    both, diff, row = load(run_dir, args.band)
    diff = correct(diff, args.fdr_q)

    n_dx = int(row['n_subjects_case'].iloc[0])
    n_non = int(row['n_subjects_control'].iloc[0])

    # Loud about the one thing that could make this run's numbers unusable.
    prov = run_dir / 'provenance.json'
    noparcel = False
    if prov.exists():
        p = json.loads(prov.read_text()).get('params', {})
        noparcel = bool(p.get('drop_parcel_term'))
    if noparcel or 'noparcel' in run_dir.parts[-2]:
        logger.warning('THIS IS A --drop-parcel-term RUN. Its omnibus values are '
                       'inflated by orders of magnitude against the full spec '
                       'and at least one band failed to converge; per-contrast '
                       'betas track the full spec closely but nothing from it '
                       'should be reported. See TASKS.md.')

    cols = ['domain', 'beta', 'se', 'ci_lo', 'ci_hi', 'p', 'p_bh_band',
            'p_bh_grid', 'reject_band', 'reject_grid', 'ci_excludes_zero']
    logger.info('\n%s', diff[cols].to_string(index=False))

    out_csv = run_dir / f'dx_{args.band}_difference.csv'
    io.write_table(diff[cols], out_csv,
                   params={'band': args.band, 'fdr_q': args.fdr_q,
                           'family_band': f'{len(diff)} circuits within '
                                          f'{args.band}',
                           'family_grid': 'all 30 circuit x band cells, from '
                                          'the pipeline',
                           'drop_parcel_term': noparcel},
                   parents=[str(run_dir / 'domain_slopes.parquet')],
                   script=SCRIPT,
                   extra={'caveat': DX_CAVEAT, 'status': DISCLAIMER,
                          'family_warning':
                              'p_bh_band corrects over this band\'s circuits '
                              'ONLY. That is legitimate if the band was '
                              'pre-specified and is a post-hoc reduction if it '
                              'was not. p_bh_grid is the pipeline\'s 30-cell '
                              'correction and is kept beside it.'})

    out = figure(both, diff, run_dir / f'fig_dx_{args.band}_difference.png',
                 args.band, args, n_dx, n_non)
    logger.info('wrote %s', out)
    io.log_analysis(f'{DX_DISPLAY} vs {NON_DX_DISPLAY} difference in the '
                    f'{args.band}-band pain slope, per circuit (EXPLORATORY)',
                    run_dir)
    print(out)


if __name__ == '__main__':
    main()

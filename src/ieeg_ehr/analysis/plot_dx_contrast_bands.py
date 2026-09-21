"""The poster figure: MDD+ minus MDD- pain slope, one panel per band.

ONE QUANTITY ONLY -- the between-group DIFFERENCE. The per-group slopes are not
here and that is the point: at 4 circuits x 6 bands they are 48 intervals, and
the two groups' interval widths are not comparable to each other anyway (unequal
n, so "one group's interval clears zero and the other's does not" is the
difference-of-significance fallacy). They are drawn as a SUPPLEMENT instead,
`fig_dx_groups_by_band.png`, for the person who asks to see them.

THE CONTROL CIRCUIT IS DROPPED, and dropping it is not cosmetic. `Control` is
Auditory + Occipital, the scheme's negative-control network and the model's
TREATMENT REFERENCE, so every other circuit's contrast is already read against
it. Removing its row removes a cell from the multiple-comparison family, and
the family therefore has to be RE-CORRECTED rather than filtered: taking the
pipeline's 30-cell BH values and then hiding five rows would leave the surviving
p_bh values corrected for tests no longer being reported. `--keep-domains` makes
the retained set explicit and it is written into the output.

Dropping the reference row does NOT change any coefficient. The fit is
untouched; `Control` remains the reference the contrasts are formed against, and
its own row was the reference's marginal value. What changes is only which
contrasts are being reported, and hence the family.

ILL-CONDITIONED BANDS ARE EXCLUDED BY NAME, from the figure AND from the family.
A non-positive-definite Hessian means the standard errors came out of a
non-invertible curvature estimate, so the interval this entire figure consists
of is not an interval estimate. Which bands those are DIFFERS BY RUN and is not
a property of the frequency: on this cohort the full specification breaks delta
and theta while dropping the parcel term breaks alpha and gamma instead (see
docs/labnotebook/2026-09-21.md). The panel is still drawn, empty and labelled,
so the layout does not silently lose a frequency.

EVERYTHING REPORTED IS BH-CORRECTED over the cells actually shown -- retained
circuits x retained bands, computed here. The uncorrected p is kept in the
output table but no marker on the figure is driven by it.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import io
from ieeg_ehr.analysis import cluster_permutation as cp
from ieeg_ehr.analysis.run_bandpower_mixed import BAND_SETS, DX_CAVEAT
from ieeg_ehr.analysis.run_domain_model import (DISCLAIMER, DOMAIN_CAVEAT,
                                                DOMAIN_COLOURS, DX_DISPLAY,
                                                DX_LABEL, MODULATORY_CAVEAT,
                                                NON_DX_DISPLAY, NON_DX_LABEL,
                                                norm_strata)

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_dx_contrast_bands.py'

#: Circuits reported by default. `Control` is the treatment reference and the
#: scheme's negative-control network, so its own contrast against itself is
#: structurally zero and reporting it only inflates the family.
DEFAULT_KEEP = ('Sensory', 'Affective', 'Cognitive', 'Modulatory')


def load_run(run_dir, band_set):
    """(per-group slopes, difference rows, per-band cells, excluded bands)."""
    run_dir = Path(run_dir)
    slopes = norm_strata(io.read_table(run_dir / 'domain_slopes.parquet',
                                       on_stale='warn'))
    cells = io.read_table(run_dir / 'domain_bands.parquet', on_stale='warn')

    warn = cells['warnings'].astype(str)
    bad = sorted(cells.loc[warn.str.contains('not positive definite'), 'band'])
    unconv = sorted(cells.loc[~cells['converged'].astype(bool), 'band'])
    excluded = sorted(set(bad) | set(unconv))
    if excluded:
        logger.warning('EXCLUDED from the figure and from the BH family: %s '
                       '(non-positive-definite Hessian and/or did not '
                       'converge). Their SEs are not interval estimates.',
                       excluded)

    both = slopes[slopes['term'] == 'pain_slope_by_stratum'].copy()
    diff = slopes[slopes['term'] == 'pain_x_dx'].copy()
    if both.empty or diff.empty:
        raise SystemExit(f'{run_dir} has no diagnosis strata -- was it fitted '
                         'with --dx-model interaction?')
    return both, diff, cells, excluded


def correct(diff, keep_domains, excluded_bands, q):
    """BH over EXACTLY the cells that will be reported.

    Recomputed, never inherited. The pipeline corrected `pain_x_dx` over its
    full 5-circuit x 6-band grid; this figure reports a subset, and a p_bh
    carried over from the larger family would be corrected for tests that are
    no longer on the figure -- conservative, but wrong, and wrong in a direction
    that is invisible to the reader.
    """
    out = diff[diff['domain'].isin(keep_domains)
               & ~diff['band'].isin(excluded_bands)].copy()
    if out.empty:
        raise SystemExit('no cells left after dropping the excluded circuits '
                         'and ill-conditioned bands')
    out = out.rename(columns={'p_bh': 'p_bh_pipeline_grid'})
    _, adj = cp.bh_fdr(out['p'].to_numpy(), q=q)
    out['p_bh'] = adj
    out['reject'] = adj <= q
    out['ci_excludes_zero_uncorrected'] = (out['ci_lo'] > 0) | (out['ci_hi'] < 0)
    logger.info('BH family = %d cells (%d circuits x %d bands), %d rejected at '
                'q=%.2f', len(out), out['domain'].nunique(),
                out['band'].nunique(), int(out['reject'].sum()), q)
    return out


def _band_label(band, band_set):
    lo, hi = BAND_SETS[band_set][band]
    return f'{band}\n{lo}-{hi} Hz'


def figure_contrast(diff, bands, doms, out_path, args, n_dx, n_non, excluded):
    """One panel per band. Circuits colour-coded. The difference, and nothing else.

    SIGNIFICANCE IS NOT ENCODED IN COLOUR, because colour is spent on circuit
    identity. A BH-significant cell gets a filled marker, a heavier interval and
    a star; everything else is the same hue, open and light. That keeps one
    visual channel for one variable each.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    y = np.arange(len(doms))
    fig, axes = plt.subplots(1, len(bands),
                             figsize=(2.55 * len(bands) + 1.6,
                                      0.52 * len(doms) + 4.0),
                             sharey=True, sharex=True, squeeze=False)

    shown = diff[diff['band'].isin(bands)]
    xmax = float(np.nanmax(np.abs(np.concatenate(
        [shown['ci_lo'].to_numpy(dtype=float),
         shown['ci_hi'].to_numpy(dtype=float)])))) * 1.12

    for j, band in enumerate(bands):
        ax = axes[0][j]
        ax.set_title(_band_label(band, args.band_set), fontsize=9.5)
        ax.axvline(0, color='0.35', lw=1.0, ls='--', zorder=1)
        ax.tick_params(labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)

        if band in excluded:
            # Frame kept, data deliberately absent, reason on the panel.
            # NO set_xticks([]) here: these axes are sharex, so clearing the
            # locator on one panel clears the tick LABELS on every panel and
            # the reader loses the effect-size scale everywhere.
            ax.text(0.5, 0.5, 'fit ill-conditioned\nEXCLUDED',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=8.5, color='#b03a2e', style='italic')
            ax.set_facecolor('#f6f6f6')
            continue

        d = diff[diff['band'] == band].set_index('domain').reindex(doms)
        for i, dom in enumerate(doms):
            if dom not in d.index or not np.isfinite(d.loc[dom, 'beta']):
                continue
            r = d.loc[dom]
            colour = DOMAIN_COLOURS.get(dom, '0.4')
            sig = bool(r['reject'])
            ax.errorbar(float(r['beta']), i,
                        xerr=1.96 * float(r['se']), fmt='o',
                        ms=8.0 if sig else 5.5,
                        lw=2.2 if sig else 1.2,
                        capsize=3 if sig else 2,
                        color=colour,
                        markerfacecolor=colour if sig else 'white',
                        markeredgecolor=colour,
                        markeredgewidth=1.6,
                        alpha=1.0 if sig else 0.75, zorder=3)
            if sig:
                ax.annotate('*', xy=(float(r['ci_hi']), i),
                            xytext=(5, 0), textcoords='offset points',
                            va='center', fontsize=16, color=colour)

        if j == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(doms, fontsize=10)
            for tick, dom in zip(ax.get_yticklabels(), doms):
                tick.set_color(DOMAIN_COLOURS.get(dom, '0.2'))
            ax.set_ylim(len(doms) - 0.5, -0.5)

    axes[0][len(bands) // 2].set_xlabel(
        f'{DX_DISPLAY} - {NON_DX_DISPLAY}   (d log10 power / pain point)',
        fontsize=10)
    if np.isfinite(xmax) and xmax > 0:
        axes[0][0].set_xlim(-xmax, xmax)
    # sharex hides tick labels on all but the last row by default; this is a
    # single row, so turn them back on explicitly for every panel.
    for ax in axes[0]:
        ax.tick_params(axis='x', labelbottom=True, labelsize=7.5)

    n_rej = int(diff['reject'].sum())
    excl = (f'  {", ".join(excluded)} excluded (ill-conditioned fit).'
            if excluded else '')
    fig.suptitle(
        f'Does pain encoding differ in MDD?   {DX_DISPLAY} - {NON_DX_DISPLAY} '
        f'pain slope, per circuit\n'
        f'{n_dx} {DX_DISPLAY} vs {n_non} {NON_DX_DISPLAY};  '
        f'filled + * = BH-significant over all {len(diff)} reported cells '
        f'({diff["domain"].nunique()} circuits x {diff["band"].nunique()} '
        f'bands), q={args.fdr_q};  {n_rej} significant',
        fontsize=12)
    fig.tight_layout(rect=(0, 0.19, 1, 0.88))
    fig.text(0.01, 0.005,
             'Positive = the pain slope is MORE POSITIVE in MDD. Both groups '
             'come from ONE three-way fit, log10_power ~ NRS_within * C(domain) '
             '* dx_state + NRS_submean with the parcel-nested random slope, as '
             'linear combinations with SEs from the fitted covariance -- '
             'neither group was refitted alone, and this contrast and its SE '
             'come straight from that fit. Bars are 95% Wald CIs on the '
             'DIFFERENCE, so unlike a per-group interval they are the right '
             'thing to read against zero -- but the marker is driven by the '
             f'BH-CORRECTED p over all {len(diff)} reported cells, not by '
             'whether the bar clears zero. The Control circuit (Auditory + '
             'Occipital) is the model\'s treatment reference and is not '
             'reported; the family was RE-CORRECTED over the reported cells '
             'rather than inherited from the pipeline\'s wider grid, and the '
             'uncorrected p and the pipeline value are both in the companion '
             f'CSV.{excl}\n'
             f'{DX_CAVEAT}\n{MODULATORY_CAVEAT}\n{DOMAIN_CAVEAT}\n{DISCLAIMER}',
             fontsize=6.3, va='bottom', ha='left', color='0.35', wrap=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def figure_groups(both, diff, bands, doms, out_path, args, n_dx, n_non,
                  excluded):
    """SUPPLEMENT: the per-group slopes the main figure deliberately omits.

    Exists so the question "but what are the actual slopes in each group" has an
    answer that is not a refit. It is explicitly labelled a supplement and
    carries NO significance marks at all -- comparing the two groups' intervals
    by eye is the fallacy the main figure exists to prevent, and this panel
    cannot support it no matter how it is drawn.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    draw = [b for b in bands if b not in excluded]
    if not draw:
        return None
    y = np.arange(len(doms))
    fig, axes = plt.subplots(1, len(draw),
                             figsize=(2.55 * len(draw) + 1.6,
                                      0.52 * len(doms) + 4.0),
                             sharey=True, sharex=True, squeeze=False)
    keep = both[both['band'].isin(draw) & both['domain'].isin(doms)]
    xmax = float(np.nanmax(np.abs(np.concatenate(
        [(keep['beta'] + 1.96 * keep['se']).to_numpy(dtype=float),
         (keep['beta'] - 1.96 * keep['se']).to_numpy(dtype=float)])))) * 1.10

    for j, band in enumerate(draw):
        ax = axes[0][j]
        for stratum, disp, off, marker in ((NON_DX_LABEL, NON_DX_DISPLAY,
                                            -0.17, 'o'),
                                           (DX_LABEL, DX_DISPLAY, +0.17, 's')):
            d = (keep[(keep['band'] == band) & (keep['stratum'] == stratum)]
                 .set_index('domain').reindex(doms))
            for i, dom in enumerate(doms):
                if dom not in d.index or not np.isfinite(d.loc[dom, 'beta']):
                    continue
                r = d.loc[dom]
                colour = DOMAIN_COLOURS.get(dom, '0.4')
                ax.errorbar(float(r['beta']), i + off,
                            xerr=1.96 * float(r['se']), fmt=marker, ms=5.0,
                            lw=1.2, capsize=2, color=colour,
                            markerfacecolor=colour if stratum == DX_LABEL
                            else 'white', markeredgecolor=colour,
                            markeredgewidth=1.4, alpha=0.9)
        ax.axvline(0, color='0.35', lw=1.0, ls='--')
        ax.set_title(_band_label(band, args.band_set), fontsize=9.5)
        ax.tick_params(labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        if j == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(doms, fontsize=10)
            for tick, dom in zip(ax.get_yticklabels(), doms):
                tick.set_color(DOMAIN_COLOURS.get(dom, '0.2'))
            ax.set_ylim(len(doms) - 0.5, -0.5)
    axes[0][len(draw) // 2].set_xlabel(
        'pain slope  (d log10 power / pain point)', fontsize=10)
    if np.isfinite(xmax) and xmax > 0:
        axes[0][0].set_xlim(-xmax, xmax)

    fig.suptitle(
        f'SUPPLEMENT to fig_dx_contrast_by_band: the per-group slopes\n'
        f'open circle = {NON_DX_DISPLAY} (n={n_non}),  filled square = '
        f'{DX_DISPLAY} (n={n_dx});  colour = circuit;  NO significance marks',
        fontsize=11.5)
    fig.tight_layout(rect=(0, 0.17, 1, 0.88))
    fig.text(0.01, 0.005,
             'THIS PANEL CANNOT BE USED TO COMPARE THE GROUPS and carries no '
             'significance marks for that reason. The two groups have unequal '
             f'n ({n_dx} vs {n_non}), so their interval widths differ for '
             'reasons that have nothing to do with the effect, and reading '
             '"one clears zero, the other does not" as a group difference is '
             'the difference-of-significance fallacy. The test of a difference '
             'is the contrast in the main figure, which has its own SE from '
             'the fitted covariance and its own BH correction. This exists '
             'only to show what the underlying slopes are.\n'
             f'{DX_CAVEAT}\n{DISCLAIMER}',
             fontsize=6.3, va='bottom', ha='left', color='0.35', wrap=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--band-set', default='paper_bands_6_hg200',
                    choices=list(BAND_SETS))
    ap.add_argument('--keep-domains', nargs='*', default=list(DEFAULT_KEEP),
                    help='Circuits to REPORT. Default drops Control, which is '
                         'the treatment reference. Changing this changes the '
                         'multiple-comparison family, which is why it is '
                         'explicit and is recorded in the output.')
    ap.add_argument('--fdr-q', type=float, default=0.05)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    both, diff, cells, excluded = load_run(run_dir, args.band_set)

    unknown = sorted(set(args.keep_domains) - set(diff['domain']))
    if unknown:
        raise SystemExit(f'--keep-domains {unknown} not in this run; it has '
                         f'{sorted(set(diff["domain"]))}')
    doms = [d for d in args.keep_domains if d in set(diff['domain'])]
    bands = [b for b in BAND_SETS[args.band_set] if b in set(diff['band'])]

    diff = correct(diff, doms, excluded, args.fdr_q)
    n_dx = int(np.nanmax(cells['n_subjects_case'].to_numpy(dtype=float)))
    n_non = int(np.nanmax(cells['n_subjects_control'].to_numpy(dtype=float)))

    prov = run_dir / 'provenance.json'
    noparcel = False
    if prov.exists():
        noparcel = bool(json.loads(prov.read_text())
                        .get('params', {}).get('drop_parcel_term'))
    if noparcel:
        logger.warning(
            'THIS IS A --drop-parcel-term RUN. It is NOT the better-conditioned '
            'specification: it trades which bands fail rather than fixing them '
            '(measured 2026-09-21 -- the full spec breaks delta and theta, this '
            'one breaks alpha and gamma), and its omnibus is inflated by orders '
            'of magnitude. The per-contrast betas here track the full spec '
            'closely (r=0.89) and the ill-conditioned bands are excluded above, '
            'so this figure is readable -- but the full specification is the '
            'one to report from.')

    cols = ['band', 'domain', 'beta', 'se', 'ci_lo', 'ci_hi', 'p', 'p_bh',
            'reject', 'ci_excludes_zero_uncorrected', 'p_bh_pipeline_grid']
    tab = diff[cols].sort_values(['band', 'domain'])
    logger.info('\n%s', tab.to_string(index=False))

    io.write_table(
        tab, run_dir / 'dx_contrast_by_band.csv',
        params={'keep_domains': doms, 'excluded_bands': excluded,
                'band_set': args.band_set, 'fdr_q': args.fdr_q,
                'bh_family': f'{len(diff)} reported cells '
                             f'({len(doms)} circuits x '
                             f'{diff["band"].nunique()} bands), recomputed here',
                'drop_parcel_term': noparcel},
        parents=[str(run_dir / 'domain_slopes.parquet')], script=SCRIPT,
        extra={'caveat': DX_CAVEAT, 'status': DISCLAIMER,
               'bh_note': 'p_bh is RECOMPUTED over the reported cells. '
                          'p_bh_pipeline_grid is the pipeline value over its '
                          'full 5-circuit x 6-band grid and is NOT what the '
                          'figure marks, because it corrects for tests this '
                          'figure does not report.',
               'excluded_bands_reason':
                   'non-positive-definite Hessian and/or non-convergence; '
                   'their SEs are not interval estimates. Which bands these '
                   'are differs by specification, not by frequency.'})

    p1 = figure_contrast(diff, bands, doms,
                         run_dir / 'fig_dx_contrast_by_band.png',
                         args, n_dx, n_non, excluded)
    logger.info('wrote %s', p1)
    p2 = figure_groups(both, diff, bands, doms,
                       run_dir / 'fig_dx_groups_by_band.png',
                       args, n_dx, n_non, excluded)
    if p2:
        logger.info('wrote %s', p2)
    io.log_analysis(f'{DX_DISPLAY} - {NON_DX_DISPLAY} pain-slope contrast by '
                    f'circuit x band, BH over {len(diff)} reported cells '
                    '(EXPLORATORY)', run_dir)
    print(p1)


if __name__ == '__main__':
    main()

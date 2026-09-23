#!/usr/bin/env python3
"""The domain x band heatmap for an lme4 domain run.

    python -m ieeg_ehr.analysis.plot_domain_lmer --run-dir <lmer run>

Rows are PROCESSING DOMAINS, columns are FREQUENCY BANDS, and each cell is that
domain's marginal pain slope in that band -- d log10 power per NRS point --
tested against ZERO. It is the lme4 counterpart of `fig_domain_summary.png` and
is drawn to the same conventions on purpose, so the two can be laid side by side
and the only thing that differs is the model.

WHY THE SAME CONVENTIONS. Diverging RdBu_r on a symmetric cap, because the
quantity is SIGNED and zero is meaningful -- a sequential ramp would hide the
sign, and a rainbow would invent structure. The value is PRINTED in every cell:
30 cells is few enough that a reader should never have to estimate a beta off a
colour ramp when the number fits. The outline marks BH-significance; stars mark
the uncorrected p, and the two are deliberately different marks because they are
different claims.

NO OMNIBUS APPEARS ON THIS FIGURE, and none is computed upstream. Each cell is
"does this domain's power track pain at all in this band", not "do the domains
differ from each other" -- the second is a question about contrasts and this
grid cannot answer it. `domain_pairs.csv` in the run holds the pairwise domain
contrasts for the reader who wants that.

A SIGNIFICANT CONTROL CELL IS NOT A CONTRADICTION. Occipital and Auditory are
the quasi-controls; an effect there argues for a global or artifactual driver
rather than nociception, so that row is read FIRST, not dropped. `--drop-domain
Control` exists for a figure that deliberately excludes it, but the default
keeps it because removing it removes the diagnostic.

EXPLORATORY. Discovery cohort only. Nominations, not findings.
"""

import argparse
import logging
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import io
from ieeg_ehr.analysis import cluster_permutation as cp

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_domain_lmer.py'

#: Low to high. NOT alphabetical, and not the order the CSV happens to be in --
#: a frequency axis out of frequency order is unreadable.
BAND_ORDER = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']
BAND_LABELS = {'high_gamma': 'high gamma'}

#: Sensory -> Affective -> Cognitive -> Modulatory -> Control, the framework's
#: own order, with the quasi-control last.
DOMAIN_ORDER = ['Sensory', 'Affective', 'Cognitive', 'Modulatory', 'Control']

DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. '
              'Not confirmed out of sample.')

SLOPE_COL = 'NRS_within.trend'


def load_slopes(run_dir, fdr_q, drop_domain=None):
    """The run's slopes, BH-corrected across the cells that will be PLOTTED.

    BH IS APPLIED HERE, NOT UPSTREAM. The R stage fits one band at a time and
    has no view of the other five, so it cannot correct across a family it
    cannot see.

    DROPPED DOMAINS LEAVE THE FAMILY BEFORE THE CORRECTION, NOT AFTER. This
    was the other way round once: BH ran over all 30 cells and `--drop-domain`
    filtered the result, so the excluded rows still inflated the threshold for
    the rows on the figure, and the footer -- counting after the drop -- named
    a family size the correction had not used.

    The caveat that goes with it: this is only honest when the exclusion is
    decided A PRIORI, which is the case here (the excluded domain is a
    different question, not a disappointing result). Dropping a domain BECAUSE
    of how its cells came out and then correcting over what is left is
    selective inference and would understate the false-discovery rate.
    """
    df = io.read_table(Path(run_dir) / 'domain_slopes.csv', on_stale='refuse')
    if SLOPE_COL not in df.columns:
        raise SystemExit(f'{run_dir}/domain_slopes.csv has no {SLOPE_COL!r}; '
                         f'columns are {list(df.columns)}')
    if drop_domain:
        df = df[~df['domain'].isin(drop_domain)].reset_index(drop=True)
        if df.empty:
            raise SystemExit(f'--drop-domain {drop_domain} removed every row.')
    m = df['p.value'].notna()
    df['p_bh'] = np.nan
    if m.any():
        _, adj = cp.bh_fdr(df.loc[m, 'p.value'].to_numpy(), q=fdr_q)
        df.loc[m, 'p_bh'] = adj
    df['p_bh_reject'] = df['p_bh'] <= fdr_q
    return df


def grid(df, domains, bands, values):
    return (df.pivot_table(index='domain', columns='band', values=values)
            .reindex(index=domains, columns=bands))


def heatmap(ax, df, domains, bands, cap, title):
    import matplotlib.pyplot as plt
    from ieeg_ehr.features import common

    arr = grid(df, domains, bands, SLOPE_COL).to_numpy(dtype=float)
    pval = grid(df, domains, bands, 'p.value').to_numpy(dtype=float)
    rej = (grid(df.assign(_r=df['p_bh_reject'].fillna(False).astype(bool)),
                domains, bands, '_r').fillna(0).to_numpy().astype(bool))

    cm = plt.get_cmap('RdBu_r').copy()
    cm.set_bad('0.85')
    im = ax.imshow(arr, aspect='auto', cmap=cm, vmin=-cap, vmax=cap,
                   interpolation='nearest')
    common.draw_mask_outline(ax, rej, linewidth=2.4)

    for i in range(len(domains)):
        for j in range(len(bands)):
            v, p = arr[i, j], pval[i, j]
            if not np.isfinite(v):
                continue
            # White on the saturated ends, where black ink disappears.
            shade = 'white' if abs(v) > 0.62 * cap else 'black'
            stars = ('***' if p < 1e-3 else '**' if p < 1e-2 else
                     '*' if p < 0.05 else '')
            ax.text(j, i - 0.13, f'{v:+.4f}', ha='center', va='center',
                    fontsize=8.5, color=shade,
                    fontweight='bold' if rej[i, j] else 'normal')
            if stars:
                ax.text(j, i + 0.26, stars, ha='center', va='center',
                        fontsize=9, color=shade)

    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([BAND_LABELS.get(b, b) for b in bands], fontsize=9.5)
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels(domains, fontsize=10)
    ax.set_xlabel('frequency band', fontsize=10)
    ax.set_title(title, fontsize=11)
    # Recessive grid: hairlines between cells, no heavy frame competing with
    # the BH outline that carries the actual claim.
    ax.set_xticks(np.arange(-0.5, len(bands), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(domains), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.2)
    ax.tick_params(which='minor', length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def figure(run_dir, df, domains, bands, args, out_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    strata = sorted(df['dx'].unique()) if 'dx' in df.columns else [None]
    cap = float(np.nanmax(np.abs(df[SLOPE_COL].to_numpy(dtype=float))))

    # THE COLOURBAR GETS ITS OWN GRIDSPEC COLUMN. Handing `ax=` to colorbar()
    # steals space from the axes it is given, and under tight_layout it landed
    # ON TOP of the last band -- high_gamma's cells were drawn and then covered,
    # which is invisible unless you look at the rendered file.
    ncol = len(strata)
    fig = plt.figure(figsize=(1.9 * len(bands) * ncol + 3.0,
                              0.82 * len(domains) + 5.0))
    gs = fig.add_gridspec(1, ncol + 1,
                          width_ratios=[1.0] * ncol + [0.035], wspace=0.10)
    axes = [fig.add_subplot(gs[0, i]) for i in range(ncol)]
    cax = fig.add_subplot(gs[0, ncol])

    for ax, stratum in zip(axes, strata):
        sub = df if stratum is None else df[df['dx'] == stratum]
        title = ('pain-power slope by processing domain'
                 if stratum is None else str(stratum))
        im = heatmap(ax, sub, domains, bands, cap, title)
        if ax is not axes[0]:
            ax.set_yticklabels([])

    cb = fig.colorbar(im, cax=cax)
    cb.set_label('d log10 power / pain point', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fit = pd.read_csv(Path(run_dir) / 'fitinfo.csv')
    dfs = df['df'].dropna()
    fig.suptitle(
        'PROCESSING DOMAIN x FREQUENCY BAND, lme4 with crossed ROI effects\n'
        'each cell is that domain\'s marginal pain slope tested against ZERO',
        fontsize=12.5)
    footer = (
        'ONE lme4 MODEL PER BAND over every channel x epoch at once: '
        'log10_power ~ 0 + domain + domain:NRS_within + domain:NRS_submean '
        '+ (1 + NRS_within || ROI) + (1 + NRS_within || subject) '
        '+ (1 + NRS_within || subj_roi) + (1 | chan_id). THE ROI TERM IS '
        'CROSSED, which is the reason this model is fitted in R at all: '
        'statsmodels evaluates every variance component within ONE grouping '
        'variable and would nest it SILENTLY. Each cell is a MARGINAL slope '
        'from emtrends(), not a coefficient. Bold + outline is BH at '
        f'q={args.fdr_q:g} over all {len(df)} cells; stars are the UNCORRECTED '
        'p (* <0.05, ** <0.01, *** <0.001) and are a different claim. '
        f'Satterthwaite df run {dfs.min():.0f}-{dfs.max():.0f}, i.e. these '
        'intervals are governed by the ~51 SUBJECTS, not the '
        f'{int(fit["n_rows"].iloc[0]):,} rows. '
        'NO OMNIBUS: whether the domains differ FROM EACH OTHER is a question '
        'about contrasts and is not on this grid -- see domain_pairs.csv. '
        # The Control note is only true when Control is ON the figure. Left
        # unconditional it told a reader to interpret a row that is not there,
        # and hid the more important fact that the diagnostic was removed.
        + ('A significant Control cell is NOT a contradiction: Occipital and '
           'Auditory are the quasi-controls, so an effect there argues for a '
           'global or artifactual driver.'
           if 'Control' in domains else
           'THE CONTROL DOMAIN (Occipital + Auditory) IS EXCLUDED from this '
           'figure. It is the quasi-control whose own slope says whether an '
           'effect is global rather than nociceptive -- in delta it is '
           'significant, so the delta column here should NOT be read as '
           'pain-specific. See the Control-inclusive figure in this run.'))

    # WRAPPED EXPLICITLY. matplotlib's `wrap=True` wraps to the FIGURE width,
    # which on a wide figure still runs the footer off the rendered edge. A
    # fixed character count against the figure's inch width is what fits.
    body = textwrap.fill(footer, width=int(13.5 * fig.get_size_inches()[0]))
    fig.text(0.012, 0.012, f'{body}\n{DISCLAIMER}',
             fontsize=6.6, va='bottom', ha='left', color='0.35')

    n_lines = body.count('\n') + 2
    fig.subplots_adjust(left=0.085, right=0.945, top=0.84,
                        bottom=0.07 + 0.020 * n_lines)
    out = Path(run_dir) / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info('wrote %s', out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True, help='an lme4 domain run')
    ap.add_argument('--drop-domain', nargs='*', default=None,
                    help='domains to leave OFF the figure, e.g. Control')
    ap.add_argument('--fdr-q', type=float, default=0.05)
    ap.add_argument('--out', default='fig_domain_lmer_heatmap.png')
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    run_dir = Path(args.run_dir)
    df = load_slopes(run_dir, args.fdr_q, drop_domain=args.drop_domain)

    # Order from the constants, INTERSECTED with what the run actually has, so
    # a scheme with fewer domains or bands plots without a row of blanks.
    domains = [d for d in DOMAIN_ORDER if d in set(df['domain'])]
    extra = sorted(set(df['domain']) - set(DOMAIN_ORDER))
    if extra:
        logger.warning('domains not in DOMAIN_ORDER, appended: %s', extra)
        domains += extra
    bands = [b for b in BAND_ORDER if b in set(df['band'])]

    logger.info('%d domains x %d bands = %d cells', len(domains), len(bands),
                len(domains) * len(bands))

    out = figure(run_dir, df, domains, bands, args, args.out)

    # The BH column is computed HERE, so it has to be saved here too -- a figure
    # whose significance marks cannot be traced to a table is not checkable.
    table_name = Path(args.out).with_suffix('').name.replace('fig_', 'table_')
    io.write_table(
        df, run_dir / f'{table_name}.csv',
        params={'fdr_q': args.fdr_q, 'dropped_domains': args.drop_domain or [],
                'bh_family': 'every cell on the figure',
                'source': str(run_dir / 'domain_slopes.csv')},
        parents=[str(run_dir / 'domain_slopes.csv')], script=SCRIPT,
        extra={'status': DISCLAIMER})
    io.log_analysis('domain x band heatmap from the lme4 domain model', run_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())

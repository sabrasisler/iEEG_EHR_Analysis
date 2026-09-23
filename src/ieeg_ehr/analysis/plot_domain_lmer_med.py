#!/usr/bin/env python3
"""Panels F1 and F2 for an lme4 MEDICATION domain run.

    python -m ieeg_ehr.analysis.plot_domain_lmer_med --run-dir <lmer med run> \
        [--figures F1 F2]

The lme4 counterpart of `domain_med_cells.figure_f1` / `figure_f2`, drawn to the
same layout (one panel per domain, bands as rows, shared x) so the two engines
can be laid side by side. The MODEL differs, and so does how each end is read:

    log10_power ~ 0 + domain:med + domain:med:NRS_within
                  + domain:NRS_submean + med_submean
                  + (1 + NRS_within || ROI) + (1 + NRS_within + med_within || subject)
                  + (1 + NRS_within || subj_roi) + (1 | chan_id)

F1 -- `domain_med_effect.csv`: emmeans(~ med | domain) at NRS_within = 0,
on-minus-off. The dosed shift in power AT THE PATIENT'S OWN MEAN PAIN. Only runs
fitted after the R stage started writing `<band>_medeff.csv` have it.

F2 -- `domain_slopes.csv`: emtrends(~ med | domain). Unlike the per-domain
statsmodels fit, `med` is a CELL-MEANS FACTOR here, so the undosed and dosed
pain slopes are each ESTIMATED directly -- there is no pbar and nothing derived
from a centred interaction. The star comes from `domain_pairs.csv` (off - on
slope), which is the interaction.

BH IS APPLIED HERE across the cells on the figure, one family per figure, and
saved beside it as `table_F1_*.csv` / `table_F2_*.csv`. Intervals are the
emmeans CIs on the run's own df (Satterthwaite unless fitinfo says otherwise).

EXPLORATORY. Discovery cohort only. Nominations, not findings.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import io
from ieeg_ehr.analysis import cluster_permutation as cp
from ieeg_ehr.analysis.domain_med_cells import (_band_labels, _grid,
                                                subject_dosed_fraction)
from ieeg_ehr.analysis.plot_domain_anatomy import DOMAIN_HUES, OTHER_GREY
from ieeg_ehr.analysis.plot_domain_lmer import BAND_ORDER, DOMAIN_ORDER

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_domain_lmer_med.py'

#: The coverage glass brain's hues (`plot_domain_anatomy`), so a domain is the
#: same colour here as on the anatomy figure. Imported, not copied: a palette
#: change there must recolour these too. Control takes that figure's grey.
DOMAIN_COLOURS = dict(zip([d for d in DOMAIN_ORDER if d != 'Control'],
                          DOMAIN_HUES), Control=OTHER_GREY)

DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. '
              'Not confirmed out of sample.')

MODEL_NOTE = (
    'ONE lme4 FIT PER BAND with every domain in it (cell means over domain x '
    'med): log10_power ~ 0 + domain:med + domain:med:NRS_within + '
    'domain:NRS_submean + med_submean + (1 + NRS_within || ROI) + '
    '(1 + NRS_within + med_within || subject) + (1 + NRS_within || subj_roi) '
    '+ (1 | chan_id). The ROI random effect is CROSSED with subject, which the '
    'statsmodels per-domain figures could not fit, so a domain estimate is '
    'guarded against one well-sampled ROI carrying it. `med` is the raw epoch '
    'dose state; med_submean holds the between-patient exposure difference out, '
    'so on-vs-off is a WITHIN-patient contrast. Medication is not randomised, '
    'it is given BECAUSE of pain: an association with the DOSED STATE, not a '
    'drug effect. Intervals are 95% emmeans CIs on the fit\'s df. ')


def bh(df, fdr_q):
    df = df.copy()
    df['p_bh'] = np.nan
    m = df['p.value'].notna()
    if m.any():
        _, adj = cp.bh_fdr(df.loc[m, 'p.value'].to_numpy(), q=fdr_q)
        df.loc[m, 'p_bh'] = adj
    df['p_bh_reject'] = df['p_bh'] <= fdr_q
    return df


def coverage(run_dir, domains):
    """Per-domain channels, subjects and pbar, from the frames the run read."""
    prov = json.loads((run_dir / 'provenance.json').read_text())
    frames_dir = Path(prov['params']['frames_dir'])
    parts = sorted(frames_dir.glob('*.parquet'))
    if not parts:
        logger.warning('no frames at %s; titles will lack coverage', frames_dir)
        return {}
    # Coverage is band-invariant (same rows in every frame), so one suffices.
    df = io.read_table(parts[0], on_stale='warn')
    out = {}
    for dom in domains:
        d = df[df['domain'] == dom]
        out[dom] = dict(
            n_channels=int(d['channel_uid'].nunique()),
            n_subjects=int(d['subject'].nunique()),
            pbar=(subject_dosed_fraction(d)
                  if {'epoch_id', 'med_state'} <= set(d.columns)
                  else float('nan')))
    return out


def _finish(fig, run_dir, name, title, note):
    import matplotlib.pyplot as plt
    fig.suptitle(title, fontsize=11.5)
    fig.tight_layout(rect=(0, 0.20, 1, 0.88))
    fig.text(0.01, 0.005, note + DISCLAIMER, fontsize=6.3, va='bottom',
             ha='left', color='0.35', wrap=True)
    out = run_dir / name
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info('wrote %s', out)


def _save_table(df, run_dir, name, source, args, family):
    io.write_table(
        df, run_dir / name,
        params={'fdr_q': args.fdr_q, 'bh_family': family,
                'source': str(run_dir / source)},
        parents=[str(run_dir / source)], script=SCRIPT,
        extra={'status': DISCLAIMER})


def figure_f1(run_dir, domains, bands, cov, subtitle, args):
    src = run_dir / 'domain_med_effect.csv'
    if not src.exists():
        raise SystemExit(f'{src} is missing: this run predates the R-side '
                         'med-effect contrast. Refit to get F1.')
    df = io.read_table(src, on_stale='refuse')
    df = bh(df[df['domain'].isin(domains)], args.fdr_q)

    fig, axes = _grid(domains, bands)
    xmax = float(np.nanmax(np.abs(df[['lower.CL', 'upper.CL']].to_numpy()))) * 1.08
    y = np.arange(len(bands))
    for ax, dom in zip(axes, domains):
        colour = DOMAIN_COLOURS.get(dom, '0.4')
        d = df[df['domain'] == dom].set_index('band').reindex(bands)
        for i, band in enumerate(bands):
            r = d.loc[band]
            if not np.isfinite(r.get('estimate', np.nan)):
                continue
            sig = bool(r['p_bh_reject'])
            ax.errorbar(r['estimate'], i,
                        xerr=[[r['estimate'] - r['lower.CL']],
                              [r['upper.CL'] - r['estimate']]],
                        fmt='o', ms=8 if sig else 5.5, lw=1.8 if sig else 1.1,
                        capsize=3, color=colour,
                        markerfacecolor=colour if sig else 'white', zorder=3)
        ax.axvline(0, color='0.5', lw=1.0, ls='--', zorder=1)
        c = cov.get(dom)
        ax.set_title(f'{dom}\n{c["n_channels"]} chan, {c["n_subjects"]} subj, '
                     f'{c["pbar"]:.0%} dosed' if c else dom,
                     fontsize=9, color=colour)
        ax.set_xlim(-xmax, xmax)
        ax.set_xlabel('d log10 power when dosed', fontsize=8)
        ax.tick_params(labelsize=7)
    _band_labels(axes[0], bands, y)

    _finish(fig, run_dir, 'fig_F1_med_effect.png',
            'F1 -- MEDICATION effect on power, lme4 domain model (one fit per '
            'band)\nshift in power when dosed, at the subject\'s own average '
            f'pain; filled = BH-significant across the {len(df)} domain x band '
            f'cells' + (f'\n{subtitle}' if subtitle else ''),
            MODEL_NOTE
            + 'F1 IS emmeans(~ med | domain) AT NRS_within = 0, on minus off. '
            'NRS_within is subject-mean-centred, so this is the dosed shift AT '
            'THAT PATIENT\'S OWN MEAN PAIN -- not zero pain, not the cohort '
            'mean. NRS_submean and med_submean do not interact with med and '
            'cancel out of the contrast. "% dosed" is the mean over subjects of '
            'each subject\'s dosed epoch proportion. BH runs across every cell '
            'on this figure; the interaction (F2) is a separate family. '
            'X SCALE IS SHARED ACROSS DOMAINS. ')
    _save_table(df, run_dir, 'table_F1_med_effect.csv', src.name, args,
                'every domain x band cell on F1')


def figure_f2(run_dir, domains, bands, cov, subtitle, args):
    from matplotlib.lines import Line2D

    sl = io.read_table(run_dir / 'domain_slopes.csv', on_stale='refuse')
    pr = io.read_table(run_dir / 'domain_pairs.csv', on_stale='refuse')
    sl = sl[sl['domain'].isin(domains)]
    pr = bh(pr[pr['domain'].isin(domains)], args.fdr_q)
    col = 'NRS_within.trend'
    und = sl[sl['med'] == 'off'].set_index(['domain', 'band'])
    dos = sl[sl['med'] == 'on'].set_index(['domain', 'band'])
    inter = pr.set_index(['domain', 'band'])

    fig, axes = _grid(domains, bands, width=2.9)
    xmax = float(np.nanmax(np.abs(sl[['lower.CL', 'upper.CL']].to_numpy()))) * 1.08
    y = np.arange(len(bands))
    OFF = 0.17
    for ax, dom in zip(axes, domains):
        colour = DOMAIN_COLOURS.get(dom, '0.4')
        for i, band in enumerate(bands):
            key = (dom, band)
            if key not in und.index or key not in dos.index:
                continue
            u, d = und.loc[key], dos.loc[key]
            sig = bool(inter.loc[key, 'p_bh_reject']) if key in inter.index \
                else False
            ax.plot([u[col], d[col]], [i - OFF, i + OFF], '-', color=colour,
                    lw=2.0 if sig else 0.9, alpha=0.9 if sig else 0.35, zorder=2)
            for r, yy, fmt, fc, ms in ((u, i - OFF, 'o', 'white', 5.5),
                                       (d, i + OFF, 'D', colour, 5.0)):
                ax.errorbar(r[col], yy,
                            xerr=[[r[col] - r['lower.CL']],
                                  [r['upper.CL'] - r[col]]],
                            fmt=fmt, ms=ms, lw=1.0, capsize=2.5, color=colour,
                            markerfacecolor=fc, zorder=3)
            if sig:
                ax.text(xmax * 0.94, i, '*', fontsize=13, color=colour,
                        ha='right', va='center', zorder=4)
        ax.axvline(0, color='0.5', lw=1.0, ls='--', zorder=1)
        for i in range(len(bands) - 1):
            ax.axhline(i + 0.5, color='0.9', lw=0.6, zorder=0)
        c = cov.get(dom)
        ax.set_title(f'{dom}\n{c["n_channels"]} chan, {c["n_subjects"]} subj'
                     if c else dom, fontsize=9, color=colour)
        ax.set_xlim(-xmax, xmax)
        ax.set_xlabel('d log10 power / pain point', fontsize=8)
        ax.tick_params(labelsize=7)
    _band_labels(axes[0], bands, y)
    axes[-1].legend(handles=[
        Line2D([], [], marker='o', color='0.35', markerfacecolor='white',
               ls='none', ms=6, label='undosed'),
        Line2D([], [], marker='D', color='0.35', markerfacecolor='0.35',
               ls='none', ms=5.5, label='dosed'),
        Line2D([], [], color='0.35', lw=2.0,
               label='* interaction BH-significant')],
        # Outside the panel: inside, it sat on the Modulatory high-band rows.
        loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=6.5,
        frameon=True, framealpha=0.9)

    _finish(fig, run_dir, 'fig_F2_pain_slope_by_dose.png',
            'F2 -- the PAIN SLOPE undosed vs dosed, lme4 domain model\n'
            'each end is emtrends(~ med | domain), estimated directly (no pbar); '
            f'* = off-minus-on slope BH-significant across {len(pr)} cells'
            + (f'\n{subtitle}' if subtitle else ''),
            MODEL_NOTE
            + 'THE TWO ENDS ARE TWO CELLS OF ONE FIT, not two fits: `med` is a '
            'cell-means factor, so each dose state has its own pain slope and '
            'no pbar is involved (unlike the statsmodels per-domain F2, which '
            'derived both ends from a centred interaction). The connector '
            'length IS the off-minus-on slope difference; its test is '
            'emmeans pairs() and SIGNIFICANCE IS MARKED FROM THAT CONTRAST\'S '
            'BH DECISION, not by comparing the two intervals -- overlapping '
            'intervals here are not a test of anything. BH runs across every '
            'cell on this figure; F1 is a separate family. X SCALE IS SHARED '
            'ACROSS DOMAINS. ')
    _save_table(pr, run_dir, 'table_F2_pain_slope_interaction.csv',
                'domain_pairs.csv', args, 'every domain x band cell on F2')


def drug_set_of(frames_dir):
    """The drug set, from the frames' level-4 scheme name (`...-opioids-...`).

    The frames-only provenance does not record it, and a silent default here
    once meant an opioid run would have been titled "analgesics". Matched on
    whole `-`-separated tokens, because 'analgesics' is a substring of
    'non_opioid_analgesics'.
    """
    from ieeg_ehr.analysis.med_state import DRUG_SETS
    tokens = frames_dir.parent.parent.name.split('-')
    hits = [t for t in tokens if t in DRUG_SETS]
    if len(hits) != 1:
        raise SystemExit(f'cannot read the drug set from {frames_dir}; '
                         'pass --drug-set')
    return hits[0]


#: 7.5 x 7.5 in, the size asked for; 300 dpi as the anatomy combined figure.
GRID_FIGSIZE = (7.5, 7.5)
GRID_DPI = 300
BAND_TEXT = {'high_gamma': 'high gamma'}


#: Vertical layout in INCHES, so one row or two come out at the same panel size
#: and the same type: margins above the first row (legend, heading, domain
#: names, coverage line), between rows (x ticks + label, next heading, domain
#: names), and below the last (x ticks + label, footnote). Two rows at these
#: values total exactly GRID_FIGSIZE's 7.5 in.
GRID_TOP_IN = 0.98
GRID_GAP_IN = 1.27
GRID_BOTTOM_IN = 1.09
GRID_AXES_H_IN = (GRID_FIGSIZE[1] - GRID_TOP_IN - GRID_GAP_IN
                  - GRID_BOTTOM_IN) / 2


def figure_grid(run_dir, domains, bands, cov, drug_set, args, dropped=(),
                rows=('F1', 'F2'), dx=None):
    """F1 over F2 as one figure: rows are the readouts, columns the domains.

    `rows=('F1', 'F2')` is the 2 x 4, 7.5 x 7.5 in figure. `rows=('F2',)`
    draws the F2 row alone at the same width, panel size and type -- for runs
    fitted before the R stage saved the med effect, which have no F1.

    The HOUSE STYLE of `plot_domain_anatomy.combined` -- grouped sizes,
    light-grey left/bottom spines, black ticks, no grid, domain-hued titles --
    and no footnote essay: the model and BH families are stated once in a
    short note, and the full record is in the tables beside it.

    Each ROW shares its x scale (the two rows are in different units), and all
    panels share the band axis.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    from matplotlib.transforms import blended_transform_factory
    from ieeg_ehr.med_analysis import style
    from ieeg_ehr.med_analysis.plot_poster_epoch_meds import GROUPED_SIZES

    rows = tuple(rows)
    # DIAGNOSIS MODE (`dx` = {'n_case', 'n_control', 'window'}): the same F2
    # row, with the two slopes being MDD- and MDD+ instead of undosed/dosed.
    # There is NO F1 for a diagnosis run and asking for one is refused: its
    # analogue would be the between-patient difference in power LEVEL, which
    # this run does not save and which is confounded with per-channel gain and
    # impedance (additive in log power -- the same place a level contrast
    # lives). A within-patient medication shift does not have that problem.
    if dx is not None and 'F1' in rows:
        raise SystemExit('a diagnosis run has no F1; draw --figures grid_f2')
    f1 = None
    if 'F1' in rows:
        f1_src = run_dir / 'domain_med_effect.csv'
        if not f1_src.exists():
            raise SystemExit(f'{f1_src} is missing; F1 needs it. Refit, or '
                             'draw the F2 row alone (--figures grid_f2).')
        f1 = bh(io.read_table(f1_src, on_stale='refuse')
                .pipe(lambda d: d[d['domain'].isin(domains)]), args.fdr_q)
    sl = io.read_table(run_dir / 'domain_slopes.csv', on_stale='refuse')
    sl = sl[sl['domain'].isin(domains)]
    pr = io.read_table(run_dir / 'domain_pairs.csv', on_stale='refuse')
    pr = bh(pr[pr['domain'].isin(domains)], args.fdr_q)
    col = 'NRS_within.trend'
    if dx is not None:
        und = sl[sl['dx'] == 'control'].set_index(['domain', 'band'])
        dos = sl[sl['dx'] == 'case'].set_index(['domain', 'band'])
    else:
        und = sl[sl['med'] == 'off'].set_index(['domain', 'band'])
        dos = sl[sl['med'] == 'on'].set_index(['domain', 'band'])
    inter = pr.set_index(['domain', 'band'])

    n_rows = len(rows)
    W = GRID_FIGSIZE[0]
    H = (GRID_TOP_IN + n_rows * GRID_AXES_H_IN + (n_rows - 1) * GRID_GAP_IN
         + GRID_BOTTOM_IN)
    if dx is not None:
        drugs = (f'MDD coded {dx["window"]}; {dx["n_case"]} MDD+, '
                 f'{dx["n_control"]} MDD−')
    else:
        window = f'{args.med_window_hours:g} h'
        drugs = f'{drug_set.replace("_", " ")}, within {window}'
    grey = '0.3'
    spec = {
        'F1': dict(
            head='Power shift when dosed',
            xlab='Δ log$_{10}$ power, dosed − undosed (at own mean pain)',
            legend=[Line2D([], [], marker='o', color=grey,
                           markerfacecolor='white', ls='none', ms=5,
                           label='n.s.'),
                    Line2D([], [], marker='o', color=grey,
                           markerfacecolor=grey, ls='none', ms=5,
                           label='BH q < .05')]),
        'F2': dict(
            head=('Pain slope, MDD− vs MDD+' if dx is not None
                  else 'Pain slope, undosed vs dosed'),
            xlab='Pain slope (Δ log$_{10}$ power per NRS point)',
            legend=[Line2D([], [], marker='o', color=grey,
                           markerfacecolor='white', ls='none', ms=4.5,
                           label='MDD−' if dx is not None else 'undosed'),
                    Line2D([], [], marker='D', color=grey,
                           markerfacecolor=grey, ls='none', ms=4,
                           label='MDD+' if dx is not None else 'dosed'),
                    Line2D([], [], marker='$*$', color=grey, ls='none', ms=7,
                           label=('slope difference BH q < .05'
                                  if dx is not None
                                  else 'slope change BH q < .05'))]),
    }

    saved = {k: getattr(style, k) for k in GROUPED_SIZES}
    for k, v in GROUPED_SIZES.items():
        setattr(style, k, v)
    try:
        fig, axes = plt.subplots(n_rows, len(domains), figsize=(W, H),
                                 sharey=True, squeeze=False)
        fig.subplots_adjust(left=0.15, right=0.985, wspace=0.10,
                            top=1 - GRID_TOP_IN / H,
                            bottom=GRID_BOTTOM_IN / H,
                            hspace=GRID_GAP_IN / GRID_AXES_H_IN)
        y = np.arange(len(bands))
        OFF = 0.18
        EB = dict(lw=1.0, capsize=2.0, zorder=3)

        def draw_f1(ax, dom, colour):
            d = f1[f1['domain'] == dom].set_index('band').reindex(bands)
            for i, band in enumerate(bands):
                r = d.loc[band]
                if not np.isfinite(r.get('estimate', np.nan)):
                    continue
                sig = bool(r['p_bh_reject'])
                ax.errorbar(r['estimate'], i,
                            xerr=[[r['estimate'] - r['lower.CL']],
                                  [r['upper.CL'] - r['estimate']]],
                            fmt='o', ms=5, color=colour,
                            markerfacecolor=colour if sig else 'white', **EB)

        def draw_f2(ax, dom, colour):
            star = blended_transform_factory(ax.transAxes, ax.transData)
            for i, band in enumerate(bands):
                key = (dom, band)
                if key not in und.index or key not in dos.index:
                    continue
                u, v = und.loc[key], dos.loc[key]
                sig = bool(inter.loc[key, 'p_bh_reject']) \
                    if key in inter.index else False
                ax.plot([u[col], v[col]], [i - OFF, i + OFF], '-',
                        color=colour, lw=1.6 if sig else 0.8,
                        alpha=0.9 if sig else 0.35, zorder=2)
                for r, yy, fmt, fc, ms in ((u, i - OFF, 'o', 'white', 4.5),
                                           (v, i + OFF, 'D', colour, 4.0)):
                    ax.errorbar(r[col], yy,
                                xerr=[[r[col] - r['lower.CL']],
                                      [r['upper.CL'] - r[col]]],
                                fmt=fmt, ms=ms, color=colour,
                                markerfacecolor=fc, **EB)
                if sig:
                    ax.text(0.97, i, '*', transform=star, ha='right',
                            va='center', fontsize=13, color=colour, zorder=4)

        draw = {'F1': draw_f1, 'F2': draw_f2}
        limits = {'F2': float(np.nanmax(np.abs(
            sl[['lower.CL', 'upper.CL']].to_numpy()))) * 1.08}
        if f1 is not None:
            limits['F1'] = float(np.nanmax(np.abs(
                f1[['lower.CL', 'upper.CL']].to_numpy()))) * 1.08

        for ri, key in enumerate(rows):
            first = ri == 0
            for j, dom in enumerate(domains):
                colour = DOMAIN_COLOURS.get(dom, '0.4')
                ax = axes[ri][j]
                draw[key](ax, dom, colour)
                style.style_axes(ax, grid_axis=None,
                                 tick_color=style.TEXT_PRIMARY)
                ax.axvline(0, color=style.ZERO_LINE_COLOR, lw=0.8, ls='--',
                           zorder=1)
                for i in range(len(bands) - 1):
                    ax.axhline(i + 0.5, color='0.93', lw=0.6, zorder=0)
                ax.set_xlim(-limits[key], limits[key])
                ax.xaxis.set_major_locator(MaxNLocator(3, symmetric=True))
                ax.tick_params(axis='x', labelsize=style.TICK_SIZE - 1.5)
                ax.tick_params(axis='y', length=0)
                # Coverage is a property of the domain, not the readout, so
                # it is printed once, under the domain name of the top row.
                c = cov.get(dom) if first else None
                ax.set_title(dom, fontsize=style.LABEL_SIZE, color=colour,
                             pad=16 if c else 6)
                if c:
                    ax.text(0.5, 1.02, f'{c["n_subjects"]} patients, '
                            f'{c["n_channels"]} electrodes',
                            transform=ax.transAxes, ha='center', va='bottom',
                            fontsize=style.TICK_SIZE - 2,
                            color=style.TEXT_MUTED)
            axes[ri][0].set_yticks(y)
            axes[ri][0].set_yticklabels([BAND_TEXT.get(x, x) for x in bands],
                                        fontsize=style.TICK_SIZE)
            axes[ri][0].set_ylim(len(bands) - 0.5, -0.5)

            # Heading, shared x label and legend, placed in inches off the
            # laid-out axes. The first row's legend sits in the top margin;
            # a later row's shares the line with its heading.
            pos0, pos1 = axes[ri][0].get_position(), axes[ri][-1].get_position()
            letter = f'{"AB"[ri]}   ' if n_rows > 1 else ''
            head = spec[key]['head'] + (f' ({drugs})' if first else '')
            fig.text(0.015, pos0.y1 + (0.56 if first else 0.34) / H,
                     letter + head, fontsize=style.LABEL_SIZE,
                     fontweight='bold', color=style.TEXT_PRIMARY,
                     ha='left', va='bottom')
            fig.text((pos0.x0 + pos1.x1) / 2, pos0.y0 - 0.41 / H,
                     spec[key]['xlab'], ha='center', va='top',
                     fontsize=style.LABEL_SIZE - 1, color=style.TEXT_PRIMARY)
            fig.legend(handles=spec[key]['legend'],
                       loc='upper right' if first else 'lower right',
                       bbox_to_anchor=(0.985, 1 - 0.04 / H) if first
                       else (0.985, pos0.y1 + 0.30 / H),
                       ncol=len(spec[key]['legend']),
                       fontsize=style.LEGEND_SIZE, frameon=False,
                       handletextpad=0.3, columnspacing=1.0)

        n_cells = len(pr)
        fig.text(0.015, 0.09 / H,
                 'lme4, one fit per band'
                 + (f' ({", ".join(dropped)} excluded from the fit)'
                    if dropped else ', every domain in the fit')
                 + '; 95% CIs; BH '
                 + ('within each row ' if n_rows > 1 else '')
                 + f'across {"its" if n_rows > 1 else "the"} {n_cells} cells.\n' + DISCLAIMER,
                 fontsize=style.FOOTNOTE_SIZE - 1, linespacing=1.3,
                 color=style.TEXT_MUTED, ha='left', va='bottom')

        name = ('fig_F2_pain_slope_by_mdd_grid.png' if dx is not None
                else 'fig_F1F2_grid.png' if 'F1' in rows
                else 'fig_F2_pain_slope_by_dose_grid.png')
        out = run_dir / name
        fig.savefig(out, dpi=GRID_DPI, facecolor='white')
        plt.close(fig)
        logger.info('wrote %s (%.2f x %.2f in, %d dpi)', out, W, H, GRID_DPI)
    finally:
        for k, v in saved.items():
            setattr(style, k, v)

    if f1 is not None:
        _save_table(f1, run_dir, 'table_F1_med_effect.csv',
                    'domain_med_effect.csv', args,
                    'every domain x band cell on F1')
    _save_table(pr, run_dir, 'table_F2_pain_slope_interaction.csv',
                'domain_pairs.csv', args, 'every domain x band cell on F2')


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True, help='an lme4 MED domain run')
    ap.add_argument('--figures', nargs='*', default=['F1', 'F2'],
                    choices=['F1', 'F2', 'grid', 'grid_f2'])
    ap.add_argument('--fdr-q', type=float, default=0.05)
    ap.add_argument('--drug-set', default=None,
                    help='default: read from the frames scheme folder name')
    ap.add_argument('--med-window-hours', type=float, default=2.0,
                    help='label only; the frames fix the actual window')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    run_dir = Path(args.run_dir)
    prov = json.loads((run_dir / 'provenance.json').read_text())
    params = prov['params']
    is_dx = bool(params.get('dx_model'))
    if not params.get('med_model') and not is_dx:
        raise SystemExit(f'{run_dir} is neither a medication nor a diagnosis '
                         'run')

    sl = io.read_table(run_dir / 'domain_slopes.csv', on_stale='refuse')
    domains = [d for d in DOMAIN_ORDER if d in set(sl['domain'])]
    bands = [b for b in BAND_ORDER if b in set(sl['band'])]
    cov = coverage(run_dir, domains)

    if is_dx:
        # Only the F2 row exists for a diagnosis run (see figure_grid).
        if set(args.figures) - {'grid_f2'}:
            raise SystemExit('a diagnosis run draws --figures grid_f2 only')
        frames = str(params.get('frames_dir', ''))
        window = ('ever' if '-mdd0d' in frames else
                  'in the 90 d before admission' if '-mdd90d' in frames
                  else 'window unknown')
        dx = dict(n_case=params.get('dx_n_case'),
                  n_control=params.get('dx_n_control'), window=window)
        figure_grid(run_dir, domains, bands, cov, None, args,
                    params.get('dropped_domains') or [], rows=('F2',), dx=dx)
        io.log_analysis('lme4 domain MDD figure grid_f2 (pain slope, MDD- vs '
                        'MDD+)', run_dir)
        return 0

    drug_set = args.drug_set or drug_set_of(Path(params['frames_dir']))
    dropped = params.get('dropped_domains') or []
    subtitle = (f'{drug_set} within {args.med_window_hours} h of the score'
                + (f'; {", ".join(dropped)} excluded from the fit'
                   if dropped else ''))

    if 'F1' in args.figures:
        figure_f1(run_dir, domains, bands, cov, subtitle, args)
    if 'F2' in args.figures:
        figure_f2(run_dir, domains, bands, cov, subtitle, args)
    if 'grid' in args.figures:
        figure_grid(run_dir, domains, bands, cov, drug_set, args, dropped)
    if 'grid_f2' in args.figures:
        figure_grid(run_dir, domains, bands, cov, drug_set, args, dropped,
                    rows=('F2',))
    io.log_analysis(f'lme4 domain med figures {"/".join(args.figures)}',
                    run_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())

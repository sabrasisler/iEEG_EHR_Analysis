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
            pbar=(subject_dosed_fraction(d) if 'epoch_id' in d.columns
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


def figure_grid(run_dir, domains, bands, cov, drug_set, args):
    """F1 over F2 in one 2 x 4 figure: rows are the two readouts, columns the
    domains. The HOUSE STYLE of `plot_domain_anatomy.combined` -- grouped sizes,
    light-grey left/bottom spines, black ticks, no grid, domain-hued titles --
    and no footnote essay: the model and BH families are stated once in a
    one-line note, and the full record is in the two tables beside it.

    Each ROW shares its x scale (the two rows are in different units), and all
    eight panels share the band axis.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    from matplotlib.transforms import blended_transform_factory
    from ieeg_ehr.med_analysis import style
    from ieeg_ehr.med_analysis.plot_poster_epoch_meds import GROUPED_SIZES

    f1_src = run_dir / 'domain_med_effect.csv'
    if not f1_src.exists():
        raise SystemExit(f'{f1_src} is missing; the grid needs F1. Refit.')
    f1 = bh(io.read_table(f1_src, on_stale='refuse')
            .pipe(lambda d: d[d['domain'].isin(domains)]), args.fdr_q)
    sl = io.read_table(run_dir / 'domain_slopes.csv', on_stale='refuse')
    sl = sl[sl['domain'].isin(domains)]
    pr = io.read_table(run_dir / 'domain_pairs.csv', on_stale='refuse')
    pr = bh(pr[pr['domain'].isin(domains)], args.fdr_q)
    col = 'NRS_within.trend'
    und = sl[sl['med'] == 'off'].set_index(['domain', 'band'])
    dos = sl[sl['med'] == 'on'].set_index(['domain', 'band'])
    inter = pr.set_index(['domain', 'band'])

    saved = {k: getattr(style, k) for k in GROUPED_SIZES}
    for k, v in GROUPED_SIZES.items():
        setattr(style, k, v)
    try:
        fig, axes = plt.subplots(2, len(domains), figsize=GRID_FIGSIZE,
                                 sharey=True, squeeze=False)
        fig.subplots_adjust(left=0.15, right=0.985, top=0.885, bottom=0.135,
                            wspace=0.10, hspace=0.62)
        y = np.arange(len(bands))
        lim_a = float(np.nanmax(np.abs(f1[['lower.CL', 'upper.CL']]
                                       .to_numpy()))) * 1.08
        lim_b = float(np.nanmax(np.abs(sl[['lower.CL', 'upper.CL']]
                                       .to_numpy()))) * 1.08
        OFF = 0.18
        EB = dict(lw=1.0, capsize=2.0, zorder=3)

        for j, dom in enumerate(domains):
            colour = DOMAIN_COLOURS.get(dom, '0.4')
            a, b = axes[0][j], axes[1][j]

            d = f1[f1['domain'] == dom].set_index('band').reindex(bands)
            for i, band in enumerate(bands):
                r = d.loc[band]
                if not np.isfinite(r.get('estimate', np.nan)):
                    continue
                sig = bool(r['p_bh_reject'])
                a.errorbar(r['estimate'], i,
                           xerr=[[r['estimate'] - r['lower.CL']],
                                 [r['upper.CL'] - r['estimate']]],
                           fmt='o', ms=5, color=colour,
                           markerfacecolor=colour if sig else 'white', **EB)

            star = blended_transform_factory(b.transAxes, b.transData)
            for i, band in enumerate(bands):
                key = (dom, band)
                if key not in und.index or key not in dos.index:
                    continue
                u, v = und.loc[key], dos.loc[key]
                sig = bool(inter.loc[key, 'p_bh_reject']) \
                    if key in inter.index else False
                b.plot([u[col], v[col]], [i - OFF, i + OFF], '-',
                       color=colour, lw=1.6 if sig else 0.8,
                       alpha=0.9 if sig else 0.35, zorder=2)
                for r, yy, fmt, fc, ms in ((u, i - OFF, 'o', 'white', 4.5),
                                           (v, i + OFF, 'D', colour, 4.0)):
                    b.errorbar(r[col], yy,
                               xerr=[[r[col] - r['lower.CL']],
                                     [r['upper.CL'] - r[col]]],
                               fmt=fmt, ms=ms, color=colour,
                               markerfacecolor=fc, **EB)
                if sig:
                    b.text(0.97, i, '*', transform=star, ha='right',
                           va='center', fontsize=13, color=colour, zorder=4)

            for ax, lim in ((a, lim_a), (b, lim_b)):
                style.style_axes(ax, grid_axis=None,
                                 tick_color=style.TEXT_PRIMARY)
                ax.axvline(0, color=style.ZERO_LINE_COLOR, lw=0.8, ls='--',
                           zorder=1)
                for i in range(len(bands) - 1):
                    ax.axhline(i + 0.5, color='0.93', lw=0.6, zorder=0)
                ax.set_xlim(-lim, lim)
                ax.xaxis.set_major_locator(MaxNLocator(3, symmetric=True))
                ax.tick_params(axis='x', labelsize=style.TICK_SIZE - 1.5)
                ax.tick_params(axis='y', length=0)
            c = cov.get(dom)
            a.set_title(dom, fontsize=style.LABEL_SIZE, color=colour, pad=16)
            if c:
                a.text(0.5, 1.02, f'{c["n_subjects"]} patients, '
                       f'{c["n_channels"]} electrodes',
                       transform=a.transAxes, ha='center', va='bottom',
                       fontsize=style.TICK_SIZE - 2, color=style.TEXT_MUTED)
            b.set_title(dom, fontsize=style.LABEL_SIZE, color=colour, pad=6)

        for row in axes:
            row[0].set_yticks(y)
            row[0].set_yticklabels([BAND_TEXT.get(x, x) for x in bands],
                                   fontsize=style.TICK_SIZE)
            row[0].set_ylim(len(bands) - 0.5, -0.5)

        # Row headings and shared x labels, placed off the laid-out axes.
        window = f'{args.med_window_hours:g} h'
        heads = (f'A   Power shift when dosed ({drug_set.replace("_", " ")}, '
                 f'within {window})',
                 'B   Pain slope, undosed vs dosed')
        xlabs = ('Δ log$_{10}$ power, dosed − undosed '
                 '(at own mean pain)',
                 'Pain slope (Δ log$_{10}$ power per NRS point)')
        for row, head, xlab in zip(axes, heads, xlabs):
            top = row[0].get_position().y1
            bot = row[0].get_position().y0
            x0, x1 = row[0].get_position().x0, row[-1].get_position().x1
            fig.text(0.015, top + (0.075 if row is axes[0] else 0.045), head,
                     fontsize=style.LABEL_SIZE, fontweight='bold',
                     color=style.TEXT_PRIMARY, ha='left', va='bottom')
            fig.text((x0 + x1) / 2, bot - 0.055, xlab, ha='center',
                     va='top', fontsize=style.LABEL_SIZE - 1,
                     color=style.TEXT_PRIMARY)

        grey = '0.3'
        fig.legend(handles=[
            Line2D([], [], marker='o', color=grey, markerfacecolor='white',
                   ls='none', ms=5, label='n.s.'),
            Line2D([], [], marker='o', color=grey, markerfacecolor=grey,
                   ls='none', ms=5, label='BH q < .05')],
            loc='upper right', bbox_to_anchor=(0.985, 0.995), ncol=2,
            fontsize=style.LEGEND_SIZE, frameon=False, handletextpad=0.3,
            columnspacing=1.0)
        b_top = axes[1][0].get_position().y1
        fig.legend(handles=[
            Line2D([], [], marker='o', color=grey, markerfacecolor='white',
                   ls='none', ms=4.5, label='undosed'),
            Line2D([], [], marker='D', color=grey, markerfacecolor=grey,
                   ls='none', ms=4, label='dosed'),
            Line2D([], [], marker='$*$', color=grey, ls='none', ms=7,
                   label='slope change BH q < .05')],
            loc='lower right', bbox_to_anchor=(0.985, b_top + 0.040), ncol=3,
            fontsize=style.LEGEND_SIZE, frameon=False, handletextpad=0.3,
            columnspacing=1.0)

        fig.text(0.015, 0.012,
                 'lme4, one fit per band, all domains in the fit; 95% CIs; BH '
                 f'within each row across its {len(f1)} cells. '
                 + DISCLAIMER, fontsize=style.FOOTNOTE_SIZE - 1,
                 color=style.TEXT_MUTED, ha='left', va='bottom')

        out = run_dir / 'fig_F1F2_grid.png'
        fig.savefig(out, dpi=GRID_DPI, facecolor='white')
        plt.close(fig)
        logger.info('wrote %s (%.1f x %.1f in, %d dpi)', out, *GRID_FIGSIZE,
                    GRID_DPI)
    finally:
        for k, v in saved.items():
            setattr(style, k, v)

    _save_table(f1, run_dir, 'table_F1_med_effect.csv', f1_src.name, args,
                'every domain x band cell on F1')
    _save_table(pr, run_dir, 'table_F2_pain_slope_interaction.csv',
                'domain_pairs.csv', args, 'every domain x band cell on F2')


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True, help='an lme4 MED domain run')
    ap.add_argument('--figures', nargs='*', default=['F1', 'F2'],
                    choices=['F1', 'F2', 'grid'])
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
    if not params.get('med_model'):
        raise SystemExit(f'{run_dir} is not a medication run')

    sl = io.read_table(run_dir / 'domain_slopes.csv', on_stale='refuse')
    domains = [d for d in DOMAIN_ORDER if d in set(sl['domain'])]
    bands = [b for b in BAND_ORDER if b in set(sl['band'])]
    cov = coverage(run_dir, domains)

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
        figure_grid(run_dir, domains, bands, cov, drug_set, args)
    io.log_analysis(f'lme4 domain med figures {"/".join(args.figures)}',
                    run_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())

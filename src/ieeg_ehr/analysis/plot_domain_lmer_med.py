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
from ieeg_ehr.analysis.plot_domain_lmer import BAND_ORDER, DOMAIN_ORDER
from ieeg_ehr.analysis.run_domain_model import DOMAIN_COLOURS

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_domain_lmer_med.py'

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


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True, help='an lme4 MED domain run')
    ap.add_argument('--figures', nargs='*', default=['F1', 'F2'],
                    choices=['F1', 'F2'])
    ap.add_argument('--fdr-q', type=float, default=0.05)
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

    frames_prov = Path(params['frames_dir']).parent / 'provenance.json'
    fp = json.loads(frames_prov.read_text())['params'] \
        if frames_prov.exists() else {}
    dropped = params.get('dropped_domains') or []
    subtitle = (f'{fp.get("drug_set", "analgesics")} within '
                f'{fp.get("med_window_hours", 2.0)} h of the score'
                + (f'; {", ".join(dropped)} excluded from the fit'
                   if dropped else ''))

    if 'F1' in args.figures:
        figure_f1(run_dir, domains, bands, cov, subtitle, args)
    if 'F2' in args.figures:
        figure_f2(run_dir, domains, bands, cov, subtitle, args)
    io.log_analysis(f'lme4 domain med figures {"/".join(args.figures)}',
                    run_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())

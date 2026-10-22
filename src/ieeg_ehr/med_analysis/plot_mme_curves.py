"""
Estimated active MME over each session, one subplot per session with an opioid.

Model and caveats: `mme_curve.py`. Factors and PK parameters, with sources:
`config/opioid_mme.py`.

One panel per (subject, session). That is one per subject for every subject but
sub-209 and sub-255, who have two admissions each: times only compare within a
session, so their two admissions get two panels. Sessions without a single
opioid dose get no panel and are logged.

Each panel:
- the active-MME curve on a 1-min grid, hours since `session_start`
- a vertical line at every opioid dose, colored by ingredient, solid for PO
  (oral and feeding tube) and dashed for IV
- a gray vertical line at every non-opioid analgesic dose (acetaminophen,
  NSAIDs), same route styles
- light shading over recorded iEEG time, where the file registry carries
  gap-aware run times (`recording_hours`); no shading where it does not
- `*` in the title when the first opioid comes within 2 h of session start:
  pre-session (OR) doses are not in the export, so that curve starts low

Tables beside the figure: the MME dose table, the unit-dose checks, the
per-session totals against a raw-file tally, and a per-session summary.

Run on Slurm, never the login node:
    python -m ieeg_ehr.med_analysis.plot_mme_curves
"""

import argparse
import logging

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ieeg_ehr import config, io
from ieeg_ehr.config import med_taxonomy, opioid_mme
from ieeg_ehr.med_analysis import load, mme_curve, output, recording_hours, style
from ieeg_ehr.med_analysis.style import plt

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_TYPE = 'mme_curves'
SCRIPT = 'ieeg_ehr/med_analysis/plot_mme_curves.py'

NCOLS = 6
PANEL_W, PANEL_H = 4.0, 2.1

ROUTE_STYLE = {'PO': '-', 'IV': '--'}
NON_OPIOID_COLOR = '#b4b2a9'
CURVE_COLOR = style.TEXT_PRIMARY
RECORDING_SHADE = '#dceaf7'

#: Every non-opioid analgesic subclass, from the taxonomy.
NON_OPIOID_SUBCLASSES = tuple(s for s in med_taxonomy.ANALGESIC_SUBCLASSES
                              if s != 'Opioids')


def session_summary(doses, sessions):
    """One row per plotted session: totals, peak, early flag, integral check."""
    early = mme_curve.early_opioid_sessions(doses)
    rows = []
    for (subject, session), d in doses.groupby(['subject', 'session']):
        s = sessions[(subject, session)]
        t, curve = s['t_h'], s['curve']
        area, expected = mme_curve.integral_check(
            d, s['session_start'], d['session_end'].iloc[0])
        rows.append(dict(
            subject=subject, session=session, n_opioid_doses=len(d),
            total_mme=float(d['mme'].sum()),
            peak_active_mme=float(curve.max()),
            peak_t_h=float(t[int(np.argmax(curve))]),
            first_opioid_h=float(d['t_h'].min()),
            early_opioid=(subject, session) in early,
            session_h=float(t[-1]),
            curve_area_mme_h=area, expected_area_mme_h=expected,
            integral_ok=bool(abs(area - expected) <= 1e-3 * max(expected, 1.0)),
            ingredients=';'.join(sorted(d['ingredient'].unique()))))
    return rows


def plot_grid(doses, non_opioid, sessions, coverage, summary, out_path):
    keys = list(sessions)
    nrows = int(np.ceil(len(keys) / NCOLS))
    fig, axes = plt.subplots(nrows, NCOLS, squeeze=False,
                             figsize=(PANEL_W * NCOLS, PANEL_H * nrows + 0.8))
    flat = axes.ravel()
    for ax in flat[len(keys):]:
        ax.set_visible(False)

    ingredients = [i for i in opioid_mme.INGREDIENT_ORDER
                   if i in set(doses['ingredient'])]
    colors = style.categorical_colors(ingredients)
    by_key = {(r['subject'], r['session']): r for r in summary}

    for ax, key in zip(flat, keys):
        s = sessions[key]
        cov = coverage.get(key)
        if cov is not None:
            start = s['session_start']
            for a, b in cov:
                ax.axvspan((a - start).total_seconds() / 3600.0,
                           (b - start).total_seconds() / 3600.0,
                           color=RECORDING_SHADE, lw=0, zorder=0)

        for d in non_opioid.get(key, ()):
            ax.axvline(d['t_h'], color=NON_OPIOID_COLOR, lw=0.6,
                       ls=ROUTE_STYLE[d['route_class']], zorder=1)
        for d in s['doses'].itertuples():
            ax.axvline(d.t_h, color=colors[d.ingredient], lw=0.7, alpha=0.9,
                       ls=ROUTE_STYLE[d.route_class], zorder=2)

        ax.plot(s['t_h'], s['curve'], color=CURVE_COLOR, lw=1.0, zorder=3)
        ax.set_ylim(0, max(float(s['curve'].max()) * 1.1, 1.0))
        span = s['t_h'][-1]
        ax.set_xlim(min(0.0, s['t_h'][0]), span)
        ax.set_xticks(np.arange(0, span + 1, 24 if span <= 168 else 48))
        style.style_axes(ax)

        row = by_key[key]
        flag = '*' if row['early_opioid'] else ''
        ax.set_title(f'sub-{key[0]} ses-{key[1]}{flag}   '
                     f'{row["total_mme"]:.0f} MME / {row["n_opioid_doses"]} dose'
                     f'{"s" if row["n_opioid_doses"] != 1 else ""}',
                     fontsize=style.TICK_SIZE, color=style.TEXT_PRIMARY,
                     loc='left', pad=3)

    for ax in axes[:, 0]:
        ax.set_ylabel('Active MME (mg)', fontsize=style.TICK_SIZE,
                      color=style.TEXT_PRIMARY)
    for ax in flat[max(len(keys) - NCOLS, 0):len(keys)]:
        ax.set_xlabel('Hours since session start', fontsize=style.TICK_SIZE,
                      color=style.TEXT_PRIMARY)

    handles = [Line2D([], [], color=CURVE_COLOR, lw=1.2,
                      label='Estimated active MME')]
    handles += [Line2D([], [], color=colors[i], lw=1.2, label=i.title())
                for i in ingredients]
    handles += [Line2D([], [], color=NON_OPIOID_COLOR, lw=1.2,
                       label='Non-opioid analgesic'),
                Line2D([], [], color=style.TEXT_MUTED, ls='-', lw=1.0,
                       label='PO / feeding tube'),
                Line2D([], [], color=style.TEXT_MUTED, ls='--', lw=1.0,
                       label='IV'),
                plt.Rectangle((0, 0), 1, 1, color=RECORDING_SHADE,
                              label='Recorded iEEG (registry)')]
    fig.legend(handles=handles, loc='upper center', ncol=len(handles),
               fontsize=style.LEGEND_SIZE, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle('Estimated active opioid exposure (MME), one panel per session',
                 fontsize=style.TITLE_SIZE, color=style.TEXT_PRIMARY,
                 x=0.005, ha='left', y=1.0 + 0.35 / fig.get_figheight())
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))
    return style.save(fig, out_path, footnote=(
        '* first opioid within 2 h of session start: pre-session (OR) doses '
        'are not in the MAR export, so the early curve is a lower bound.  '
        'PO/IV bolus one-compartment, IV fentanyl three-compartment, '
        'effect site for fentanyl, hydromorphone and morphine, every dose '
        'peaking at its MME; '
        'MME factors CDC 2022 (oral), NDARC TR 329 (IV).'))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    output.add_output_arguments(parser)
    parser.set_defaults(question=config.MED_PK_QUESTION)
    args = parser.parse_args()

    io.warn_if_dirty()

    paths = config.med_admin_files()
    admin = load.load_administrations(
        paths=paths, subclasses=med_taxonomy.ANALGESIC_SUBCLASSES)

    doses = mme_curve.opioid_doses(admin)
    checks = mme_curve.unit_dose_checks()
    totals = mme_curve.totals_check(doses, paths)
    logger.info('unit-dose checks:\n%s', checks.to_string(index=False))
    bad_totals = totals[(totals['mme_diff'].abs() > 1e-6) | (totals['n_diff'] != 0)]
    logger.info('total MME: pipeline %.1f vs raw tally %.1f; %d session(s) '
                'differ:\n%s', totals['mme_pipeline'].sum(),
                totals['mme_raw'].sum(), len(bad_totals),
                bad_totals.to_string(index=False))

    sessions = {}
    for (subject, session), d in doses.groupby(['subject', 'session']):
        start, end = d['session_start'].iloc[0], d['session_end'].iloc[0]
        t = mme_curve.time_grid(start, end, d)
        sessions[(subject, session)] = dict(
            doses=d, t_h=t, curve=mme_curve.exposure_curve(d, t),
            session_start=start)
    without = sorted(set(zip(admin['subject'], admin['session'])) - set(sessions))
    logger.info('%d session(s) with an opioid dose get a panel; %d analgesic '
                'session(s) without one do not: %s', len(sessions),
                len(without), without)

    non = admin[admin['level2'].isin(NON_OPIOID_SUBCLASSES)].copy()
    non['route_class'] = non['route'].map(opioid_mme.ROUTE_CLASS)
    unmapped_routes = sorted(set(non.loc[non['route_class'].isna(), 'route']))
    if unmapped_routes:
        raise KeyError(f'non-opioid route(s) with no line style: {unmapped_routes}')
    non['t_h'] = (non['taken_dt'] - non['session_start']).dt.total_seconds() / 3600.0
    non_opioid = {k: g[['t_h', 'route_class']].to_dict('records')
                  for k, g in non.groupby(['subject', 'session'])}

    cov = recording_hours.session_coverage(admin)
    coverage = {(r.subject, r.session): r.intervals for r in cov.itertuples()
                if r.method == recording_hours.METHOD_REGISTRY}

    summary = session_summary(doses, sessions)
    failed = [(r['subject'], r['session']) for r in summary if not r['integral_ok']]
    if failed or not checks['auc_ok'].all():
        raise AssertionError(f'superposition integral failed for {failed}; '
                             f'unit-dose AUC ok: {checks["auc_ok"].to_dict()}')

    run_dir = output.resolve_run_dir(args, OUTPUT_TYPE)
    plot_grid(doses, non_opioid, sessions, coverage, summary,
              run_dir / 'mme_curves_by_session.png')

    parents = output.source_parents(paths)
    subjects = sorted(doses['subject'].unique())
    common = dict(script=SCRIPT, params=vars(args), parents=parents,
                  subjects=subjects)
    output.write_table(doses.drop(columns=['session_start', 'session_end']),
                       run_dir, 'opioid_doses_mme', **common)
    output.write_table(checks, run_dir, 'unit_dose_checks', **common)
    output.write_table(totals, run_dir, 'mme_totals_vs_raw', **common)
    output.write_table(pd.DataFrame(summary), run_dir, 'session_summary', **common)

    output.write_run(
        run_dir, SCRIPT, args, doses, paths,
        extra={'n_panels': len(sessions),
               'sessions_without_opioid': [f'{s}/{e}' for s, e in without],
               'grid_step_min': mme_curve.GRID_STEP_H * 60},
        description=(f'Active-MME curves, {len(sessions)} sessions, one panel '
                     f'each; unit-dose and superposition checks'))


if __name__ == '__main__':
    main()

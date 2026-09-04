"""
Fig 7 — how often a pain assessment was followed by an analgesic.

The mirror of Fig 5, and the more clinically direct question. Fig 5 takes a
dose and asks what the score was before it; this takes an ASSESSMENT and asks
whether a dose followed within 30 minutes, and which drug. Stacked bars, one
per pain score, each summing to 100% of the assessments recorded at that score:
the coloured segments are the drug given, the grey cap is "no analgesic".

WHY THE DENOMINATOR IS THE THING TO WATCH. This figure's y axis is a share of
ASSESSMENTS, not of doses, so its bars are only as trustworthy as the count of
assessments behind them — and that count is wildly uneven: 1,700 assessments at
score 0 against 58 at score 10. Every bar therefore carries its n above it. A
tall segment over n=58 is three or four patients.

WHAT COUNTS AS OBSERVABLE. An assessment is scored only if its session has a
MAR export (otherwise "no drug" is unobserved, not false) and only if the
session does not end inside the window (otherwise absence is unearned). See
`pain_link.session_bounds` and `pain_link.response_by_assessment`; the counts
that were dropped are on the figure.

TWO ASSESSMENTS INSIDE ONE WINDOW. Settled by attribution rather than by
guessing: each dose belongs to its nearest preceding assessment, which is the
same thing as truncating a window at the next assessment, so nothing is
double-counted and the percentages partition. Full argument in
`pain_link.response_by_assessment.__doc__`. `--exclude-clustered` re-runs
without the 7.8% of assessments that have a neighbour inside the window.

NOT CAUSAL, and the direction here is even more tempting to over-read than in
Fig 5. A dose following an assessment does not mean the assessment caused it:
scheduled drugs land on a clock regardless, and the assessment is frequently
charted BECAUSE the dose was being given. This measures co-occurrence in the
chart, in one direction, nothing more.

Run on Slurm, never the login node:
    python -m ieeg_ehr.med_analysis.plot_pain_score_response
"""

import argparse
import logging

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.config import med_taxonomy
from ieeg_ehr.med_analysis import load, output, pain_link, style
from ieeg_ehr.med_analysis.style import plt

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_TYPE = 'pain_score_response'
SCRIPT = 'ieeg_ehr/med_analysis/plot_pain_score_response.py'

#: The named drugs, in the order used by Figs 5 and 6 so a colour means the
#: same drug across the whole set.
DEFAULT_DRUGS = ('ACETAMINOPHEN', 'HYDROCODONE-ACETAMINOPHEN', 'OXYCODONE',
                 'HYDROMORPHONE', 'FENTANYL', 'KETOROLAC', 'TRAMADOL')

#: Categories that are not a single named drug. `OTHER` keeps "no analgesic"
#: honest — without it a morphine dose would read as nothing having happened.
#: `MULTI` exists because 4.8% of responded-to assessments have two distinct
#: drugs, and a stacked bar has to assign each assessment exactly one segment
#: for the bar to mean 100% of assessments.
OTHER = 'Other analgesic'
MULTI = 'Two analgesics'
NONE = 'No analgesic'


def categorize(per_assessment, drugs):
    """One segment per assessment: the drug, OTHER, MULTI, or NONE."""
    named = set(drugs)

    def one(row):
        if row.n_drugs == 0:
            return NONE
        if row.n_drugs > 1:
            return MULTI
        drug = row.drugs[0]
        return drug if drug in named else OTHER

    out = per_assessment.copy()
    out['category'] = [one(r) for r in out.itertuples()]
    return out


def response_table(categorized, drugs):
    """pain_score x category counts over the full 0-10 scale, plus percentages."""
    order = list(drugs) + [OTHER, MULTI, NONE]
    counts = (categorized.pivot_table(index='pain_score', columns='category',
                                      values='score_dt', aggfunc='count')
              .reindex(range(pain_link.PAIN_SCORE_MIN,
                             pain_link.PAIN_SCORE_MAX + 1))
              .reindex(columns=order)
              .fillna(0).astype(int))
    counts.index.name = 'pain_score'
    totals = counts.sum(axis=1)
    # A score nobody was ever assessed at has no denominator; leave it NaN
    # rather than dividing by zero and drawing an empty 0% bar as if measured.
    pct = counts.divide(totals.replace(0, np.nan), axis=0) * 100.0
    return counts, pct, totals


def plot_stacked(counts, pct, totals, out_path, drugs, *, window_minutes,
                 n_subjects, stats):
    order = list(drugs) + [OTHER, MULTI, NONE]
    colors = style.categorical_colors(list(drugs) + [OTHER])
    colors[MULTI] = style.TEXT_MUTED      # a mixture, so deliberately neutral
    colors[NONE] = style.GRID_COLOR       # absence, the lightest thing here

    scores = counts.index.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11.5, 6.5))

    # Category keys are the raw MAR drug names; the legend shows them title-cased
    # so they sit consistently beside "Other analgesic" / "No analgesic".
    label_of = {cat: (cat if cat in (OTHER, MULTI, NONE) else cat.title())
                for cat in order}

    bottom = np.zeros(len(scores))
    for cat in order:
        vals = pct[cat].fillna(0).to_numpy()
        ax.bar(scores, vals, bottom=bottom, width=0.72, color=colors[cat],
               label=label_of[cat], zorder=3, edgecolor='white', linewidth=0.4)
        bottom += vals

    # The share with ANY analgesic is the top of the coloured stack, which is
    # the number this figure exists to report — labelled so it is read off
    # rather than estimated against the grid.
    any_pct = 100.0 - pct[NONE].fillna(0).to_numpy()
    for x, y, n in zip(scores, any_pct, totals.to_numpy()):
        if n == 0:
            continue
        ax.annotate(f'{y:.0f}%', (x, y), textcoords='offset points',
                    xytext=(0, 3), ha='center', va='bottom',
                    fontsize=style.TICK_SIZE, color=style.TEXT_PRIMARY)
        ax.annotate(f'n={n}', (x, 101), ha='center', va='bottom',
                    fontsize=style.FOOTNOTE_SIZE + 1, color=style.TEXT_MUTED)

    ax.set_xticks(range(pain_link.PAIN_SCORE_MIN, pain_link.PAIN_SCORE_MAX + 1))
    ax.set_xlim(pain_link.PAIN_SCORE_MIN - 0.6, pain_link.PAIN_SCORE_MAX + 0.6)
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))

    style.style_axes(ax, grid_axis='y')
    style.label_axes(
        ax, 'Pain score recorded (0-10)',
        '% of assessments at that score',
        f'Was an analgesic given within {window_minutes} min of a pain '
        f'assessment?\n'
        f'{stats["n_assessments_scored"]} assessments, {n_subjects} subjects; '
        f'{100 * stats["frac_with_any_analgesic"]:.1f}% followed by an '
        f'analgesic overall')
    leg = ax.legend(title='Given within the window', frameon=False,
                    fontsize=style.LEGEND_SIZE,
                    title_fontsize=style.LEGEND_SIZE,
                    loc='upper left', bbox_to_anchor=(1.01, 1.0))
    leg._legend_box.align = 'left'

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return style.save(
        fig, out_path,
        footnote=(
            f'n above each bar = assessments at that score, the denominator; '
            f'each dose is attributed to its nearest preceding assessment, so '
            f'no dose is counted twice\n'
            f'excluded: {stats["n_dropped_session_has_no_mar"]} '
            f'assessment{"" if stats["n_dropped_session_has_no_mar"] == 1 else "s"} '
            f'in sessions with no MAR export, '
            f'{stats["n_dropped_right_censored"]} whose window ran past session '
            f'end. NOT causal — an assessment is often charted because a dose '
            f'was being given'))


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    output.add_output_arguments(parser)
    parser.add_argument('--drugs', nargs='+', default=list(DEFAULT_DRUGS),
                        help='drugs to colour separately; any other analgesic '
                             'is pooled into "%s"' % OTHER)
    parser.add_argument('--window-minutes', type=int,
                        default=pain_link.WINDOW_MINUTES,
                        help='how long after an assessment a dose still counts '
                             'as following it (default: %(default)s)')
    parser.add_argument('--exclude-clustered', action='store_true',
                        help='drop assessments that have another assessment '
                             'inside the window (7.8%% of them) instead of '
                             'relying on nearest-preceding attribution. '
                             'Sensitivity check; see '
                             'pain_link.response_by_assessment')
    parser.set_defaults(question=config.MED_PAIN_QUESTION)
    return parser


def main():
    args = build_parser().parse_args()

    io.warn_if_dirty()

    paths = config.med_admin_files()
    # Unfiltered: this frame defines which sessions are OBSERVABLE, so it must
    # be every session with a MAR export, not only those with analgesics.
    admin_all = load.load_administrations(paths=paths)
    bounds = pain_link.session_bounds(admin_all)

    # Every analgesic, not just the coloured ones — "no analgesic" has to mean
    # no analgesic.
    analgesics = admin_all[admin_all['level2'].isin(
        set(med_taxonomy.ANALGESIC_SUBCLASSES))].copy()
    drugs = load.select_drugs(analgesics, drugs=args.drugs)

    scores = pain_link.load_pain_scores()
    per, stats = pain_link.response_by_assessment(
        scores, analgesics, bounds, window_minutes=args.window_minutes,
        exclude_clustered=args.exclude_clustered)
    if per.empty:
        raise SystemExit('no observable assessment survived the exclusions')

    categorized = categorize(per, drugs)
    counts, pct, totals = response_table(categorized, drugs)

    run_dir = output.resolve_run_dir(args, OUTPUT_TYPE)
    parents = output.source_parents(paths)
    params = vars(args)
    subjects = sorted(categorized['subject'].unique())

    output.write_table(counts.reset_index(), run_dir, 'assessment_counts',
                       SCRIPT, params=params, parents=parents,
                       subjects=subjects, extra=stats)
    output.write_table(pct.round(3).reset_index(), run_dir,
                       'assessment_percent', SCRIPT, params=params,
                       parents=parents, extra=stats)
    output.write_table(
        pd.DataFrame({'pain_score': totals.index, 'n_assessments': totals.to_numpy()}),
        run_dir, 'assessment_denominator', SCRIPT, params=params,
        parents=parents, extra=stats)
    # Row level, so any percentage here is re-checkable without a re-run.
    output.write_table(
        categorized[['subject', 'session', 'score_dt', 'pain_score', 'category',
                     'n_drugs', 'n_administrations', 'clustered',
                     'next_gap_minutes']],
        run_dir, 'assessment_response', SCRIPT, params=params, parents=parents,
        extra=stats)

    plot_stacked(counts, pct, totals, run_dir / 'fig7_response_by_pain_score.png',
                 drugs, window_minutes=args.window_minutes,
                 n_subjects=len(subjects), stats=stats)

    output.write_run(
        run_dir, SCRIPT, args, categorized, paths,
        extra={
            'drugs': list(drugs),
            'response': stats,
            'interpretation_note': (
                'DESCRIPTIVE and NOT causal, in a direction that invites '
                'over-reading: a dose following an assessment does not mean '
                'the assessment prompted it. Scheduled drugs land on a clock, '
                'and an assessment is frequently charted because a dose was '
                'being given. Nomination, not a finding (CLAUDE.md).'),
        },
        description=(f'analgesic response within {args.window_minutes} min of a '
                     f'pain assessment, {len(drugs)} drugs, '
                     f'{stats["n_assessments_scored"]} assessments, '
                     f'{100 * stats["frac_with_any_analgesic"]:.1f}% responded, '
                     f'n={len(subjects)}'))


if __name__ == '__main__':
    main()

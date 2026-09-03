"""
Fig 5 — administrations by the pain score that preceded them.

Grouped bars: x = the most recent pain score in the 30 minutes before a dose,
y = how many times each of the four most-administered analgesics was given at
that score. One bar per drug per score.

Two panels are written, and they answer different questions:

  fig5a_admin_by_pain_score.png            raw counts — how much dosing happens
                                           at each score
  fig5b_admin_by_pain_score_normalized.png each drug scaled to its own total —
                                           WHERE on the scale a given drug is used

5a is the requested view. 5b exists because acetaminophen and
hydrocodone-acetaminophen have three to four times the administrations of
hydromorphone, so on raw counts they dominate every bar group and a
lower-volume drug's shape is unreadable — which is the actual question when
asking how a drug relates to pain. Neither replaces the other.

Read `pain_link` before reading this file. The 30-minute window is an inclusion
criterion (unmatched administrations are dropped, not imputed), a same-minute
score counts as prior, and NONE of this is causal — an assessment is often
charted because a PRN dose was requested, so the arrow can point either way.

Run on Slurm, never the login node:
    python -m ieeg_ehr.med_analysis.plot_pain_score_bars
"""

import argparse
import logging

import numpy as np

from ieeg_ehr import config, io
from ieeg_ehr.med_analysis import load, output, pain_link, style
from ieeg_ehr.med_analysis.style import plt

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_TYPE = 'pain_score_bars'
SCRIPT = 'ieeg_ehr/med_analysis/plot_pain_score_bars.py'

#: The four most-administered analgesics, in descending administration count
#: (Fig 1's table: 643, 493, 216, 157). Fixed order so the bar within a group
#: means the same drug in every group and across both panels.
DEFAULT_DRUGS = ('ACETAMINOPHEN', 'HYDROCODONE-ACETAMINOPHEN', 'OXYCODONE',
                 'HYDROMORPHONE')


def plot_grouped_bars(counts, summary, out_path, *, normalize, window_minutes,
                      n_subjects, n_dropped, n_total):
    drugs = list(counts.columns)
    colors = style.categorical_colors(drugs)
    scores = counts.index.to_numpy(dtype=float)

    values = counts.astype(float)
    if normalize:
        totals = values.sum(axis=0).replace(0, np.nan)
        values = values.divide(totals, axis=1) * 100.0

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bars are placed side by side inside a 0.8-wide slot centred on the score,
    # leaving 0.2 of clear space between adjacent scores so the groups read as
    # groups rather than as one continuous run of bars.
    width = 0.8 / len(drugs)
    for i, drug in enumerate(drugs):
        offset = (i - (len(drugs) - 1) / 2) * width
        row = summary.loc[summary['drug'] == drug]
        n = int(row['n_linked'].iloc[0]) if len(row) else 0
        median = row['score_median'].iloc[0] if len(row) and n else float('nan')
        label = (f'{drug.title()}  (n={n}, median {median:g})' if n
                 else f'{drug.title()}  (n=0)')
        ax.bar(scores + offset, values[drug].to_numpy(), width=width,
               color=colors[drug], label=label, zorder=3,
               edgecolor='white', linewidth=0.4)

    ax.set_xticks(range(pain_link.PAIN_SCORE_MIN, pain_link.PAIN_SCORE_MAX + 1))
    ax.set_xlim(pain_link.PAIN_SCORE_MIN - 0.6, pain_link.PAIN_SCORE_MAX + 0.6)

    ylabel = ("% of that drug's linked administrations" if normalize
              else 'Administrations')
    title_tail = ('scaled within drug' if normalize else 'raw counts')
    style.style_axes(ax, grid_axis='y')
    style.label_axes(
        ax,
        f'Most recent pain score in the {window_minutes} min before the dose '
        f'(0-10)',
        ylabel,
        f'Analgesic administrations by the preceding pain score — {title_tail}\n'
        f'{int(counts.to_numpy().sum())} administrations with an assessment in '
        f'the preceding {window_minutes} min, {n_subjects} subjects')
    leg = ax.legend(title='Medication', fontsize=style.LEGEND_SIZE,
                    title_fontsize=style.LEGEND_SIZE, frameon=False,
                    loc='upper left')
    leg._legend_box.align = 'left'

    fig.tight_layout()
    # The excluded count belongs ON the image: this panel is a subset of the
    # administrations, and a reader comparing its totals to Fig 1's needs to
    # know why they differ without going to the provenance file.
    return style.save(
        fig, out_path,
        footnote=(f'{n_dropped} of {n_total} administrations excluded — no pain '
                  f'assessment in the {window_minutes} min before the dose; '
                  f'a same-minute score counts as prior. NOT causal: an '
                  f'assessment is often charted because a dose was requested'))


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    output.add_output_arguments(parser)
    parser.add_argument('--drugs', nargs='+', default=list(DEFAULT_DRUGS),
                        help='medications to compare, as they appear in the MAR '
                             '(default: the four most-administered analgesics)')
    parser.add_argument('--window-minutes', type=int,
                        default=pain_link.WINDOW_MINUTES,
                        help='an administration needs a pain assessment this '
                             'recent to be included at all (default: '
                             '%(default)s)')
    parser.add_argument('--strict-prior', action='store_true',
                        help='require the assessment to be STRICTLY before the '
                             'dose, excluding same-minute scores. Drops roughly '
                             'half the sample and does not change the shape; '
                             'see pain_link.__doc__')
    parser.set_defaults(question=config.MED_PAIN_QUESTION)
    return parser


def main():
    args = build_parser().parse_args()

    io.warn_if_dirty()

    paths = config.med_admin_files()
    admin = load.load_administrations(paths=paths)
    admin = admin[admin['drug'].isin(set(args.drugs))].copy()
    if admin.empty:
        raise SystemExit(f'no administrations for drugs {args.drugs}')

    scores = pain_link.load_pain_scores()
    linked, stats = pain_link.link_to_prior_score(
        admin, scores, window_minutes=args.window_minutes,
        allow_exact=not args.strict_prior)
    if linked.empty:
        raise SystemExit('no administration had a pain score in the window')

    # Plot the drugs in the order given, not the order pandas groups them in.
    drugs = [d for d in args.drugs if d in set(linked['drug'])]
    counts = pain_link.counts_by_score(linked, drugs)
    summary = pain_link.per_drug_summary(linked, drugs)

    run_dir = output.resolve_run_dir(args, OUTPUT_TYPE)
    parents = output.source_parents(paths)
    params = vars(args)

    output.write_table(counts.reset_index(), run_dir, 'counts_by_pain_score',
                       SCRIPT, params=params, parents=parents,
                       subjects=sorted(linked['subject'].unique()),
                       extra=stats)
    output.write_table(summary, run_dir, 'per_drug_summary', SCRIPT,
                       params=params, parents=parents, extra=stats)
    # The row-level table, so any of this is re-checkable without a re-run.
    output.write_table(
        linked[['subject', 'session', 'drug', 'route', 'dose', 'dose_unit',
                'taken_dt', 'score_dt', 'gap_minutes', 'pain_score']],
        run_dir, 'linked_administrations', SCRIPT, params=params,
        parents=parents, extra=stats)

    common = dict(window_minutes=args.window_minutes,
                  n_subjects=linked['subject'].nunique(),
                  n_dropped=stats['n_dropped_no_recent_score'],
                  n_total=stats['n_administrations_in'])
    plot_grouped_bars(counts, summary,
                      run_dir / 'fig5a_admin_by_pain_score.png',
                      normalize=False, **common)
    plot_grouped_bars(counts, summary,
                      run_dir / 'fig5b_admin_by_pain_score_normalized.png',
                      normalize=True, **common)

    output.write_run(
        run_dir, SCRIPT, args, linked, paths,
        extra={
            'drugs': drugs,
            'linkage': stats,
            'per_drug_median_prior_score': dict(
                zip(summary['drug'], summary.get('score_median', []))),
            'interpretation_note': (
                'DESCRIPTIVE and NOT causal. A pain score preceding a dose does '
                'not make it the reason for the dose: scheduled drugs are given '
                'on a clock whatever the assessment says, and an assessment is '
                'often charted because a PRN dose was requested. Nomination, '
                'not a finding (CLAUDE.md).'),
        },
        description=(f'analgesic administrations by preceding pain score, '
                     f'{len(drugs)} drugs, {stats["n_linked"]} linked '
                     f'administrations within {args.window_minutes} min, '
                     f'n={linked["subject"].nunique()}'))


if __name__ == '__main__':
    main()

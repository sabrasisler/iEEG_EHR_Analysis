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
  fig5c_pain_score_violin.png              the same distributions per drug, as
                                           violins: one distribution per drug
                                           side by side, for reading the
                                           ordering off directly

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

#: The analgesics with enough administrations to carry a distribution, in
#: descending administration count (Fig 1's table: 643, 493, 216, 157, 94, 47,
#: 39). Fixed order so a given bar position means the same drug in every group
#: and across all three panels.
#:
#: Morphine (25 administrations, 9 linked, 2 subjects) and ibuprofen (2 and 2)
#: are deliberately absent: at that size a distribution is two patients'
#: habits, not a dosing pattern, and a violin of two identical points is not
#: even estimable. Excluded by omission from this list so the omission is
#: visible, rather than by a threshold that silently drops them.
DEFAULT_DRUGS = ('ACETAMINOPHEN', 'HYDROCODONE-ACETAMINOPHEN', 'OXYCODONE',
                 'HYDROMORPHONE', 'FENTANYL', 'KETOROLAC', 'TRAMADOL')

#: A violin needs a distribution. Below this the KDE is either meaningless or,
#: for a single repeated value, singular and raising — so such a drug is left
#: out of 5c and named in its footnote, while 5a/5b still show its bars.
MIN_VIOLIN_N = 5


def plot_grouped_bars(counts, summary, out_path, *, normalize, window_minutes,
                      n_subjects, n_dropped, n_total):
    drugs = list(counts.columns)
    colors = style.categorical_colors(drugs)
    scores = counts.index.to_numpy(dtype=float)

    values = counts.astype(float)
    if normalize:
        totals = values.sum(axis=0).replace(0, np.nan)
        values = values.divide(totals, axis=1) * 100.0

    # Wider as drugs are added: eleven score groups times N bars, so at seven
    # drugs this is 77 bars and a 10-in frame makes each one a hairline.
    fig, ax = plt.subplots(figsize=(min(10 + 1.1 * max(0, len(drugs) - 4), 16), 6))

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

    # Reserve the bottom band: the footnote runs to two lines and would
    # otherwise sit on top of the axis label.
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    # The excluded count belongs ON the image: this panel is a subset of the
    # administrations, and a reader comparing its totals to Fig 1's needs to
    # know why they differ without going to the provenance file.
    return style.save(
        fig, out_path,
        footnote=(f'{n_dropped} of {n_total} administrations excluded — no pain '
                  f'assessment in the {window_minutes} min before the dose; '
                  f'a same-minute score counts as prior\n'
                  f'NOT causal: an assessment is often charted because a dose '
                  f'was requested'))


def violin_eligible(summary, min_n=MIN_VIOLIN_N):
    """Drugs with enough linked administrations to estimate a density.

    Kept as a plain function rather than inlined so the threshold is testable:
    the failure it prevents is a raise from a singular KDE, which only shows up
    with real data.
    """
    return [row.drug for row in summary.itertuples()
            if getattr(row, 'n_linked', 0) >= min_n]


def _drug_tick_label(drug, n):
    """Drug name for an x tick, wrapped at the hyphen so it fits under a violin."""
    name = drug.title()
    if len(name) > 14 and '-' in name:
        name = name.replace('-', '-\n', 1)
    return f'{name}\nn={n}'


def plot_violin(linked, summary, out_path, *, window_minutes, n_subjects,
                n_dropped, n_total):
    """One distribution per drug: x = drug, y = the score before the dose."""
    eligible = violin_eligible(summary)
    omitted = [d for d in summary['drug'] if d not in eligible]
    drugs = eligible
    colors = style.categorical_colors(drugs)
    data = [linked.loc[linked['drug'] == d, 'pain_score'].to_numpy(dtype=float)
            for d in drugs]

    # ~1.9 in per violin, so seven drugs do not squeeze into a four-drug frame.
    fig, ax = plt.subplots(figsize=(max(8.5, 1.9 * len(drugs)), 6))

    parts = ax.violinplot(data, positions=range(len(drugs)), widths=0.78,
                          showmedians=False, showextrema=False)
    for body, drug in zip(parts['bodies'], drugs):
        body.set_facecolor(colors[drug])
        body.set_edgecolor(colors[drug])
        body.set_alpha(0.32)
        body.set_linewidth(1.2)

    # The scores are INTEGERS 0-10, so a violin outline is a kernel density of
    # discrete data and its smoothness is an artifact of the estimator rather
    # than structure in the ratings — it will also bulge past 0 and 10, where no
    # rating can exist. The jittered points go on top for exactly that reason:
    # they land in eleven horizontal rows and make the granularity visible
    # instead of letting the outline imply a continuous scale. Jitter is on x
    # ONLY. Jittering y would move a point off the score that was charted.
    rng = np.random.default_rng(0)
    for i, (drug, values) in enumerate(zip(drugs, data)):
        ax.scatter(i + rng.uniform(-0.17, 0.17, size=len(values)), values,
                   s=7, color=colors[drug], alpha=0.30, linewidth=0, zorder=3)

    # Median and IQR drawn explicitly, because the ordering claim rests on them
    # and a KDE's fattest point is not its median.
    for i, drug in enumerate(drugs):
        row = summary.loc[summary['drug'] == drug].iloc[0]
        ax.plot([i, i], [row['score_q1'], row['score_q3']],
                color=style.TEXT_PRIMARY, linewidth=1.2, zorder=4,
                solid_capstyle='round')
        ax.plot([i - 0.28, i + 0.28], [row['score_median']] * 2,
                color=style.TEXT_PRIMARY, linewidth=2.2, zorder=5,
                solid_capstyle='round')

    ax.set_xticks(range(len(drugs)))
    ax.set_xticklabels([
        _drug_tick_label(d, int(summary.loc[summary['drug'] == d,
                                           'n_linked'].iloc[0]))
        for d in drugs], fontsize=style.TICK_SIZE)
    ax.set_xlim(-0.6, len(drugs) - 0.4)
    ax.set_yticks(range(pain_link.PAIN_SCORE_MIN, pain_link.PAIN_SCORE_MAX + 1))
    ax.set_ylim(pain_link.PAIN_SCORE_MIN - 0.6, pain_link.PAIN_SCORE_MAX + 0.6)

    style.style_axes(ax, grid_axis='y')
    # No x label: the tick labels are drug names, so "Medication" underneath
    # them is redundant, and with a wrapped two-line name it collides.
    style.label_axes(
        ax, None,
        f'Pain score in the {window_minutes} min before the dose (0-10)',
        'Preceding pain score by medication\n'
        f'{len(linked)} administrations with an assessment in the preceding '
        f'{window_minutes} min, {n_subjects} subjects')

    # Reserve the bottom band explicitly: the tick labels run to three lines
    # for a wrapped drug name plus its n, and the default margin lets them
    # collide with the footnote.
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    omitted_note = (f'; {", ".join(d.title() for d in omitted)} omitted '
                    f'(fewer than {MIN_VIOLIN_N} linked administrations — no '
                    f'estimable density)' if omitted else '')
    return style.save(
        fig, out_path,
        footnote=(f'bar = median, vertical line = IQR, points = individual '
                  f'administrations (x-jittered only); scores are integers, so '
                  f'the violin outline is a KDE of discrete data\n'
                  f'{n_dropped} of {n_total} administrations excluded — no '
                  f'assessment in the {window_minutes} min before the dose'
                  + omitted_note))


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
    admin_all = load.load_administrations(paths=paths)
    # Validated against the WHOLE corpus before filtering, so a misspelled drug
    # raises here — with the available names — instead of quietly becoming a
    # figure with one fewer panel than its caption claims.
    requested = load.select_drugs(admin_all, drugs=args.drugs)
    admin = admin_all[admin_all['drug'].isin(set(requested))].copy()
    if admin.empty:
        raise SystemExit(f'no administrations for drugs {requested}')

    scores = pain_link.load_pain_scores()
    linked, stats = pain_link.link_to_prior_score(
        admin, scores, window_minutes=args.window_minutes,
        allow_exact=not args.strict_prior)
    if linked.empty:
        raise SystemExit('no administration had a pain score in the window')

    # Plot the drugs in the order given, not the order pandas groups them in.
    drugs = [d for d in requested if d in set(linked['drug'])]
    lost = [d for d in requested if d not in set(linked['drug'])]
    if lost:
        logger.warning('%s: no administration had an assessment within %d min, '
                       'so they are absent from these panels',
                       ', '.join(lost), args.window_minutes)
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
    plot_violin(linked, summary, run_dir / 'fig5c_pain_score_violin.png',
                **common)

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

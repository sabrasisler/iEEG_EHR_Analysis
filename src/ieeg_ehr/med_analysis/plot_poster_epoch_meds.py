"""
Poster panels A-E — analgesic exposure around pain epochs.

Five standalone figures at poster sizes, each written separately so a panel can
be dropped into a layout on its own:

  A  figA_subjects_by_drug.png     % of subjects with each drug in the 2 h
                                   before at least one of their pain epochs
  B  figB_medicated_epochs.png     per subject, the % of their epochs that are
                                   dosed — one bar per subject
  C  figC_dose_probability.png     within-subject P(dose | pain score) in the
                                   2 h AFTER a score, averaged over subjects
  D  figD_pain_dosed_vs_undosed.png    mean pain in dosed vs undosed epochs,
                                       paired within subject
  E  figE_pain_deviation.png       the same, as deviation from that session's
                                   own mean pain

COHORT. Discovery only, by default and by design: the unit here is a pain
epoch, so `CLAUDE.md`'s hold-out gate applies (unlike the pure-EHR figures,
where DECISIONS 2026-09-03 call 4 reasoned it did not). `--split` exists but
cannot name the hold-out.

THE CONFOUND, which every panel here shares and D and E state outright. Dosing
is not randomised: the patient in more pain is the one who gets a dose. So a
HIGHER pain score in dosed epochs is the expected result of confounding by
indication and is not evidence that analgesia failed. Panel E removes
between-subject differences by centring on each session's own mean, which
helps with "who gets dosed" but does nothing about "when within a session",
and that is the direction the bias runs. These are nominations.

Run on Slurm, never the login node:
    python -m ieeg_ehr.med_analysis.plot_poster_epoch_meds
"""

import argparse
import logging

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.config import med_taxonomy
from ieeg_ehr.med_analysis import epoch_meds, load, output, style
from ieeg_ehr.med_analysis.style import plt

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

EVENT = 'pain'
QUESTION = 'medication_exposure'
OUTPUT_TYPE = 'poster_panels'
SCRIPT = 'ieeg_ehr/med_analysis/plot_poster_epoch_meds.py'

DEFAULT_DRUGS = ('ACETAMINOPHEN', 'HYDROCODONE-ACETAMINOPHEN', 'OXYCODONE',
                 'HYDROMORPHONE', 'FENTANYL', 'KETOROLAC', 'TRAMADOL')

DOSED_COLOR = '#eb6834'
UNDOSED_COLOR = '#2a78d6'


def _class_colors(drug_order, summary):
    classes = [c for c in med_taxonomy.ANALGESIC_SUBCLASS_ORDER
               if c in set(summary['level2'])]
    classes += [c for c in summary['level2'].unique() if c not in classes]
    return classes, style.categorical_colors(classes)


# --------------------------------------------------------------- panel A ---
def panel_a_table(per_epoch_drug, epochs, drugs, class_of):
    """Per drug: subjects with it in the window before >=1 epoch.

    `class_of` is passed in rather than looked up from `per_epoch_drug`,
    because a drug with zero exposed epochs has no row there to read a class
    from — and a drug nobody was exposed to is a real 0% bar that belongs on
    the figure, not a missing one.
    """
    n_subjects = epochs['subject'].nunique()
    rows = []
    for drug in drugs:
        sub = per_epoch_drug[per_epoch_drug['drug'] == drug]
        rows.append({
            'drug': drug,
            'level2': class_of.get(drug, 'Unknown'),
            'n_subjects': sub['subject'].nunique(),
            'pct_subjects': 100.0 * sub['subject'].nunique() / n_subjects,
            'n_epochs_exposed': sub['epoch_id'].nunique(),
        })
    return pd.DataFrame(rows), n_subjects


def plot_panel_a(summary, n_subjects, out_path, window_hours):
    classes, colors = _class_colors(list(summary['drug']), summary)
    ordered = (summary.assign(_c=summary['level2'].map(
                                  {c: i for i, c in enumerate(classes)}))
               .sort_values(['_c', 'pct_subjects'], ascending=[True, False])
               .reset_index(drop=True))

    fig, ax = plt.subplots(figsize=(16, 10))
    x = np.arange(len(ordered), dtype=float)
    ax.bar(x, ordered['pct_subjects'], width=0.7, zorder=3,
           color=[colors[c] for c in ordered['level2']],
           edgecolor='white')
    for xi, v, n in zip(x, ordered['pct_subjects'], ordered['n_subjects']):
        ax.annotate(f'{v:.0f}%\n({n})', (xi, v), textcoords='offset points',
                    xytext=(0, 6), ha='center', va='bottom',
                    fontsize=style.TICK_SIZE, color=style.TEXT_MUTED)

    # Rotated rather than wrapped: at poster type a 25 pt drug name is wide
    # enough that seven of them collide horizontally whatever the wrapping.
    ax.set_xticks(x)
    ax.set_xticklabels([d.title() for d in ordered['drug']],
                       rotation=28, ha='right', fontsize=style.TICK_SIZE)
    ax.set_ylim(0, 100)
    style.style_axes(ax, grid_axis='y')
    # Titles stay short and stay within ~45 characters a line. At poster type
    # a long title is wider than the axes, and `bbox_inches='tight'` then
    # stretches the canvas to fit it, leaving the panel adrift in whitespace.
    # Cohort size is deliberately absent from every panel — it is stated once
    # on the poster instead of five times on the figures.
    style.label_axes(ax, None, '% of subjects',
                     f'Analgesic exposure before a pain epoch\n'
                     f'{window_hours:g} h window', title_loc='center')
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in classes]
    # Upper LEFT: the tallest bar is on the right, and at poster size the
    # legend is big enough to sit on its value label.
    ax.legend(handles, classes, frameon=False, fontsize=style.LEGEND_SIZE,
              loc='upper left')
    fig.tight_layout()
    return style.save(fig, out_path)


# --------------------------------------------------------------- panel B ---
MULTI_CLASS = 'Multiple classes'


def panel_b_table(per_epoch, per_epoch_drug):
    """Per subject: % of epochs dosed, split into MUTUALLY EXCLUSIVE classes.

    An epoch can be preceded by drugs of more than one class, so classes
    cannot simply be stacked — the segments would sum past the share of epochs
    that were dosed at all, and the bar would stop meaning "% dosed". Each
    dosed epoch is therefore assigned to exactly one segment: its class if
    only one was given, `Multiple classes` otherwise. The bar total is then
    still the share of that subject's epochs with any analgesic.
    """
    classes = (per_epoch_drug.groupby(['subject', 'epoch_id'])['level2']
               .agg(lambda v: sorted(set(v))).rename('classes').reset_index())
    classes['segment'] = [c[0] if len(c) == 1 else MULTI_CLASS
                          for c in classes['classes']]

    totals = (per_epoch.groupby('subject')
              .agg(n_epochs=('dosed', 'size'), n_dosed=('dosed', 'sum'))
              .reset_index())
    counts = (classes.groupby(['subject', 'segment']).size()
              .unstack(fill_value=0))

    out = totals.merge(counts, on='subject', how='left').fillna(0)
    for seg in list(med_taxonomy.ANALGESIC_SUBCLASS_ORDER) + [MULTI_CLASS]:
        if seg not in out.columns:
            out[seg] = 0
        out[f'pct_{seg}'] = 100.0 * out[seg] / out['n_epochs']
    out['pct_dosed'] = 100.0 * out['n_dosed'] / out['n_epochs']
    return out.sort_values('pct_dosed', ascending=False).reset_index(drop=True)


def plot_panel_b(per_subject, out_path, window_hours):
    segments = [c for c in med_taxonomy.ANALGESIC_SUBCLASS_ORDER
                if per_subject.get(f'pct_{c}', pd.Series(dtype=float)).sum() > 0]
    segments.append(MULTI_CLASS)
    colors = style.categorical_colors(
        list(med_taxonomy.ANALGESIC_SUBCLASS_ORDER))
    colors[MULTI_CLASS] = style.TEXT_MUTED   # a mixture, so neutral

    fig, ax = plt.subplots(figsize=(16, 9))
    x = np.arange(len(per_subject), dtype=float)
    bottom = np.zeros(len(per_subject))
    for seg in segments:
        vals = per_subject[f'pct_{seg}'].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, width=0.85, color=colors[seg],
               label=seg, zorder=3, edgecolor='white', linewidth=0.3)
        bottom += vals

    median = float(per_subject['pct_dosed'].median())
    ax.axhline(median, color=style.TEXT_PRIMARY, linestyle='--', zorder=4)
    ax.annotate(f'median {median:.0f}%', (len(per_subject) - 0.5, median),
                textcoords='offset points', xytext=(-6, 8), ha='right',
                va='bottom', fontsize=style.LEGEND_SIZE,
                color=style.TEXT_PRIMARY)

    # One tick per subject would be unreadable at poster size and says nothing
    # — the subjects are anonymous and the ordering is the message.
    ax.set_xticks([])
    ax.set_xlim(-0.8, len(per_subject) - 0.2)
    ax.set_ylim(0, 100)
    style.style_axes(ax, grid_axis='y')
    style.label_axes(
        ax, 'Subjects (sorted)', '% of pain epochs dosed',
        f'Medicated pain epochs per subject\n'
        f'analgesic in the previous {window_hours:g} h', title_loc='center')
    ax.legend(frameon=False, fontsize=style.LEGEND_SIZE, loc='upper right')
    fig.tight_layout()
    return style.save(fig, out_path)


# --------------------------------------------------------------- panel C ---
def panel_c_table(per_subject_score):
    """Across subjects, at each score: mean of the within-subject P(dose)."""
    rows = []
    for score, g in per_subject_score.groupby('pain_score'):
        p = g['p_dose']
        rows.append({
            'pain_score': float(score),
            'n_subjects': len(g),
            'mean_p_dose': float(p.mean()),
            'sem_p_dose': float(p.std(ddof=1) / np.sqrt(len(p)))
                          if len(p) > 1 else float('nan'),
            'median_p_dose': float(p.median()),
            'n_assessments': int(g['n_assessments'].sum()),
        })
    return pd.DataFrame(rows).sort_values('pain_score').reset_index(drop=True)


ALL_ANALGESICS = 'All analgesics'


def plot_panel_c(summaries, per_subject_all, out_path, window_hours):
    """One line per medication class, plus a line for any analgesic.

    `summaries` is an ordered mapping label -> the per-score summary frame.
    The three class lines do NOT sum to the all-analgesics line: an assessment
    followed by both an opioid and acetaminophen counts once in each class and
    once overall, which is the right reading of "probability a drug of this
    class followed" and the wrong one to add up.
    """
    colors = style.categorical_colors(
        list(med_taxonomy.ANALGESIC_SUBCLASS_ORDER))
    colors[ALL_ANALGESICS] = style.TEXT_PRIMARY

    fig, ax = plt.subplots(figsize=(14, 9))

    # Individual subjects sit behind the all-analgesics line only. Four
    # scatter clouds would bury the lines they are meant to support, and this
    # is the line the spread matters for. Unlabelled, deliberately.
    rng = np.random.default_rng(0)
    ax.scatter(per_subject_all['pain_score']
               + rng.uniform(-0.18, 0.18, len(per_subject_all)),
               100 * per_subject_all['p_dose'],
               s=26, color=style.AXIS_COLOR, alpha=0.45, linewidth=0, zorder=2)

    for label, summary in summaries.items():
        width = 2.2 if label == ALL_ANALGESICS else 1.4
        ax.errorbar(summary['pain_score'], 100 * summary['mean_p_dose'],
                    yerr=100 * summary['sem_p_dose'], marker='o',
                    color=colors[label], ecolor=colors[label], capsize=4,
                    linewidth=width, label=label,
                    zorder=5 if label == ALL_ANALGESICS else 4)

    ax.set_xticks(range(0, 11))
    ax.set_xlim(-0.6, 10.6)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    style.style_axes(ax, grid_axis='y')
    style.label_axes(
        ax, 'Pain score', f'P(dose within {window_hours:g} h)  [%]',
        'Probability of a dose after a pain score', title_loc='center')
    # Upper left: the curves rise left-to-right, so that corner is the one
    # region no line passes through.
    ax.legend(frameon=False, fontsize=style.LEGEND_SIZE, loc='upper left')
    fig.tight_layout()
    return style.save(fig, out_path)


# ------------------------------------------------------------ panels D/E ---
def paired_stats(paired):
    """Median difference, sign consistency, and a within-subject test.

    A paired Wilcoxon over SUBJECTS, never over epochs: epochs within a
    subject are not independent, and CLAUDE.md asks for per-subject effects
    and sign consistency rather than a pooled p-value that ignores that
    structure.
    """
    diff = paired['difference'].to_numpy(dtype=float)
    out = {
        'n_subjects': int(len(diff)),
        'median_difference': float(np.median(diff)) if len(diff) else None,
        'mean_difference': float(np.mean(diff)) if len(diff) else None,
        'n_higher_in_dosed': int((diff > 0).sum()),
        'n_lower_in_dosed': int((diff < 0).sum()),
        'n_tied': int((diff == 0).sum()),
    }
    if len(diff) >= 6 and np.any(diff != 0):
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(diff)
        out['wilcoxon_stat'] = float(stat)
        out['wilcoxon_p'] = float(p)
    return out


def plot_paired(paired, stats, out_path, ylabel, title, zero_line=False,
                show_p=True):
    fig, ax = plt.subplots(figsize=(11, 9))

    xs = np.array([0.0, 1.0])
    for row in paired.itertuples():
        ax.plot(xs, [row.undosed, row.dosed], color=style.AXIS_COLOR,
                alpha=0.5, zorder=2, linewidth=1.2)
    ax.scatter(np.full(len(paired), 0.0), paired['undosed'], s=60,
               color=UNDOSED_COLOR, alpha=0.75, zorder=3, linewidth=0)
    ax.scatter(np.full(len(paired), 1.0), paired['dosed'], s=60,
               color=DOSED_COLOR, alpha=0.75, zorder=3, linewidth=0)

    # The group summary sits ON its group, not beside it. Offset sideways it
    # read as a stray extra point rather than as a summary of the column it
    # belongs to. A white edge and a high zorder keep it legible on top of the
    # subject points instead of needing its own x position.
    for x, col, color in ((0.0, 'undosed', UNDOSED_COLOR),
                          (1.0, 'dosed', DOSED_COLOR)):
        m = float(paired[col].mean())
        sem = float(paired[col].std(ddof=1) / np.sqrt(len(paired)))
        ax.errorbar([x], [m], yerr=[sem], marker='s', color=color,
                    ecolor=style.TEXT_PRIMARY, capsize=8, markersize=20,
                    markeredgecolor='white', markeredgewidth=2.5,
                    elinewidth=2.5, zorder=6)

    if zero_line:
        ax.axhline(0, color=style.ZERO_LINE_COLOR, linestyle='--', zorder=1)

    ax.set_xticks(xs)
    ax.set_xticklabels(['Undosed', 'Dosed'], fontsize=style.LABEL_SIZE)
    ax.set_xlim(-0.45, 1.45)
    style.style_axes(ax, grid_axis='y')

    # The statistics go INSIDE the axes. In the title they ran to three lines
    # and `bbox_inches='tight'` stretched the canvas sideways to fit them.
    # Cohort size is not among them — it lives on the poster, not the panel.
    p = stats.get('wilcoxon_p')
    lines = [f'median diff {stats["median_difference"]:+.2f}',
             f'higher when dosed: '
             f'{stats["n_higher_in_dosed"]}/{stats["n_subjects"]}']
    if p is not None and show_p:
        lines.append(f'Wilcoxon p={p:.1e}')
    ax.text(0.98, 0.98, '\n'.join(lines), transform=ax.transAxes,
            ha='right', va='top', fontsize=style.LEGEND_SIZE,
            color=style.TEXT_PRIMARY, linespacing=1.4)

    style.label_axes(ax, None, ylabel, title, title_loc='center')
    fig.tight_layout()
    return style.save(fig, out_path)


# ------------------------------------------------------------------ main ---
def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    output.add_output_arguments(parser)
    parser.add_argument('--drugs', nargs='+', default=list(DEFAULT_DRUGS),
                        help='drugs to show in panel A')
    parser.add_argument('--window-hours', type=float,
                        default=epoch_meds.WINDOW_HOURS,
                        help='exposure window (default: %(default)s)')
    parser.add_argument('--split', default='discovery',
                        help='cohort split; the hold-out is not reachable')
    parser.add_argument('--epoch-minutes', type=int, default=None,
                        help='epoch definition to read (default: the '
                             'configured one)')
    parser.set_defaults(question=QUESTION)
    return parser


def main():
    args = build_parser().parse_args()
    io.warn_if_dirty()
    style.use_poster(True)

    epochs = epoch_meds.load_epochs(split=args.split,
                                    minutes_before=args.epoch_minutes)
    admin_all = load.load_administrations(paths=config.med_admin_files())
    analgesics = epoch_meds.load_analgesics()

    epochs, n_no_mar = epoch_meds.restrict_to_observable(epochs, admin_all)
    if epochs.empty:
        raise SystemExit('no epoch survived the cohort and MAR filters')

    per_epoch, per_epoch_drug = epoch_meds.exposure_before_epochs(
        epochs, analgesics, window_hours=args.window_hours)
    per_epoch = epoch_meds.subject_deviation(per_epoch)

    drugs = load.select_drugs(analgesics, drugs=args.drugs)

    run_dir = config.analysis_run_dir(
        question=args.question, output_type=OUTPUT_TYPE,
        run_name=args.run_name or OUTPUT_TYPE, event=EVENT)
    if args.scratch:
        run_dir = config.PLOTS_ROOT / 'poster_epoch_meds'
    run_dir.mkdir(parents=True, exist_ok=True)

    paths = config.med_admin_files()
    parents = output.source_parents(paths)
    params = vars(args)
    subjects = sorted(per_epoch['subject'].unique())

    # A
    class_of = (analgesics.drop_duplicates('drug')
                .set_index('drug')['level2'].to_dict())
    a_table, n_subjects = panel_a_table(per_epoch_drug, per_epoch, drugs,
                                        class_of)
    plot_panel_a(a_table, n_subjects, run_dir / 'figA_subjects_by_drug.png',
                 args.window_hours)
    # B
    b_table = panel_b_table(per_epoch, per_epoch_drug)
    plot_panel_b(b_table, run_dir / 'figB_medicated_epochs.png',
                 args.window_hours)
    # C — one curve per class, plus one for any analgesic. Each is computed on
    # its own administration subset, so the nearest-preceding attribution is
    # applied within that subset rather than inherited from the pooled one.
    summaries, per_subject_all, c_long = {}, None, []
    for label in list(med_taxonomy.ANALGESIC_SUBCLASS_ORDER) + [ALL_ANALGESICS]:
        subset = (analgesics if label == ALL_ANALGESICS
                  else analgesics[analgesics['level2'] == label])
        if subset.empty:
            continue
        per_score = epoch_meds.dose_probability_by_score(
            epochs, subset, window_hours=args.window_hours)
        summaries[label] = panel_c_table(per_score)
        c_long.append(per_score.assign(med_group=label))
        if label == ALL_ANALGESICS:
            per_subject_all = per_score
    c_table = pd.concat([s.assign(med_group=k) for k, s in summaries.items()],
                        ignore_index=True)
    per_subject_score = pd.concat(c_long, ignore_index=True)
    plot_panel_c(summaries, per_subject_all,
                 run_dir / 'figC_dose_probability.png', args.window_hours)
    # D
    d_paired, d_dropped = epoch_meds.paired_by_subject(per_epoch, 'pain_score')
    d_stats = paired_stats(d_paired)
    plot_paired(d_paired, d_stats, run_dir / 'figD_pain_dosed_vs_undosed.png',
                'Mean pain score', 'Pain in dosed vs undosed epochs')
    # E
    e_paired, e_dropped = epoch_meds.paired_by_subject(per_epoch,
                                                       'pain_deviation')
    e_stats = paired_stats(e_paired)
    plot_paired(e_paired, e_stats, run_dir / 'figE_pain_deviation.png',
                'Pain minus session mean',
                'Pain relative to the session mean',
                zero_line=True, show_p=False)

    stats = {
        'split': args.split,
        'window_hours': args.window_hours,
        'n_epochs': int(len(per_epoch)),
        'n_subjects': len(subjects),
        'n_sessions': int(per_epoch.groupby(['subject', 'session']).ngroups),
        'n_epochs_dropped_no_mar': n_no_mar,
        'n_epochs_dosed': int(per_epoch['dosed'].sum()),
        'frac_epochs_dosed': round(float(per_epoch['dosed'].mean()), 4),
        'panel_d': d_stats, 'panel_d_subjects_one_arm_only': d_dropped,
        'panel_e': e_stats, 'panel_e_subjects_one_arm_only': e_dropped,
    }

    for df, name in ((a_table, 'panelA_subjects_by_drug'),
                     (b_table, 'panelB_medicated_epochs'),
                     (c_table, 'panelC_dose_probability'),
                     (per_subject_score, 'panelC_per_subject'),
                     (d_paired, 'panelD_paired'),
                     (e_paired, 'panelE_paired')):
        output.write_table(df, run_dir, name, SCRIPT, params=params,
                           parents=parents, extra=stats)
    output.write_table(
        per_epoch[['subject', 'session', 'epoch_id', 'pain_time', 'pain_score',
                   'session_mean_pain', 'pain_deviation', 'n_doses', 'n_drugs',
                   'dosed']],
        run_dir, 'epoch_exposure', SCRIPT, params=params, parents=parents,
        subjects=subjects, extra=stats)

    io.write_run_provenance(
        run_dir, script=SCRIPT, params=params, parents=parents,
        subjects=subjects,
        extra={
            **stats,
            'unit': 'pain epoch (5 min before a pain assessment)',
            'cohort_note': (
                'DISCOVERY ONLY. The unit is a pain epoch, so the CLAUDE.md '
                'hold-out gate applies here even though no neural power is '
                'read.'),
            'interpretation_note': (
                'CONFOUNDED BY INDICATION and not causal: the patient in more '
                'pain is the one who gets dosed, so higher pain in dosed '
                'epochs is the expected artefact, not evidence that analgesia '
                'failed. Panel E removes between-subject differences only. '
                'Nominations, not findings (CLAUDE.md).'),
        })
    io.log_analysis(
        f'poster panels A-E: analgesic exposure around pain epochs, '
        f'{args.window_hours:g} h window, {len(per_epoch)} epochs, '
        f'n={len(subjects)} {args.split} subjects', run_dir)
    logger.info('figures + provenance -> %s', run_dir)


if __name__ == '__main__':
    main()

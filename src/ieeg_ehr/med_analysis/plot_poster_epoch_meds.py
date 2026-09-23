"""
Poster panels — analgesic exposure around pain epochs.

Standalone panels, each written separately so one can be placed on its own,
plus a combined 2x2 for the panels that carry the story:

  A   figA_subjects_by_drug.png      % of subjects with each drug in the 2 h
                                     before >=1 of their pain epochs, led by
                                     an "Any analgesic" bar
  B   figB_medicated_epochs.png      per subject, the % of their epochs that
                                     are dosed, split by medication class
  C   figC_dose_probability.png      P(dose within 30 min) by pain score, one
                                     curve per class plus any analgesic
  C2  figC2_dose_probability_deviation.png   the same against the deviation
                                     from that subject's own mean pain
  D   figD_pain_dosed_vs_undosed.png mean pain in dosed vs undosed epochs
  E   figE_pain_deviation.png        the same, as pain minus subject mean
  --  figABC2E_grouped.png           A, B, C2 and E in one 2x2, no subplot
                                     titles, with a gap between the rows for
                                     text

TWO WINDOWS, deliberately. Exposure (A, B, D, E) asks what a patient was under
when the score was given, and uses 2 h. The dose-probability panels (C, C2) ask
whether a score was ACTED on, which is a much tighter question, and use 30 min.
Mixing them in one figure would be wrong; keeping them in different panels with
the window named on each axis is not.

COHORT. The pain study's 51 subjects (`--cohort pain-study`), intersected with
the discovery split rather than replacing it, so no cohort file can smuggle a
hold-out subject past the gate. The unit is a pain epoch, so that gate applies
here even though no neural power is read.

THE CONFOUND. Dosing is not randomised: the patient in more pain is the one who
gets dosed. Higher pain in dosed epochs is the expected result of confounding
by indication, NOT evidence that analgesia failed. E and C2 remove
between-subject differences by centring on each subject's own mean; neither
touches the within-subject timing, which is the direction the bias runs.
Nominations, not findings.

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

ANY_ANALGESIC = 'Any analgesic'
ALL_ANALGESICS = 'All analgesics'
MULTI_CLASS = 'Multiple classes'

#: Display order for these panels: acetaminophen first, then opioids. It is
#: the escalation order a clinician reads, and it puts the most-given class
#: first. Deliberately NOT `med_taxonomy.ANALGESIC_SUBCLASS_ORDER`, which is
#: administration-count order and still drives Figs 1-6 — changing that
#: constant would silently reorder figures that are already agreed.
#:
#: COLOUR is still assigned from the taxonomy order, so Opioids stays blue and
#: Acetaminophen orange exactly as in every earlier figure. Order and colour
#: are decoupled on purpose: re-ordering a legend should not repaint it.
POSTER_CLASS_ORDER = ('Acetaminophen', 'Opioids', 'NSAIDs')

#: Axis wording for the centred pain score, shared by C2 and E so the two
#: cannot describe the same quantity differently. "Pain minus subject mean"
#: read as a single subtraction; this is each epoch's score expressed
#: relative to that subject's own average, which is what centring means.
SUBJECT_CENTRED_LABEL = 'Subject-centered pain score'


def _axes(ax, grid=None):
    """Poster axis style: no grid, black tick labels."""
    style.style_axes(ax, grid_axis=grid, tick_color=style.TEXT_PRIMARY)


def _class_palette():
    colors = style.categorical_colors(
        list(med_taxonomy.ANALGESIC_SUBCLASS_ORDER))
    colors[MULTI_CLASS] = style.TEXT_MUTED
    colors[ANY_ANALGESIC] = style.TEXT_PRIMARY
    colors[ALL_ANALGESICS] = style.TEXT_PRIMARY
    return colors


def _wrap_drug(name):
    """Break a long drug name after its hyphen.

    Only the combination products are long enough to matter, and they all
    carry a hyphen at the natural break. Wrapping shortens the diagonal reach
    of a rotated tick label, which is what makes one name look far longer
    than its neighbours.
    """
    title = name.title()
    if len(title) > 16 and '-' in title:
        return title.replace('-', '-\n', 1)
    return title


def _p_text(p):
    """`p = 3.6 x 10^-9`, without naming the test."""
    if p is None:
        return None
    if p >= 0.001:
        return f'$p$ = {p:.3f}'
    exponent = int(np.floor(np.log10(p)))
    mantissa = p / (10 ** exponent)
    return rf'$p$ = {mantissa:.1f} $\times$ 10$^{{{exponent}}}$'


def _sig_bracket(ax, x1, x2, text):
    """A bracket spanning the two groups, labelled above the data."""
    lo, hi = ax.get_ylim()
    span = hi - lo
    y = max(ax.get_lines()[0].get_ydata().max() if ax.get_lines() else lo, hi)
    y = hi + span * 0.02
    tick = span * 0.025
    ax.plot([x1, x1, x2, x2], [y, y + tick, y + tick, y],
            color=style.TEXT_PRIMARY, linewidth=2.0, zorder=8,
            clip_on=False)
    ax.annotate(text, ((x1 + x2) / 2, y + tick), textcoords='offset points',
                xytext=(0, 6), ha='center', va='bottom',
                fontsize=style.LEGEND_SIZE, color=style.TEXT_PRIMARY,
                annotation_clip=False)
    ax.set_ylim(lo, hi + span * 0.16)


# --------------------------------------------------------------- panel A ---
def panel_a_table(per_epoch_drug, per_epoch, drugs, class_of):
    """Per drug: subjects with it in the window before >=1 epoch.

    `Any analgesic` leads the table because it is the headline: the per-drug
    bars answer "which drug", but the number a reader wants first is how much
    of the cohort was exposed at all. It is a set union over drugs, never a
    sum — most subjects received more than one.

    `class_of` is passed in because a drug with zero exposed epochs has no row
    in `per_epoch_drug` to read a class from, and a drug nobody was exposed to
    is a real 0% bar rather than a missing one.

    ACETAMINOPHEN here is single-ingredient only. The combination products are
    separate `drug` strings classed as Opioids, so neither this table nor the
    class colouring folds hydrocodone-acetaminophen into acetaminophen.
    """
    n_subjects = per_epoch['subject'].nunique()
    rows = [{
        'drug': ANY_ANALGESIC,
        'level2': ANY_ANALGESIC,
        'n_subjects': per_epoch.loc[per_epoch['dosed'], 'subject'].nunique(),
        'pct_subjects': 100.0 * per_epoch.loc[per_epoch['dosed'],
                                              'subject'].nunique() / n_subjects,
        'n_epochs_exposed': int(per_epoch['dosed'].sum()),
    }]
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


def draw_panel_a(ax, summary):
    colors = _class_palette()
    classes = [c for c in POSTER_CLASS_ORDER if c in set(summary['level2'])]

    head = summary[summary['drug'] == ANY_ANALGESIC]
    rest = (summary[summary['drug'] != ANY_ANALGESIC]
            .assign(_c=lambda d: d['level2'].map(
                {c: i for i, c in enumerate(classes)}))
            .sort_values(['_c', 'pct_subjects'], ascending=[True, False]))
    ordered = pd.concat([head, rest], ignore_index=True)

    x = np.arange(len(ordered), dtype=float)
    ax.bar(x, ordered['pct_subjects'], width=0.7, zorder=3,
           color=[colors[c] for c in ordered['level2']], edgecolor='white')
    for xi, v in zip(x, ordered['pct_subjects']):
        ax.annotate(f'{v:.0f}%', (xi, v), textcoords='offset points',
                    xytext=(0, 6), ha='center', va='bottom',
                    fontsize=style.TICK_SIZE, color=style.TEXT_PRIMARY)

    ax.set_xticks(x)
    ax.set_xticklabels([d if d == ANY_ANALGESIC else _wrap_drug(d)
                        for d in ordered['drug']],
                       rotation=28, ha='right', fontsize=style.TICK_SIZE)
    ax.set_ylim(0, 105)
    _axes(ax)
    style.label_axes(ax, None, '% of subjects')
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c])
               for c in [ANY_ANALGESIC] + classes]
    # Upper CENTRE: the two tall bars are the leading "Any analgesic" and
    # acetaminophen near the right, so centre is the only corner-ish space
    # that clears both a bar and its value label.
    ax.legend(handles, [ANY_ANALGESIC] + classes, frameon=False,
              fontsize=style.LEGEND_SIZE, loc='upper center')
    return ordered


# --------------------------------------------------------------- panel B ---
def panel_b_table(per_epoch, per_epoch_drug):
    """Per subject: % of epochs dosed, split into MUTUALLY EXCLUSIVE classes.

    An epoch can be preceded by drugs of more than one class, so classes
    cannot simply be stacked — the segments would sum past the share of epochs
    dosed at all and the bar would stop meaning "% dosed". Each dosed epoch is
    assigned exactly one segment: its class, or `Multiple classes`.
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
    for seg in list(POSTER_CLASS_ORDER) + [MULTI_CLASS]:
        if seg not in out.columns:
            out[seg] = 0
        out[f'pct_{seg}'] = 100.0 * out[seg] / out['n_epochs']
    out['pct_dosed'] = 100.0 * out['n_dosed'] / out['n_epochs']
    return out.sort_values('pct_dosed', ascending=False).reset_index(drop=True)


def draw_panel_b(ax, per_subject):
    colors = _class_palette()
    segments = [c for c in POSTER_CLASS_ORDER
                if per_subject.get(f'pct_{c}', pd.Series(dtype=float)).sum() > 0]
    segments.append(MULTI_CLASS)

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

    # Anonymous subjects: a tick per bar would be unreadable and say nothing.
    ax.set_xticks([])
    ax.set_xlim(-0.8, len(per_subject) - 0.2)
    ax.set_ylim(0, 100)
    _axes(ax)
    style.label_axes(ax, 'Subjects', '% of pain epochs dosed')
    ax.legend(frameon=False, fontsize=style.LEGEND_SIZE, loc='upper right')


# ------------------------------------------------------------ panels C/C2 ---
def panel_c_table(per_subject_value, value_col):
    """Across subjects, at each x: mean of the within-subject P(dose)."""
    rows = []
    for value, g in per_subject_value.groupby(value_col):
        p = g['p_dose']
        rows.append({
            value_col: float(value),
            'n_subjects': len(g),
            'mean_p_dose': float(p.mean()),
            'sem_p_dose': (float(p.std(ddof=1) / np.sqrt(len(p)))
                           if len(p) > 1 else float('nan')),
            'median_p_dose': float(p.median()),
            'n_assessments': int(g['n_assessments'].sum()),
        })
    return pd.DataFrame(rows).sort_values(value_col).reset_index(drop=True)


def draw_panel_c(ax, summaries, value_col, xlabel, window_minutes,
                 min_subjects=5):
    """One curve per medication class, plus one for any analgesic.

    The class curves do NOT sum to the all-analgesics curve: an assessment
    followed by both an opioid and acetaminophen counts once in each and once
    overall, which is the right reading of "probability a drug of this class
    followed" and the wrong one to add up.

    `min_subjects` trims x positions that rest on a handful of subjects. On
    the deviation axis the tails are single subjects with one extreme epoch,
    and at poster size a 0%-or-100% point there reads as a result.
    """
    colors = _class_palette()
    for label, summary in summaries.items():
        keep = summary[summary['n_subjects'] >= min_subjects]
        if keep.empty:
            continue
        width = 2.4 if label == ALL_ANALGESICS else 1.5
        ax.errorbar(keep[value_col], 100 * keep['mean_p_dose'],
                    yerr=100 * keep['sem_p_dose'], marker='o',
                    color=colors[label], ecolor=colors[label], capsize=4,
                    linewidth=width, label=label,
                    zorder=5 if label == ALL_ANALGESICS else 4)

    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    _axes(ax)
    style.label_axes(ax, xlabel, f'P(dose within {window_minutes:g} min)  [%]')
    # The curves rise left to right, so the upper left is the one region no
    # line passes through.
    ax.legend(frameon=False, fontsize=style.LEGEND_SIZE, loc='upper left')


# ------------------------------------------------------------ panels D/E ---
def paired_stats(paired):
    """Median difference, sign consistency, and a within-subject test.

    A paired Wilcoxon over SUBJECTS, never over epochs: epochs within a
    subject are not independent, and CLAUDE.md asks for per-subject effects
    and sign consistency rather than a pooled p-value that ignores that.
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


def draw_paired(ax, paired, stats, ylabel, zero_line=False, show_p=True,
                show_summary=True, show_violin=False, show_bracket=False):
    """Paired subject means, dosed vs undosed.

    `show_violin` draws the distribution behind the points; `show_summary`
    draws a mean +/- SEM marker on each group. They are alternatives rather
    than additions — a violin already shows where the mass is, and the marker
    on top of it reads as a third, unexplained thing.
    """
    xs = np.array([0.0, 1.0])

    if show_violin:
        data = [paired['undosed'].to_numpy(dtype=float),
                paired['dosed'].to_numpy(dtype=float)]
        parts = ax.violinplot(data, positions=xs, widths=0.62,
                              showmedians=False, showextrema=False)
        for body, color in zip(parts['bodies'], (UNDOSED_COLOR, DOSED_COLOR)):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.28)
            body.set_linewidth(1.5)
            body.set_zorder(1)

    for row in paired.itertuples():
        ax.plot(xs, [row.undosed, row.dosed], color=style.AXIS_COLOR,
                alpha=0.5, zorder=2, linewidth=1.2)
    ax.scatter(np.full(len(paired), 0.0), paired['undosed'], s=60,
               color=UNDOSED_COLOR, alpha=0.75, zorder=3, linewidth=0)
    ax.scatter(np.full(len(paired), 1.0), paired['dosed'], s=60,
               color=DOSED_COLOR, alpha=0.75, zorder=3, linewidth=0)

    # The group summary sits ON its group. Offset sideways it read as a stray
    # extra point rather than a summary of the column it belongs to.
    if show_summary:
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
    _axes(ax)

    # Statistics inside the axes; in the title they ran to three lines and
    # bbox_inches='tight' stretched the canvas sideways to fit them. Cohort
    # size is not among them — it lives on the poster, not the panel. The
    # test is not named anywhere on the figure: the p value is the number a
    # reader needs, and which test produced it belongs in the caption and the
    # provenance, both of which say Wilcoxon signed-rank.
    lines = [f'median diff {stats["median_difference"]:+.2f}',
             f'higher when dosed: '
             f'{stats["n_higher_in_dosed"]}/{stats["n_subjects"]}']
    p = stats.get('wilcoxon_p')
    if p is not None and show_p and not show_bracket:
        lines.append(_p_text(p))
    # The bracket owns the top centre-right, so the stats move left when one
    # is drawn rather than sitting on top of it.
    # With a bracket, the stats go to the LOWER right: the bracket owns the
    # top of the panel, and the space below the dosed violin is empty in every
    # version of this figure (dosed subject means sit above the undosed ones).
    # Upper left collided with the bracket's left riser once the panel shrank.
    if show_bracket:
        tx, ty, tha, tva = 0.98, 0.02, 'right', 'bottom'
    else:
        tx, ty, tha, tva = 0.98, 0.98, 'right', 'top'
    ax.text(tx, ty, '\n'.join(lines), transform=ax.transAxes,
            ha=tha, va=tva, fontsize=style.LEGEND_SIZE,
            color=style.TEXT_PRIMARY, linespacing=1.4)
    style.label_axes(ax, None, ylabel)

    # Drawn last: the bracket sizes itself against the final y limits.
    if show_bracket and p is not None:
        _sig_bracket(ax, xs[0], xs[1], _p_text(p))


# ------------------------------------------------------ standalone wrappers ---
def _save(fig, out_path, title=None):
    if title:
        fig.suptitle(title, fontsize=style.TITLE_SIZE,
                     color=style.TEXT_PRIMARY, ha='center')
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        fig.tight_layout()
    return style.save(fig, out_path)


def plot_panel_a(summary, out_path, window_hours):
    fig, ax = plt.subplots(figsize=(16, 10))
    draw_panel_a(ax, summary)
    return _save(fig, out_path,
                 f'Analgesic exposure before a pain epoch\n'
                 f'{window_hours:g} h window')


def plot_panel_b(per_subject, out_path, window_hours):
    fig, ax = plt.subplots(figsize=(16, 9))
    draw_panel_b(ax, per_subject)
    return _save(fig, out_path,
                 f'Medicated pain epochs per subject\n'
                 f'analgesic in the previous {window_hours:g} h')


def plot_panel_c(summaries, out_path, value_col, xlabel, window_minutes,
                 title):
    fig, ax = plt.subplots(figsize=(14, 9))
    draw_panel_c(ax, summaries, value_col, xlabel, window_minutes)
    return _save(fig, out_path, title)


def plot_paired(paired, stats, out_path, ylabel, title, zero_line=False,
                show_p=True, show_summary=True, show_violin=False,
                show_bracket=False):
    fig, ax = plt.subplots(figsize=(11, 9))
    draw_paired(ax, paired, stats, ylabel, zero_line=zero_line, show_p=show_p,
                show_summary=show_summary, show_violin=show_violin,
                show_bracket=show_bracket)
    return _save(fig, out_path, title)


#: Type for the grouped figure. It is printed at 12.5 x 7 in, half the linear
#: size the standalone poster panels were drawn at, so it gets its own scale:
#: the POSTER sizes (30 / 25 / 21 pt) in a 3-in-tall panel leave no room for
#: data. These are chosen for a panel read at poster distance at that size.
GROUPED_SIZES = dict(TITLE_SIZE=13, LABEL_SIZE=11, TICK_SIZE=9.5,
                     LEGEND_SIZE=8.5, FOOTNOTE_SIZE=7, DPI=300)
GROUPED_FIGSIZE = (12.5, 7.0)

#: Short panel titles for the grouped figure: enough to say what each panel
#: is without a caption, and short enough to sit above a 6-in-wide panel.
GROUPED_TITLES = (
    'Exposure before a pain epoch (2 h)',
    'Medicated epochs per subject',
    'Dose after a pain score (30 min)',
    'Pain in dosed vs undosed epochs',
)


def plot_grouped(a_table, b_table, c2_summaries, e_paired, e_stats, out_path,
                 score_window_minutes):
    """A, B, C2 and E in one 2x2 at exactly 12.5 x 7 in.

    Saved WITHOUT `bbox_inches='tight'`, unlike everything else here: tight
    cropping recomputes the canvas from whatever the text extends to, so the
    file would come out at some other size. The layout is fitted inside the
    fixed canvas instead, which is what a poster slot of a given size needs.

    The row gap is deliberately wide: it is where the poster's own text goes.
    """
    saved = {k: getattr(style, k) for k in GROUPED_SIZES}
    style.use_poster(False)            # screen line weights suit this size
    for k, v in GROUPED_SIZES.items():
        setattr(style, k, v)
    try:
        fig, axes = plt.subplots(2, 2, figsize=GROUPED_FIGSIZE)
        draw_panel_a(axes[0][0], a_table)
        draw_panel_b(axes[0][1], b_table)
        draw_panel_c(axes[1][0], c2_summaries, 'pain_deviation',
                     SUBJECT_CENTRED_LABEL, score_window_minutes)
        draw_paired(axes[1][1], e_paired, e_stats, SUBJECT_CENTRED_LABEL,
                    zero_line=True, show_summary=False, show_violin=True,
                    show_bracket=True)
        for ax, title in zip(axes.ravel(), GROUPED_TITLES):
            # pad clears the significance bracket, which is drawn above E's
            # axes and would otherwise run into E's title.
            ax.set_title(title, fontsize=style.TITLE_SIZE,
                         color=style.TEXT_PRIMARY, loc='center', pad=14)
        fig.tight_layout(h_pad=3.5)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=style.DPI, facecolor='white')
        plt.close(fig)
    finally:
        style.use_poster(True)
        for k, v in saved.items():
            setattr(style, k, v)
    return out_path


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
                        help='exposure window for A, B, D, E (default: '
                             '%(default)s)')
    parser.add_argument('--score-window-minutes', type=float, default=30.0,
                        help='response window for C and C2 (default: '
                             '%(default)s)')
    parser.add_argument('--split', default='discovery',
                        help='cohort split; the hold-out is not reachable')
    parser.add_argument('--cohort', default='pain-study',
                        choices=('pain-study', 'split'),
                        help='pain-study = the 51 subjects of the continuous-'
                             'pain regression, intersected with the split')
    parser.add_argument('--epoch-minutes', type=int, default=None)
    parser.set_defaults(question=QUESTION)
    return parser


def main():
    args = build_parser().parse_args()
    io.warn_if_dirty()
    style.use_poster(True)

    epochs = epoch_meds.load_epochs(split=args.split, cohort=args.cohort,
                                    minutes_before=args.epoch_minutes)
    admin_all = load.load_administrations(paths=config.med_admin_files())
    analgesics = epoch_meds.load_analgesics()

    epochs, n_no_mar = epoch_meds.restrict_to_observable(epochs, admin_all)
    if epochs.empty:
        raise SystemExit('no epoch survived the cohort and MAR filters')

    per_epoch, per_epoch_drug = epoch_meds.exposure_before_epochs(
        epochs, analgesics, window_hours=args.window_hours)
    per_epoch = epoch_meds.subject_deviation(per_epoch, by='subject')
    epochs_dev = epoch_meds.subject_deviation(epochs, by='subject')

    drugs = load.select_drugs(analgesics, drugs=args.drugs)
    class_of = (analgesics.drop_duplicates('drug')
                .set_index('drug')['level2'].to_dict())

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

    a_table, n_subjects = panel_a_table(per_epoch_drug, per_epoch, drugs,
                                        class_of)
    plot_panel_a(a_table, run_dir / 'figA_subjects_by_drug.png',
                 args.window_hours)

    b_table = panel_b_table(per_epoch, per_epoch_drug)
    plot_panel_b(b_table, run_dir / 'figB_medicated_epochs.png',
                 args.window_hours)

    # C and C2 — same curves, two x axes, both at the tighter response window.
    c_summaries, c2_summaries, c_long = {}, {}, []
    for label in list(POSTER_CLASS_ORDER) + [ALL_ANALGESICS]:
        subset = (analgesics if label == ALL_ANALGESICS
                  else analgesics[analgesics['level2'] == label])
        if subset.empty:
            continue
        by_score = epoch_meds.dose_probability_by_value(
            epochs_dev, subset, window_minutes=args.score_window_minutes,
            value_col='pain_score')
        by_dev = epoch_meds.dose_probability_by_value(
            epochs_dev, subset, window_minutes=args.score_window_minutes,
            value_col='pain_deviation')
        c_summaries[label] = panel_c_table(by_score, 'pain_score')
        c2_summaries[label] = panel_c_table(by_dev, 'pain_deviation')
        c_long.append(by_score.assign(med_group=label))

    plot_panel_c(c_summaries, run_dir / 'figC_dose_probability.png',
                 'pain_score', 'Pain score', args.score_window_minutes,
                 'Probability of a dose after a pain score')
    plot_panel_c(c2_summaries,
                 run_dir / 'figC2_dose_probability_deviation.png',
                 'pain_deviation', SUBJECT_CENTRED_LABEL,
                 args.score_window_minutes,
                 'Probability of a dose after a pain score')

    d_paired, d_dropped = epoch_meds.paired_by_subject(per_epoch, 'pain_score')
    d_stats = paired_stats(d_paired)
    plot_paired(d_paired, d_stats, run_dir / 'figD_pain_dosed_vs_undosed.png',
                'Mean pain score', 'Pain in dosed vs undosed epochs')

    e_paired, e_dropped = epoch_meds.paired_by_subject(per_epoch,
                                                       'pain_deviation')
    e_stats = paired_stats(e_paired)
    plot_paired(e_paired, e_stats, run_dir / 'figE_pain_deviation.png',
                SUBJECT_CENTRED_LABEL, 'Pain relative to the subject mean',
                zero_line=True, show_summary=False, show_violin=True,
                show_bracket=True)

    plot_grouped(a_table, b_table, c2_summaries, e_paired, e_stats,
                 run_dir / 'figABC2E_grouped.png', args.score_window_minutes)

    stats = {
        'split': args.split,
        'cohort': args.cohort,
        'window_hours': args.window_hours,
        'score_window_minutes': args.score_window_minutes,
        'n_epochs': int(len(per_epoch)),
        'n_subjects': len(subjects),
        'n_sessions': int(per_epoch.groupby(['subject', 'session']).ngroups),
        'n_epochs_dropped_no_mar': n_no_mar,
        'n_epochs_dosed': int(per_epoch['dosed'].sum()),
        'frac_epochs_dosed': round(float(per_epoch['dosed'].mean()), 4),
        'panel_d': d_stats, 'panel_d_subjects_one_arm_only': d_dropped,
        'panel_e': e_stats, 'panel_e_subjects_one_arm_only': e_dropped,
        'deviation_centred_on': 'subject',
    }

    c_table = pd.concat([s.assign(med_group=k) for k, s in c_summaries.items()],
                        ignore_index=True)
    c2_table = pd.concat([s.assign(med_group=k)
                          for k, s in c2_summaries.items()], ignore_index=True)
    for df, name in ((a_table, 'panelA_subjects_by_drug'),
                     (b_table, 'panelB_medicated_epochs'),
                     (c_table, 'panelC_dose_probability'),
                     (c2_table, 'panelC2_dose_probability_deviation'),
                     (pd.concat(c_long, ignore_index=True),
                      'panelC_per_subject'),
                     (d_paired, 'panelD_paired'),
                     (e_paired, 'panelE_paired')):
        output.write_table(df, run_dir, name, SCRIPT, params=params,
                           parents=parents, extra=stats)
    output.write_table(
        per_epoch[['subject', 'session', 'epoch_id', 'pain_time', 'pain_score',
                   'mean_pain', 'pain_deviation', 'n_doses', 'n_drugs',
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
                'The pain study cohort (51 subjects of the continuous-pain '
                'regression), INTERSECTED with the discovery split so no '
                'cohort file can reach a hold-out subject. The unit is a pain '
                'epoch, so the CLAUDE.md gate applies even though no neural '
                'power is read.'),
            'acetaminophen_note': (
                'ACETAMINOPHEN is single-ingredient only. Hydrocodone-, '
                'oxycodone- and codeine-acetaminophen are separate drug '
                'strings classed as Opioids, so they are counted neither in '
                'the acetaminophen bar nor in the Acetaminophen class.'),
            'interpretation_note': (
                'CONFOUNDED BY INDICATION and not causal: the patient in more '
                'pain is the one who gets dosed, so higher pain in dosed '
                'epochs is the expected artefact, not evidence that analgesia '
                'failed. Nominations, not findings (CLAUDE.md).'),
        })
    io.log_analysis(
        f'poster panels: analgesic exposure around pain epochs, '
        f'{args.window_hours:g} h exposure / {args.score_window_minutes:g} min '
        f'response, {len(per_epoch)} epochs, n={len(subjects)} '
        f'{args.cohort} subjects', run_dir)
    logger.info('figures + provenance -> %s', run_dir)


if __name__ == '__main__':
    main()

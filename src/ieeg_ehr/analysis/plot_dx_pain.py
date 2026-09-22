"""Do patients with an MDD diagnosis report more pain?

This is the STRATUM DESCRIPTION that has to be read before any map split by
diagnosis, and it is a question about the predictor, not about power -- so it
needs no view, no QC mask and no model, and it runs in seconds.

WHY IT COMES FIRST. The diagnosis contrast asks whether pain is encoded
DIFFERENTLY in depressed patients. If depressed patients simply report higher
pain, or report over a narrower range, then a difference in the fitted slope can
come from the predictor's distribution rather than from neural encoding:

  - a DIFFERENT MEAN is largely harmless to the within-subject slope, because
    `NRS_within` is subject-mean-centred and `NRS_submean` carries the between-
    subject contrast as a nuisance term -- but it is still the first thing a
    reader will ask about, so it is measured rather than asserted;
  - a DIFFERENT WITHIN-SUBJECT SPREAD is NOT harmless. The slope is estimated
    over whatever range each patient actually spans, so an arm with less
    within-subject variation gets noisier slopes, and a systematic difference in
    spread produces a systematic difference in precision that looks like a
    difference in effect;
  - a DIFFERENT NUMBER OF RATINGS does the same thing through n.

So all three are plotted, not just the mean.

THE DENOMINATOR IS EVERY CHARTED RATING, not the epochs that survived QC into
the model. That is the right denominator for "does this patient report more
pain" and the wrong one for anything about power; `dx_state.pain_by_subject`
says the same thing where the numbers are produced. A run that wants the
model's epochs instead should read `inventory_subjects.parquet` from the
band-power run, which is built from the scores that entered it.

Usage:

    python -m ieeg_ehr.analysis.plot_dx_pain --question mdd
    python -m ieeg_ehr.analysis.plot_dx_pain --dx-window-days 0   # 'ever'
    # one standalone violin+scatter figure per metric, MDD coded EVER:
    python -m ieeg_ehr.analysis.plot_dx_pain --dx-window-days 0 --figure violin
"""

import argparse
import logging
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import dx_state, reference_run

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_dx_pain.py'

QUESTION = 'mdd'
OUTPUT_TYPE = 'stratum_description'
RUN_NAME = 'dx_pain_levels'

DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. '
              'Not confirmed out of sample.')

#: The four quantities, and what each one would mean for the slope contrast.
METRICS = [
    ('nrs_mean', 'Mean reported pain (NRS)',
     'A between-subject difference. Absorbed by NRS_submean, so it does not '
     'bias the within-subject slope -- but it is the first thing asked about.'),
    ('nrs_sd', 'Within-subject SD of pain',
     'THE ONE THAT MATTERS. The slope is estimated over the range a patient '
     'actually spans, so a systematic difference here is a systematic '
     'difference in precision that can masquerade as a difference in effect.'),
    ('n_reports', 'Number of charted ratings',
     'Acts on the slope through n, the same way spread does.'),
    ('frac_zero', 'Fraction of ratings at NRS = 0',
     'A floor effect concentrated in one arm would shorten its usable range '
     'without changing the SD much.'),
]


def arm_labels(args, n_case, n_ctrl):
    """x-tick labels for (control, case), in that order.

    'control' is wrong here: the negative arm is not a matched control group,
    it is everyone in the cohort without the code. MDD- / MDD+ says exactly
    that and nothing more.
    """
    dx = args.dx.upper()
    return [f'{dx}−\nn={n_ctrl}', f'{dx}+\nn={n_case}']


def compare(pain, labels):
    """Per metric: both strata, the difference, Welch t and Mann-Whitney.

    BOTH tests, because they fail differently: Welch is the right test for a
    mean difference with unequal variances and is what the reader expects, and
    Mann-Whitney does not assume the metric is symmetric -- `n_reports` and
    `frac_zero` are both visibly skewed on this cohort. When the two disagree,
    that disagreement is the finding.
    """
    from scipy import stats

    d = pain.merge(labels[['subject_id', 'dx_state']], on='subject_id', how='inner')
    rows = []
    for col, label, _ in METRICS:
        a = d.loc[d['dx_state'], col].dropna()
        b = d.loc[~d['dx_state'], col].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        # Hedges' g: Cohen's d with the small-sample correction, which matters
        # at n = 17 vs 34 (the factor is ~0.98 here, small but free).
        n1, n2 = len(a), len(b)
        sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1))
                     / (n1 + n2 - 2))
        dcoh = (a.mean() - b.mean()) / sp if sp > 0 else np.nan
        g = dcoh * (1 - 3 / (4 * (n1 + n2) - 9)) if np.isfinite(dcoh) else np.nan
        rows.append({
            'metric': col, 'label': label,
            'n_case': n1, 'n_control': n2,
            'case_mean': float(a.mean()), 'control_mean': float(b.mean()),
            'case_median': float(a.median()), 'control_median': float(b.median()),
            'case_sd': float(a.std(ddof=1)), 'control_sd': float(b.std(ddof=1)),
            'difference': float(a.mean() - b.mean()),
            'hedges_g': float(g),
            'welch_t': float(stats.ttest_ind(a, b, equal_var=False).statistic),
            'welch_p': float(stats.ttest_ind(a, b, equal_var=False).pvalue),
            'mannwhitney_p': float(stats.mannwhitneyu(
                a, b, alternative='two-sided').pvalue),
        })
    return d, pd.DataFrame(rows)


def figure(d, table, out_path, args, n_case, n_ctrl):
    """One column per metric: every subject as a point, plus the group summary.

    RAW POINTS, NOT A BAR CHART. n = 17 and 34; a bar of the mean with an error
    bar would hide the whole distribution and imply a precision these group
    sizes do not have. The strip plot is the data.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)          # jitter only, fixed so it is stable
    fig, axs = plt.subplots(1, len(METRICS), figsize=(3.5 * len(METRICS), 5.6),
                            squeeze=False)
    colors = {False: '#4a7fb5', True: '#b03a2e'}

    for ax, (col, label, _why) in zip(axs[0], METRICS):
        row = table[table['metric'] == col]
        for i, state in enumerate((False, True)):
            v = d.loc[d['dx_state'] == state, col].dropna().to_numpy()
            if not len(v):
                continue
            x = i + rng.uniform(-0.13, 0.13, size=len(v))
            ax.scatter(x, v, s=26, alpha=0.75, color=colors[state],
                       edgecolor='white', linewidth=0.6, zorder=3)
            ax.hlines(np.mean(v), i - 0.28, i + 0.28, color='0.15', lw=2.2,
                      zorder=4)
            ax.hlines(np.median(v), i - 0.22, i + 0.22, color='0.15', lw=1.0,
                      ls=':', zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(arm_labels(args, n_case, n_ctrl), fontsize=9)
        ax.set_xlim(-0.55, 1.55)
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        if len(row):
            r = row.iloc[0]
            ax.set_xlabel(f"diff {r['difference']:+.2f}   g={r['hedges_g']:+.2f}\n"
                          f"Welch p={r['welch_p']:.3g}   MW p={r['mannwhitney_p']:.3g}",
                          fontsize=8.5)

    window = (f'{args.dx_window_days} d before admission'
              if args.dx_window_days else 'ever')
    fig.suptitle(
        f'Do {args.dx.upper()} patients report more pain?  '
        f'({args.dx.upper()} = coded {window}, sources: {args.dx_sources})\n'
        'thick line = mean, dotted = median, one point = one subject',
        fontsize=12)
    fig.tight_layout(rect=(0, 0.13, 1, 0.90))
    fig.text(0.01, 0.005,
             'Every charted rating counts here, not only the epochs that '
             'survived QC into a model -- that is the right denominator for '
             '"does this patient report more pain" and the wrong one for '
             'anything about power. The SECOND panel is the one that bears on '
             'the slope contrast: a within-subject spread difference changes '
             'how precisely each arm\'s slope can be estimated, whereas a mean '
             'difference is absorbed by NRS_submean. '
             f'Two-sided tests, uncorrected across the {len(METRICS)} metrics.\n'
             f'{DISCLAIMER}',
             fontsize=6.5, va='bottom', ha='left', color='0.35', wrap=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def figure_violin(d, table, out_path, args, n_case, n_ctrl, metric='nrs_mean'):
    """ONE metric, as a violin with every subject drawn on top of it.

    The violin is a kernel density and at n = 17 vs 34 it is a smoothing
    assumption, not data -- so the points stay on top of it and the mean and
    median stay drawn as lines. The violin carries the shape; the scatter
    carries the evidence.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    label = dict((c, lb) for c, lb, _ in METRICS)[metric]
    ylabel = {'nrs_mean': 'Mean reported pain (NRS, 0-10)',
              'nrs_sd': 'Within-subject SD of reported pain (NRS)',
              'n_reports': 'Number of charted ratings',
              'frac_zero': 'Fraction of ratings at NRS = 0'}.get(metric, label)
    headline = {
        'nrs_mean': 'report more pain?',
        'nrs_sd': 'vary more in the pain they report?',
        'n_reports': 'have more charted pain ratings?',
        'frac_zero': 'report no pain more often?',
    }.get(metric, f'differ in {label.lower()}?')
    # Which caveat the reader needs depends on which metric is drawn: a mean
    # difference is absorbed by NRS_submean, a spread difference is not.
    caveat = ('A mean difference is largely absorbed by NRS_submean in the '
              'slope models; the quantity that bears hardest on the slope '
              'contrast is within-subject SPREAD, which is in the companion '
              'table.'
              if metric != 'nrs_sd' else
              'THIS is the quantity that bears hardest on the slope contrast: '
              'each patient\'s slope is estimated over the range they actually '
              'span, so a systematic spread difference is a systematic '
              'precision difference that can masquerade as a difference in '
              'neural encoding.')

    rng = np.random.default_rng(0)          # jitter only, fixed so it is stable
    fig, ax = plt.subplots(figsize=(5.4, 5.6))
    colors = {False: '#4a7fb5', True: '#b03a2e'}

    values, positions = [], []
    for i, state in enumerate((False, True)):
        v = d.loc[d['dx_state'] == state, metric].dropna().to_numpy()
        if not len(v):
            continue
        values.append(v)
        positions.append(i)

    parts = ax.violinplot(values, positions=positions, widths=0.72,
                          showmeans=False, showmedians=False, showextrema=False)
    for body, i in zip(parts['bodies'], positions):
        body.set_facecolor(colors[bool(i)])
        body.set_edgecolor(colors[bool(i)])
        body.set_alpha(0.22)
        body.set_linewidth(1.0)

    for v, i in zip(values, positions):
        x = i + rng.uniform(-0.10, 0.10, size=len(v))
        ax.scatter(x, v, s=30, alpha=0.8, color=colors[bool(i)],
                   edgecolor='white', linewidth=0.6, zorder=3)
        ax.hlines(np.mean(v), i - 0.26, i + 0.26, color='0.15', lw=2.2, zorder=4)
        ax.hlines(np.median(v), i - 0.20, i + 0.20, color='0.15', lw=1.0,
                  ls=':', zorder=4)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(arm_labels(args, n_case, n_ctrl), fontsize=10)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=9)
    ax.spines[['top', 'right']].set_visible(False)

    row = table[table['metric'] == metric]
    if len(row):
        r = row.iloc[0]
        ax.set_xlabel(f"diff {r['difference']:+.2f}   g={r['hedges_g']:+.2f}   "
                      f"Welch p={r['welch_p']:.3g}   MW p={r['mannwhitney_p']:.3g}",
                      fontsize=9)

    window = (f'{args.dx_window_days} d before admission'
              if args.dx_window_days else 'ever')
    fig.suptitle(
        f'Do {args.dx.upper()}+ patients {headline}  '
        f'[{args.dx.upper()}+ = coded {window}, sources: {args.dx_sources}]\n'
        'violin = kernel density, thick line = mean, dotted = median, '
        'one point = one subject',
        fontsize=11)
    fig.tight_layout(rect=(0, 0.13, 1, 0.92))
    # Wrapped explicitly rather than with wrap=True: this figure is narrow
    # enough that matplotlib's auto-wrap runs the footnote off the right edge.
    note = textwrap.fill(
        'Every charted rating counts here, not only the epochs that survived '
        'QC into a model -- that is the right denominator for "does this '
        f'patient report more pain" and the wrong one for anything about '
        f'power. {caveat} Two-sided tests, uncorrected across the '
        f'{len(METRICS)} metrics in dx_pain_comparison.csv.', width=108)
    fig.text(0.01, 0.005, f'{note}\n{DISCLAIMER}',
             fontsize=6.5, va='bottom', ha='left', color='0.35')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


#: Reference marks on the recency axis, in days before admission. These are
#: conventions a reader already carries, not anything the data picks out.
RECENCY_GUIDES = [(30, '1 mo'), (90, '3 mo'), (365, '1 y'),
                  (365 * 5, '5 y'), (365 * 10, '10 y')]

#: A code dated exactly ON `session_start` is 0 days before it, which has no
#: place on a log axis. Only exact zeros are moved here, and the floor is set
#: an order of magnitude below the smallest real value on this cohort (~0.35 d)
#: so that clipping never silently swallows a genuine sub-day recency.
RECENCY_FLOOR_DAYS = 0.02

#: Annotate an empty stretch of the recency axis when the jump across it is at
#: least this many fold. A gap this size means every window threshold inside it
#: selects the SAME subjects, which is the most decision-relevant thing the
#: recency distribution can say about `--dx-window-days`.
RECENCY_GAP_RATIO = 10.0


def figure_recency(d, labels, out_path, args, n_case, n_ctrl):
    """How STALE is each MDD label, and does staleness track reported pain?

    This is the picture behind the window parameter. `--dx-window-days 90`
    draws a vertical line through the left panel and calls everything to its
    right a control; running at `0` ('ever') keeps them all as cases. Neither
    is obviously right, and the spread of these points is the reason the
    choice matters.

    LOG AXIS, BECAUSE THE RANGE IS LOGARITHMIC. Recency here runs from days to
    decades -- problem-list entries are not re-dated, so a code can legitimately
    sit years before the admission (dx_state docstring, and data_sop.md 5.2:
    the per-subject offset makes absolute dates fiction but leaves intervals
    true, which is exactly what is plotted). On a linear axis every recent
    subject collapses onto the left spine.

    THE RIGHT PANEL IS THE ONE WITH A CLAIM IN IT. If reported pain fell with
    label staleness, the 'ever' arm would be diluting the contrast and the
    90-day window would be doing real work. If it is flat, the window is
    costing arm size and buying nothing.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    case_color, ctrl_color = '#b03a2e', '#4a7fb5'
    r = labels[labels['dx_state']][['subject_id', 'last_days_before',
                                    'n_clinical_codes']].dropna(
        subset=['last_days_before']).copy()
    n_missing = int(labels['dx_state'].sum()) - len(r)
    r['days'] = r['last_days_before'].clip(lower=RECENCY_FLOOR_DAYS)
    n_floored = int((r['last_days_before'] < RECENCY_FLOOR_DAYS).sum())
    # Billing-only labels are called out because the module docstring is blunt
    # that they are the weakest evidence in the arm; a stale billing-only code
    # is the weakest point on this figure and should be visible as such.
    r['billing_only'] = r['n_clinical_codes'] == 0
    r = r.sort_values('days').reset_index(drop=True)

    # The widest multiplicative hole in the recency distribution. Every window
    # threshold inside it picks the same subjects, so if the hole is wide the
    # window parameter has far fewer distinct settings than it appears to.
    gap = None
    v = r['days'].to_numpy()
    if len(v) > 2:
        ratios = v[1:] / np.maximum(v[:-1], RECENCY_FLOOR_DAYS)
        i = int(np.argmax(ratios))
        if ratios[i] >= RECENCY_GAP_RATIO:
            gap = (float(v[i]), float(v[i + 1]), int(i + 1))

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(12.4, 6.0), gridspec_kw={'width_ratios': [1.15, 1.0]})

    # ---- left: one row per case subject, sorted by recency -----------------
    y = np.arange(len(r))
    axL.axvspan(0, 90, color=case_color, alpha=0.07, zorder=0)
    axL.hlines(y, r['days'].min() * 0.6, r['days'], color='0.75', lw=1.0,
               zorder=1)
    for mask, marker, lbl in ((~r['billing_only'], 'o', 'has a clinical code'),
                              (r['billing_only'], 'D', 'billing code only')):
        if mask.any():
            axL.scatter(r.loc[mask, 'days'], y[mask.to_numpy()], s=34,
                        marker=marker, color=case_color, edgecolor='white',
                        linewidth=0.6, zorder=3, label=lbl)
    for days, lbl in RECENCY_GUIDES:
        axL.axvline(days, color='0.55', lw=0.8, ls=':', zorder=2)
        axL.text(days, len(r) + 0.4, lbl, fontsize=7.5, color='0.4',
                 ha='center', va='bottom')
    axL.set_xscale('log')
    axL.set_xlim(r['days'].min() * 0.6, max(r['days'].max() * 1.6, 400))
    axL.set_ylim(-1, len(r) + 1.5)
    floor_note = (f'; {n_floored} code(s) dated exactly at admission plotted '
                  f'at {RECENCY_FLOOR_DAYS} d' if n_floored else '')
    axL.set_xlabel(f'Days from most recent {args.dx.upper()} code to '
                   f'admission  (log scale{floor_note})', fontsize=9)
    axL.set_ylabel(f'{args.dx.upper()}+ subjects, sorted by recency', fontsize=10)
    axL.set_yticks([])
    axL.spines[['top', 'right', 'left']].set_visible(False)
    axL.tick_params(labelsize=8)
    axL.legend(fontsize=8, loc='lower right', frameon=False)

    within = int((r['last_days_before'] <= 90).sum())
    title = (f'{within} of {len(r)} {args.dx.upper()}+ subjects were last '
             f'coded within 90 d; {len(r) - within} carry only an older code')
    if gap:
        lo, hi, n_below = gap
        title += (f'\nbut the distribution is BIMODAL: nothing between '
                  f'{lo:.2g} d and {hi:.0f} d, so every window in that range '
                  f'picks the same {n_below}')
    axL.set_title(title, fontsize=9.5)

    # A span rather than a shaded region: a second fill on top of the 90 d
    # band produced three overlapping tones and read as the subject of the
    # figure, which it is not -- the points are.
    if gap:
        lo, hi, _ = gap
        y_arrow = len(r) - 1.2
        axL.annotate('', xy=(lo, y_arrow), xytext=(hi, y_arrow),
                     arrowprops=dict(arrowstyle='<->', color='0.4', lw=1.1))
        axL.text(np.sqrt(lo * hi), y_arrow - 0.5,
                 f'no subject coded in here ({lo:.2g}–{hi:.0f} d)',
                 fontsize=8, color='0.4', ha='center', va='top')

    # ---- right: does a stale label go with less reported pain? -------------
    m = r.merge(d[['subject_id', 'nrs_mean']], on='subject_id', how='inner')
    ctrl = d.loc[~d['dx_state'], 'nrs_mean'].dropna()
    if len(ctrl):
        axR.axhspan(ctrl.mean() - ctrl.std(ddof=1),
                    ctrl.mean() + ctrl.std(ddof=1),
                    color=ctrl_color, alpha=0.12, zorder=0)
        axR.axhline(ctrl.mean(), color=ctrl_color, lw=1.6, zorder=1)
        axR.text(0.015, ctrl.mean(), f'{args.dx.upper()}− mean ± SD '
                 f'(n={n_ctrl})', transform=axR.get_yaxis_transform(),
                 fontsize=8, color=ctrl_color, va='bottom', ha='left')
    axR.axvspan(0, 90, color=case_color, alpha=0.07, zorder=0)
    axR.scatter(m['days'], m['nrs_mean'], s=40, color=case_color,
                edgecolor='white', linewidth=0.6, zorder=3)
    for days, _lbl in RECENCY_GUIDES:
        axR.axvline(days, color='0.55', lw=0.8, ls=':', zorder=2)

    # Spearman, not Pearson: recency is heavily right-skewed even in logs and
    # n is 26, so the rank statistic is the honest one.
    sub = f'n={len(m)}'
    if len(m) > 2:
        from scipy import stats
        rho, p = stats.spearmanr(m['days'], m['nrs_mean'])
        sub = f'Spearman rho={rho:+.2f}, p={p:.3g}, n={len(m)}'
    axR.set_xscale('log')
    if len(m):
        axR.set_xlim(m['days'].min() * 0.6, max(m['days'].max() * 1.6, 400))
    axR.set_xlabel(f'Days from most recent {args.dx.upper()} code to '
                   'admission (log scale)', fontsize=9)
    axR.set_ylabel('Mean reported pain (NRS, 0-10)', fontsize=10)
    axR.set_title(f'Does a staler label go with less reported pain?\n{sub}',
                  fontsize=10)
    axR.spines[['top', 'right']].set_visible(False)
    axR.tick_params(labelsize=8)

    window = (f'{args.dx_window_days} d before admission'
              if args.dx_window_days else 'ever')
    fig.suptitle(f'How recent is the {args.dx.upper()} label?  '
                 f'[{args.dx.upper()}+ = coded {window}, sources: '
                 f'{args.dx_sources}; {n_case} {args.dx.upper()}+ / '
                 f'{n_ctrl} {args.dx.upper()}−]', fontsize=12)
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))

    stale = ('Recency is the quantity the --dx-window-days parameter '
             'thresholds on: at a 90 d window every subject to the RIGHT of '
             'the shaded band is relabelled a control. MDD is recurrent and '
             'problem-list entries are not re-dated, so an old code is not '
             'evidence of no current depression -- which is why the wide '
             '"ever" arm is the primary here and the narrow window is the '
             'sensitivity check. Intervals are exact under the per-subject '
             'time offset even though absolute dates are not.')
    n_subday = int((r['last_days_before'] < 1).sum())
    if n_subday:
        stale += (f' READ THE LEFT CLUSTER CAREFULLY: {n_subday} subject(s) '
                  'have their most recent code dated LESS THAN A DAY before '
                  'session_start. The window is right-closed at session_start '
                  'precisely so the admission cannot define its own '
                  'predictor, but EHR dates appear to be day-resolution, so a '
                  'code entered during the admission day can still land just '
                  'inside it. Those labels may be the encounter the iEEG '
                  'comes from, not prior history.')
    if n_missing:
        stale += (f' {n_missing} {args.dx.upper()}+ subject(s) omitted: a '
                  'matching code with no parseable date.')
    fig.text(0.01, 0.005, f'{textwrap.fill(stale, width=168)}\n{DISCLAIMER}',
             fontsize=6.5, va='bottom', ha='left', color='0.35')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dx', choices=list(dx_state.CONDITIONS), default='mdd')
    ap.add_argument('--dx-window-days', type=int,
                    default=dx_state.DEFAULT_WINDOW_DAYS,
                    help='Days before session_start in which a code counts. '
                         '0 means EVER.')
    ap.add_argument('--dx-sources', choices=list(dx_state.SOURCE_SETS),
                    default='any')
    ap.add_argument('--reference-run', default=str(reference_run.CONTPAIN_HEATMAP),
                    help='Where the subject list comes from, so this figure '
                         'describes the SAME cohort the models are fitted on.')
    ap.add_argument('--figure', choices=('panels', 'violin', 'recency'),
                    default='panels',
                    help="'panels' = all four metrics in one strip-plot "
                         "figure; 'violin' = ONE STANDALONE FIGURE PER "
                         "--metric, each a violin with the subjects "
                         "scattered on it; 'recency' = how stale each case's "
                         "most recent code is, and whether that tracks pain "
                         "(read it at --dx-window-days 0).")
    ap.add_argument('--metric', nargs='+', choices=[c for c, _, _ in METRICS],
                    default=[c for c, _, _ in METRICS],
                    help='Which metrics --figure violin draws, one figure '
                         'each. Ignored for panels.')
    ap.add_argument('--question', default=QUESTION)
    ap.add_argument('--run-name', default=RUN_NAME)
    ap.add_argument('--run-dir', default=None)
    args = ap.parse_args()

    ref = reference_run.load(args.reference_run)
    subjects = sorted(ref.subjects)
    bare = sorted(str(s).replace('sub-', '') for s in subjects)
    logger.info('cohort from %s: %d subjects', args.reference_run, len(bare))

    dx_all = dx_state.load_diagnoses(subjects=bare)
    labels, detail = dx_state.subject_labels(
        dx_all, condition=args.dx, window_days=args.dx_window_days,
        sources=args.dx_sources)
    labels = labels[labels['subject_id'].isin({f'sub-{s}' for s in bare})]
    pain = dx_state.pain_by_subject(subjects=bare)

    d, table = compare(pain, labels)
    # Recency rides along in the per-subject CSV rather than in a file of its
    # own: it is one more column about the same subjects, and the figure that
    # uses it should not be the only place the numbers exist.
    d = d.merge(labels[['subject_id', 'last_days_before', 'first_days_before',
                        'n_clinical_codes', 'n_billing_codes']],
                on='subject_id', how='left')
    n_case = int(d['dx_state'].sum())
    n_ctrl = int((~d['dx_state']).sum())
    logger.info('\n%s', table.to_string(index=False))

    run_dir = (Path(args.run_dir) if args.run_dir else
               config.analysis_run_dir(question=args.question,
                                       output_type=OUTPUT_TYPE,
                                       run_name=args.run_name))
    run_dir.mkdir(parents=True, exist_ok=True)

    params = {'condition': args.dx, 'window_days': args.dx_window_days,
              'source_set': args.dx_sources, 'figure': args.figure,
              'metrics': list(args.metric) if args.figure == 'violin' else
                         [c for c, _, _ in METRICS],
              'denominator': 'every charted pain rating, not QC-surviving epochs'}
    io.write_table(table, run_dir / 'dx_pain_comparison.csv', params=params,
                   script=SCRIPT, extra={'caveat': dx_state.CONDITIONS[args.dx]
                                         ['description']})
    io.write_table(d, run_dir / 'dx_pain_by_subject.csv', params=params,
                   script=SCRIPT)
    io.write_table(dx_state.stratum_summary(labels, pain),
                   run_dir / 'dx_stratum_summary.csv', params=params, script=SCRIPT)

    if args.figure == 'recency':
        if args.dx_window_days:
            logger.warning('--figure recency under a %d d window: recency is '
                           'bounded by the window by construction and the '
                           'figure will only show that. Re-run with '
                           '--dx-window-days 0 for the real distribution.',
                           args.dx_window_days)
        outs = [figure_recency(d, labels,
                               run_dir / 'fig_dx_recency.png',
                               args, n_case, n_ctrl)]
    elif args.figure == 'violin':
        outs = [figure_violin(d, table,
                              run_dir / f'fig_dx_pain_{m}_violin.png',
                              args, n_case, n_ctrl, metric=m)
                for m in args.metric]
    else:
        outs = [figure(d, table, run_dir / 'fig_dx_pain_levels.png',
                       args, n_case, n_ctrl)]
    for out in outs:
        logger.info('wrote %s', out)

    io.write_run_provenance(
        run_dir, script=SCRIPT, params=params,
        parents=[str(Path(args.reference_run) / 'provenance.json')],
        subjects=subjects,
        extra={'status': DISCLAIMER, 'n_case': n_case, 'n_control': n_ctrl})
    io.log_analysis(f'{args.dx.upper()} vs control pain-report levels, '
                    f'{n_case} vs {n_ctrl} subjects (EXPLORATORY)', run_dir)
    print(run_dir)


if __name__ == '__main__':
    main()

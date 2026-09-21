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
"""

import argparse
import logging
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
        ax.set_xticklabels([f'control\nn={n_ctrl}',
                            f'{args.dx.upper()}\nn={n_case}'], fontsize=9)
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
    n_case = int(d['dx_state'].sum())
    n_ctrl = int((~d['dx_state']).sum())
    logger.info('\n%s', table.to_string(index=False))

    run_dir = (Path(args.run_dir) if args.run_dir else
               config.analysis_run_dir(question=args.question,
                                       output_type=OUTPUT_TYPE,
                                       run_name=args.run_name))
    run_dir.mkdir(parents=True, exist_ok=True)

    params = {'condition': args.dx, 'window_days': args.dx_window_days,
              'source_set': args.dx_sources,
              'denominator': 'every charted pain rating, not QC-surviving epochs'}
    io.write_table(table, run_dir / 'dx_pain_comparison.csv', params=params,
                   script=SCRIPT, extra={'caveat': dx_state.CONDITIONS[args.dx]
                                         ['description']})
    io.write_table(d, run_dir / 'dx_pain_by_subject.csv', params=params,
                   script=SCRIPT)
    io.write_table(dx_state.stratum_summary(labels, pain),
                   run_dir / 'dx_stratum_summary.csv', params=params, script=SCRIPT)

    out = figure(d, table, run_dir / 'fig_dx_pain_levels.png', args, n_case, n_ctrl)
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

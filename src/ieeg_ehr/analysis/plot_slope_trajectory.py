#!/usr/bin/env python3
"""
Does each subject's 1/f slope go UP or DOWN with pain? Three views the violins
cannot give.

WHY THIS EXISTS. `plot_slope_violin.py` draws three MARGINAL distributions side
by side, which is the right figure for "where does each level sit" and the wrong
one for "which direction does a subject move". A violin has thrown the pairing
away by the time it is drawn: the same three violins are produced by every
subject rising slightly and by half rising while half fall. Since the effect
(+0.36 z, 70% of cells) lives entirely in the pairing, the pairing has to be on
the page.

THREE FIGURES, from ONE table, answering three different questions:

  *_paired.png    One line per subject across none -> low -> high, coloured by its
                  own direction. Answers "how many go up, how many go down, and
                  how consistent is it" by inspection. Group mean +- SEM overlaid.

  *_beta.png      THE EFFECT SIZE. Per subject, per region, an OLS of slope on the
                  CONTINUOUS pain score; one dot is one subject's beta. Regions on
                  the x axis so they can be compared at a glance, zero line drawn.
                  This is what CLAUDE.md's sweep rule asks for: per-subject effect
                  sizes plus sign-consistency plus contributing n, and NOT a
                  pooled p-value that ignores per-subject structure.

  *_ribbon.png    Mean +- SEM across subjects against the continuous 0-10 score.
                  Shows whether the relationship is actually MONOTONIC or whether
                  the 3-level summary is hiding a non-monotonicity -- the thing
                  binarizing can never tell you.

WHY REGRESSION IS LEGITIMATE HERE, given the outcome looks categorical.
It only looks categorical because we made it so. `pain_score` is a genuine 0-10
report and is carried per epoch in the slope tables; none/low/high is a
BINARIZATION applied on top of it (registry AXIS 7, which lists `graded` -- keep
continuous, no binarization -- as an axis value in its own right). Regressing on
the score is not imposing linearity on categories, it is declining to discretize.
Measured on this cohort: median 8 distinct scores and 44 epochs per subject, 54
of 56 subjects with >= 3 distinct scores.

WITHIN SUBJECT, ALWAYS. One beta per subject, then the DISTRIBUTION of betas is
the result. A single regression pooling every epoch from every subject would be
pseudo-replication, and worse, it would be open to Simpson's paradox: subjects
differ in both their mean slope and their mean reported pain, so a pooled fit can
show a slope of the opposite sign to every individual subject's.

THE CONFOUNDS ARE UNCHANGED. A tilt that tracks pain is also what sleep and what
EMG would produce (SCRATCHPAD.md, docs/labnotebook/2026-08-06.md). A prettier
figure does not address either. EXPLORATORY, discovery cohort.

Run on Slurm, never the login node:
    python -m ieeg_ehr.analysis.plot_slope_trajectory --view-dir <slope view>
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import view_tables

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_TYPE = 'slope_trajectory'

# Direction colours. Warm = the spectrum FLATTENS as pain rises (slope less
# negative), cool = it steepens. Chosen far apart in hue rather than as a
# light/dark ramp so direction survives a greyscale print and a colour-vision
# deficiency; they are not the pain-level palette, because these encode a
# DERIVED direction, not a level, and reusing that palette would imply otherwise.
UP_COLOUR = '#c1483f'
DOWN_COLOUR = '#2e6f95'


def per_subject_beta(epoch_tables, min_epochs=8, min_distinct_scores=3,
                     min_score_range=0):
    """OLS of slope on pain_score, one fit per (subject, region).

    Centred sums rather than the naive sum-of-products form, matching
    views/aperiodic.py and axes.BaselineAccumulator: the naive form cancels badly
    when the spread is small next to the mean.

    A (subject, region) is refused -- NaN, not a number -- unless it has enough
    epochs, enough DISTINCT pain scores, and a wide enough score RANGE. The last
    two are the floors an epoch count alone would miss: 40 epochs that are all
    0-pain and one 7 give a beta determined entirely by a single epoch, and 40
    epochs spanning only scores 6-7 give a beta divided by a tiny x-variance.

    That is not hypothetical. Measured 2026-08-07 over 658 cells, |beta|
    correlates NEGATIVELY with both n_epochs (-0.23) and score range (-0.30): the
    biggest betas came from the thinnest fits. The largest of all (sub-090 PCC,
    beta 0.092, 13 epochs, 4 distinct scores) implies 0.56 slope units over a
    0->7 swing -- roughly 290% of the entire between-subject spread, which no
    physiology produces. The median |beta| of 0.010 is ~37% of that spread and is
    the honest number.
    """
    df = epoch_tables[['subject_id', 'region', 'epoch_id', 'pain_score', 'slope']].copy()
    df = df.dropna(subset=['pain_score', 'slope'])
    keys = ['subject_id', 'region']
    g = df.groupby(keys, dropna=False)

    df['xc'] = df['pain_score'] - g['pain_score'].transform('mean')
    df['yc'] = df['slope'] - g['slope'].transform('mean')
    df['xy'] = df['xc'] * df['yc']
    df['xx'] = df['xc'] ** 2
    df['yy'] = df['yc'] ** 2

    out = (df.groupby(keys, dropna=False)
           .agg(sxy=('xy', 'sum'), sxx=('xx', 'sum'), syy=('yy', 'sum'),
                n_epochs=('slope', 'size'), n_scores=('pain_score', 'nunique'),
                score_range=('pain_score', lambda s: s.max() - s.min()),
                mean_slope=('slope', 'mean'))
           .reset_index())

    usable = ((out['n_epochs'] >= min_epochs)
              & (out['n_scores'] >= min_distinct_scores)
              & (out['score_range'] > min_score_range)
              & (out['sxx'] > 0))
    with np.errstate(invalid='ignore', divide='ignore'):
        out['beta'] = np.where(usable, out['sxy'] / out['sxx'].replace(0, np.nan), np.nan)
        out['r'] = np.where(usable & (out['syy'] > 0),
                            out['sxy'] / np.sqrt(out['sxx'] * out['syy']), np.nan)
    dropped = int((~usable).sum())
    if dropped:
        logger.info('%d/%d (subject, region) cells refused a beta (< %d epochs, '
                    '< %d distinct pain scores, or score range <= %g)',
                    dropped, len(out), min_epochs, min_distinct_scores,
                    min_score_range)
    out['subject'] = out['subject_id']
    return out.dropna(subset=['beta'])


def plot_paired(subject_values, regions, panels, out_path, ncols=4):
    """One line per subject across the pain levels; colour = that subject's direction."""
    wide = subject_values.pivot_table(index=['subject_id', 'region'],
                                      columns='pain_bin', values='value')
    wide = wide.reindex(columns=panels).dropna()

    nrows = int(np.ceil(len(regions) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.0 * nrows),
                             sharey=True, squeeze=False)
    flat = axes.ravel()
    x = np.arange(len(panels))

    for ax, region in zip(flat, regions):
        rows = wide.xs(region, level='region') if region in wide.index.get_level_values(1) \
            else wide.iloc[0:0]
        if rows.empty:
            ax.set_visible(False)
            continue
        delta = rows[panels[-1]] - rows[panels[0]]
        for (_, r), d in zip(rows.iterrows(), delta):
            ax.plot(x, r.to_numpy(dtype=float),
                    color=(UP_COLOUR if d > 0 else DOWN_COLOUR),
                    linewidth=0.7, alpha=0.45, zorder=2)

        mean = rows.mean()
        sem = rows.std(ddof=1) / np.sqrt(len(rows))
        ax.errorbar(x, mean.to_numpy(dtype=float), yerr=sem.to_numpy(dtype=float),
                    color='black', linewidth=2.0, marker='o', markersize=4,
                    capsize=3, zorder=4)
        ax.axhline(0, color='0.5', linewidth=0.7, linestyle=':', zorder=1)

        n_up = int((delta > 0).sum())
        ax.set_title(f'{region}\n{n_up}/{len(rows)} up ({100 * n_up / len(rows):.0f}%)',
                     fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(panels, fontsize=8)
        ax.tick_params(labelsize=8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    for ax in flat[len(regions):]:
        ax.set_visible(False)

    fig.supylabel('1/f slope (within-subject z)', fontsize=9)
    fig.suptitle('1/f slope trajectory per subject — one line = one subject', fontsize=12)
    fig.text(0.01, -0.01,
             f'One LINE is one subject. Red = that subject\'s slope FLATTENS from '
             f'{panels[0]} to {panels[-1]} (less negative); blue = steepens. Black = '
             f'mean ± SEM across subjects. The percentage is the share of subjects '
             f'moving up, which is the sign-consistency the violins hide.\n'
             f'y is within-subject z, pooled across levels so no level is its own '
             f'reference. EXPLORATORY, discovery cohort — NOMINATIONS, NOT FINDINGS.',
             ha='left', va='top', fontsize=6, color='0.25')
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info('Wrote %s', out_path)


def plot_beta(betas, regions, out_path, min_subjects):
    """One panel, regions on x, one dot per subject: the per-subject regression beta."""
    import seaborn as sns

    keep = [r for r in regions
            if betas.loc[betas['region'] == r, 'beta'].notna().sum() >= min_subjects]
    data = betas[betas['region'].isin(keep)]

    fig, ax = plt.subplots(figsize=(1.0 * len(keep) + 3, 6))
    sns.violinplot(data=data, x='region', y='beta', order=keep, ax=ax,
                   inner='quartile', color='0.88', cut=0, linewidth=1, saturation=1)
    for i, region in enumerate(keep):
        vals = data.loc[data['region'] == region, 'beta'].to_numpy()
        offsets = (np.linspace(-0.16, 0.16, len(vals)) if len(vals) > 1
                   else np.zeros(1))
        ax.scatter(i + offsets, vals, s=16, zorder=3, linewidths=0.3,
                   edgecolors='black',
                   c=[UP_COLOUR if v > 0 else DOWN_COLOUR for v in vals])

    ax.axhline(0, color='black', linewidth=1.0, zorder=2)
    labels = []
    for region in keep:
        vals = data.loc[data['region'] == region, 'beta']
        labels.append(f'{region}\nn={len(vals)}  {100 * (vals > 0).mean():.0f}%↑')
    ax.set_xticks(range(len(keep)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_xlabel('')
    ax.set_ylabel('β — change in 1/f slope per 1 point of reported pain', fontsize=9)
    ax.set_title('Per-subject regression of 1/f slope on the CONTINUOUS pain score\n'
                 'one dot = one subject', fontsize=11)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

    fig.text(0.01, -0.02,
             'β > 0 (red) = that subject\'s spectrum FLATTENS as reported pain rises. '
             'Each β is one OLS fit within one subject and one region, on the raw 0-10 '
             'score — never a pooled fit across subjects, which would be '
             'pseudo-replication and open to Simpson\'s paradox.\n'
             'The %↑ under each region is the sign-consistency across subjects, which '
             'is the statistic that matters here, not a p-value. '
             'EXPLORATORY, discovery cohort — NOMINATIONS, NOT FINDINGS.',
             ha='left', va='top', fontsize=6, color='0.25')
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info('Wrote %s', out_path)
    return keep


def plot_ribbon(epoch_z, regions, out_path, min_subjects, ncols=4):
    """Mean +- SEM across subjects against the CONTINUOUS pain score.

    Two-stage on purpose: collapse to one value per (subject, score) FIRST, then
    average across subjects. Averaging the raw epochs would weight a subject with
    30 epochs at a given score thirty times as heavily as one with a single epoch.
    """
    per_subject = (epoch_z.groupby(['region', 'pain_score', 'subject_id'])['z']
                   .mean().reset_index())
    stats = (per_subject.groupby(['region', 'pain_score'])['z']
             .agg(mean='mean', sd=lambda s: s.std(ddof=1), n='count').reset_index())
    stats['sem'] = stats['sd'] / np.sqrt(stats['n'])
    stats = stats[stats['n'] >= min_subjects]

    nrows = int(np.ceil(len(regions) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.0 * nrows),
                             sharey=True, sharex=True, squeeze=False)
    flat = axes.ravel()

    for ax, region in zip(flat, regions):
        s = stats[stats['region'] == region].sort_values('pain_score')
        ax.axhline(0, color='0.5', linewidth=0.7, linestyle=':', zorder=1)
        if not s.empty:
            ax.fill_between(s['pain_score'], s['mean'] - s['sem'], s['mean'] + s['sem'],
                            color=UP_COLOUR, alpha=0.22, zorder=2)
            ax.plot(s['pain_score'], s['mean'], color=UP_COLOUR, linewidth=1.6,
                    marker='o', markersize=3, zorder=3)
            # n varies a lot across the score axis (10s are rare), and a ribbon
            # that hides that invites reading its right-hand end as if it were as
            # well-supported as its left.
            for _, row in s.iterrows():
                ax.annotate(f'{int(row["n"])}', (row['pain_score'], row['mean']),
                            textcoords='offset points', xytext=(0, 6),
                            ha='center', fontsize=4.5, color='0.4')
        ax.set_title(region, fontsize=9)
        ax.tick_params(labelsize=8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    for ax in flat[len(regions):]:
        ax.set_visible(False)

    fig.supxlabel('Reported pain score (0-10)', fontsize=9)
    fig.supylabel('1/f slope (within-subject z)', fontsize=9)
    fig.suptitle('1/f slope against the continuous pain score — mean ± SEM across subjects',
                 fontsize=12)
    fig.text(0.01, -0.01,
             f'Subject-weighted: collapsed to one value per (subject, score) BEFORE '
             f'averaging, so a subject with many epochs at one score does not dominate '
             f'it. Small grey number = subjects contributing to that point; points '
             f'backed by < {min_subjects} subjects are not drawn.\n'
             f'This is the view that shows whether the relationship is actually '
             f'MONOTONIC — the thing a 3-level binarization cannot tell you. '
             f'EXPLORATORY, discovery cohort — NOMINATIONS, NOT FINDINGS.',
             ha='left', va='top', fontsize=6, color='0.25')
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info('Wrote %s', out_path)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--view-dir', required=True, help='A SLOPE view directory.')
    ap.add_argument('--pain-bin-scheme', choices=list(view_tables.PANELS),
                    default='subject_relative')
    ap.add_argument('--min-subjects', type=int, default=8)
    ap.add_argument('--min-epochs', type=int, default=4,
                    help='Floor for the within-subject z scale (the paired figure).')
    ap.add_argument('--min-epochs-beta', type=int, default=11,
                    help='Floor for a per-subject regression (> 10 epochs).')
    ap.add_argument('--min-distinct-scores', type=int, default=3,
                    help='A beta needs this many DISTINCT pain scores. The floor an '
                         'epoch count alone would miss. Deliberately NOT raised '
                         'alongside the range floor (Sabra, 2026-08-07).')
    ap.add_argument('--min-score-range', type=float, default=4,
                    help='A beta needs a pain-score range GREATER than this. Betas '
                         'correlate -0.30 with range, so the extreme values came '
                         'from cells spanning almost no pain at all.')
    ap.add_argument('--min-none-epochs', type=int, default=5,
                    help='Exclude a SUBJECT with fewer than this many 0-pain epochs; '
                         'their 0-pain reference is noisier than the effect.')
    ap.add_argument('--min-r2', type=float, default=None,
                    help='Drop (epoch, region) rows whose mean FIT r2 is below this.')
    ap.add_argument('--exclude-zero-pain', action='store_true',
                    help='Fit the betas on the NONZERO scores only (1-10). THE key '
                         'sensitivity check, not a cosmetic one: the biggest single '
                         'step is 0 -> any pain, which is also exactly what the sleep '
                         'confound would produce, since a patient who reports a score '
                         'is awake. If beta survives here the relationship is graded '
                         'WITHIN pain and cannot be only a wake/sleep contrast. '
                         'Measured 2026-08-07: it survives, attenuated (median beta '
                         '+0.0068 -> +0.0048, subjects positive 75.9%% -> 69.6%%). '
                         'Affects the beta figure only; the paired and ribbon figures '
                         'still show every level.')
    ap.add_argument('--ncols', type=int, default=4)
    ap.add_argument('--run-name', default=None)
    view_tables.add_output_arguments(ap)
    args = ap.parse_args()

    view_dir = Path(args.view_dir)
    io.warn_if_dirty()

    epoch_tables, epoch_paths = view_tables.load_view_tables(view_dir, 'epochs')
    subject_paths = sorted(view_dir.glob('view_subject_sub-*.parquet'))
    view_params, view = view_tables.view_params_from(subject_paths or epoch_paths)

    if view_params.get('metric') != 'aperiodic_slope':
        raise SystemExit(
            f'--view-dir does not hold slope tables (params[metric]='
            f'{view_params.get("metric")!r}). Build one with '
            '`python -m ieeg_ehr.views.build_pain_epoch_slope`.')

    if args.min_r2 is not None:
        before = len(epoch_tables)
        epoch_tables = epoch_tables[epoch_tables['r2'] >= args.min_r2]
        logger.info('r2 >= %.2f keeps %d/%d rows', args.min_r2, len(epoch_tables), before)

    n_before = epoch_tables['subject_id'].nunique()
    epoch_tables, excluded_thin = view_tables.exclude_thin_baseline_subjects(
        epoch_tables, args.min_none_epochs)
    logger.info('subjects: %d -> %d after the 0-pain floor', n_before,
                epoch_tables['subject_id'].nunique())

    panels = [b for b in config.pain_bin_order(args.pain_bin_scheme)
              if b in set(epoch_tables['pain_bin'])]
    epoch_z = view_tables.within_subject_z(epoch_tables, 'slope',
                                           min_epochs=args.min_epochs)
    subject_values = view_tables.subject_level(epoch_z, panels, value_col='z')

    roi_regions = view_tables.roi_regions_for(view_params)
    regions, per_region, below = view_tables.regions_by_min_subjects(
        subject_values, panels, roi_regions, args.min_subjects)
    if not regions:
        raise SystemExit(f'no region has >= {args.min_subjects} subjects in every level')

    beta_input = (epoch_tables[epoch_tables['pain_score'] > 0] if args.exclude_zero_pain
                  else epoch_tables)
    if args.exclude_zero_pain:
        logger.info('betas on NONZERO scores only: %d of %d epoch rows',
                    len(beta_input), len(epoch_tables))
    betas = per_subject_beta(beta_input, min_epochs=args.min_epochs_beta,
                             min_distinct_scores=args.min_distinct_scores,
                             min_score_range=args.min_score_range)

    if not args.view_scheme:
        args.view_scheme = (view.scheme_code if view is not None else 'unknown')
    run_dir = view_tables.resolve_run_dir(args, OUTPUT_TYPE, view,
                                          run_name=args.run_name or 'discovery')
    logger.info('run dir: %s', run_dir)

    subjects = sorted(subject_values['subject'].unique())
    lo, hi = view_params.get('fit_lo_hz'), view_params.get('fit_hi_hz')

    plot_paired(subject_values, regions, panels, run_dir / 'slope_paired_by_region.png',
                ncols=args.ncols)
    beta_regions = plot_beta(betas, roi_regions, run_dir / 'slope_beta_by_region.png',
                             args.min_subjects)
    ribbon = plot_ribbon(epoch_z, regions, run_dir / 'slope_ribbon_by_score.png',
                         args.min_subjects, ncols=args.ncols)

    # The numbers behind the figures, so a claim can be checked without re-running.
    summary = (betas.groupby('region')
               .agg(n_subjects=('beta', 'size'), beta_mean=('beta', 'mean'),
                    beta_median=('beta', 'median'),
                    frac_positive=('beta', lambda s: float((s > 0).mean())),
                    r_median=('r', 'median'))
               .reset_index().sort_values('frac_positive', ascending=False))
    logger.info('\n%s', summary.round(4).to_string(index=False))

    overall = betas['beta']
    logger.info('ALL (subject, region) betas: n=%d, median %.5f, frac>0 %.3f',
                len(overall), float(overall.median()), float((overall > 0).mean()))
    per_subj = betas.groupby('subject_id')['beta'].mean()
    logger.info('per SUBJECT (mean over regions): n=%d, frac>0 %.3f',
                len(per_subj), float((per_subj > 0).mean()))

    fig_params = {
        'unit_of_observation': 'one subject',
        'beta_definition': 'OLS of per-epoch 1/f slope on the raw 0-10 pain score, '
                           'fit WITHIN one subject and one region; never pooled.',
        'min_epochs_beta': args.min_epochs_beta,
        'min_distinct_scores': args.min_distinct_scores,
        'min_score_range': args.min_score_range,
        'min_none_epochs': args.min_none_epochs,
        'excluded_thin_baseline_subjects': excluded_thin,
        'beta_excludes_zero_pain': args.exclude_zero_pain,
        'beta_frac_positive_cells': float((overall > 0).mean()),
        'beta_frac_positive_subjects': float((per_subj > 0).mean()),
        'sign_convention': 'beta > 0 = spectrum flattens as pain rises',
        'confounds_not_excluded': ['sleep/state (NREM steepens the slope)',
                                   'EMG (broadband HF power flattens the fit)'],
    }
    for name, table in (('subject_betas', betas), ('beta_summary', summary),
                        ('ribbon_points', ribbon)):
        io.write_table(table, run_dir / f'{name}.parquet', kind='table',
                       script='ieeg_ehr/analysis/plot_slope_trajectory.py',
                       params={**view_params, **fig_params},
                       parents=[io.parent_ref(p, digest=False) for p in epoch_paths],
                       subjects=subjects)

    io.write_run_provenance(
        run_dir, script='ieeg_ehr/analysis/plot_slope_trajectory.py',
        params={**vars(args), 'view_params': view_params, **fig_params},
        parents=[io.parent_ref(p, digest=False) for p in epoch_paths + subject_paths],
        subjects=subjects,
        extra={'panels': panels, 'regions_plotted': regions,
               'regions_in_beta_figure': beta_regions,
               'n_subjects_per_region': {r: int(per_region[r]) for r in regions},
               'regions_below_floor': below,
               'status': 'EXPLORATORY nomination, not a finding '
                         '(CLAUDE.md; pending P2.6 FREEZE)'},
    )
    io.log_analysis(f'1/f slope trajectory: paired + per-subject regression on the '
                    f'continuous score + ribbon ({lo:g}-{hi:g} Hz fit), '
                    f'{len(regions)} regions, n={len(subjects)}', run_dir)
    logger.info('3 figures + tables + provenance -> %s', run_dir)


if __name__ == '__main__':
    main()

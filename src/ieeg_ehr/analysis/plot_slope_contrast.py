#!/usr/bin/env python3
"""
The CHANGE in 1/f slope with pain: `high - none` and `low - none`, one violin
each, one dot per subject, one panel per region, against a measured noise floor.

WHAT THIS ADDS TO THE OTHER TWO SLOPE FIGURES. `plot_slope_violin.py` draws three
marginal distributions and `plot_slope_trajectory.py` draws each subject's
trajectory; neither draws the CONTRAST itself as a distribution against zero,
which is the quantity a reader actually wants to test. Here the null hypothesis is
drawn on the page: zero.

NATIVE SLOPE UNITS, not z. The pairing has already removed the between-subject
offset -- that is what a difference does -- so the standardization the other
figures need to be legible would only cost interpretability here. A value of
+0.05 is 0.05 decades of power per decade of frequency, against a between-subject
SD of 0.194 and an epoch-to-epoch SD of 0.149.

THE NOISE FLOOR IS THE POINT. Every panel carries a shaded band and a
`floor_ratio`, following the convention in docs/cluster_permutation.md section 6:
measure how large a contrast the pipeline produces from NOTHING, report the ratio
beside the effect, and apply no hard gate. Two floors are reported because they
answer different questions and they disagree (measured 2026-08-07):

    group mean  +0.0515  vs floor 0.0064  ->  8.1x   THE GROUP EFFECT IS REAL
    median cell  0.0641  vs floor 0.0427  ->  1.5x   ONE DOT IS NOT READABLE

That gap is the honest reading of this figure and is why the band is drawn at the
CELL floor: it is the scale against which an individual dot should be judged, and
a reader looking at dots will otherwise judge them against the group.

ELIGIBILITY (agreed 2026-08-07, see view_tables.exclude_thin_baseline_subjects):
subjects need >= 5 zero-pain epochs, and a cell needs >= 10 epochs in the pain bin
being contrasted. The zero-pain floor is a SUBJECT-level criterion because a thin
baseline is a property of the subject, and its error was measured larger than the
effect itself.

EXPLORATORY, discovery cohort. The sleep and EMG confounds
(docs/labnotebook/2026-08-06.md) are untouched by any of this.

Run on Slurm, never the login node:
    python -m ieeg_ehr.analysis.plot_slope_contrast --view-dir <slope view>
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
from ieeg_ehr.analysis import contrast_stats, view_tables
from ieeg_ehr.features import common

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_TYPE = 'slope_contrast'


def plot_contrast_grid(contrasts, summaries, regions, order, out_path, ncols=4):
    """One panel per region; one violin per contrast; one dot per subject."""
    subjects = sorted(contrasts['subject'].unique())
    colour = common.subject_color_map(subjects)

    nrows = int(np.ceil(len(regions) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.1 * nrows),
                             sharey=True, squeeze=False)
    flat = axes.ravel()

    for ax, region in zip(flat, regions):
        rows = contrasts[contrasts['region'] == region]

        # The null band FIRST, so the dots sit on top of it. Drawn at the CELL
        # floor: this panel shows one dot per subject, so the scale a reader needs
        # is the one an individual cell is judged against, not the group's.
        cell_floor = summaries.loc[summaries['region'] == region, 'floor_cell']
        if len(cell_floor) and np.isfinite(cell_floor.iloc[0]):
            f = float(cell_floor.iloc[0])
            ax.axhspan(-f, f, color='0.75', alpha=0.35, zorder=0, linewidth=0)

        ax.axhline(0, color='black', linewidth=0.9, zorder=1)
        # `pain_bin` is the column the shared violin helper keys on; the contrast
        # label is put there so this figure can reuse it unchanged.
        drawn = rows.rename(columns={'contrast': 'pain_bin'})
        common.draw_seaborn_violin_with_subject_dots(
            ax, drawn, colour, value_col='value', pain_bins=order)

        bits = []
        for contrast in order:
            s = summaries[(summaries['region'] == region)
                          & (summaries['contrast'] == contrast)]
            if s.empty:
                continue
            r = s.iloc[0]
            bits.append(f'{contrast.split("-")[0]}: {100 * r["frac_positive"]:.0f}%↑ '
                        f'{r["floor_ratio"]:.1f}×')
        n = rows['subject'].nunique()
        ax.set_title(f'{region}  (n={n})\n' + '  '.join(bits), fontsize=7.5)
        ax.tick_params(labelsize=8)
        ax.set_ylabel('')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    for ax in flat[len(regions):]:
        ax.set_visible(False)

    fig.supylabel('Change in 1/f slope (decades power per decade Hz)', fontsize=9)
    fig.suptitle('Change in 1/f slope vs the subject\'s own 0-pain epochs', fontsize=12)
    fig.text(0.01, -0.01,
             'One dot = one subject. Positive = that subject\'s spectrum FLATTENS '
             'relative to their own 0-pain epochs. Grey band = the noise floor from '
             'permuting pain labels within each subject and region — the size of a '
             'contrast this pipeline produces from nothing, at the SINGLE-CELL '
             'level. A dot inside the band is not distinguishable from noise.\n'
             'Panel title gives, per contrast, the share of subjects above zero and '
             'the GROUP mean as a multiple of the group floor. The group ratio is '
             'much larger than any single dot\'s — the group effect is supported, an '
             'individual subject\'s is not. '
             'EXPLORATORY, discovery cohort — NOMINATIONS, NOT FINDINGS.',
             ha='left', va='top', fontsize=6, color='0.25')
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info('Wrote %s', out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--view-dir', required=True, help='A SLOPE view directory.')
    ap.add_argument('--pain-bin-scheme', choices=list(view_tables.PANELS),
                    default='subject_relative')
    ap.add_argument('--min-subjects', type=int, default=8)
    ap.add_argument('--min-none-epochs', type=int, default=5,
                    help='Exclude a SUBJECT with fewer than this many 0-pain epochs. '
                         'Their baseline error was measured LARGER than the effect '
                         '(median SEM 0.083 vs an effect of 0.052).')
    ap.add_argument('--min-epochs-per-bin', type=int, default=10,
                    help='A cell needs this many epochs in the PAIN bin being '
                         'contrasted. The 0-pain side is governed by '
                         '--min-none-epochs at the subject level instead.')
    ap.add_argument('--min-r2', type=float, default=None)
    ap.add_argument('--n-perm', type=int, default=contrast_stats.DEFAULT_N_PERM)
    ap.add_argument('--seed', type=int, default=0)
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
    epoch_tables, excluded = view_tables.exclude_thin_baseline_subjects(
        epoch_tables, args.min_none_epochs)
    logger.info('subjects: %d -> %d after the 0-pain floor', n_before,
                epoch_tables['subject_id'].nunique())

    order = [f'{b}-none' for b in config.pain_bin_order(args.pain_bin_scheme)
             if b != 'none' and b in set(epoch_tables['pain_bin'])]
    if not order:
        raise SystemExit('no non-zero pain bin present to contrast against none')
    logger.info('contrasts: %s', order)

    frames, summaries = [], []
    for contrast in order:
        a = contrast.split('-')[0]
        frames.append(contrast_stats.paired_contrast(
            epoch_tables, a, 'none', value_col='slope',
            min_a=args.min_epochs_per_bin, min_b=0))
        logger.info('%s: permuting %d times...', contrast, args.n_perm)
        s = contrast_stats.permutation_null(
            epoch_tables, a, 'none', value_col='slope',
            min_a=args.min_epochs_per_bin, min_b=0,
            n_perm=args.n_perm, seed=args.seed, by_region=True)
        s['contrast'] = contrast
        summaries.append(s)

    contrasts = pd.concat(frames, ignore_index=True)
    summaries = pd.concat(summaries, ignore_index=True)
    if contrasts.empty:
        raise SystemExit('no cells survive the epoch floors')

    roi_regions = view_tables.roi_regions_for(view_params)
    counts = (contrasts.groupby(['region', 'contrast'])['subject'].nunique()
              .unstack('contrast').reindex(columns=order))
    per_region = counts.min(axis=1, skipna=False).fillna(0).astype(int)
    regions = [r for r in roi_regions if per_region.get(r, 0) >= args.min_subjects]
    below = {r: int(per_region.get(r, 0)) for r in roi_regions
             if 0 < per_region.get(r, 0) < args.min_subjects}
    if below:
        logger.info('%d region(s) below the %d-subject floor: %s',
                    len(below), args.min_subjects, below)
    if not regions:
        raise SystemExit(f'no region has >= {args.min_subjects} subjects')
    logger.info('%d region(s) plotted', len(regions))

    if not args.view_scheme:
        args.view_scheme = (view.scheme_code if view is not None else 'unknown')
    run_dir = view_tables.resolve_run_dir(args, OUTPUT_TYPE, view,
                                          run_name=args.run_name or 'discovery')
    logger.info('run dir: %s', run_dir)

    logger.info('\n%s', summaries[summaries['region'].isin(regions)]
                .sort_values(['contrast', 'floor_ratio'], ascending=[True, False])
                [['contrast', 'region', 'n_subjects', 'observed_mean', 'frac_positive',
                  'floor_group', 'floor_ratio', 'perm_p']].round(4).to_string(index=False))

    # The pooled (all-region) floor too: it is the number quoted in prose, and
    # recomputing it by hand later would risk quoting a different one.
    pooled = []
    for contrast in order:
        a = contrast.split('-')[0]
        p = contrast_stats.permutation_null(
            epoch_tables, a, 'none', value_col='slope',
            min_a=args.min_epochs_per_bin, min_b=0,
            n_perm=args.n_perm, seed=args.seed, by_region=False)
        p['contrast'] = contrast
        pooled.append(p)
    pooled = pd.concat(pooled, ignore_index=True)
    logger.info('\nPOOLED across regions:\n%s', pooled[
        ['contrast', 'n_subjects', 'observed_mean', 'frac_positive', 'floor_group',
         'floor_ratio', 'observed_median_abs_cell', 'floor_cell', 'cell_floor_ratio',
         'perm_p']].round(4).to_string(index=False))

    subjects = sorted(contrasts['subject'].unique())
    fig_params = {
        'unit_of_observation': 'one subject (mean over that subject\'s epochs)',
        'value_units': 'decades of power per decade of frequency (native slope)',
        'sign_convention': 'positive = spectrum flattens relative to 0-pain',
        'min_none_epochs': args.min_none_epochs,
        'excluded_thin_baseline_subjects': excluded,
        'min_epochs_per_bin': args.min_epochs_per_bin,
        'n_perm': args.n_perm,
        'seed': args.seed,
        'null_definition':
            'pain_bin labels permuted WITHIN each (subject, region), preserving that '
            'cell\'s epoch count, per-bin group sizes and slope distribution, and '
            'destroying only the association with pain.',
        'floor_note':
            'floor_group = 95th pct of |group mean| under the null; floor_cell = '
            'median |single cell| under the null. Reported, never gated -- any '
            'multiplier would be arbitrary (docs/cluster_permutation.md 6).',
        'confounds_not_excluded': ['sleep/state', 'EMG'],
    }
    for name, table in (('subject_contrasts', contrasts),
                        ('contrast_summary', summaries),
                        ('contrast_summary_pooled', pooled)):
        io.write_table(table, run_dir / f'{name}.parquet', kind='table',
                       script='ieeg_ehr/analysis/plot_slope_contrast.py',
                       params={**view_params, **fig_params},
                       parents=[io.parent_ref(p, digest=False) for p in epoch_paths],
                       subjects=subjects)

    io.write_run_provenance(
        run_dir, script='ieeg_ehr/analysis/plot_slope_contrast.py',
        params={**vars(args), 'view_params': view_params, **fig_params},
        parents=[io.parent_ref(p, digest=False) for p in epoch_paths + subject_paths],
        subjects=subjects,
        extra={'contrasts': order, 'regions_plotted': regions,
               'n_subjects_per_region': {r: int(per_region[r]) for r in regions},
               'regions_below_floor': below, 'roi_regions': roi_regions,
               'status': 'EXPLORATORY nomination, not a finding '
                         '(CLAUDE.md; pending P2.6 FREEZE)'},
    )

    plot_contrast_grid(contrasts, summaries, regions, order,
                       run_dir / 'slope_contrast_by_region.png', ncols=args.ncols)

    io.log_analysis(f'1/f slope CONTRAST violins ({" & ".join(order)}) vs a permutation '
                    f'noise floor, {len(regions)} regions, n={len(subjects)}', run_dir)
    logger.info('figure + tables + provenance -> %s', run_dir)


if __name__ == '__main__':
    main()
